#!/usr/bin/env python3
"""
Benchmark 10 — Multi-GPU Training Scaling

Measures how much buying a second identical GPU speeds up training.

Methods compared (for each model):
  1. Single-GPU baseline   — 1 GPU, plain training with grad accumulation
  2. DDP (2 GPU)           — DistributedDataParallel, NCCL all-reduce
  3. FSDP ZeRO-2 (2 GPU)  — SHARD_GRAD_OP, shards gradients & optimizer states

Scaling efficiency (%) = throughput_2GPU / (2 × throughput_1GPU) × 100

Models tested
─────────────
  • ResNet-50   — vision (CNN)
  • BERT-base   — NLP encoder (transformer)
  • GPT-2 Large — LLM decoder (transformer, ~774M params)

If only one GPU is present the benchmark still completes, reporting
single-GPU baselines and noting that multi-GPU results require ≥ 2 GPUs.

Results saved to:
  <output_dir>/multi_gpu_scaling.csv
  <output_dir>/multi_gpu_scaling.json
"""

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.models import resnet50
from transformers import (
    BertConfig, BertForSequenceClassification,
    GPT2Config, GPT2LMHeadModel,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_utils import (
    add_common_args,
    check_cuda_available,
    get_gpu_info,
    gpu_thermal_warmup,
    print_gpu_banner,
    save_results,
    set_reproducibility,
    set_tf32,
)
from benchmarks.config import (
    MULTIGPU_GRAD_ACCUM_STEPS,
    MULTIGPU_ITERATIONS,
    MULTIGPU_LLM_BATCH_PER_GPU,
    MULTIGPU_LLM_CONFIG,
    MULTIGPU_LLM_SEQ_LENGTH,
    MULTIGPU_NLP_BATCH_PER_GPU,
    MULTIGPU_VISION_BATCH_PER_GPU,
    MULTIGPU_WARMUP,
    NLP_MODELS,
    NLP_SEQ_LENGTH,
    NLP_VOCAB_SIZE,
    RANDOM_SEED,
    VISION_IMAGE_SIZE,
    VISION_NUM_CLASSES,
)


# ──────────────────────────── helpers ─────────────────────────────────────────

def _find_free_port() -> int:
    """Find an available TCP port for rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ──────────────────────────── model builders ──────────────────────────────────

def _build_resnet50() -> nn.Module:
    return resnet50(weights=None, num_classes=VISION_NUM_CLASSES)


def _build_bert_base() -> nn.Module:
    cfg = BertConfig(**NLP_MODELS["bert-base"],
                     vocab_size=NLP_VOCAB_SIZE, num_labels=2)
    return BertForSequenceClassification(cfg)


def _build_gpt2() -> nn.Module:
    cfg = GPT2Config(**MULTIGPU_LLM_CONFIG)
    # Suppress "loss_type=None" warning emitted by transformers logger
    if getattr(cfg, "loss_type", "MISSING") is None:
        cfg.loss_type = "ForCausalLMLoss"
    import logging as _logging
    _tf_logger = _logging.getLogger("transformers.modeling_utils")
    _prev_level = _tf_logger.level
    _tf_logger.setLevel(_logging.ERROR)
    try:
        model = GPT2LMHeadModel(cfg)  # FP32; AMP autocast handles FP16 compute
    finally:
        _tf_logger.setLevel(_prev_level)
    return model


# ──────────────────────────── micro-step functions ────────────────────────────
# Each does forward + backward only (no optimizer step).

def _microstep_resnet(model, batch_per_gpu, device):
    x = torch.randn(batch_per_gpu, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE,
                     device=device)
    y = torch.zeros(batch_per_gpu, dtype=torch.long, device=device)
    with torch.amp.autocast("cuda"):
        loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()


def _microstep_bert(model, batch_per_gpu, device):
    ids = torch.randint(0, NLP_VOCAB_SIZE,
                        (batch_per_gpu, NLP_SEQ_LENGTH), device=device)
    mask = torch.ones(batch_per_gpu, NLP_SEQ_LENGTH,
                      dtype=torch.long, device=device)
    labels = torch.zeros(batch_per_gpu, dtype=torch.long, device=device)
    with torch.amp.autocast("cuda"):
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    loss.backward()


def _microstep_gpt2(model, batch_per_gpu, device):
    vocab = MULTIGPU_LLM_CONFIG["vocab_size"]
    ids = torch.randint(0, vocab,
                        (batch_per_gpu, MULTIGPU_LLM_SEQ_LENGTH), device=device)
    labels = torch.randint(0, vocab,
                           (batch_per_gpu, MULTIGPU_LLM_SEQ_LENGTH), device=device)
    with torch.amp.autocast("cuda"):
        loss = model(input_ids=ids, labels=labels).loss
    loss.backward()


# Module-level registries (must be picklable for mp.spawn)
_MODEL_BUILDERS = {
    "resnet50":   _build_resnet50,
    "bert-base":  _build_bert_base,
    "gpt2-large": _build_gpt2,
}

_MICROSTEPS = {
    "resnet50":   _microstep_resnet,
    "bert-base":  _microstep_bert,
    "gpt2-large": _microstep_gpt2,
}


# ──────────────────────────── DDP training worker ─────────────────────────────

def _ddp_train_worker(rank, world_size, port, model_key, batch_per_gpu,
                      warmup, iterations, seed, grad_accum, result_dict):
    """Spawned DDP training worker — one process per GPU.

    Standard DistributedDataParallel with NCCL backend.
    Uses gradient accumulation with ``model.no_sync()`` so only the last
    micro-step triggers the all-reduce, amortising PCIe cost.

    Additional DDP flags:
      • gradient_as_bucket_view — avoids gradient-to-bucket copy
      • bucket_cap_mb=100      — fewer, larger allreduce calls (good for PCIe)
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            device_id=torch.device(f"cuda:{rank}"))
    torch.cuda.set_device(rank)

    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = _MODEL_BUILDERS[model_key]().to(rank)
    model = DDP(model, device_ids=[rank],
                gradient_as_bucket_view=True,
                bucket_cap_mb=100)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    micro_fn = _MICROSTEPS[model_key]

    def _one_iter():
        optimizer.zero_grad(set_to_none=True)
        for micro in range(grad_accum):
            if micro < grad_accum - 1:
                with model.no_sync():
                    micro_fn(model, batch_per_gpu, rank)
            else:
                micro_fn(model, batch_per_gpu, rank)
        optimizer.step()

    try:
        for _ in range(warmup):
            _one_iter()
        torch.cuda.synchronize(rank)
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(iterations):
            _one_iter()
        torch.cuda.synchronize(rank)
        dist.barrier()
        elapsed = time.perf_counter() - t0

        if rank == 0:
            result_dict["elapsed"] = elapsed
    finally:
        dist.destroy_process_group()


# ──────────────────────────── FSDP training worker ────────────────────────────

def _get_wrap_policy(model_key):
    """Return FSDP auto-wrap policy for per-block sharding & comm overlap."""
    if model_key == "gpt2-large":
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        return ModuleWrapPolicy({GPT2Block})
    elif model_key == "bert-base":
        from transformers.models.bert.modeling_bert import BertLayer
        return ModuleWrapPolicy({BertLayer})
    elif model_key == "resnet50":
        from torchvision.models.resnet import Bottleneck
        return ModuleWrapPolicy({Bottleneck})
    return None


def _fsdp_train_worker(rank, world_size, port, model_key, batch_per_gpu,
                       warmup, iterations, seed, grad_accum, result_dict):
    """Spawned FSDP training worker — one process per GPU.

    Uses SHARD_GRAD_OP (ZeRO Stage-2): gradients and optimizer states
    are sharded across GPUs, while parameters stay unsharded between
    forward and backward.  Each GPU holds only 1/N of the gradients,
    freeing VRAM for larger batches.

    Auto-wraps transformer/bottleneck blocks for backward-prefetch overlap.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            device_id=torch.device(f"cuda:{rank}"))
    torch.cuda.set_device(rank)

    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = _MODEL_BUILDERS[model_key]()          # built on CPU
    model = FSDP(model,
                 sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                 auto_wrap_policy=_get_wrap_policy(model_key),
                 forward_prefetch=True,
                 device_id=rank,
                 use_orig_params=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    micro_fn = _MICROSTEPS[model_key]

    def _one_iter():
        optimizer.zero_grad(set_to_none=True)
        for micro in range(grad_accum):
            if micro < grad_accum - 1:
                with model.no_sync():
                    micro_fn(model, batch_per_gpu, rank)
            else:
                micro_fn(model, batch_per_gpu, rank)
        optimizer.step()

    try:
        for _ in range(warmup):
            _one_iter()
        torch.cuda.synchronize(rank)
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(iterations):
            _one_iter()
        torch.cuda.synchronize(rank)
        dist.barrier()
        elapsed = time.perf_counter() - t0

        if rank == 0:
            result_dict["elapsed"] = elapsed
    finally:
        dist.destroy_process_group()


# ──────────────────────────── training benchmarks ─────────────────────────────

def _bench_training_single(model_key, batch_per_gpu, warmup, iterations,
                           grad_accum=MULTIGPU_GRAD_ACCUM_STEPS, device=0):
    """Single-GPU training baseline with gradient accumulation + AMP."""
    model = _MODEL_BUILDERS[model_key]().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    micro_fn = _MICROSTEPS[model_key]

    def _one_iter():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            micro_fn(model, batch_per_gpu, device)
        optimizer.step()

    for _ in range(warmup):
        _one_iter()
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _one_iter()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    samples = iterations * batch_per_gpu * grad_accum
    del model, optimizer
    torch.cuda.empty_cache()
    return samples / elapsed


def _bench_training_multi(method, model_key, n_gpus, batch_per_gpu,
                          warmup, iterations,
                          grad_accum=MULTIGPU_GRAD_ACCUM_STEPS,
                          seed=RANDOM_SEED):
    """Multi-GPU training — ``method`` is 'DDP' or 'FSDP'."""
    worker_fn = _ddp_train_worker if method == "DDP" else _fsdp_train_worker
    port = _find_free_port()
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        worker_fn,
        args=(n_gpus, port, model_key, batch_per_gpu,
              warmup, iterations, seed, grad_accum, result_dict),
        nprocs=n_gpus,
        join=True,
    )

    elapsed = result_dict["elapsed"]
    total_samples = iterations * batch_per_gpu * n_gpus * grad_accum
    throughput = total_samples / elapsed

    torch.cuda.empty_cache()
    return throughput


# ──────────────────────────── main ────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-GPU Training Scaling Benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    sep = "=" * 72
    print(sep)
    print("Benchmark 10 — Multi-GPU Training Scaling")
    print(f"Methods: 1-GPU baseline  |  DDP  |  FSDP ZeRO-2")
    print(f"Grad accumulation steps: {MULTIGPU_GRAD_ACCUM_STEPS}")
    print(sep)

    check_cuda_available()
    n_gpus_available = torch.cuda.device_count()
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)
    set_reproducibility(args.seed)
    set_tf32(True)

    print(f"\nAvailable GPUs: {n_gpus_available}")

    if not args.skip_thermal_warmup:
        gpu_thermal_warmup(torch.device(f"cuda:{args.device}"))

    # Check whether all GPUs are the same model
    can_multi = False
    if n_gpus_available >= 2:
        gpu_names = [torch.cuda.get_device_name(i)
                     for i in range(n_gpus_available)]
        for i in range(n_gpus_available):
            vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_names[i]}  ({vram:.1f} GiB)")
        if len(set(gpu_names)) == 1:
            can_multi = True
        else:
            print("\n  WARNING: GPUs are NOT identical — multi-GPU tests "
                  "will be skipped.")
            print("           Meaningful scaling requires matching GPUs.\n")
    else:
        print("  NOTE: Only 1 GPU — single-GPU baselines only.")
        print("        Multi-GPU scaling requires ≥ 2 GPUs.\n")

    all_rows: list[dict] = []

    specs = [
        {"key": "resnet50",   "label": "ResNet-50",
         "bpg": MULTIGPU_VISION_BATCH_PER_GPU},
        {"key": "bert-base",  "label": "BERT-base",
         "bpg": MULTIGPU_NLP_BATCH_PER_GPU},
        {"key": "gpt2-large", "label": "GPT-2 Large",
         "bpg": MULTIGPU_LLM_BATCH_PER_GPU},
    ]

    # Methods to benchmark: always single; DDP + FSDP if ≥ 2 identical GPUs
    methods = ["single"]
    if can_multi:
        methods += ["DDP", "FSDP"]

    baseline: dict[str, float] = {}  # model_key -> 1-GPU throughput

    for spec in specs:
        key   = spec["key"]
        label = spec["label"]
        bpg   = spec["bpg"]

        print(f"\n{'─' * 72}")
        print(f"  {label}")
        print(f"{'─' * 72}")

        for method in methods:
            n_gpu = 1 if method == "single" else 2
            total_batch = bpg * n_gpu
            gpu_tag = f"[{n_gpu} GPU{'s' if n_gpu > 1 else ''}]"

            print(f"  {gpu_tag}  {label} TRAINING ({method})  "
                  f"batch={total_batch} ({bpg}/GPU)...",
                  end=" ", flush=True)
            try:
                if method == "single":
                    tput = _bench_training_single(
                        key, bpg, MULTIGPU_WARMUP, MULTIGPU_ITERATIONS)
                else:
                    tput = _bench_training_multi(
                        method, key, n_gpu, bpg,
                        MULTIGPU_WARMUP, MULTIGPU_ITERATIONS,
                        seed=args.seed)

                if method == "single":
                    baseline[key] = tput
                base = baseline.get(key, tput)
                eff = tput / (n_gpu * base) * 100 if n_gpu > 1 else 100.0
                speedup = tput / base if base > 0 else 0.0

                if n_gpu > 1:
                    suffix = (f"  (efficiency: {eff:.0f}%, "
                              f"speedup: {speedup:.2f}×)")
                else:
                    suffix = ""
                print(f"{tput:.1f} samples/s{suffix}")

                all_rows.append({
                    "model": key, "mode": "training", "method": method,
                    "n_gpus": n_gpu, "batch_per_gpu": bpg,
                    "total_batch": total_batch,
                    "throughput_samples_per_sec": round(tput, 2),
                    "scaling_efficiency_pct": round(eff, 1),
                    "speedup": round(speedup, 3),
                    "status": "success",
                })
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                short = str(e).split("\n")[0][:120]
                print(f"OOM: {short}")
                all_rows.append({
                    "model": key, "mode": "training", "method": method,
                    "n_gpus": n_gpu, "status": f"OOM: {short}",
                })
            except Exception as e:
                torch.cuda.empty_cache()
                msg = str(e) or repr(e)
                short = msg.strip().split("\n")[0][:200]
                print(f"ERROR: {short}")
                all_rows.append({
                    "model": key, "mode": "training", "method": method,
                    "n_gpus": n_gpu, "status": f"error: {short}",
                })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Multi-GPU Training Scaling Summary")
    print(sep)
    fmt_hdr = (f"  {'Model':>12}  {'Method':>8}  {'GPUs':>5}  "
               f"{'Throughput':>14}  {'Efficiency':>10}  {'Speedup':>8}")
    print(fmt_hdr)
    print(f"  {'─'*12}  {'─'*8}  {'─'*5}  {'─'*14}  {'─'*10}  {'─'*8}")
    for r in all_rows:
        if r.get("status") == "success":
            tput_str = f"{r['throughput_samples_per_sec']:>10.1f} s/s"
            method = r.get("method", "")
            eff_str = f"{r['scaling_efficiency_pct']:>9.1f}%"
            spd_str = f"{r['speedup']:>7.2f}×" if r["n_gpus"] > 1 else "    —  "
            print(f"  {r['model']:>12}  {method:>8}  "
                  f"{r['n_gpus']:>5}  "
                  f"{tput_str:>14}  "
                  f"{eff_str}  {spd_str}")
    print(sep)

    # Normalise row keys so every dict has the same fields (CSV requires it)
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in all_rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    for r in all_rows:
        for k in all_keys:
            r.setdefault(k, "")

    csv_path, json_path = save_results(
        all_rows,
        "multi_gpu_scaling",
        gpu_info,
        extra_meta={"n_gpus_available": n_gpus_available,
                    "can_multi_gpu": can_multi,
                    "methods_tested": methods,
                    "grad_accum_steps": MULTIGPU_GRAD_ACCUM_STEPS},
        output_dir=args.output_dir,
    )
    print(f"\nResults saved:\n  CSV  → {csv_path}\n  JSON → {json_path}\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
