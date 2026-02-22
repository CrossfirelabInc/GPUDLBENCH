#!/usr/bin/env python3
"""
Benchmark 10 — Multi-GPU Scaling

Training:    DistributedDataParallel (DDP) with NCCL backend
Inference:   DataParallel (DP) — no gradient all-reduce needed
Generation:  Single-GPU only (tensor parallelism required for true multi-GPU)

Scaling efficiency (%) = throughput_N / (N × throughput_1) × 100

Models tested
─────────────
  • ResNet-50   — vision training + inference
  • BERT-base   — NLP training + inference
  • GPT-2 Large — LLM training + inference + token generation

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
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
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
    MULTIGPU_LLM_GEN_TOKENS,
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
    """Find an available TCP port for DDP rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _sync_all(device_ids: list[int]) -> None:
    for d in device_ids:
        torch.cuda.synchronize(d)


# ──────────────────────────── model builders ──────────────────────────────────

def _build_resnet50() -> nn.Module:
    return resnet50(weights=None, num_classes=VISION_NUM_CLASSES)


def _build_bert_base() -> nn.Module:
    cfg = BertConfig(**NLP_MODELS["bert-base"],
                     vocab_size=NLP_VOCAB_SIZE, num_labels=2)
    return BertForSequenceClassification(cfg)


def _build_gpt2() -> nn.Module:
    cfg = GPT2Config(**MULTIGPU_LLM_CONFIG)
    return GPT2LMHeadModel(cfg)  # FP32; AMP autocast handles FP16 compute


# ──────────────────────────── micro-step functions ────────────────────────────
# Each returns a scalar loss (forward + backward only, no optimizer step).

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
    "resnet50":  _build_resnet50,
    "bert-base": _build_bert_base,
    "gpt2-large":  _build_gpt2,
}

_MICROSTEPS = {
    "resnet50":  _microstep_resnet,
    "bert-base": _microstep_bert,
    "gpt2-large":  _microstep_gpt2,
}


# ──────────────────────────── DDP training worker ─────────────────────────────

def _ddp_train_worker(rank, world_size, port, model_key, batch_per_gpu,
                      warmup, iterations, seed, grad_accum, result_dict):
    """Spawned DDP training worker — one process per GPU.

    Uses gradient accumulation with ``model.no_sync()`` to overlap
    computation with communication.  Only the last micro-step in each
    accumulation window triggers the NCCL all-reduce, amortising the
    PCIe transfer cost over *grad_accum* forward/backward passes.

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

    # --- helper: one full "iteration" = grad_accum micro-steps + optimizer ---
    def _one_iter():
        optimizer.zero_grad(set_to_none=True)
        for micro in range(grad_accum):
            if micro < grad_accum - 1:
                with model.no_sync():        # skip allreduce
                    micro_fn(model, batch_per_gpu, rank)
            else:
                micro_fn(model, batch_per_gpu, rank)  # allreduce here
        optimizer.step()

    try:
        # Warmup
        for _ in range(warmup):
            _one_iter()

        torch.cuda.synchronize(rank)
        dist.barrier()

        # Timed benchmark
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
    """Single-GPU training with gradient accumulation + AMP.

    Uses the same grad_accum count as DDP so the per-iteration sample
    count is identical and scaling efficiency is an apples-to-apples
    comparison.
    """
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


def _bench_training_ddp(model_key, n_gpus, batch_per_gpu, warmup, iterations,
                        grad_accum=MULTIGPU_GRAD_ACCUM_STEPS,
                        seed=RANDOM_SEED):
    """Multi-GPU DDP training — spawns one process per GPU, returns throughput."""
    port = _find_free_port()
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        _ddp_train_worker,
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


# ──────────────────────────── inference benchmark (DataParallel) ──────────────

def _bench_inference(model_key, device_ids, batch_per_gpu, warmup, iterations):
    """DataParallel inference benchmark (DP is fine for eval — no gradient sync)."""
    n_gpu = len(device_ids)
    total_batch = batch_per_gpu * n_gpu
    primary = torch.device(f"cuda:{device_ids[0]}")

    model = _MODEL_BUILDERS[model_key]().to(primary)
    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    # Pre-generate fixed inputs
    if model_key == "resnet50":
        inputs = torch.randn(total_batch, 3, VISION_IMAGE_SIZE,
                             VISION_IMAGE_SIZE, device=primary)
        run = lambda: model(inputs)
    elif model_key == "bert-base":
        ids = torch.randint(0, NLP_VOCAB_SIZE,
                            (total_batch, NLP_SEQ_LENGTH), device=primary)
        mask = torch.ones(total_batch, NLP_SEQ_LENGTH,
                          dtype=torch.long, device=primary)
        run = lambda: model(input_ids=ids, attention_mask=mask)
    elif model_key == "gpt2-large":
        vocab = MULTIGPU_LLM_CONFIG["vocab_size"]
        ids = torch.randint(0, vocab,
                            (total_batch, MULTIGPU_LLM_SEQ_LENGTH), device=primary)
        run = lambda: model(input_ids=ids)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message=".*gather along dimension 0.*")
        # Warmup
        for _ in range(warmup):
            run()
        _sync_all(device_ids)

        # Timed benchmark
        times: list[float] = []
        for _ in range(iterations):
            _sync_all(device_ids)
            t0 = time.perf_counter()
            run()
            _sync_all(device_ids)
            times.append(time.perf_counter() - t0)

    times.sort()
    med_t = times[len(times) // 2]
    throughput = total_batch / med_t

    del model
    torch.cuda.empty_cache()
    return throughput


# ──────────────────────────── LLM generation benchmark ────────────────────────

def _bench_generation(batch_per_gpu, warmup, iterations, device=0):
    """Single-GPU token generation benchmark (GPT-2 Large).

    Multi-GPU generation requires tensor parallelism — not benchmarked here.
    """
    model = _build_gpt2().to(device)
    model.eval()

    vocab = MULTIGPU_LLM_CONFIG["vocab_size"]
    ids = torch.randint(0, vocab,
                        (batch_per_gpu, MULTIGPU_LLM_SEQ_LENGTH), device=device)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")

        # Warmup
        for _ in range(warmup):
            model.generate(input_ids=ids,
                           max_new_tokens=MULTIGPU_LLM_GEN_TOKENS,
                           do_sample=False,
                           pad_token_id=vocab - 1)
        torch.cuda.synchronize(device)

        # Timed benchmark
        times: list[float] = []
        for _ in range(iterations):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model.generate(input_ids=ids,
                           max_new_tokens=MULTIGPU_LLM_GEN_TOKENS,
                           do_sample=False,
                           pad_token_id=vocab - 1)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

    times.sort()
    med_t = times[len(times) // 2]
    total_new_tokens = MULTIGPU_LLM_GEN_TOKENS * batch_per_gpu
    tokens_per_sec = total_new_tokens / med_t

    del model
    torch.cuda.empty_cache()
    return tokens_per_sec


# ──────────────────────────── main ────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU Scaling Benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    sep = "=" * 72
    print(sep)
    print("Benchmark 10 — Multi-GPU Scaling")
    print(f"Training: DDP (NCCL, grad_accum={MULTIGPU_GRAD_ACCUM_STEPS})  |  Inference: DataParallel")
    print(sep)

    check_cuda_available()
    n_gpus_available = torch.cuda.device_count()
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)
    set_reproducibility(args.seed)
    set_tf32(True)

    print(f"\nAvailable GPUs: {n_gpus_available}")

    # Stabilise GPU clocks/thermals before any timed measurements
    if not args.skip_thermal_warmup:
        gpu_thermal_warmup(torch.device(f"cuda:{args.device}"))

    # Check whether all GPUs are the same model
    gpus_identical = True
    if n_gpus_available >= 2:
        gpu_names = [torch.cuda.get_device_name(i)
                     for i in range(n_gpus_available)]
        for i in range(n_gpus_available):
            vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_names[i]}  ({vram:.1f} GiB)")
        if len(set(gpu_names)) != 1:
            gpus_identical = False
            print("\n  WARNING: GPUs are NOT identical — multi-GPU scaling "
                  "tests will be skipped.")
            print("           Meaningful scaling requires matching GPUs.\n")

    if n_gpus_available < 2:
        print("  NOTE: Only 1 GPU found — will run single-GPU baselines only.")
        print("        Multi-GPU scaling results require ≥ 2 GPUs.\n")

    # GPU counts: 1, 2, 4, ... up to n_gpus_available
    gpu_counts: list[int] = [1]
    if gpus_identical:
        c = 2
        while c <= n_gpus_available:
            gpu_counts.append(c)
            c *= 2

    all_rows: list[dict] = []

    specs = [
        {"key": "resnet50",  "label": "ResNet-50",
         "bpg": MULTIGPU_VISION_BATCH_PER_GPU},
        {"key": "bert-base", "label": "BERT-base",
         "bpg": MULTIGPU_NLP_BATCH_PER_GPU},
        {"key": "gpt2-large",  "label": "GPT-2 Large",
         "bpg": MULTIGPU_LLM_BATCH_PER_GPU, "llm_gen": True},
    ]

    baseline: dict[tuple, float] = {}  # (model_key, mode) -> 1-GPU throughput

    for spec in specs:
        key   = spec["key"]
        label = spec["label"]
        bpg   = spec["bpg"]

        print(f"\n{'─' * 72}")
        print(f"  {label}")
        print(f"{'─' * 72}")

        for n_gpu in gpu_counts:
            device_ids = list(range(n_gpu))
            total_batch = bpg * n_gpu
            gpu_tag = f"[{n_gpu} GPU{'s' if n_gpu > 1 else ''}]"

            # ── Training ──────────────────────────────────────────────
            method_train = "DDP" if n_gpu > 1 else "single"
            print(f"  {gpu_tag}  {label} TRAINING ({method_train})  "
                  f"batch={total_batch} ({bpg}/GPU)...",
                  end=" ", flush=True)
            try:
                if n_gpu == 1:
                    tput = _bench_training_single(
                        key, bpg, MULTIGPU_WARMUP, MULTIGPU_ITERATIONS)
                else:
                    tput = _bench_training_ddp(
                        key, n_gpu, bpg, MULTIGPU_WARMUP, MULTIGPU_ITERATIONS,
                        seed=args.seed)

                if n_gpu == 1:
                    baseline[(key, "train")] = tput
                base = baseline.get((key, "train"), tput)
                eff = tput / (n_gpu * base) * 100 if n_gpu > 1 else 100.0

                suffix = f"  (efficiency: {eff:.0f}%)" if n_gpu > 1 else ""
                print(f"{tput:.1f} samples/s{suffix}")

                all_rows.append({
                    "model": key, "mode": "training", "method": method_train,
                    "n_gpus": n_gpu, "batch_per_gpu": bpg,
                    "total_batch": total_batch,
                    "throughput_samples_per_sec": round(tput, 2),
                    "scaling_efficiency_pct": round(eff, 1),
                    "status": "success",
                })
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                short = str(e).split("\n")[0][:120]
                print(f"OOM: {short}")
                all_rows.append({
                    "model": key, "mode": "training", "n_gpus": n_gpu,
                    "status": f"OOM: {short}",
                })
            except Exception as e:
                torch.cuda.empty_cache()
                # mp.spawn wraps errors; use repr() for non-empty message
                msg = str(e) or repr(e)
                short = msg.strip().split("\n")[0][:200]
                print(f"ERROR: {short}")
                all_rows.append({
                    "model": key, "mode": "training", "n_gpus": n_gpu,
                    "status": f"error: {short}",
                })

            # ── Inference (DataParallel) ──────────────────────────────
            print(f"  {gpu_tag}  {label} INFERENCE (DP)  "
                  f"batch={total_batch} ({bpg}/GPU)...",
                  end=" ", flush=True)
            try:
                tput = _bench_inference(
                    key, device_ids, bpg,
                    MULTIGPU_WARMUP, MULTIGPU_ITERATIONS)

                if n_gpu == 1:
                    baseline[(key, "infer")] = tput
                base = baseline.get((key, "infer"), tput)
                eff = tput / (n_gpu * base) * 100 if n_gpu > 1 else 100.0

                suffix = f"  (efficiency: {eff:.0f}%)" if n_gpu > 1 else ""
                print(f"{tput:.1f} samples/s{suffix}")

                all_rows.append({
                    "model": key, "mode": "inference", "method": "DP",
                    "n_gpus": n_gpu, "batch_per_gpu": bpg,
                    "total_batch": total_batch,
                    "throughput_samples_per_sec": round(tput, 2),
                    "scaling_efficiency_pct": round(eff, 1),
                    "status": "success",
                })
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                short = str(e).split("\n")[0][:120]
                print(f"OOM: {short}")
                all_rows.append({
                    "model": key, "mode": "inference", "n_gpus": n_gpu,
                    "status": f"OOM: {short}",
                })
            except Exception as e:
                torch.cuda.empty_cache()
                short = str(e).split("\n")[0][:120]
                print(f"ERROR: {short}")
                all_rows.append({
                    "model": key, "mode": "inference", "n_gpus": n_gpu,
                    "status": f"error: {short}",
                })

            # ── Token Generation (LLM only, single-GPU) ──────────────
            if spec.get("llm_gen") and n_gpu == 1:
                print(f"  {gpu_tag}  {label} GENERATION  "
                      f"batch={bpg} prompt={MULTIGPU_LLM_SEQ_LENGTH} "
                      f"gen={MULTIGPU_LLM_GEN_TOKENS}...",
                      end=" ", flush=True)
                try:
                    tput = _bench_generation(bpg, 2, 10)
                    print(f"{tput:.1f} tokens/s")
                    all_rows.append({
                        "model": key, "mode": "generation", "method": "single",
                        "n_gpus": 1, "batch_per_gpu": bpg, "total_batch": bpg,
                        "tokens_per_sec": round(tput, 2),
                        "scaling_efficiency_pct": 100.0,
                        "status": "success",
                    })
                except Exception as e:
                    torch.cuda.empty_cache()
                    short = str(e).split("\n")[0][:120]
                    print(f"ERROR: {short}")
                    all_rows.append({
                        "model": key, "mode": "generation", "n_gpus": 1,
                        "status": f"error: {short}",
                    })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Multi-GPU Scaling Summary")
    print(sep)
    fmt_hdr = (f"  {'Model':>12}  {'Mode':>12}  {'Method':>8}  "
               f"{'GPUs':>5}  {'Throughput':>14}  {'Efficiency':>10}")
    print(fmt_hdr)
    print(f"  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*5}  {'─'*14}  {'─'*10}")
    for r in all_rows:
        if r.get("status") == "success":
            if r["mode"] == "generation":
                tput_str = f"{r['tokens_per_sec']:>10.1f} t/s"
            else:
                tput_str = f"{r['throughput_samples_per_sec']:>10.1f} s/s"
            method = r.get("method", "")
            print(f"  {r['model']:>12}  {r['mode']:>12}  {method:>8}  "
                  f"{r['n_gpus']:>5}  "
                  f"{tput_str:>14}  "
                  f"{r['scaling_efficiency_pct']:>9.1f}%")
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
                    "gpu_counts_tested": gpu_counts,
                    "training_method": "DDP (NCCL)",
                    "inference_method": "DataParallel"},
        output_dir=args.output_dir,
    )
    print(f"\nResults saved:\n  CSV  → {csv_path}\n  JSON → {json_path}\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
