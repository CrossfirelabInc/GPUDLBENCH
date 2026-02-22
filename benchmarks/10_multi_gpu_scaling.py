#!/usr/bin/env python3
"""
Benchmark 10 — Multi-GPU Scaling

Measures training and inference throughput as GPU count scales from 1 to N
using torch.nn.DataParallel (single-process, single-node).  Reports:
  • Absolute throughput (samples/s) per GPU count
  • Scaling efficiency (%)  = throughput_N / (N × throughput_1) × 100

Models tested
─────────────
  • ResNet-50   — vision training + inference
  • BERT-base   — NLP training + inference

If only one GPU is present the benchmark still completes, reporting
single-GPU baselines and noting that multi-GPU results require ≥ 2 GPUs.

Results saved to:
  <output_dir>/multi_gpu_scaling.csv
  <output_dir>/multi_gpu_scaling.json
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet50
from transformers import (
    BertConfig, BertForSequenceClassification,
    LlamaConfig, LlamaForCausalLM,
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
    VISION_IMAGE_SIZE,
    VISION_NUM_CLASSES,
)


# ──────────────────────────── model builders ──────────────────────────────────

def _build_resnet50() -> nn.Module:
    model = resnet50(weights=None, num_classes=VISION_NUM_CLASSES)
    return model


def _build_bert_base() -> nn.Module:
    cfg = BertConfig(**NLP_MODELS["bert-base"],
                     vocab_size=NLP_VOCAB_SIZE,
                     num_labels=2)
    return BertForSequenceClassification(cfg)


def _build_llama() -> nn.Module:
    cfg = LlamaConfig(**MULTIGPU_LLM_CONFIG)
    model = LlamaForCausalLM(cfg)
    return model.half()  # FP16 to fit in VRAM for training


# ──────────────────────────── benchmark core ──────────────────────────────────

def _sync_all(device_ids: list[int]) -> None:
    for d in device_ids:
        torch.cuda.synchronize(d)


def _bench_training(
    model_name: str,
    model_factory,
    device_ids: list[int],
    batch_per_gpu: int,
    warmup: int,
    iterations: int,
    make_inputs_fn,
    compute_loss_fn,
) -> float:
    """Generic DataParallel training benchmark."""
    n_gpu = len(device_ids)
    total_batch = batch_per_gpu * n_gpu
    primary = torch.device(f"cuda:{device_ids[0]}")

    model = model_factory().to(primary)
    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Pre-generate fixed inputs to avoid random-generation overhead in timed loop
    inputs = make_inputs_fn(total_batch, primary)

    print(f"  [{n_gpu} GPU{'s' if n_gpu > 1 else ''}]  {model_name} TRAINING  "
          f"batch={total_batch} ({batch_per_gpu}/GPU)...", end=" ", flush=True)

    # Warmup (suppress DataParallel scalar-gather warning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*gather along dimension 0.*")
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss_fn(model, inputs)
            loss.backward()
            optimizer.step()
        _sync_all(device_ids)

        # Benchmark
        times: list[float] = []
        for _ in range(iterations):
            _sync_all(device_ids)
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss_fn(model, inputs)
            loss.backward()
            optimizer.step()
            _sync_all(device_ids)
            times.append(time.perf_counter() - t0)

    times.sort()
    med_t = times[len(times) // 2]
    throughput = total_batch / med_t
    print(f"{throughput:.1f} samples/s")

    del model, optimizer
    torch.cuda.empty_cache()
    return throughput


def _bench_inference(
    model_name: str,
    model_factory,
    device_ids: list[int],
    batch_per_gpu: int,
    warmup: int,
    iterations: int,
    make_inputs_fn,
) -> float:
    """Generic DataParallel inference benchmark."""
    n_gpu = len(device_ids)
    total_batch = batch_per_gpu * n_gpu
    primary = torch.device(f"cuda:{device_ids[0]}")

    model = model_factory().to(primary)
    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    # Pre-generate fixed inputs to avoid random-generation overhead in timed loop
    inputs = make_inputs_fn(total_batch, primary)

    print(f"  [{n_gpu} GPU{'s' if n_gpu > 1 else ''}]  {model_name} INFERENCE "
          f"batch={total_batch} ({batch_per_gpu}/GPU)...", end=" ", flush=True)

    with torch.no_grad():
        for _ in range(warmup):
            model(**inputs) if isinstance(inputs, dict) else model(inputs)
        _sync_all(device_ids)

        times: list[float] = []
        for _ in range(iterations):
            _sync_all(device_ids)
            t0 = time.perf_counter()
            model(**inputs) if isinstance(inputs, dict) else model(inputs)
            _sync_all(device_ids)
            times.append(time.perf_counter() - t0)

    times.sort()
    med_t = times[len(times) // 2]
    throughput = total_batch / med_t
    print(f"{throughput:.1f} samples/s")

    del model
    torch.cuda.empty_cache()
    return throughput


def _bench_llm_generation(
    model_name: str,
    model_factory,
    device_ids: list[int],
    batch_per_gpu: int,
    warmup: int,
    iterations: int,
    seq_length: int = MULTIGPU_LLM_SEQ_LENGTH,
    gen_tokens: int = MULTIGPU_LLM_GEN_TOKENS,
) -> float:
    """Measure LLM token generation throughput (tokens/sec) across GPUs.

    Uses model.generate() with a prompt of *seq_length* tokens and generates
    *gen_tokens* new tokens.  Reports total tokens/sec across all GPUs.
    """
    n_gpu = len(device_ids)
    total_batch = batch_per_gpu * n_gpu
    primary = torch.device(f"cuda:{device_ids[0]}")

    model = model_factory().to(primary)
    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    input_ids = torch.randint(0, MULTIGPU_LLM_CONFIG["vocab_size"],
                               (total_batch, seq_length), device=primary)
    attention_mask = torch.ones_like(input_ids)

    print(f"  [{n_gpu} GPU{'s' if n_gpu > 1 else ''}]  {model_name} GENERATION "
          f"batch={total_batch} ({batch_per_gpu}/GPU) "
          f"prompt={seq_length} gen={gen_tokens}...", end=" ", flush=True)

    # For generation we need the unwrapped model (DataParallel doesn't support .generate)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    with torch.no_grad():
        # Warmup
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pad_token_id.*")
            for _ in range(warmup):
                raw_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                )
            _sync_all(device_ids)

            # Benchmark
            times: list[float] = []
            for _ in range(iterations):
                _sync_all(device_ids)
                t0 = time.perf_counter()
                raw_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                )
                _sync_all(device_ids)
                times.append(time.perf_counter() - t0)

    times.sort()
    med_t = times[len(times) // 2]
    total_new_tokens = gen_tokens * total_batch
    tokens_per_sec = total_new_tokens / med_t
    print(f"{tokens_per_sec:.1f} tokens/s")

    del model, raw_model
    torch.cuda.empty_cache()
    return tokens_per_sec


# ──────────────────────────── input factories ─────────────────────────────────

def _vision_inputs(batch: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, device=device)


def _vision_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    target = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    logits = model(x)
    return nn.functional.cross_entropy(logits, target)


def _nlp_inputs(batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids":      torch.randint(0, NLP_VOCAB_SIZE, (batch, NLP_SEQ_LENGTH), device=device),
        "attention_mask": torch.ones(batch, NLP_SEQ_LENGTH, dtype=torch.long,   device=device),
        "labels":         torch.zeros(batch, dtype=torch.long,                   device=device),
    }


def _llm_inputs(batch: int, device: torch.device) -> dict[str, torch.Tensor]:
    vocab = MULTIGPU_LLM_CONFIG["vocab_size"]
    return {
        "input_ids":      torch.randint(0, vocab, (batch, MULTIGPU_LLM_SEQ_LENGTH), device=device),
        "attention_mask": torch.ones(batch, MULTIGPU_LLM_SEQ_LENGTH, dtype=torch.long, device=device),
        "labels":         torch.randint(0, vocab, (batch, MULTIGPU_LLM_SEQ_LENGTH), device=device),
    }


def _llm_loss(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**inputs)
    return out.loss.mean()


def _nlp_loss(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**inputs)
    # DataParallel gathers per-GPU scalar losses into a vector;
    # .mean() collapses it back to a scalar for .backward().
    return out.loss.mean()


# ──────────────────────────── main ────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU Scaling Benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    sep = "=" * 72
    print(sep)
    print("Benchmark 10 — Multi-GPU Scaling")
    print("DataParallel training & inference: ResNet-50 + BERT-base")
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
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus_available)]
        for i in range(n_gpus_available):
            vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_names[i]}  ({vram:.1f} GiB)")
        if len(set(gpu_names)) != 1:
            gpus_identical = False
            print("\n  WARNING: GPUs are NOT identical — multi-GPU scaling "
                  "tests will be skipped.")
            print("           DataParallel requires matching GPUs for "
                  "meaningful results.\n")

    if n_gpus_available < 2:
        print("  NOTE: Only 1 GPU found — will run single-GPU baselines only.")
        print("        Multi-GPU scaling results require ≥ 2 GPUs.\n")

    # Determine GPU counts to test (powers of 2 up to n_gpus_available)
    # Only include multi-GPU counts if GPUs are identical
    gpu_counts: list[int] = [1]
    if gpus_identical:
        c = 2
        while c <= n_gpus_available:
            gpu_counts.append(c)
            c *= 2

    all_rows: list[dict] = []

    # Per-model specs: (key, label, factory, train_loss_fn, infer_inputs_fn, train_inputs_fn, bpg)
    specs = [
        {
            "key":          "resnet50",
            "label":        "ResNet-50",
            "factory":      _build_resnet50,
            "inputs_train": _vision_inputs,
            "inputs_infer": _vision_inputs,
            "loss_fn":      _vision_loss,
            "bpg":          MULTIGPU_VISION_BATCH_PER_GPU,
        },
        {
            "key":          "bert-base",
            "label":        "BERT-base",
            "factory":      _build_bert_base,
            "inputs_train": _nlp_inputs,
            "inputs_infer": _nlp_inputs,
            "loss_fn":      _nlp_loss,
            "bpg":          MULTIGPU_NLP_BATCH_PER_GPU,
        },
        {
            "key":          "llama-1b",
            "label":        "Llama-1B",
            "factory":      _build_llama,
            "inputs_train": _llm_inputs,
            "inputs_infer": _llm_inputs,
            "loss_fn":      _llm_loss,
            "bpg":          MULTIGPU_LLM_BATCH_PER_GPU,
            "llm_gen":      True,
        },
    ]

    baseline: dict[str, dict[str, float]] = {}  # {model_key: {"train": x, "infer": x}}

    for spec in specs:
        key    = spec["key"]
        label  = spec["label"]
        print(f"\n{'─'*72}")
        print(f"  {label}")
        print(f"{'─'*72}")

        baseline.setdefault(key, {})

        for n_gpu in gpu_counts:
            device_ids = list(range(n_gpu))

            # Training
            try:
                tput_train = _bench_training(
                    label, spec["factory"], device_ids,
                    spec["bpg"], MULTIGPU_WARMUP, MULTIGPU_ITERATIONS,
                    spec["inputs_train"], spec["loss_fn"],
                )
                if n_gpu == 1:
                    baseline[key]["train"] = tput_train
                eff_train = (tput_train / (n_gpu * baseline[key].get("train", tput_train)) * 100
                             if n_gpu > 1 else 100.0)
                all_rows.append({
                    "model": key,
                    "mode": "training",
                    "n_gpus": n_gpu,
                    "batch_per_gpu": spec["bpg"],
                    "total_batch": spec["bpg"] * n_gpu,
                    "throughput_samples_per_sec": round(tput_train, 2),
                    "scaling_efficiency_pct": round(eff_train, 1),
                    "status": "success",
                })
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                short = str(e).split("\n")[0][:120]
                print(f"    OOM (training, {n_gpu} GPU): {short}")
                all_rows.append({"model": key, "mode": "training", "n_gpus": n_gpu,
                                  "status": f"OOM: {short}"})
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"    ERROR (training, {n_gpu} GPU): {e}")
                all_rows.append({"model": key, "mode": "training", "n_gpus": n_gpu,
                                  "status": f"error: {e}"})

            # Inference
            try:
                tput_infer = _bench_inference(
                    label, spec["factory"], device_ids,
                    spec["bpg"], MULTIGPU_WARMUP, MULTIGPU_ITERATIONS,
                    spec["inputs_infer"],
                )
                if n_gpu == 1:
                    baseline[key]["infer"] = tput_infer
                eff_infer = (tput_infer / (n_gpu * baseline[key].get("infer", tput_infer)) * 100
                             if n_gpu > 1 else 100.0)
                all_rows.append({
                    "model": key,
                    "mode": "inference",
                    "n_gpus": n_gpu,
                    "batch_per_gpu": spec["bpg"],
                    "total_batch": spec["bpg"] * n_gpu,
                    "throughput_samples_per_sec": round(tput_infer, 2),
                    "scaling_efficiency_pct": round(eff_infer, 1),
                    "status": "success",
                })
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                short = str(e).split("\n")[0][:120]
                print(f"    OOM (inference, {n_gpu} GPU): {short}")
                all_rows.append({"model": key, "mode": "inference", "n_gpus": n_gpu,
                                  "status": f"OOM: {short}"})
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"    ERROR (inference, {n_gpu} GPU): {e}")
                all_rows.append({"model": key, "mode": "inference", "n_gpus": n_gpu,
                                  "status": f"error: {e}"})

            # Token generation (LLM models only)
            if spec.get("llm_gen"):
                try:
                    tput_gen = _bench_llm_generation(
                        label, spec["factory"], device_ids,
                        spec["bpg"], 2, 10,
                    )
                    if n_gpu == 1:
                        baseline[key]["generation"] = tput_gen
                    eff_gen = (tput_gen / (n_gpu * baseline[key].get("generation", tput_gen)) * 100
                               if n_gpu > 1 else 100.0)
                    all_rows.append({
                        "model": key,
                        "mode": "generation",
                        "n_gpus": n_gpu,
                        "batch_per_gpu": spec["bpg"],
                        "total_batch": spec["bpg"] * n_gpu,
                        "tokens_per_sec": round(tput_gen, 2),
                        "scaling_efficiency_pct": round(eff_gen, 1),
                        "status": "success",
                    })
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    short = str(e).split("\n")[0][:120]
                    print(f"    OOM (generation, {n_gpu} GPU): {short}")
                    all_rows.append({"model": key, "mode": "generation", "n_gpus": n_gpu,
                                      "status": f"OOM: {short}"})
                except Exception as e:
                    torch.cuda.empty_cache()
                    print(f"    ERROR (generation, {n_gpu} GPU): {e}")
                    all_rows.append({"model": key, "mode": "generation", "n_gpus": n_gpu,
                                      "status": f"error: {e}"})

    # Summary
    print(f"\n{sep}")
    print("Multi-GPU Scaling Summary")
    print(f"{sep}")
    print(f"  {'Model':>12}  {'Mode':>12}  {'GPUs':>5}  {'Throughput':>14}  {'Efficiency':>10}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*5}  {'─'*14}  {'─'*10}")
    for r in all_rows:
        if r.get("status") == "success":
            if r["mode"] == "generation":
                tput_str = f"{r['tokens_per_sec']:>10.1f} t/s"
            else:
                tput_str = f"{r['throughput_samples_per_sec']:>10.1f} s/s"
            print(f"  {r['model']:>12}  {r['mode']:>12}  "
                  f"{r['n_gpus']:>5}  "
                  f"{tput_str:>14}  "
                  f"{r['scaling_efficiency_pct']:>9.1f}%")
    print(sep)

    csv_path, json_path = save_results(
        all_rows,
        "multi_gpu_scaling",
        gpu_info,
        extra_meta={"n_gpus_available": n_gpus_available,
                    "gpu_counts_tested": gpu_counts},
        output_dir=args.output_dir,
    )
    print(f"\nResults saved:\n  CSV  → {csv_path}\n  JSON → {json_path}\n")


if __name__ == "__main__":
    main()
