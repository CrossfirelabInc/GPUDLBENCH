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
from pathlib import Path

import torch
import torch.nn as nn

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
    from torchvision.models import resnet50
    model = resnet50(weights=None, num_classes=VISION_NUM_CLASSES)
    return model


def _build_bert_base() -> nn.Module:
    from transformers import BertConfig, BertForSequenceClassification
    cfg = BertConfig(**NLP_MODELS["bert-base"],
                     vocab_size=NLP_VOCAB_SIZE,
                     num_labels=2)
    return BertForSequenceClassification(cfg)


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

    # Warmup
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


def _nlp_loss(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**inputs)
    return out.loss


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
    gpu_thermal_warmup(torch.device(f"cuda:{args.device}"))

    # Check whether all GPUs are identical (same model name and VRAM)
    gpus_identical = True
    if n_gpus_available >= 2:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus_available)]
        gpu_vrams = [
            torch.cuda.get_device_properties(i).total_memory
            for i in range(n_gpus_available)
        ]
        for i in range(n_gpus_available):
            print(f"  GPU {i}: {gpu_names[i]}  "
                  f"({gpu_vrams[i] / 1024**3:.1f} GiB)")
        if len(set(gpu_names)) != 1 or len(set(gpu_vrams)) != 1:
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

    # Summary
    print(f"\n{sep}")
    print("Multi-GPU Scaling Summary")
    print(f"{sep}")
    print(f"  {'Model':>12}  {'Mode':>9}  {'GPUs':>5}  {'Samples/s':>11}  {'Efficiency':>10}")
    print(f"  {'─'*12}  {'─'*9}  {'─'*5}  {'─'*11}  {'─'*10}")
    for r in all_rows:
        if r.get("status") == "success":
            print(f"  {r['model']:>12}  {r['mode']:>9}  "
                  f"{r['n_gpus']:>5}  "
                  f"{r['throughput_samples_per_sec']:>11.1f}  "
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
