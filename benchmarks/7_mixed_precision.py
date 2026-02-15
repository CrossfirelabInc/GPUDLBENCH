#!/usr/bin/env python3
"""
Benchmark 7 — Mixed Precision Speedup Analysis

Compares FP32 vs FP16 vs BF16 training performance using proper AMP.
Measures speedup ratios to show Tensor Core utilisation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertForSequenceClassification, BertConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import (
    MIXED_PREC_NLP_BATCH,
    MIXED_PREC_NLP_ITERS,
    MIXED_PREC_VISION_BATCH,
    MIXED_PREC_VISION_ITERS,
    NLP_MODELS,
    NLP_SEQ_LENGTH,
    NLP_VOCAB_SIZE,
    TRAINING_PRECISIONS,
)
from benchmarks.benchmark_utils import (
    add_common_args,
    check_cuda_available,
    filter_precisions_for_gpu,
    get_amp_context,
    get_gpu_info,
    get_grad_scaler,
    GPUMonitor,
    print_gpu_banner,
    save_results,
    set_reproducibility,
    set_tf32,
)


def _time_vision_training(
    model_fn,
    precision: str,
    batch_size: int,
    iterations: int,
    device: torch.device,
) -> float:
    """Return total time (seconds) for `iterations` training steps."""
    model = model_fn(weights=None).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = get_grad_scaler(precision)
    amp_ctx = get_amp_context(precision)

    images = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 1000, (batch_size,), device=device)

    # Warmup
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            loss = criterion(model(images), targets)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            loss = criterion(model(images), targets)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    del model, optimizer, criterion, images, targets, scaler
    torch.cuda.empty_cache()
    return elapsed


def benchmark_vision_mixed_precision(
    precisions: List[str],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Compare precision modes for vision models."""

    print("\n" + "=" * 70)
    print("Vision Models: Mixed Precision Comparison")
    print("=" * 70)
    print()

    vision_models = {"resnet50": models.resnet50, "resnet101": models.resnet101}
    batch_size = MIXED_PREC_VISION_BATCH
    iters = MIXED_PREC_VISION_ITERS
    results: List[Dict[str, Any]] = []

    for model_name, model_fn in vision_models.items():
        print(f"{model_name}:")
        timings: Dict[str, float] = {}

        for prec in precisions:
            print(f"  {prec.upper()}...", end=" ")
            try:
                elapsed = _time_vision_training(model_fn, prec, batch_size, iters, device)
                throughput = (batch_size * iters) / elapsed
                timings[prec] = elapsed
                print(f"{throughput:.1f} img/s")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("OOM")
                    torch.cuda.empty_cache()
                else:
                    raise

        fp32_time = timings.get("fp32")
        for prec in precisions:
            if prec in timings:
                throughput = (batch_size * iters) / timings[prec]
                speedup = fp32_time / timings[prec] if fp32_time and timings[prec] > 0 else 0
                results.append({
                    "model_type": "vision",
                    "model": model_name,
                    "precision": prec.upper(),
                    "throughput": round(throughput, 1),
                    "speedup_vs_fp32": round(speedup, 2),
                })

    return results


def _time_nlp_training(
    config: BertConfig,
    precision: str,
    batch_size: int,
    iterations: int,
    device: torch.device,
) -> float:
    """Return total time (seconds) for NLP training steps."""
    model = BertForSequenceClassification(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = get_grad_scaler(precision)
    amp_ctx = get_amp_context(precision)

    input_ids = torch.randint(0, config.vocab_size, (batch_size, NLP_SEQ_LENGTH), device=device)
    attention_mask = torch.ones(batch_size, NLP_SEQ_LENGTH, dtype=torch.long, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)

    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    del model, optimizer, input_ids, attention_mask, labels, scaler
    torch.cuda.empty_cache()
    return elapsed


def benchmark_nlp_mixed_precision(
    precisions: List[str],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Compare precision modes for NLP models."""

    print("\n" + "=" * 70)
    print("NLP Models: Mixed Precision Comparison")
    print("=" * 70)
    print()

    batch_size = MIXED_PREC_NLP_BATCH
    iters = MIXED_PREC_NLP_ITERS
    results: List[Dict[str, Any]] = []

    for model_name, cfg_dict in NLP_MODELS.items():
        print(f"{model_name}:")

        config = BertConfig(
            vocab_size=NLP_VOCAB_SIZE,
            max_position_embeddings=512,
            num_labels=2,
            **cfg_dict,
        )

        timings: Dict[str, float] = {}
        for prec in precisions:
            print(f"  {prec.upper()}...", end=" ")
            try:
                elapsed = _time_nlp_training(config, prec, batch_size, iters, device)
                throughput = (batch_size * iters) / elapsed
                timings[prec] = elapsed
                print(f"{throughput:.1f} samples/s")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("OOM")
                    torch.cuda.empty_cache()
                else:
                    raise

        fp32_time = timings.get("fp32")
        for prec in precisions:
            if prec in timings:
                throughput = (batch_size * iters) / timings[prec]
                speedup = fp32_time / timings[prec] if fp32_time and timings[prec] > 0 else 0
                results.append({
                    "model_type": "nlp",
                    "model": model_name,
                    "precision": prec.upper(),
                    "throughput": round(throughput, 1),
                    "speedup_vs_fp32": round(speedup, 2),
                })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed Precision Speedup Analysis")
    add_common_args(parser)
    args = parser.parse_args()

    print("=" * 70)
    print("Mixed Precision Speedup Analysis")
    print("FP32 vs FP16 vs BF16 Comparison")
    print("=" * 70)
    print()

    check_cuda_available()
    device = torch.device(f"cuda:{args.device}")
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)

    set_reproducibility(args.seed)
    set_tf32(True)

    precisions = args.precisions or TRAINING_PRECISIONS
    precisions = filter_precisions_for_gpu(precisions, args.device)

    print(f"Precisions: {precisions}")
    print()

    monitor = None
    if not args.no_monitor:
        monitor = GPUMonitor(device_id=args.device)
        monitor.start()

    vision_results = benchmark_vision_mixed_precision(precisions, device)
    nlp_results = benchmark_nlp_mixed_precision(precisions, device)
    all_results = vision_results + nlp_results

    hw_stats = monitor.stop() if monitor else {}

    csv_path, json_path = save_results(
        all_results, "mixed_precision", gpu_info,
        extra_meta={"hw_monitor": hw_stats} if hw_stats else None,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("Mixed Precision Analysis Complete!")
    print("=" * 70)
    print(f"\nResults saved to:\n  - {csv_path}\n  - {json_path}\n")

    print("Summary:")
    print("-" * 70)
    for r in all_results:
        print(f"  {r['model']:20s} {r['precision']:4s}  {r['throughput']:>8.1f} {'img' if r['model_type']=='vision' else 'samples'}/s  ({r['speedup_vs_fp32']:.2f}x vs FP32)")

    fp32_only = [r for r in all_results if r["precision"] == "FP32"]
    non_fp32 = [r for r in all_results if r["precision"] != "FP32"]
    if non_fp32:
        avg_speedup = sum(r["speedup_vs_fp32"] for r in non_fp32) / len(non_fp32)
        print(f"\n  Average mixed-precision speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
