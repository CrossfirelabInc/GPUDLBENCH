#!/usr/bin/env python3
"""
Benchmark 1 — Vision Training (ResNet-50, ResNet-101)

Measures training throughput (images/second) using synthetic data.
Uses proper AMP (torch.amp.autocast + GradScaler) for FP16/BF16.
Based on Lambda Labs methodology.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

# Add parent dir so `benchmarks.config` / `benchmarks.benchmark_utils` resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import (
    VISION_IMAGE_SIZE,
    VISION_MODELS,
    VISION_NUM_CLASSES,
    VISION_TRAIN_ITERATIONS,
    VISION_TRAIN_WARMUP,
    TRAINING_PRECISIONS,
)
from benchmarks.benchmark_utils import (
    add_common_args,
    check_cuda_available,
    filter_precisions_for_gpu,
    get_amp_context,
    get_gpu_info,
    get_grad_scaler,
    get_vision_train_batch_sizes,
    gpu_thermal_warmup,
    GPUMonitor,
    print_gpu_banner,
    save_results,
    set_reproducibility,
    set_tf32,
)


def benchmark_model(
    model_name: str,
    precision: str,
    batch_size: int,
    device: torch.device,
    warmup: int = VISION_TRAIN_WARMUP,
    iterations: int = VISION_TRAIN_ITERATIONS,
) -> dict:
    """Benchmark a single (model, precision, batch_size) configuration."""

    tag = f"{model_name} {precision.upper()} BS={batch_size}"
    print(f"  Testing: {tag}...", end=" ", flush=True)

    try:
        # Create model in FP32 (AMP handles precision internally)
        model = getattr(models, model_name)(weights=None).to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = get_grad_scaler(precision)

        # Synthetic data (always FP32; autocast converts on the fly)
        images = torch.randn(batch_size, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, device=device)
        targets = torch.randint(0, VISION_NUM_CLASSES, (batch_size,), device=device)

        amp_ctx = get_amp_context(precision)

        # Warmup
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize()

        # Benchmark
        times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        throughput = batch_size / avg_time

        print(f"{throughput:.1f} img/s")

        del model, optimizer, criterion, images, targets, scaler
        torch.cuda.empty_cache()

        return {
            "model": model_name,
            "precision": precision.upper(),
            "batch_size": batch_size,
            "avg_time_ms": round(avg_time * 1000, 2),
            "throughput_img_per_sec": round(throughput, 1),
            "status": "success",
        }

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM")
            torch.cuda.empty_cache()
            return {
                "model": model_name,
                "precision": precision.upper(),
                "batch_size": batch_size,
                "avg_time_ms": None,
                "throughput_img_per_sec": None,
                "status": "oom",
            }
        print(f"ERROR: {e}")
        raise


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Training Benchmark")
    add_common_args(parser)
    parser.add_argument("--models", nargs="+", default=VISION_MODELS,
                        help="Models to test (default: resnet50 resnet101)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        help="Override batch sizes (default: auto based on VRAM)")
    parser.add_argument("--warmup", type=int, default=VISION_TRAIN_WARMUP)
    parser.add_argument("--iterations", type=int, default=VISION_TRAIN_ITERATIONS)
    args = parser.parse_args()

    print("=" * 70)
    print("Vision Training Benchmark - ResNet Models")
    print("=" * 70)
    print()

    check_cuda_available()
    device = torch.device(f"cuda:{args.device}")
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)

    set_reproducibility(args.seed)
    set_tf32(True)  # Let TF32 be on (Ampere+), realistic default

    precisions = args.precisions or TRAINING_PRECISIONS
    precisions = filter_precisions_for_gpu(precisions, args.device)
    batch_sizes = args.batch_sizes or get_vision_train_batch_sizes(gpu_info["vram_gb"])

    print(f"Precisions: {precisions}")
    print(f"Batch sizes: {batch_sizes}")
    print()

    # Stabilise GPU clocks/thermals before any timed measurements
    gpu_thermal_warmup(device)

    # Optional power/thermal monitoring
    monitor = None
    if not args.no_monitor:
        monitor = GPUMonitor(device_id=args.device)
        monitor.start()

    results: list[dict] = []

    for model_name in args.models:
        print(f"\n{model_name.upper()}")
        print("-" * 70)
        for prec in precisions:
            for bs in batch_sizes:
                result = benchmark_model(model_name, prec, bs, device,
                                         warmup=args.warmup, iterations=args.iterations)
                results.append(result)
                if result["status"] == "oom":
                    break  # larger batch sizes will also OOM

    # Stop monitoring
    hw_stats = {}
    if monitor:
        hw_stats = monitor.stop()

    # Save
    csv_path, json_path = save_results(
        results, "training_vision", gpu_info,
        extra_meta={"hw_monitor": hw_stats} if hw_stats else None,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("Vision Training Benchmark Complete!")
    print("=" * 70)
    print(f"\nResults saved to:\n  - {csv_path}\n  - {json_path}\n")

    # Summary
    print("Summary (Best Results):")
    print("-" * 70)
    for model_name in args.models:
        for prec in precisions:
            hits = [r for r in results
                    if r["model"] == model_name and r["precision"] == prec.upper()
                    and r["status"] == "success"]
            if hits:
                best = max(hits, key=lambda x: x["throughput_img_per_sec"] or 0)
                print(f"  {model_name} {prec.upper()}: {best['throughput_img_per_sec']:.1f} img/s (BS={best['batch_size']})")


if __name__ == "__main__":
    main()
