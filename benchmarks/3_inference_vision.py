#!/usr/bin/env python3
"""
Benchmark 3 — Vision Inference (ResNet-50, ResNet-101)

Measures inference latency (ms/image at BS=1) and throughput (images/s).
Uses torch.inference_mode() and AMP autocast for FP16/BF16.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import (
    VISION_IMAGE_SIZE,
    VISION_MODELS,
    VISION_INFER_ITERATIONS,
    VISION_INFER_WARMUP,
    INFERENCE_PRECISIONS,
)
from benchmarks.benchmark_utils import (
    add_common_args,
    check_cuda_available,
    filter_precisions_for_gpu,
    get_amp_context,
    get_gpu_info,
    get_vision_infer_batch_sizes,
    GPUMonitor,
    print_gpu_banner,
    save_results,
    set_reproducibility,
    set_tf32,
)

MODEL_FNS = {
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}


def benchmark_model(
    model_name: str,
    precision: str,
    batch_size: int,
    device: torch.device,
    warmup: int = VISION_INFER_WARMUP,
    iterations: int = VISION_INFER_ITERATIONS,
) -> Dict[str, Any]:
    """Benchmark a single inference configuration."""

    tag = f"{model_name} {precision.upper()} BS={batch_size}"
    print(f"  Testing: {tag}...", end=" ", flush=True)

    try:
        model = MODEL_FNS[model_name](weights=None).to(device)
        model.eval()

        images = torch.randn(batch_size, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE, device=device)
        amp_ctx = get_amp_context(precision)

        # Warmup
        with torch.inference_mode():
            for _ in range(warmup):
                with amp_ctx:
                    _ = model(images)

        torch.cuda.synchronize()

        # Benchmark
        times: List[float] = []
        with torch.inference_mode():
            for _ in range(iterations):
                start = time.perf_counter()
                with amp_ctx:
                    _ = model(images)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        throughput = batch_size / avg_time
        # True latency only meaningful at BS=1; at higher BS report amortized
        latency_per_image = avg_time * 1000 / batch_size

        print(f"{throughput:.1f} img/s, {latency_per_image:.2f} ms/img")

        del model, images
        torch.cuda.empty_cache()

        return {
            "model": model_name,
            "precision": precision.upper(),
            "batch_size": batch_size,
            "avg_time_ms": round(avg_time * 1000, 2),
            "latency_ms_per_image": round(latency_per_image, 2),
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
                "latency_ms_per_image": None,
                "throughput_img_per_sec": None,
                "status": "oom",
            }
        print(f"ERROR: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Inference Benchmark")
    add_common_args(parser)
    parser.add_argument("--models", nargs="+", default=list(VISION_MODELS.keys()))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=VISION_INFER_WARMUP)
    parser.add_argument("--iterations", type=int, default=VISION_INFER_ITERATIONS)
    args = parser.parse_args()

    print("=" * 70)
    print("Vision Inference Benchmark - ResNet Models")
    print("=" * 70)
    print()

    check_cuda_available()
    device = torch.device(f"cuda:{args.device}")
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)

    set_reproducibility(args.seed)
    set_tf32(True)

    precisions = args.precisions or INFERENCE_PRECISIONS
    precisions = filter_precisions_for_gpu(precisions, args.device)
    batch_sizes = args.batch_sizes or get_vision_infer_batch_sizes(gpu_info["vram_gb"])

    print(f"Precisions: {precisions}")
    print(f"Batch sizes: {batch_sizes}")
    print()

    monitor = None
    if not args.no_monitor:
        monitor = GPUMonitor(device_id=args.device)
        monitor.start()

    results: List[Dict[str, Any]] = []

    for model_name in args.models:
        print(f"\n{model_name.upper()}")
        print("-" * 70)
        for prec in precisions:
            for bs in batch_sizes:
                result = benchmark_model(model_name, prec, bs, device,
                                         warmup=args.warmup, iterations=args.iterations)
                results.append(result)

    hw_stats = monitor.stop() if monitor else {}

    csv_path, json_path = save_results(
        results, "inference_vision", gpu_info,
        extra_meta={"hw_monitor": hw_stats} if hw_stats else None,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("Vision Inference Benchmark Complete!")
    print("=" * 70)
    print(f"\nResults saved to:\n  - {csv_path}\n  - {json_path}\n")

    print("Summary (BS=1 Latency, Best Throughput):")
    print("-" * 70)
    for model_name in args.models:
        for prec in precisions:
            bs1 = [r for r in results
                   if r["model"] == model_name and r["precision"] == prec.upper()
                   and r["batch_size"] == 1 and r["status"] == "success"]
            if bs1:
                print(f"  {model_name} {prec.upper()} (BS=1): {bs1[0]['latency_ms_per_image']:.2f} ms/img")
            hits = [r for r in results
                    if r["model"] == model_name and r["precision"] == prec.upper()
                    and r["status"] == "success"]
            if hits:
                best = max(hits, key=lambda x: x["throughput_img_per_sec"] or 0)
                print(f"  {model_name} {prec.upper()} (Best): {best['throughput_img_per_sec']:.1f} img/s (BS={best['batch_size']})")


if __name__ == "__main__":
    main()
