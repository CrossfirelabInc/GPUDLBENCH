#!/usr/bin/env python3
"""
Benchmark 8 — Object Detection Training (Faster R-CNN)

Measures training throughput (images/second) for Faster R-CNN with ResNet-50 FPN.
Uses AMP for FP16/BF16 (required because Faster R-CNN internals break with raw .half()).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import (
    DETECTION_IMAGE_SIZE,
    DETECTION_ITERATIONS,
    DETECTION_NUM_CLASSES,
    DETECTION_WARMUP,
    TRAINING_PRECISIONS,
)
from benchmarks.benchmark_utils import (
    add_common_args,
    check_cuda_available,
    filter_precisions_for_gpu,
    get_amp_context,
    get_detection_batch_sizes,
    get_gpu_info,
    get_grad_scaler,
    GPUMonitor,
    print_gpu_banner,
    save_results,
    set_reproducibility,
    set_tf32,
)


def _make_dummy_targets(batch_size: int, device: torch.device, num_boxes: int = 5) -> list:
    """Create valid dummy detection targets."""
    targets = []
    for _ in range(batch_size):
        # Generate valid boxes: (x1, y1, x2, y2) where x2>x1, y2>y1, within image
        x1 = torch.rand(num_boxes, device=device) * (DETECTION_IMAGE_SIZE - 50)
        y1 = torch.rand(num_boxes, device=device) * (DETECTION_IMAGE_SIZE - 50)
        x2 = x1 + torch.rand(num_boxes, device=device) * 49 + 1  # width 1–50
        y2 = y1 + torch.rand(num_boxes, device=device) * 49 + 1  # height 1–50
        x2 = x2.clamp(max=DETECTION_IMAGE_SIZE)
        y2 = y2.clamp(max=DETECTION_IMAGE_SIZE)
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        targets.append({
            "boxes": boxes,
            "labels": torch.randint(1, DETECTION_NUM_CLASSES, (num_boxes,), device=device),
        })
    return targets


def benchmark_model(
    precision: str,
    batch_size: int,
    device: torch.device,
    warmup: int = DETECTION_WARMUP,
    iterations: int = DETECTION_ITERATIONS,
) -> Dict[str, Any]:
    """Benchmark Faster R-CNN at a given precision and batch size."""

    tag = f"Faster R-CNN {precision.upper()} BS={batch_size}"
    print(f"  Testing: {tag}...", end=" ", flush=True)

    try:
        # Always create model in FP32; AMP handles casting
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=DETECTION_NUM_CLASSES)
        model = model.to(device)
        model.train()

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        scaler = get_grad_scaler(precision)
        amp_ctx = get_amp_context(precision)

        # Synthetic data — always FP32 images (AMP casts internally)
        images = [torch.randn(3, DETECTION_IMAGE_SIZE, DETECTION_IMAGE_SIZE, device=device)
                  for _ in range(batch_size)]
        targets = _make_dummy_targets(batch_size, device)

        # Warmup
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            if scaler:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

        torch.cuda.synchronize()

        # Benchmark
        times: List[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            if scaler:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        throughput = batch_size / avg_time
        print(f"{throughput:.2f} img/s")

        del model, optimizer, images, targets, scaler
        torch.cuda.empty_cache()

        return {
            "model": "faster-rcnn-resnet50",
            "precision": precision.upper(),
            "batch_size": batch_size,
            "avg_time_ms": round(avg_time * 1000, 2),
            "throughput_img_per_sec": round(throughput, 2),
            "status": "success",
        }

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM")
            torch.cuda.empty_cache()
            return {
                "model": "faster-rcnn-resnet50",
                "precision": precision.upper(),
                "batch_size": batch_size,
                "avg_time_ms": None,
                "throughput_img_per_sec": None,
                "status": "oom",
            }
        print(f"ERROR: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Object Detection Training Benchmark")
    add_common_args(parser)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=DETECTION_WARMUP)
    parser.add_argument("--iterations", type=int, default=DETECTION_ITERATIONS)
    args = parser.parse_args()

    print("=" * 70)
    print("Object Detection Training Benchmark - Faster R-CNN")
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
    batch_sizes = args.batch_sizes or get_detection_batch_sizes(gpu_info["vram_gb"])

    print(f"Precisions: {precisions}")
    print(f"Batch sizes: {batch_sizes}")
    print()

    monitor = None
    if not args.no_monitor:
        monitor = GPUMonitor(device_id=args.device)
        monitor.start()

    results: List[Dict[str, Any]] = []

    print("FASTER R-CNN")
    print("-" * 70)
    for prec in precisions:
        for bs in batch_sizes:
            result = benchmark_model(prec, bs, device,
                                     warmup=args.warmup, iterations=args.iterations)
            results.append(result)

    hw_stats = monitor.stop() if monitor else {}

    csv_path, json_path = save_results(
        results, "training_detection", gpu_info,
        extra_meta={"hw_monitor": hw_stats} if hw_stats else None,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("Object Detection Training Benchmark Complete!")
    print("=" * 70)
    print(f"\nResults saved to:\n  - {csv_path}\n  - {json_path}\n")

    print("Summary (Best Results):")
    print("-" * 70)
    for prec in precisions:
        hits = [r for r in results if r["precision"] == prec.upper() and r["status"] == "success"]
        if hits:
            best = max(hits, key=lambda x: x["throughput_img_per_sec"] or 0)
            print(f"  Faster R-CNN {prec.upper()}: {best['throughput_img_per_sec']:.2f} img/s (BS={best['batch_size']})")


if __name__ == "__main__":
    main()
