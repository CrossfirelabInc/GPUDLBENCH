#!/usr/bin/env python3
"""
Benchmark 8 — Object Detection Training (Faster R-CNN + Mask R-CNN)

Measures training throughput (images/second) for:
  • Faster R-CNN with ResNet-50 FPN  — 2-stage detection
  • Mask R-CNN with ResNet-50 FPN    — detection + instance segmentation

Uses AMP for FP16/BF16 (required because detection model internals break with
raw .half()).  Both models share the same backbone and FPN, so the comparison
isolates the cost of the mask prediction head.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import (
    DETECTION_IMAGE_SIZE,
    DETECTION_ITERATIONS,
    DETECTION_NUM_CLASSES,
    DETECTION_WARMUP,
    RANDOM_SEED,
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
    gpu_thermal_warmup,
    GPUMonitor,
    print_gpu_banner,
    save_results,
    set_reproducibility,
    set_tf32,
)


def _make_dummy_boxes(batch_size: int, device: torch.device,
                      num_boxes: int = 5) -> list[dict[str, torch.Tensor]]:
    """Create valid dummy bounding-box targets (shared by both models)."""
    targets = []
    for _ in range(batch_size):
        x1 = torch.rand(num_boxes, device=device) * (DETECTION_IMAGE_SIZE - 50)
        y1 = torch.rand(num_boxes, device=device) * (DETECTION_IMAGE_SIZE - 50)
        x2 = x1 + torch.rand(num_boxes, device=device) * 49 + 1
        y2 = y1 + torch.rand(num_boxes, device=device) * 49 + 1
        x2 = x2.clamp(max=DETECTION_IMAGE_SIZE)
        y2 = y2.clamp(max=DETECTION_IMAGE_SIZE)
        targets.append({
            "boxes":  torch.stack([x1, y1, x2, y2], dim=1),
            "labels": torch.randint(1, DETECTION_NUM_CLASSES, (num_boxes,), device=device),
        })
    return targets


def _make_dummy_targets(batch_size: int, device: torch.device,
                        num_boxes: int = 5,
                        with_masks: bool = False) -> list[dict[str, torch.Tensor]]:
    """
    Create dummy detection targets. Pass *with_masks=True* for Mask R-CNN,
    which additionally requires a binary mask per instance.
    """
    targets = _make_dummy_boxes(batch_size, device, num_boxes)
    if with_masks:
        for t in targets:
            n = t["labels"].shape[0]
            # One H×W binary mask per instance
            t["masks"] = torch.zeros(
                n, DETECTION_IMAGE_SIZE, DETECTION_IMAGE_SIZE,
                dtype=torch.uint8, device=device,
            )
    return targets


# Fixed RPN/ROI kwargs to ensure constant compute per iteration.
# Without these, NMS produces a *variable* number of proposals depending
# on model weights, which is the main source of run-to-run variance.
_FIXED_RPN_KWARGS: dict = {
    "rpn_pre_nms_top_n_train":   512,
    "rpn_post_nms_top_n_train":  512,
    "rpn_pre_nms_top_n_test":    512,
    "rpn_post_nms_top_n_test":   512,
    "box_batch_size_per_image":  256,
    "rpn_batch_size_per_image":  256,
}

# Registry of supported detection models
_MODEL_REGISTRY: dict = {
    "faster-rcnn-resnet50": {
        "factory": fasterrcnn_resnet50_fpn,
        "label":   "Faster R-CNN (ResNet-50 FPN)",
        "masks":   False,
    },
    "mask-rcnn-resnet50": {
        "factory": maskrcnn_resnet50_fpn,
        "label":   "Mask R-CNN  (ResNet-50 FPN)",
        "masks":   True,
    },
}


def benchmark_model(
    model_key: str,
    precision: str,
    batch_size: int,
    device: torch.device,
    warmup: int = DETECTION_WARMUP,
    iterations: int = DETECTION_ITERATIONS,
) -> dict:
    """Benchmark a detection model (Faster R-CNN or Mask R-CNN) at a given precision."""

    spec = _MODEL_REGISTRY[model_key]
    tag = f"{spec['label']} {precision.upper()} BS={batch_size}"
    print(f"  Testing: {tag}...", end=" ", flush=True)

    try:
        # Always create model in FP32; AMP handles casting.
        # Pass fixed RPN/ROI counts to keep compute constant per iteration.
        model = spec["factory"](
            weights=None,
            num_classes=DETECTION_NUM_CLASSES,
            **_FIXED_RPN_KWARGS,
        )
        model = model.to(device)
        model.train()

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        scaler = get_grad_scaler(precision)
        amp_ctx = get_amp_context(precision)

        # Synthetic data — always FP32 images (AMP casts internally)
        images = [torch.randn(3, DETECTION_IMAGE_SIZE, DETECTION_IMAGE_SIZE, device=device)
                  for _ in range(batch_size)]

        # Warmup — regenerate targets each iteration with fixed seed
        for i in range(warmup):
            torch.manual_seed(RANDOM_SEED + i)
            targets = _make_dummy_targets(batch_size, device, with_masks=spec["masks"])
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

        # Benchmark — regenerate targets each iteration with fixed seed
        times: list[float] = []
        for i in range(iterations):
            torch.manual_seed(RANDOM_SEED + warmup + i)
            targets = _make_dummy_targets(batch_size, device, with_masks=spec["masks"])
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
            "model": model_key,
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
                "model": model_key,
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
    print("Benchmark 8 — Object Detection Training")
    print("Models: Faster R-CNN (ResNet-50 FPN) + Mask R-CNN (ResNet-50 FPN)")
    print("=" * 70)
    print()

    check_cuda_available()
    device = torch.device(f"cuda:{args.device}")
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)

    set_reproducibility(args.seed)
    set_tf32(True)

    # Suppress noisy torch._dynamo recompilation warnings from ROI Align —
    # varying proposal counts trigger recompiles that are harmless but verbose.
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

    precisions = args.precisions or TRAINING_PRECISIONS
    precisions = filter_precisions_for_gpu(precisions, args.device)
    batch_sizes = args.batch_sizes or get_detection_batch_sizes(gpu_info["vram_gb"])

    print(f"Precisions: {precisions}")
    print(f"Batch sizes: {batch_sizes}")
    print()

    # ── GPU thermal warmup ─────────────────────────────────────────────
    # Run a dummy matmul workload to bring GPU clocks and temperature to
    # a stable operating point before any measurements.
    gpu_thermal_warmup(device)

    monitor = None
    if not args.no_monitor:
        monitor = GPUMonitor(device_id=args.device)
        monitor.start()

    results: list[dict] = []

    for model_key, spec in _MODEL_REGISTRY.items():
        print(spec["label"].upper())
        print("-" * 70)
        for prec in precisions:
            for bs in batch_sizes:
                result = benchmark_model(model_key, prec, bs, device,
                                         warmup=args.warmup,
                                         iterations=args.iterations)
                results.append(result)
                if result["status"] == "oom":
                    break  # larger batch sizes will also OOM
        print()

    hw_stats = monitor.stop() if monitor else {}

    csv_path, json_path = save_results(
        results, "training_detection", gpu_info,
        extra_meta={"hw_monitor": hw_stats} if hw_stats else None,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("Benchmark 8 — Object Detection Training Complete!")
    print("=" * 70)
    print(f"\nResults saved to:\n  - {csv_path}\n  - {json_path}\n")

    print("Summary (Best Throughput per Model/Precision):")
    print("-" * 70)
    for model_key, spec in _MODEL_REGISTRY.items():
        for prec in precisions:
            hits = [
                r for r in results
                if r["model"] == model_key
                and r["precision"] == prec.upper()
                and r["status"] == "success"
            ]
            if hits:
                best = max(hits, key=lambda x: x["throughput_img_per_sec"] or 0)
                label = spec["label"]
                print(f"  {label} {prec.upper():>5s}: "
                      f"{best['throughput_img_per_sec']:.2f} img/s "
                      f"(BS={best['batch_size']})")


if __name__ == "__main__":
    main()
