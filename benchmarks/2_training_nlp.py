#!/usr/bin/env python3
"""
Benchmark 2 — NLP Training (BERT-base, BERT-large)

Measures training throughput (samples/second) using synthetic data.
Uses proper AMP (torch.amp.autocast + GradScaler) for FP16/BF16.
Based on Lambda Labs methodology.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import BertForSequenceClassification, BertConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import (
    NLP_MODELS,
    NLP_NUM_LABELS,
    NLP_SEQ_LENGTH,
    NLP_TRAIN_ITERATIONS,
    NLP_TRAIN_WARMUP,
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
    get_nlp_train_batch_sizes,
    GPUMonitor,
    print_gpu_banner,
    save_results,
    set_reproducibility,
    set_tf32,
)


def benchmark_model(
    model_name: str,
    model_config: dict,
    precision: str,
    batch_size: int,
    device: torch.device,
    warmup: int = NLP_TRAIN_WARMUP,
    iterations: int = NLP_TRAIN_ITERATIONS,
) -> Dict[str, Any]:
    """Benchmark a single (model, precision, batch_size) configuration."""

    tag = f"{model_name} {precision.upper()} BS={batch_size}"
    print(f"  Testing: {tag}...", end=" ", flush=True)

    try:
        config = BertConfig(
            vocab_size=NLP_VOCAB_SIZE,
            max_position_embeddings=512,
            num_labels=NLP_NUM_LABELS,
            **model_config,
        )

        # Model is always FP32 on GPU — AMP handles internal casting
        model = BertForSequenceClassification(config).to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scaler = get_grad_scaler(precision)
        amp_ctx = get_amp_context(precision)

        # Synthetic data (long tensors; no casting needed)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, NLP_SEQ_LENGTH), device=device)
        attention_mask = torch.ones(batch_size, NLP_SEQ_LENGTH, dtype=torch.long, device=device)
        labels = torch.randint(0, NLP_NUM_LABELS, (batch_size,), device=device)

        # Warmup
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        torch.cuda.synchronize()

        # Benchmark
        times: List[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
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

        print(f"{throughput:.1f} samples/s")

        del model, optimizer, input_ids, attention_mask, labels, scaler
        torch.cuda.empty_cache()

        return {
            "model": model_name,
            "precision": precision.upper(),
            "batch_size": batch_size,
            "seq_length": NLP_SEQ_LENGTH,
            "avg_time_ms": round(avg_time * 1000, 2),
            "throughput_samples_per_sec": round(throughput, 1),
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
                "seq_length": NLP_SEQ_LENGTH,
                "avg_time_ms": None,
                "throughput_samples_per_sec": None,
                "status": "oom",
            }
        print(f"ERROR: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP Training Benchmark")
    add_common_args(parser)
    parser.add_argument("--models", nargs="+", default=list(NLP_MODELS.keys()),
                        help="Models to test (default: bert-base bert-large)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        help="Override batch sizes (default: auto based on VRAM)")
    parser.add_argument("--warmup", type=int, default=NLP_TRAIN_WARMUP)
    parser.add_argument("--iterations", type=int, default=NLP_TRAIN_ITERATIONS)
    args = parser.parse_args()

    print("=" * 70)
    print("NLP Training Benchmark - BERT Models")
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
    print(f"Sequence Length: {NLP_SEQ_LENGTH}")
    print()

    monitor = None
    if not args.no_monitor:
        monitor = GPUMonitor(device_id=args.device)
        monitor.start()

    results: List[Dict[str, Any]] = []

    for model_name in args.models:
        model_config = NLP_MODELS[model_name]
        batch_sizes = args.batch_sizes or get_nlp_train_batch_sizes(gpu_info["vram_gb"], model_name)

        print(f"\n{model_name.upper()}")
        print(f"Batch sizes: {batch_sizes}")
        print("-" * 70)

        for prec in precisions:
            for bs in batch_sizes:
                result = benchmark_model(model_name, model_config, prec, bs, device,
                                         warmup=args.warmup, iterations=args.iterations)
                results.append(result)

    hw_stats = monitor.stop() if monitor else {}

    csv_path, json_path = save_results(
        results, "training_nlp", gpu_info,
        extra_meta={"hw_monitor": hw_stats} if hw_stats else None,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("NLP Training Benchmark Complete!")
    print("=" * 70)
    print(f"\nResults saved to:\n  - {csv_path}\n  - {json_path}\n")

    print("Summary (Best Results):")
    print("-" * 70)
    for model_name in args.models:
        for prec in precisions:
            hits = [r for r in results
                    if r["model"] == model_name and r["precision"] == prec.upper()
                    and r["status"] == "success"]
            if hits:
                best = max(hits, key=lambda x: x["throughput_samples_per_sec"] or 0)
                print(f"  {model_name} {prec.upper()}: {best['throughput_samples_per_sec']:.1f} samples/s (BS={best['batch_size']})")


if __name__ == "__main__":
    main()
