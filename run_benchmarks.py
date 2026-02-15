#!/usr/bin/env python3
"""
Run all 9 benchmarks sequentially and generate a summary report.

Usage:
    python run_benchmarks.py              # run all
    python run_benchmarks.py --skip 5 9   # skip benchmarks 5 and 9
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


BENCHMARKS = [
    (1, "Vision Training (ResNet-50, ResNet-101)",    "benchmarks/1_training_vision.py"),
    (2, "NLP Training (BERT-base, BERT-large)",       "benchmarks/2_training_nlp.py"),
    (3, "Vision Inference (ResNet-50, ResNet-101)",    "benchmarks/3_inference_vision.py"),
    (4, "NLP Inference (BERT-base, BERT-large)",       "benchmarks/4_inference_nlp.py"),
    (5, "LLM Token Generation (llama.cpp)",           "benchmarks/5_llm_tokens_per_sec.py"),
    (6, "VRAM Limitation Tests",                       "benchmarks/6_vram_limits.py"),
    (7, "Mixed Precision Analysis",                    "benchmarks/7_mixed_precision.py"),
    (8, "Object Detection Training (Faster R-CNN)",    "benchmarks/8_training_detection.py"),
    (9, "Quick Smoke Tests (PyTorch micro-benchmarks)","benchmarks/9_quick_benchmarks.py"),
]


def check_gpu():
    """Print basic GPU info or exit if no GPU found."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed. Run install.py first.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    print(f"GPU:  {name}")
    print(f"VRAM: {vram:.1f} GB")
    print()


def run_benchmark(num, name, script, total):
    """Run a single benchmark script. Returns 'pass', 'fail', or 'skip'."""
    path = Path(script)
    if not path.exists():
        print(f"  SKIP: {script} not found")
        return "skip"

    print(f"[{num}/{total}] {name}")
    print("-" * 50)
    try:
        subprocess.run(
            [sys.executable, str(path), "--output-dir", "results"],
            check=True,
        )
        print(f"  PASS: {name}")
        return "pass"
    except subprocess.CalledProcessError as e:
        print(f"  FAIL: {name} (exit code {e.returncode})")
        return "fail"
    except KeyboardInterrupt:
        print(f"\n  Interrupted during {name}")
        return "fail"


def main():
    parser = argparse.ArgumentParser(description="Run GPU DL Benchmark Suite")
    parser.add_argument("--skip", nargs="*", type=int, default=[],
                        help="Benchmark numbers to skip (e.g. --skip 5 9)")
    args = parser.parse_args()

    skip_set = set(args.skip)

    print("=" * 60)
    print("GPU DL Benchmark Suite")
    print("=" * 60)
    print()

    check_gpu()

    Path("results").mkdir(exist_ok=True)

    total = len(BENCHMARKS)
    passed = 0
    failed = 0
    skipped = 0
    start = time.time()

    for num, name, script in BENCHMARKS:
        print()
        if num in skip_set:
            print(f"[{num}/{total}] {name} -- skipped by user")
            skipped += 1
            continue

        result = run_benchmark(num, name, script, total)
        if result == "pass":
            passed += 1
        elif result == "fail":
            failed += 1
        else:
            skipped += 1

    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Time:    {hours}h {minutes}m {seconds}s")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print()

    # Generate report
    report_script = Path("utils/generate_report.py")
    if report_script.exists():
        print("Generating report...")
        try:
            subprocess.run([sys.executable, str(report_script)], check=True)
        except subprocess.CalledProcessError:
            print("  WARNING: report generation failed")
    print()
    print("Results are in results/")
    print("  cat results/benchmark_summary.md")


if __name__ == "__main__":
    main()
