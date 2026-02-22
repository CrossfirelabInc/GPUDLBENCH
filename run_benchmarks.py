#!/usr/bin/env python3
"""
Run all 10 benchmarks sequentially and generate a summary report.

Usage:
    python run_benchmarks.py                       # run all (default models)
    python run_benchmarks.py --model-set popular   # use popular model set for LLM benchmark
    python run_benchmarks.py --skip 5 10           # skip benchmarks 5 and 10
"""

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import torch


class TeeWriter:
    """Duplicate writes to both a terminal stream and a log file.

    Handles \\r (carriage-return) overwrites gracefully: animation frames
    that use \\r without a trailing \\n are sent to the terminal only.
    The final line (\\r + \\n) is logged with the \\r stripped so the log
    file stays clean.
    """

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        # Always send everything to the real terminal
        self.stream.write(data)
        # For the log file, filter out carriage-return animation frames
        if "\r" in data:
            if "\n" not in data:
                # Pure CR overwrite (spinner frame) — skip the log
                return
            # Final line from a spinner (e.g. "\r  message... OK\n")
            # Strip the CR so the log file gets a single clean line
            data = data.replace("\r", "")
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()


BENCHMARKS = [
    (1,  "Vision Training (ResNet-50, ResNet-101)",              "benchmarks/1_training_vision.py"),
    (2,  "NLP Training (BERT-base, BERT-large)",                 "benchmarks/2_training_nlp.py"),
    (3,  "Vision Inference (ResNet-50, ResNet-101)",             "benchmarks/3_inference_vision.py"),
    (4,  "NLP Inference (BERT-base, BERT-large)",                "benchmarks/4_inference_nlp.py"),
    (5,  "LLM Token Generation (llama.cpp)",                     "benchmarks/5_llm_tokens_per_sec.py"),
    (6,  "VRAM Limitation Tests",                                "benchmarks/6_vram_limits.py"),
    (7,  "Compute Stress — GEMM Throughput",                     "benchmarks/7_gemm_stress.py"),
    (8,  "Object Detection (Faster R-CNN + Mask R-CNN)",         "benchmarks/8_training_detection.py"),
    (9,  "GPU Fundamentals (Bandwidth / FFT / HPC)",             "benchmarks/9_gpu_fundamentals.py"),
    (10, "Multi-GPU Scaling (DDP Training, DP Inference)","benchmarks/10_multi_gpu_scaling.py"),
]


def check_gpu():
    """Print basic GPU info and software versions, or exit if no GPU found."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed. Run install.py first.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU:  {name}")
    print(f"VRAM: {vram:.1f} GB")

    # Software versions
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA (PyTorch): {torch.version.cuda}")
    if torch.backends.cudnn.is_available():
        print(f"cuDNN:          {torch.backends.cudnn.version()}")

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            print(f"NVIDIA Driver:  {r.stdout.strip().split(chr(10))[0]}")
    except Exception:
        pass

    # nvcc (CUDA toolkit) version
    try:
        if shutil.which("nvcc"):
            r = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5,
            )
            m = re.search(r"release (\d+\.\d+)", r.stdout)
            if m:
                print(f"CUDA Toolkit:   {m.group(1)} (nvcc)")
    except Exception:
        pass
    print()


def make_session_id() -> str:
    """Generate a human-readable session ID: YYYYMMDD_HHMMSS_<short-uuid>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{ts}_{short}"


def run_benchmark(num, name, script, total, output_dir: str = "results", extra_args: list = None):
    """Run a single benchmark script. Returns 'pass', 'fail', or 'skip'."""
    path = Path(script)
    if not path.exists():
        print(f"  SKIP: {script} not found")
        return "skip"

    print(f"[{num}/{total}] {name}")
    print("-" * 50)
    try:
        cmd = [sys.executable, "-u", str(path), "--output-dir", output_dir]
        if extra_args:
            cmd.extend(extra_args)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
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
    parser.add_argument("--model-set", type=str, default="default",
                        choices=["default", "popular"],
                        help="LLM model set for benchmark 5 (default: default)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: minimal batch size, fewer iterations, "
                             "skip benchmarks 6/9/10 for a fast ~5min run")
    args = parser.parse_args()

    skip_set = set(args.skip)
    if args.demo:
        skip_set |= {6, 9, 10}

    # ── Session setup ─────────────────────────────────────────────────
    session_id = make_session_id()
    session_dir = Path("results") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Tee all console output to a log file in the session directory
    console_log_path = session_dir / "console.log"
    _console_log_file = open(console_log_path, "w", buffering=1)
    sys.stdout = TeeWriter(sys.__stdout__, _console_log_file)
    sys.stderr = TeeWriter(sys.__stderr__, _console_log_file)

    # Maintain a "latest" symlink for convenience
    latest_link = Path("results") / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(session_id)
    except OSError:
        pass  # Symlinks may fail on some systems

    print("=" * 60)
    print("GPU DL Benchmark Suite")
    if args.demo:
        print(">>> DEMO MODE — fast run, minimal batch sizes <<<")
    print("=" * 60)
    print(f"Session: {session_id}")
    print(f"Output:  {session_dir}")
    print()

    check_gpu()

    # ── GPU thermal warmup before all benchmarks ─────────────────
    # Bring all GPUs to a stable thermal/clock state once, then skip
    # per-benchmark warmups to save time and reduce variance.
    n_gpus = torch.cuda.device_count()
    print(f"Warming up {n_gpus} GPU(s) to stabilise clocks/thermals (60s)...")

    def _warmup_gpu(dev_id: int) -> None:
        dev = torch.device(f"cuda:{dev_id}")
        a = torch.randn(4096, 4096, device=dev, dtype=torch.float32)
        b = torch.randn(4096, 4096, device=dev, dtype=torch.float32)
        end_t = time.perf_counter() + 60
        while time.perf_counter() < end_t:
            torch.mm(a, b)
        torch.cuda.synchronize(dev)
        del a, b

    import threading as _th
    if n_gpus > 1:
        threads = [_th.Thread(target=_warmup_gpu, args=(i,)) for i in range(n_gpus)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        _warmup_gpu(0)

    torch.cuda.empty_cache()
    print(f"GPU warm-up complete ({n_gpus} device(s)).\n")

    total = len(BENCHMARKS)
    passed = 0
    failed = 0
    skipped = 0
    run_start = datetime.now()
    start = time.time()

    for num, name, script in BENCHMARKS:
        print()
        if num in skip_set:
            print(f"[{num}/{total}] {name} -- skipped by user")
            skipped += 1
            continue

        # Pass --model-set to the LLM benchmark (benchmark 5)
        extra = ["--skip-thermal-warmup"]
        if num == 5:
            extra.extend(["--model-set", args.model_set])
        if args.demo:
            extra.append("--demo")
        result = run_benchmark(num, name, script, total,
                               output_dir=str(session_dir),
                               extra_args=extra or None)
        if result == "pass":
            passed += 1
        elif result == "fail":
            failed += 1
        else:
            skipped += 1

    run_end = datetime.now()
    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    # Write session metadata
    session_meta = {
        "session_id": session_id,
        "run_start": run_start.isoformat(),
        "run_end": run_end.isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }
    meta_path = session_dir / "session_meta.json"
    with open(meta_path, "w") as _f:
        json.dump(session_meta, _f, indent=2)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Session: {session_id}")
    print(f"  Started: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Ended:   {run_end.strftime('%Y-%m-%d %H:%M:%S')}")
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
            subprocess.run(
                [sys.executable, str(report_script), "--results-dir", str(session_dir)],
                check=True,
            )
        except subprocess.CalledProcessError:
            print("  WARNING: report generation failed")
    print()
    print(f"Results are in {session_dir}/")
    print(f"  cat {session_dir}/benchmark_summary.md")
    print(f"  (also available via: results/latest/benchmark_summary.md)")

    # Close the console log file
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _console_log_file.close()
    print(f"Console log saved to {console_log_path}")


if __name__ == "__main__":
    main()
