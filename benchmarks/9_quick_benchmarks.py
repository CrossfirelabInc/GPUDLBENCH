#!/usr/bin/env python3
"""
Benchmark 9 — Quick GPU Smoke Tests

Lightweight, dependency-free (PyTorch only) tests that run in < 2 minutes.
Use these to quickly validate GPU health and get ballpark performance numbers.

Tests:
  1. Matrix Multiplication Throughput (FP32 / FP16 / BF16)
  2. Memory Bandwidth (Host↔Device, Device↔Device)
  3. Kernel Launch Latency
  4. Convolution Throughput (ResNet-style)
  5. Attention Throughput (Transformer-style)
"""

import argparse
import json
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
#  Shared helpers (minimal — no dependency on benchmark_utils)
# ---------------------------------------------------------------------------

def _sync():
    torch.cuda.synchronize()


def _timed(fn, warmup: int = 5, repeats: int = 20) -> float:
    """Run *fn* several times and return median wall-time in seconds."""
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
#  Individual tests
# ---------------------------------------------------------------------------

def test_matmul(device: torch.device) -> dict:
    """Square matrix multiply throughput for FP32, FP16, and (optionally) BF16."""
    N = 4096
    results = {}
    dtypes = {"FP32": torch.float32, "FP16": torch.float16}
    cc = torch.cuda.get_device_capability(device)
    if cc[0] >= 8:
        dtypes["BF16"] = torch.bfloat16

    for label, dt in dtypes.items():
        a = torch.randn(N, N, device=device, dtype=dt)
        b = torch.randn(N, N, device=device, dtype=dt)
        t = _timed(lambda: torch.mm(a, b))
        flops = 2 * N ** 3
        tflops = flops / t / 1e12
        results[label] = {"time_s": round(t, 5), "tflops": round(tflops, 2)}
        print(f"  MatMul {N}x{N} {label:4s}: {tflops:7.2f} TFLOPS  ({t*1000:.2f} ms)")
        del a, b

    return results


def test_memory_bandwidth(device: torch.device) -> dict:
    """Host-Device copy and Device-Device copy bandwidth."""
    size_mb = 256
    numel = size_mb * 1024 * 1024 // 4  # float32 elements
    results = {}

    # Host -> Device
    host = torch.randn(numel, dtype=torch.float32, pin_memory=True)
    dev = torch.empty(numel, dtype=torch.float32, device=device)
    t = _timed(lambda: dev.copy_(host, non_blocking=False), warmup=3, repeats=10)
    bw = size_mb / t / 1024  # GB/s
    results["h2d_gbps"] = round(bw, 2)
    print(f"  Host -> Device ({size_mb} MB):  {bw:7.2f} GB/s")
    del host

    # Device -> Device
    src = torch.randn(numel, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    t = _timed(lambda: dst.copy_(src), warmup=3, repeats=10)
    bw = size_mb / t / 1024
    results["d2d_gbps"] = round(bw, 2)
    print(f"  Device -> Device ({size_mb} MB): {bw:7.2f} GB/s")
    del src, dst, dev

    return results


def test_kernel_launch_latency(device: torch.device) -> dict:
    """Measure empty-kernel launch overhead."""
    a = torch.zeros(1, device=device)
    t = _timed(lambda: a.add_(0), warmup=50, repeats=200)
    us = t * 1e6
    print(f"  Kernel launch latency:       {us:7.2f} us")
    del a
    return {"latency_us": round(us, 2)}


def test_conv_throughput(device: torch.device) -> dict:
    """3x3 convolution throughput (ResNet-style block)."""
    batch, c_in, h, w, c_out = 32, 64, 56, 56, 64
    x = torch.randn(batch, c_in, h, w, device=device)
    conv = torch.nn.Conv2d(c_in, c_out, 3, padding=1, bias=False).to(device)
    t = _timed(lambda: conv(x))
    img_per_sec = batch / t
    print(f"  Conv2d 3x3 ({c_in}->{c_out}, {h}x{w}): {img_per_sec:,.0f} img/s  ({t*1000:.2f} ms)")
    del x, conv
    return {"img_per_sec": round(img_per_sec, 1), "time_ms": round(t * 1000, 2)}


def test_attention_throughput(device: torch.device) -> dict:
    """Multi-head self-attention throughput (Transformer-style)."""
    batch, seq, heads, dim = 8, 512, 8, 64
    embed = heads * dim
    x = torch.randn(batch, seq, embed, device=device)
    attn = torch.nn.MultiheadAttention(embed, heads, batch_first=True).to(device)

    def _fwd():
        attn(x, x, x, need_weights=False)

    t = _timed(_fwd)
    samples_per_sec = batch / t
    print(f"  MHA (seq={seq}, heads={heads}, dim={dim}): {samples_per_sec:,.0f} samples/s  ({t*1000:.2f} ms)")
    del x, attn
    return {"samples_per_sec": round(samples_per_sec, 1), "time_ms": round(t * 1000, 2)}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Quick GPU Smoke Tests")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for result files (default: results)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        return

    gpu_name = torch.cuda.get_device_name(device)
    vram_gb = torch.cuda.get_device_properties(device).total_mem / (1024 ** 3)
    cc = torch.cuda.get_device_capability(device)

    print("=" * 70)
    print("Quick GPU Smoke Tests")
    print("=" * 70)
    print(f"GPU : {gpu_name}")
    print(f"VRAM: {vram_gb:.1f} GB")
    print(f"CC  : {cc[0]}.{cc[1]}")
    print()

    all_results: dict = {
        "gpu": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "results": {},
    }

    tests = [
        ("Matrix Multiplication", "matmul", test_matmul),
        ("Memory Bandwidth", "memory_bandwidth", test_memory_bandwidth),
        ("Kernel Launch Latency", "kernel_launch", test_kernel_launch_latency),
        ("Convolution Throughput", "conv_throughput", test_conv_throughput),
        ("Attention Throughput", "attention_throughput", test_attention_throughput),
    ]

    for title, key, fn in tests:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
        try:
            all_results["results"][key] = fn(device)
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            all_results["results"][key] = {"error": str(e)}
        finally:
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("Quick Tests Summary")
    print(f"{'='*70}")

    matmul = all_results["results"].get("matmul", {})
    fp32 = matmul.get("FP32", {}).get("tflops", 0)
    fp16 = matmul.get("FP16", {}).get("tflops", 0)
    mem = all_results["results"].get("memory_bandwidth", {})
    print(f"  MatMul FP32:      {fp32:>8.2f} TFLOPS")
    print(f"  MatMul FP16:      {fp16:>8.2f} TFLOPS")
    if "BF16" in matmul:
        print(f"  MatMul BF16:      {matmul['BF16']['tflops']:>8.2f} TFLOPS")
    print(f"  H2D Bandwidth:    {mem.get('h2d_gbps', 0):>8.2f} GB/s")
    print(f"  D2D Bandwidth:    {mem.get('d2d_gbps', 0):>8.2f} GB/s")
    print()

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "quick_benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
