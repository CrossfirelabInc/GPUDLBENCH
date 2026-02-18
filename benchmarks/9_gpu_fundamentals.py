#!/usr/bin/env python3
"""
Benchmark 9 — GPU Fundamentals

A comprehensive characterisation of low-level GPU capabilities spanning
memory, PCIe, compute kernels, FFT, scientific simulations, and DL
primitives.  Useful as a baseline and for roofline analysis.

Tests
─────
  1. Device memory bandwidth sweep   — D2D copy,  16 MB → 2 GB
  2. PCIe bandwidth                  — H2D / D2H / Bidirectional
  3. Kernel launch latency           — 10 000 no-op kernel launches
  4. FFT throughput                  — FP32 + FP64, 1 K → 256 K points
  5. N-body gravity                  — all-pairs, FP32 + FP64
  6. Sparse matrix multiply (SpMM)   — CSR, varying density
  7. 2-D heat stencil (5-point)      — roll-based, FP32
  8. Parallel reduction              — torch.sum, various sizes
  9. Conv2d throughput               — 3×3 / 5×5 / 7×7 kernels
 10. Multi-head attention            — SDPA, FP32 + FP16

Results saved to:
  <output_dir>/gpu_fundamentals.csv
  <output_dir>/gpu_fundamentals.json
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_utils import (
    add_common_args,
    check_cuda_available,
    get_gpu_info,
    print_gpu_banner,
    save_results,
    set_reproducibility,
)
from benchmarks.config import (
    FUND_BW_SIZES,
    FUND_FFT_SIZES,
    FUND_NBODY_N,
    FUND_NBODY_STEPS,
    FUND_PCIE_SIZE,
    FUND_REPEATS,
    FUND_STENCIL_SIZE,
    FUND_STENCIL_STEPS,
    FUND_WARMUP,
)


# ──────────────────────────── helpers ─────────────────────────────────────────

def _sync() -> None:
    torch.cuda.synchronize()


def _median_time(fn, warmup: int = FUND_WARMUP,
                 repeats: int = FUND_REPEATS) -> float:
    """Return median wall-clock time (seconds)."""
    for _ in range(warmup):
        fn()
    _sync()
    ts: List[float] = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def _row(category: str, test: str, metric: str, value: float, unit: str,
         dtype: str = "", notes: str = "") -> Dict[str, Any]:
    return {
        "category": category,
        "test": test,
        "dtype": dtype,
        "metric": metric,
        "value": round(value, 4),
        "unit": unit,
        "notes": notes,
    }


# ──────────────────────────── 1. Memory bandwidth ─────────────────────────────

def bench_memory_bandwidth(device: torch.device) -> List[Dict[str, Any]]:
    print("\n── 1. Device Memory Bandwidth (D2D copy)")
    print(f"  {'Size':>10}  {'GB/s':>10}")
    rows: List[Dict[str, Any]] = []
    for size_bytes in FUND_BW_SIZES:
        n_floats = size_bytes // 4
        try:
            src = torch.randn(n_floats, device=device, dtype=torch.float32)
            dst = torch.empty_like(src)
            t = _median_time(lambda: dst.copy_(src))
            gb_per_s = 2 * size_bytes / t / 1e9  # read + write
            label = f"{size_bytes // (1024*1024)} MB"
            print(f"  {label:>10}  {gb_per_s:>10.1f}")
            rows.append(_row("memory_bandwidth", f"d2d_copy_{label}", "bandwidth",
                             gb_per_s, "GB/s", "FP32", f"size={label}"))
            del src, dst
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {size_bytes // (1024*1024)} MB  OOM")
                torch.cuda.empty_cache()
                break
    return rows


# ──────────────────────────── 2. PCIe bandwidth ───────────────────────────────

def bench_pcie_bandwidth(device: torch.device) -> List[Dict[str, Any]]:
    print("\n── 2. PCIe Bandwidth (Host ↔ Device)")
    size = FUND_PCIE_SIZE
    n_floats = size // 4
    label = f"{size // (1024*1024)} MB"

    cpu_pinned = torch.randn(n_floats, pin_memory=True)
    gpu_buf = torch.empty(n_floats, device=device, dtype=torch.float32)
    cpu_out = torch.empty(n_floats, pin_memory=True)

    # H2D
    t_h2d = _median_time(lambda: gpu_buf.copy_(cpu_pinned))
    h2d_gbs = size / t_h2d / 1e9

    # D2H
    t_d2h = _median_time(lambda: cpu_out.copy_(gpu_buf))
    d2h_gbs = size / t_d2h / 1e9

    print(f"  H2D ({label}): {h2d_gbs:.1f} GB/s")
    print(f"  D2H ({label}): {d2h_gbs:.1f} GB/s")

    del cpu_pinned, gpu_buf, cpu_out
    return [
        _row("pcie", "h2d", "bandwidth", h2d_gbs, "GB/s", notes=label),
        _row("pcie", "d2h", "bandwidth", d2h_gbs, "GB/s", notes=label),
    ]


# ──────────────────────────── 3. Kernel launch latency ────────────────────────

def bench_kernel_launch_latency(device: torch.device) -> List[Dict[str, Any]]:
    LAUNCHES = 10_000
    print(f"\n── 3. Kernel Launch Latency ({LAUNCHES:,} no-op launches)")
    tiny = torch.ones(1, device=device)
    for _ in range(200):
        _ = tiny + tiny  # warmup
    _sync()

    t0 = time.perf_counter()
    for _ in range(LAUNCHES):
        _ = tiny + tiny
    _sync()
    total = time.perf_counter() - t0
    latency_us = total / LAUNCHES * 1e6
    print(f"  Avg per launch: {latency_us:.2f} µs")
    del tiny
    return [_row("kernel_launch", "latency_noopx10k", "latency", latency_us, "µs")]


# ──────────────────────────── 4. FFT throughput ───────────────────────────────

def bench_fft(device: torch.device) -> List[Dict[str, Any]]:
    print("\n── 4. FFT Throughput")
    print(f"  {'Size':>8}  {'Dtype':>5}  {'GB/s':>8}  {'GFLOPS':>8}")
    rows: List[Dict[str, Any]] = []
    for n in FUND_FFT_SIZES:
        for dtype, label in [(torch.float32, "FP32"), (torch.float64, "FP64")]:
            try:
                x = torch.randn(n, dtype=dtype, device=device)
                t = _median_time(lambda: torch.fft.rfft(x))
                # FLOPs for FFT: ~5 N log2(N)
                flops = 5.0 * n * (n.bit_length() - 1)
                gflops = flops / t / 1e9
                gb_s = n * x.element_size() / t / 1e9
                print(f"  {n:>8}  {label:>5}  {gb_s:>8.1f}  {gflops:>8.1f}")
                rows.append(_row("fft", f"rfft_{n}", "throughput", gflops,
                                 "GFLOPS", label, f"n={n}"))
                del x
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {n:>8}  {label:>5}  ERROR: {e}")
    return rows


# ──────────────────────────── 5. N-body gravity ───────────────────────────────

def _nbody_step(pos: torch.Tensor, mass: torch.Tensor,
                dt: float = 0.01) -> torch.Tensor:
    """One Euler step of all-pairs gravity: O(N²) via broadcasting."""
    # pos: (N, 3), mass: (N,)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)          # (N, N, 3)
    dist_sq = (diff ** 2).sum(-1) + 1e-6                # (N, N) softened
    dist_cu = dist_sq * dist_sq.sqrt()                  # |r|³
    m = mass.unsqueeze(0)                               # (1, N)
    accel = -(diff * m.unsqueeze(-1) / dist_cu.unsqueeze(-1)).sum(1)  # (N, 3)
    return pos + accel * dt


def bench_nbody(device: torch.device, N: int = FUND_NBODY_N,
                steps: int = FUND_NBODY_STEPS) -> List[Dict[str, Any]]:
    print(f"\n── 5. N-body Gravity Simulation (N={N}, steps={steps})")
    rows: List[Dict[str, Any]] = []
    for dtype, label in [(torch.float32, "FP32"), (torch.float64, "FP64")]:
        try:
            pos  = torch.randn(N, 3,  dtype=dtype, device=device)
            mass = torch.rand( N,     dtype=dtype, device=device) + 0.1

            def _run():
                p = pos
                for _ in range(steps):
                    p = _nbody_step(p, mass)
                return p

            t = _median_time(_run, warmup=2, repeats=5)
            particle_steps_per_s = N * steps / t
            print(f"  {label}: {particle_steps_per_s/1e6:.2f} M particle-steps/s  "
                  f"({t*1000:.1f} ms/run)")
            rows.append(_row("nbody", f"gravity_{label.lower()}", "throughput",
                             particle_steps_per_s / 1e6, "M_particle_steps_per_s",
                             label, f"N={N}, steps={steps}"))
            del pos, mass
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {label}: OOM for N={N}")
                torch.cuda.empty_cache()
    return rows


# ──────────────────────────── 6. Sparse matrix multiply (SpMM) ────────────────

def bench_spmm(device: torch.device) -> List[Dict[str, Any]]:
    print("\n── 6. Sparse Matrix Multiply (SpMM, CSR)")
    print(f"  {'Dim':>8}  {'Density':>8}  {'GFLOPS':>8}")
    rows: List[Dict[str, Any]] = []
    configs = [
        (4096,  0.01),
        (4096,  0.05),
        (8192,  0.01),
    ]
    for dim, density in configs:
        try:
            nnz = int(dim * dim * density)
            indices = torch.stack([
                torch.randint(0, dim, (nnz,)),
                torch.randint(0, dim, (nnz,)),
            ])
            values  = torch.randn(nnz)
            sparse  = torch.sparse_coo_tensor(indices, values, (dim, dim)).to(device)
            sparse  = sparse.to_sparse_csr()
            dense   = torch.randn(dim, 512, device=device)

            t = _median_time(lambda: torch.mm(sparse, dense))
            flops = 2.0 * nnz * 512
            gflops = flops / t / 1e9
            label = f"{dim}×{dim} d={density}"
            print(f"  {label:>8}  {density:>8.2f}  {gflops:>8.1f}")
            rows.append(_row("spmm", f"spmm_{dim}_d{int(density*100)}pct",
                             "throughput", gflops, "GFLOPS", "FP32",
                             f"dim={dim}, density={density}"))
            del sparse, dense
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {dim}×{dim} d={density}  SKIP ({e})")
    return rows


# ──────────────────────────── 7. 2-D heat stencil ─────────────────────────────

def bench_stencil(device: torch.device,
                  size: int = FUND_STENCIL_SIZE,
                  steps: int = FUND_STENCIL_STEPS) -> List[Dict[str, Any]]:
    """5-point 2D heat-equation stencil via roll operations."""
    print(f"\n── 7. 2-D Heat Stencil (5-point, {size}×{size}, {steps} steps)")
    grid = torch.randn(1, 1, size, size, device=device, dtype=torch.float32)
    kernel = torch.tensor([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=torch.float32,
                           device=device).reshape(1, 1, 3, 3)

    def _run():
        g = grid
        for _ in range(steps):
            g = g + 0.25 * F.conv2d(g, kernel, padding=1)
        return g

    t = _median_time(_run, warmup=3, repeats=10)
    cell_updates = size * size * steps
    giga_updates = cell_updates / t / 1e9
    print(f"  {giga_updates:.2f} G cell-updates/s  ({t*1000:.1f} ms/run)")
    del grid, kernel
    torch.cuda.empty_cache()
    return [_row("stencil", f"heat2d_{size}x{size}", "throughput",
                 giga_updates, "G_cell_updates_per_s", "FP32",
                 f"size={size}, steps={steps}")]


# ──────────────────────────── 8. Parallel reduction ──────────────────────────

def bench_reduction(device: torch.device) -> List[Dict[str, Any]]:
    print("\n── 8. Parallel Reduction (torch.sum)")
    print(f"  {'Elements':>12}  {'GB/s':>8}")
    rows: List[Dict[str, Any]] = []
    sizes = [1 << e for e in range(20, 29, 2)]  # 1M, 4M, 16M, 64M, 256M
    for n in sizes:
        try:
            x = torch.randn(n, device=device, dtype=torch.float32)
            t = _median_time(lambda: torch.sum(x))
            gb_s = n * 4 / t / 1e9
            label = f"{n // 1_000_000}M"
            print(f"  {label:>12}  {gb_s:>8.1f}")
            rows.append(_row("reduction", f"sum_{label}", "bandwidth", gb_s, "GB/s",
                             "FP32", f"n={n}"))
            del x
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
    return rows


# ──────────────────────────── 9. Conv2d throughput ───────────────────────────

def bench_conv2d(device: torch.device) -> List[Dict[str, Any]]:
    print("\n── 9. Conv2d Throughput (224×224, batch=32)")
    print(f"  {'Kernel':>8}  {'Channels':>10}  {'GFLOPS':>8}")
    rows: List[Dict[str, Any]] = []
    configs = [
        (3, 64, 64),
        (3, 256, 256),
        (5, 64, 64),
        (7, 64, 64),
    ]
    H, W, BS = 224, 224, 32
    for k, c_in, c_out in configs:
        try:
            x   = torch.randn(BS, c_in, H, W, device=device)
            w   = torch.randn(c_out, c_in, k, k, device=device)
            pad = k // 2
            t = _median_time(lambda: F.conv2d(x, w, padding=pad))
            flops = 2.0 * BS * c_out * c_in * k * k * H * W
            gflops = flops / t / 1e9
            label = f"{k}×{k}"
            chan_label = f"{c_in}→{c_out}"
            print(f"  {label:>8}  {chan_label:>10}  {gflops:>8.1f}")
            rows.append(_row("conv2d", f"conv2d_k{k}_c{c_in}to{c_out}",
                             "throughput", gflops, "GFLOPS", "FP32",
                             f"k={k}, C={c_in}→{c_out}"))
            del x, w
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
    return rows


# ──────────────────────────── 10. Multi-head attention ───────────────────────

def bench_attention(device: torch.device) -> List[Dict[str, Any]]:
    print("\n── 10. Scaled Dot-Product Attention (SDPA)")
    print(f"  {'SeqLen':>8}  {'Heads':>6}  {'Dtype':>5}  {'ms':>8}  {'GFLOPS':>8}")
    rows: List[Dict[str, Any]] = []
    BS = 8
    configs = [
        (512,   12, 64),
        (1024,  12, 64),
        (2048,   8, 64),
    ]
    for seq_len, n_heads, d_head in configs:
        for dtype, label in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
            try:
                d = n_heads * d_head
                q = torch.randn(BS, n_heads, seq_len, d_head, dtype=dtype, device=device)
                k = torch.randn(BS, n_heads, seq_len, d_head, dtype=dtype, device=device)
                v = torch.randn(BS, n_heads, seq_len, d_head, dtype=dtype, device=device)
                t = _median_time(
                    lambda: F.scaled_dot_product_attention(q, k, v)
                )
                # FLOPs: 2×BS×heads×SeqLen²×d_head (QK^T) + 2×BS×heads×SeqLen²×d_head (AV)
                flops = 4.0 * BS * n_heads * seq_len * seq_len * d_head
                gflops = flops / t / 1e9
                print(f"  {seq_len:>8}  {n_heads:>6}  {label:>5}  "
                      f"{t*1000:>8.2f}  {gflops:>8.1f}")
                rows.append(_row("attention", f"sdpa_s{seq_len}_h{n_heads}_{label.lower()}",
                                 "throughput", gflops, "GFLOPS", label,
                                 f"seq={seq_len}, heads={n_heads}, d_head={d_head}"))
                del q, k, v
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
    return rows


# ──────────────────────────── main ────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Fundamentals Benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    sep = "=" * 72
    print(sep)
    print("Benchmark 9 — GPU Fundamentals")
    print("Memory · PCIe · Latency · FFT · N-body · SpMM · Stencil · Conv · Attention")
    print(sep)

    check_cuda_available()
    device = torch.device(f"cuda:{args.device}")
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)
    set_reproducibility(args.seed)

    all_rows: List[Dict[str, Any]] = []

    all_rows += bench_memory_bandwidth(device)
    all_rows += bench_pcie_bandwidth(device)
    all_rows += bench_kernel_launch_latency(device)
    all_rows += bench_fft(device)
    all_rows += bench_nbody(device)
    all_rows += bench_spmm(device)
    all_rows += bench_stencil(device)
    all_rows += bench_reduction(device)
    all_rows += bench_conv2d(device)
    all_rows += bench_attention(device)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Summary")
    print(sep)
    categories = {}
    for r in all_rows:
        categories.setdefault(r["category"], []).append(r)
    for cat, rows in categories.items():
        best = max(rows, key=lambda x: x["value"])
        print(f"  {cat:<26}  best: {best['value']:>10.2f} {best['unit']:<26}  [{best['test']}]")
    print(sep)

    csv_path, json_path = save_results(
        all_rows,
        "gpu_fundamentals",
        gpu_info,
        extra_meta={"total_tests": len(all_rows)},
        output_dir=args.output_dir,
    )
    print(f"\nResults saved:\n  CSV  → {csv_path}\n  JSON → {json_path}\n")


if __name__ == "__main__":
    main()
