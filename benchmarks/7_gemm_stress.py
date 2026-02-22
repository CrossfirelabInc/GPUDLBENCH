#!/usr/bin/env python3
"""
Benchmark 7 — GEMM Compute Stress

Sweeps square matrix-multiplication (GEMM) across sizes 512 → 16384 for
FP64, FP32, FP16, BF16 (and FP8 on Ada/Hopper/Blackwell).  Reports sustained
TFLOPS per size/precision and a peak-TFLOPS summary table — a simple
roofline-style compute characterisation.

Results saved to:
  <output_dir>/gemm_stress.csv
  <output_dir>/gemm_stress.json
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_utils import (
    add_common_args,
    check_cuda_available,
    cuda_sync,
    get_gpu_info,
    gpu_thermal_warmup,
    median_time,
    print_gpu_banner,
    save_results,
    set_reproducibility,
)
from benchmarks.config import (
    DEMO_GEMM_REPEATS,
    DEMO_GEMM_SIZES,
    DEMO_GEMM_WARMUP,
    GEMM_REPEATS,
    GEMM_SIZES,
    GEMM_WARMUP,
)


# ─── GEMM kernels ─────────────────────────────────────────────────────────────

def _bench_gemm_dtype(N: int, dtype: torch.dtype, device: torch.device) -> dict:
    """Square NxN @ NxN GEMM for the given dtype."""
    a = torch.randn(N, N, device=device, dtype=dtype)
    b = torch.randn(N, N, device=device, dtype=dtype)
    t = median_time(lambda: torch.mm(a, b), GEMM_WARMUP, GEMM_REPEATS)
    flops = 2.0 * N ** 3          # multiply-add counts as 2 FLOPs
    tflops = flops / t / 1e12
    mem_gb = 2.0 * N * N * a.element_size() / 1e9
    del a, b
    torch.cuda.empty_cache()
    return {
        "matrix_size": N,
        "time_ms": round(t * 1_000, 3),
        "tflops": round(tflops, 3),
        "matrix_mem_gb": round(mem_gb, 4),
    }


def _bench_gemm_fp8(N: int, device: torch.device) -> dict | None:
    """FP8 GEMM via torch._scaled_mm (Ada/Hopper/Blackwell only)."""
    if not hasattr(torch, "float8_e4m3fn"):
        return None
    dt = torch.float8_e4m3fn
    a_fp16 = torch.randn(N, N, device=device, dtype=torch.float16)
    b_fp16 = torch.randn(N, N, device=device, dtype=torch.float16)
    # _scaled_mm requires: A row-major (contiguous) × B column-major (.t()).
    # Do NOT call .contiguous() on B after .t() — that converts it back to
    # row-major, which cuBLASLt rejects under deterministic mode.
    a8 = a_fp16.to(dt).contiguous()
    b8 = b_fp16.to(dt).t()
    scale_a = torch.ones(1, device=device, dtype=torch.float32)
    scale_b = torch.ones(1, device=device, dtype=torch.float32)

    def _fn() -> None:
        torch._scaled_mm(a8, b8, scale_a=scale_a, scale_b=scale_b,
                         out_dtype=torch.float16)

    t = median_time(_fn, GEMM_WARMUP, GEMM_REPEATS)
    flops = 2.0 * N ** 3
    tflops = flops / t / 1e12
    del a_fp16, b_fp16, a8, b8
    torch.cuda.empty_cache()
    return {
        "matrix_size": N,
        "time_ms": round(t * 1_000, 3),
        "tflops": round(tflops, 3),
        "matrix_mem_gb": round(2.0 * N * N * 1 / 1e9, 4),  # FP8 = 1 byte
    }


# ─── precision sweep ──────────────────────────────────────────────────────────

def _run_precision(label: str, spec, device: torch.device,
                   sizes: list[int]) -> list[dict]:
    """Sweep all sizes for one precision.  Returns a list of result rows."""
    bar = "─" * 72
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  {'Size':>13}  {'TFLOPS':>10}  {'Time (ms)':>10}  {'Matrices (GB)':>14}")
    print(f"  {'─'*13}  {'─'*10}  {'─'*10}  {'─'*14}")

    rows: list[dict] = []
    for N in sizes:
        try:
            if spec == "fp8":
                r = _bench_gemm_fp8(N, device)
                if r is None:
                    print(f"  {N:>6}×{N:<6}  FP8 not available on this GPU")
                    break
            else:
                r = _bench_gemm_dtype(N, spec, device)
            r["precision"] = label
            rows.append(r)
            print(f"  {N:>6}×{N:<6}  "
                  f"{r['tflops']:>10.2f}  "
                  f"{r['time_ms']:>10.2f}  "
                  f"{r['matrix_mem_gb']:>14.4f}")
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg:
                print(f"  {N:>6}×{N:<6}  OOM — stopping this precision")
                torch.cuda.empty_cache()
                break
            print(f"  {N:>6}×{N:<6}  ERROR: {exc}")
    return rows


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GEMM Compute Stress Benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    # Demo mode overrides
    if args.demo:
        global GEMM_WARMUP, GEMM_REPEATS
        GEMM_WARMUP = DEMO_GEMM_WARMUP
        GEMM_REPEATS = DEMO_GEMM_REPEATS
        print("*** DEMO MODE — reduced sizes and repetitions ***\n")

    separator = "=" * 72
    print(separator)
    print("Benchmark 7 — GEMM Compute Stress")
    print("Roofline sweep: FP64 / FP32 / FP16 / BF16 (/ FP8 if supported)")
    print(separator)

    check_cuda_available()
    device = torch.device(f"cuda:{args.device}")
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)
    set_reproducibility(args.seed)

    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    sizes: list[int] = DEMO_GEMM_SIZES if args.demo else GEMM_SIZES

    # Build precision list based on capabilities
    if args.demo:
        precisions: list[tuple[str, ...]] = [("FP16", torch.float16)]
    else:
        precisions: list[tuple[str, ...]] = [
            ("FP64", torch.float64),
            ("FP32", torch.float32),
            ("FP16", torch.float16),
        ]
        if cc_major >= 8:
            precisions.append(("BF16", torch.bfloat16))
        # FP8 requires CC ≥ 8.9  (Ada Lovelace / Hopper / Blackwell)
        if cc_major > 8 or (cc_major == 8 and cc_minor >= 9):
            precisions.append(("FP8", "fp8"))

    print(f"\nMatrix sizes : {sizes}")
    print(f"Precisions   : {[p[0] for p in precisions]}")
    print(f"Warmup / runs: {GEMM_WARMUP} / {GEMM_REPEATS}")

    # Stabilise GPU clocks/thermals before any timed measurements
    if not args.skip_thermal_warmup:
        gpu_thermal_warmup(device)

    all_rows: list[dict] = []
    peak: dict[str, float] = {}

    for label, spec in precisions:
        rows = _run_precision(label, spec, device, sizes)
        all_rows.extend(rows)
        if rows:
            peak[label] = max(r["tflops"] for r in rows)

    # Summary
    print(f"\n{separator}")
    print("Peak TFLOPS Summary")
    print(separator)
    for label, tflops in peak.items():
        bar_len = int(tflops / max(peak.values()) * 40) if peak else 0
        bar_str = "█" * bar_len
        print(f"  {label:>5s}  {tflops:>8.2f} TFLOPS  {bar_str}")
    print(separator)

    csv_path, json_path = save_results(
        all_rows,
        "gemm_stress",
        gpu_info,
        extra_meta={"peak_tflops": peak, "matrix_sizes": sizes,
                    "warmup": GEMM_WARMUP, "repeats": GEMM_REPEATS},
        output_dir=args.output_dir,
    )
    print(f"\nResults saved:\n  CSV  → {csv_path}\n  JSON → {json_path}\n")


if __name__ == "__main__":
    main()
