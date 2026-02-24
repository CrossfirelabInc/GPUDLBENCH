#!/usr/bin/env python3
"""
Report Generator — GPU AI Benchmark Suite

Reads all JSON result files from a session directory and produces:
  - Markdown summary  (benchmark_summary.md)
  - CSV summary       (benchmark_summary.csv)
  - JSON blob         (benchmark_summary.json)
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("report")

DEFAULT_RESULTS_DIR = Path("results")


# ═════════════════════════════════════════════════════════════════════════════
#  Load results (with specific exception handling)
# ═════════════════════════════════════════════════════════════════════════════

def _load_json(filename: str, results_dir: Path = DEFAULT_RESULTS_DIR) -> dict | None:
    """Load a single JSON result file, returning None on failure."""
    path = results_dir / filename
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info(f"Not found (skipped): {path}")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error in {path}: {e}")
    except PermissionError as e:
        logger.warning(f"Permission denied: {path}: {e}")
    return None


def load_results(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, Any]:
    """Load all benchmark result JSON files."""
    return {
        "training_vision": _load_json("training_vision.json", results_dir),
        "training_nlp": _load_json("training_nlp.json", results_dir),
        "inference_vision": _load_json("inference_vision.json", results_dir),
        "inference_nlp": _load_json("inference_nlp.json", results_dir),
        "llm": _load_json("llm_tokens_per_sec.json", results_dir),
        "vram_limits": _load_json("vram_limits.json", results_dir),
        "training_detection": _load_json("training_detection.json", results_dir),
        "gemm_stress": _load_json("gemm_stress.json", results_dir),
        "gpu_fundamentals": _load_json("gpu_fundamentals.json", results_dir),
        "multi_gpu_scaling": _load_json("multi_gpu_scaling.json", results_dir),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  GPU info extraction
# ═════════════════════════════════════════════════════════════════════════════

def _gpu_meta(data: dict[str, Any]) -> dict[str, Any]:
    """Extract gpu name, vram, compute capability, and system_info from the first available result file."""
    for d in data.values():
        if d and "gpu" in d:
            return {
                "gpu_name": d.get("gpu", "Unknown GPU"),
                "vram_gb": d.get("vram_gb", 0),
                "compute_capability": d.get("compute_capability", "N/A"),
                "system_info": d.get("system_info", {}),
            }
    return {"gpu_name": "Unknown GPU", "vram_gb": 0, "compute_capability": "N/A", "system_info": {}}


# ═════════════════════════════════════════════════════════════════════════════
#  Markdown report
# ═════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(data: dict[str, Any], session_id: str = "", session_meta: dict[str, Any] | None = None) -> str:
    meta = _gpu_meta(data)
    lines: list[str] = []

    lines.append("# GPU AI Benchmark Results\n")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    if session_id:
        lines.append(f"**Session**: `{session_id}`\n")

    # ── Run timing ──────────────────────────────────────────────────────
    if session_meta:
        run_start = session_meta.get("run_start", "N/A")
        run_end = session_meta.get("run_end", "N/A")
        elapsed = session_meta.get("elapsed_seconds", 0)
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        lines.append(f"**Run Start**: {run_start}  ")
        lines.append(f"**Run End**: {run_end}  ")
        lines.append(f"**Duration**: {h}h {m}m {s}s  ")
        p = session_meta.get("passed", 0)
        f_ = session_meta.get("failed", 0)
        sk = session_meta.get("skipped", 0)
        lines.append(f"**Benchmarks**: {p} passed, {f_} failed, {sk} skipped\n")

    # ── Hardware ──────────────────────────────────────────────────────────
    lines.append("## System Environment\n")
    lines.append("### Hardware\n")
    lines.append(f"| Component | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| GPU | {meta['gpu_name']} |")
    lines.append(f"| VRAM | {meta['vram_gb']:.1f} GB |")
    lines.append(f"| Compute Capability | {meta['compute_capability']} |")

    si = meta.get("system_info", {})

    # Power limit info
    power_limit = si.get("gpu_power_limit_w")
    power_default = si.get("gpu_power_default_w")
    power_max = si.get("gpu_power_max_w")
    if power_limit is not None:
        pl_str = f"{power_limit:.0f} W"
        if power_default and abs(power_limit - power_default) > 1:
            pl_str += f" (**throttled** — default: {power_default:.0f} W)"
        lines.append(f"| Power Limit | {pl_str} |")
    if power_max is not None:
        lines.append(f"| Max Power Limit | {power_max:.0f} W |")

    if si:
        lines.append(f"| OS | {si.get('os', 'N/A')} |")
        lines.append("")

        # ── Software versions ─────────────────────────────────────────────
        lines.append("### Software Versions\n")
        lines.append("| Package | Version |")
        lines.append("|---------|---------|")
        lines.append(f"| NVIDIA Driver | {si.get('nvidia_driver_version', 'N/A')} |")
        lines.append(f"| CUDA (PyTorch) | {si.get('pytorch_cuda_version', 'N/A')} |")
        lines.append(f"| CUDA Toolkit (nvcc) | {si.get('cuda_toolkit_version', 'N/A')} |")
        lines.append(f"| cuDNN | {si.get('cudnn_version', 'N/A')} |")
        lines.append(f"| PyTorch | {si.get('pytorch_version', 'N/A')} |")
        lines.append(f"| Python | {si.get('python_version', 'N/A')} |")
        for pkg in ['transformers', 'accelerate', 'numpy', 'pillow', 'datasets']:
            ver = si.get(f'{pkg}_version')
            if ver:
                lines.append(f"| {pkg} | {ver} |")
    lines.append("")
    lines.append("---\n")

    # ── Training Vision ───────────────────────────────────────────────────
    tv = data.get("training_vision")
    if tv:
        lines.append("## Training Performance — Vision\n")
        lines.append("| Model | Precision | Best Throughput | Batch Size |")
        lines.append("|-------|-----------|-----------------|------------|")
        for model in ["resnet50", "resnet101"]:
            for prec in ["FP32", "FP16", "BF16"]:
                hits = [r for r in tv["results"]
                        if r["model"] == model and r["precision"] == prec and r["status"] == "success"]
                if hits:
                    best = max(hits, key=lambda x: x.get("throughput_img_per_sec", 0) or 0)
                    lines.append(f"| {model} | {prec} | {best['throughput_img_per_sec']:.1f} img/s | {best['batch_size']} |")
        lines.append("")

    # ── Training NLP ──────────────────────────────────────────────────────
    tn = data.get("training_nlp")
    if tn:
        lines.append("## Training Performance — NLP\n")
        lines.append("| Model | Precision | Best Throughput | Batch Size |")
        lines.append("|-------|-----------|-----------------|------------|")
        for model in ["bert-base", "bert-large"]:
            for prec in ["FP32", "FP16", "BF16"]:
                hits = [r for r in tn["results"]
                        if r["model"] == model and r["precision"] == prec and r["status"] == "success"]
                if hits:
                    best = max(hits, key=lambda x: x.get("throughput_samples_per_sec", 0) or 0)
                    lines.append(f"| {model} | {prec} | {best['throughput_samples_per_sec']:.1f} samples/s | {best['batch_size']} |")
        lines.append("")

    # ── Training Detection ────────────────────────────────────────────────
    td = data.get("training_detection")
    if td:
        lines.append("## Training Performance — Detection\n")
        lines.append("| Model | Precision | Best Throughput | Batch Size |")
        lines.append("|-------|-----------|-----------------|------------|")
        for prec in ["FP32", "FP16", "BF16"]:
            hits = [r for r in td["results"]
                    if r["precision"] == prec and r["status"] == "success"]
            if hits:
                best = max(hits, key=lambda x: x.get("throughput_img_per_sec", 0) or 0)
                lines.append(f"| {best['model']} | {prec} | {best['throughput_img_per_sec']:.2f} img/s | {best['batch_size']} |")
        lines.append("")

    # ── Inference Vision ──────────────────────────────────────────────────
    iv = data.get("inference_vision")
    if iv:
        lines.append("## Inference Performance — Vision\n")
        lines.append("| Model | Precision | BS=1 Latency | Best Throughput | Best BS |")
        lines.append("|-------|-----------|-------------|-----------------|---------|")
        for model in ["resnet50", "resnet101"]:
            for prec in ["FP32", "FP16", "BF16"]:
                bs1 = [r for r in iv["results"]
                       if r["model"] == model and r["precision"] == prec
                       and r["batch_size"] == 1 and r["status"] == "success"]
                hits = [r for r in iv["results"]
                        if r["model"] == model and r["precision"] == prec and r["status"] == "success"]
                lat = f"{bs1[0]['latency_ms_per_image']:.2f} ms" if bs1 else "N/A"
                if hits:
                    best = max(hits, key=lambda x: x.get("throughput_img_per_sec", 0) or 0)
                    lines.append(f"| {model} | {prec} | {lat} | {best['throughput_img_per_sec']:.1f} img/s | {best['batch_size']} |")
        lines.append("")

    # ── Inference NLP ─────────────────────────────────────────────────────
    inlp = data.get("inference_nlp")
    if inlp:
        lines.append("## Inference Performance — NLP\n")
        lines.append("| Model | Precision | BS=1 Latency | Best Throughput | Best BS |")
        lines.append("|-------|-----------|-------------|-----------------|---------|")
        for model in ["bert-base", "bert-large"]:
            for prec in ["FP32", "FP16", "BF16"]:
                bs1 = [r for r in inlp["results"]
                       if r["model"] == model and r["precision"] == prec
                       and r["batch_size"] == 1 and r["status"] == "success"]
                hits = [r for r in inlp["results"]
                        if r["model"] == model and r["precision"] == prec and r["status"] == "success"]
                lat = f"{bs1[0]['latency_ms_per_sample']:.2f} ms" if bs1 else "N/A"
                if hits:
                    best = max(hits, key=lambda x: x.get("throughput_samples_per_sec", 0) or 0)
                    lines.append(f"| {model} | {prec} | {lat} | {best['throughput_samples_per_sec']:.1f} s/s | {best['batch_size']} |")
        lines.append("")

    # ── LLM Performance ──────────────────────────────────────────────────
    llm = data.get("llm")
    if llm:
        lines.append("## LLM Performance (Tokens/Second)\n")
        lines.append("| Model | Size | Quantization | Tokens/sec | TTFT (ms) | Status |")
        lines.append("|-------|------|--------------|------------|-----------|--------|")
        for r in llm["results"]:
            tps = f"{r['tokens_per_second']:.1f}" if r.get("tokens_per_second") else "-"
            ttft = f"{r['time_to_first_token_ms']:.0f}" if r.get("time_to_first_token_ms") else "-"
            lines.append(f"| {r['model']} | {r['size_gb']:.1f}GB | {r['quantization']} | {tps} | {ttft} | {r['status']} |")
        lines.append("")

    # ── VRAM Limits ───────────────────────────────────────────────────────
    vram = data.get("vram_limits")
    if vram:
        lines.append("## VRAM Capabilities\n")
        lines.append(f"- **Maximum Model Size**: ~{vram.get('max_model_size_label', 'N/A')} parameters")
        mc = vram.get("max_context_length", 0)
        lines.append(f"- **Maximum Context Length** (7B model): {mc:,} tokens")
        lines.append(f"- **Simultaneous 7B Models**: {vram.get('max_simultaneous_7b_models', 'N/A')}")
        lines.append("")

    # ── llmfit — Best Loadable LLMs for this GPU ─────────────────────────
    vram_gb = meta.get("vram_gb", 0)
    llmfit_path = Path(__file__).resolve().parent.parent / "benchmarks" / "llmfit_vram_tiers.json"
    if vram_gb > 0 and llmfit_path.exists():
        try:
            with llmfit_path.open() as _f:
                llmfit_data = json.load(_f)
            tiers = llmfit_data.get("vram_tiers", {})
            tier_keys = sorted(tiers.keys(), key=lambda k: int(k))
            matched_tier = None
            for tk in tier_keys:
                if int(tk) <= vram_gb:
                    matched_tier = tk
            if matched_tier is None and tier_keys:
                matched_tier = tier_keys[0]
            if matched_tier and matched_tier in tiers:
                tier = tiers[matched_tier]
                lines.append(f"## LLM Fit Analysis ({tier['label']} VRAM tier)\n")
                lines.append(f"*Based on [llmfit](https://github.com/AlexsJones/llmfit) model database — "
                             f"best models for {vram_gb:.0f} GB VRAM*\n")

                # Top recommended models
                top = tier.get("top_models", [])
                if top:
                    lines.append("### Best Recommended Models\n")
                    lines.append("| Model | Params | Quant | VRAM | Fit | Use Case |")
                    lines.append("|-------|--------|-------|------|-----|----------|")
                    for m in top:
                        lines.append(f"| {m['name']} | {m['params_b']}B | {m['best_quant']} | "
                                     f"{m['vram_required_gb']}GB | {m['fit']} | {m['use_case']} |")
                    lines.append("")

                # Largest quantized models
                lq = tier.get("largest_quantized", [])
                if lq:
                    lines.append("### Largest Loadable Models (aggressive quantization)\n")
                    lines.append("| Model | Params | Quant | VRAM | Mode | Note |")
                    lines.append("|-------|--------|-------|------|------|------|")
                    for m in lq:
                        vr = m.get("vram_required_gb", "?")
                        vr_str = f"{vr}GB" if isinstance(vr, (int, float)) else str(vr)
                        lines.append(f"| {m['name']} | {m['params_b']}B | {m['best_quant']} | "
                                     f"{vr_str} | {m.get('run_mode', '')} | {m.get('note', '')} |")
                    lines.append("")

                ld = tier.get("largest_dense_model")
                lm = tier.get("largest_moe_model")
                if ld:
                    lines.append(f"- **Largest Dense Model**: {ld}")
                if lm:
                    lines.append(f"- **Largest MoE Model**: {lm}")
                if ld or lm:
                    lines.append("")
        except Exception:
            pass

    # ── GEMM Compute Stress ───────────────────────────────────────────────
    gemm = data.get("gemm_stress")
    if gemm:
        lines.append("## GEMM Compute Stress\n")
        peak = gemm.get("peak_tflops", {})
        if peak:
            lines.append("### Peak TFLOPS by Precision\n")
            lines.append("| Precision | Peak TFLOPS |")
            lines.append("|-----------|-------------|")
            for prec, val in peak.items():
                lines.append(f"| {prec} | {val:.2f} |")
            lines.append("")
        # Per-size detail table
        gemm_rows = gemm.get("results", [])
        if gemm_rows:
            lines.append("### Per-Size Results\n")
            lines.append("| Precision | Size | TFLOPS | Time (ms) |")
            lines.append("|-----------|------|--------|-----------|")
            for r in gemm_rows:
                lines.append(f"| {r.get('precision','')} | {r['matrix_size']}×{r['matrix_size']} | {r['tflops']:.2f} | {r['time_ms']:.2f} |")
            lines.append("")

    # ── GPU Fundamentals ──────────────────────────────────────────────────
    fund = data.get("gpu_fundamentals")
    if fund:
        lines.append("## GPU Fundamentals\n")
        fund_rows = fund.get("results", [])

        # Memory Bandwidth
        bw = [r for r in fund_rows if r.get("category") == "memory_bandwidth"]
        if bw:
            peak_bw = max(r["value"] for r in bw)
            lines.append(f"- **Peak Memory Bandwidth (D2D)**: {peak_bw:.1f} GB/s")

        # PCIe
        h2d = [r for r in fund_rows if r.get("category") == "pcie" and "h2d" in r.get("test", "")]
        d2h = [r for r in fund_rows if r.get("category") == "pcie" and "d2h" in r.get("test", "")]
        if h2d:
            lines.append(f"- **PCIe H2D Bandwidth**: {max(r['value'] for r in h2d):.2f} GB/s")
        if d2h:
            lines.append(f"- **PCIe D2H Bandwidth**: {max(r['value'] for r in d2h):.2f} GB/s")

        # Kernel launch latency
        kl = [r for r in fund_rows if r.get("category") == "kernel_launch"]
        if kl:
            lines.append(f"- **Kernel Launch Latency**: {min(r['value'] for r in kl):.2f} μs")

        # Attention SDPA
        attn = [r for r in fund_rows if r.get("category") == "attention"]
        if attn:
            lines.append("")
            lines.append("### Attention (SDPA) Throughput\n")
            lines.append("| Config | Dtype | TFLOPS |")
            lines.append("|--------|-------|--------|")
            for r in attn:
                tflops = r["value"] / 1000
                lines.append(f"| {r['notes']} | {r['dtype']} | {tflops:.1f} |")

        # Reduction
        red = [r for r in fund_rows if r.get("category") == "reduction"]
        if red:
            peak_red = max(r["value"] for r in red)
            lines.append(f"\n- **Peak Reduction Bandwidth**: {peak_red:.1f} GB/s")

        # FFT
        fft32 = [r for r in fund_rows if r.get("category") == "fft" and r.get("dtype") == "FP32"]
        fft64 = [r for r in fund_rows if r.get("category") == "fft" and r.get("dtype") == "FP64"]
        if fft32:
            lines.append(f"- **Peak FFT FP32**: {max(r['value'] for r in fft32):.1f} GFLOPS")
        if fft64:
            lines.append(f"- **Peak FFT FP64**: {max(r['value'] for r in fft64):.1f} GFLOPS")

        # SpMM
        spmm = [r for r in fund_rows if r.get("category") == "spmm"]
        if spmm:
            lines.append(f"- **Peak SpMM**: {max(r['value'] for r in spmm):.1f} GFLOPS")

        lines.append("")

    # ── Multi-GPU Scaling ─────────────────────────────────────────────────
    mgpu = data.get("multi_gpu_scaling")
    if mgpu:
        mgpu_rows = mgpu.get("results", [])
        success_rows = [r for r in mgpu_rows if r.get("status") == "success"]
        if success_rows:
            lines.append("## Multi-GPU Training Scaling\n")
            lines.append("| Model | Method | GPUs | Throughput | Efficiency | Speedup |")
            lines.append("|-------|--------|------|------------|------------|---------|")
            for r in success_rows:
                model = r.get("model", "")
                method = r.get("method", "")
                n_gpus = r.get("n_gpus", 1)
                tput = r.get("throughput_samples_per_sec", 0)
                eff = r.get("scaling_efficiency_pct")
                spd = r.get("speedup")
                eff_str = f"{eff:.1f}%" if eff is not None else "—"
                spd_str = f"{spd:.2f}×" if spd is not None and n_gpus > 1 else "—"
                lines.append(f"| {model} | {method} | {n_gpus} | {tput:.1f} s/s | {eff_str} | {spd_str} |")
            lines.append("")

    # ── HW Monitor stats (if available) ──────────────────────────────────
    any_hw = False
    for d in data.values():
        if d and d.get("hw_monitor"):
            hw = d["hw_monitor"]
            if not any_hw:
                lines.append("## Hardware Monitoring\n")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                any_hw = True
            lines.append(f"| Avg Power | {hw.get('avg_power_w', 'N/A')} W |")
            lines.append(f"| Max Power | {hw.get('max_power_w', 'N/A')} W |")
            lines.append(f"| Avg Temperature | {hw.get('avg_temp_c', 'N/A')} C |")
            lines.append(f"| Max Temperature | {hw.get('max_temp_c', 'N/A')} C |")
            break
    if any_hw:
        lines.append("")

    # ── DLPerf Estimate ──────────────────────────────────────────────────
    if tv:
        fp32_hits = [r for r in tv["results"]
                     if r["model"] == "resnet50" and r["precision"] == "FP32" and r["status"] == "success"]
        if fp32_hits:
            best = max(fp32_hits, key=lambda x: x.get("throughput_img_per_sec", 0) or 0)
            tp = best["throughput_img_per_sec"]
            dlperf = tp / 13.0
            lines.append("## DLPerf Score (Estimated)\n")
            lines.append(f"Based on ResNet-50 FP32 training throughput ({tp:.1f} img/s):\n")
            lines.append(f"**Estimated DLPerf**: {dlperf:.1f}\n")
            lines.append("> DLPerf formula: `ResNet-50 FP32 throughput / 13.0` (approximate Vast.ai methodology)")
            lines.append("")

    lines.append("---\n")
    lines.append("*Report generated by GPU AI Benchmark Suite*\n")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  CSV summary
# ═════════════════════════════════════════════════════════════════════════════

def generate_csv_summary(data: dict[str, Any]) -> list[dict[str, str]]:
    meta = _gpu_meta(data)
    gpu = meta["gpu_name"]
    vram = meta["vram_gb"]
    si = meta.get("system_info", {})
    rows: list[dict[str, str]] = []

    def _add(test_type: str, metric: str, value: str):
        rows.append({
            "gpu": gpu,
            "vram_gb": vram,
            "nvidia_driver": si.get("nvidia_driver_version", ""),
            "cuda_pytorch": si.get("pytorch_cuda_version", ""),
            "cuda_toolkit_nvcc": si.get("cuda_toolkit_version", ""),
            "pytorch_version": si.get("pytorch_version", ""),
            "test_type": test_type,
            "metric": metric,
            "value": value,
        })

    # Training Vision
    tv = data.get("training_vision")
    if tv:
        for r in tv["results"]:
            if r["status"] == "success":
                _add("Training Vision", f"{r['model']} {r['precision']} BS={r['batch_size']}",
                     f"{r['throughput_img_per_sec']:.1f} img/s")

    # Training NLP
    tn = data.get("training_nlp")
    if tn:
        for r in tn["results"]:
            if r["status"] == "success":
                _add("Training NLP", f"{r['model']} {r['precision']} BS={r['batch_size']}",
                     f"{r['throughput_samples_per_sec']:.1f} samples/s")

    # Training Detection
    td = data.get("training_detection")
    if td:
        for r in td["results"]:
            if r["status"] == "success":
                _add("Training Detection", f"{r['model']} {r['precision']} BS={r['batch_size']}",
                     f"{r['throughput_img_per_sec']:.2f} img/s")

    # Inference Vision
    iv = data.get("inference_vision")
    if iv:
        for r in iv["results"]:
            if r["status"] == "success":
                _add("Inference Vision", f"{r['model']} {r['precision']} BS={r['batch_size']}",
                     f"{r['throughput_img_per_sec']:.1f} img/s")

    # Inference NLP
    inlp = data.get("inference_nlp")
    if inlp:
        for r in inlp["results"]:
            if r["status"] == "success":
                _add("Inference NLP", f"{r['model']} {r['precision']} BS={r['batch_size']}",
                     f"{r['throughput_samples_per_sec']:.1f} samples/s")

    # GEMM Stress
    gemm = data.get("gemm_stress")
    if gemm:
        peak = gemm.get("peak_tflops", {})
        for prec, val in peak.items():
            _add("GEMM Stress", f"{prec} Peak", f"{val:.2f} TFLOPS")

    # GPU Fundamentals
    fund = data.get("gpu_fundamentals")
    if fund:
        for r in fund.get("results", []):
            cat = r.get("category", "")
            _add("GPU Fundamentals", f"{cat} — {r.get('test', '')}",
                 f"{r['value']:.2f} {r.get('unit', '')}")

    # LLM
    llm = data.get("llm")
    if llm:
        for r in llm["results"]:
            if r.get("tokens_per_second"):
                _add("LLM", r["model"], f"{r['tokens_per_second']:.1f} t/s")

    # Multi-GPU Scaling
    mgpu = data.get("multi_gpu_scaling")
    if mgpu:
        for r in mgpu.get("results", []):
            if r.get("status") == "success":
                method = r.get("method", "")
                model = r.get("model", "")
                n = r.get("n_gpus", 1)
                tput = r.get("throughput_samples_per_sec", 0)
                eff = r.get("scaling_efficiency_pct")
                eff_str = f" eff={eff:.0f}%" if eff is not None and n > 1 else ""
                _add("Multi-GPU Scaling",
                     f"{model} {method} {n}GPU",
                     f"{tput:.1f} s/s{eff_str}")

    return rows


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Benchmark Report")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                        help="Directory containing JSON result files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Derive session_id from the directory name (if it's a session subfolder)
    session_id = results_dir.name if results_dir != DEFAULT_RESULTS_DIR else ""

    print("=" * 70)
    print("Generating Benchmark Report")
    if session_id:
        print(f"  Session: {session_id}")
    print("=" * 70)
    print()

    data = load_results(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load session metadata (run timing, pass/fail counts)
    session_meta: dict[str, Any] | None = None
    meta_path = results_dir / "session_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                session_meta = json.load(f)
        except Exception:
            pass

    # Markdown
    print("Generating Markdown report...")
    md = generate_markdown_report(data, session_id=session_id, session_meta=session_meta)
    md_path = results_dir / "benchmark_summary.md"
    md_path.write_text(md, encoding="utf-8")

    # CSV
    print("Generating CSV summary...")
    csv_rows = generate_csv_summary(data)
    csv_path = results_dir / "benchmark_summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    # JSON
    print("Generating JSON summary...")
    json_blob = data.copy()
    if session_id:
        json_blob["session_id"] = session_id
    if session_meta:
        json_blob["session_meta"] = session_meta
    json_path = results_dir / "benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_blob, f, indent=2, default=str)

    print(f"\nGenerated files:")
    print(f"  - {md_path}")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print()


if __name__ == "__main__":
    main()
