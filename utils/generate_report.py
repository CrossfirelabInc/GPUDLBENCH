#!/usr/bin/env python3
"""
Report Generator — GPU AI Benchmark Suite

Reads all JSON result files from results/ and produces:
  - Markdown summary  (benchmark_summary.md)
  - CSV summary       (benchmark_summary.csv)
  - JSON blob         (benchmark_summary.json)
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("report")

RESULTS_DIR = Path("results")


# ═════════════════════════════════════════════════════════════════════════════
#  Load results (with specific exception handling)
# ═════════════════════════════════════════════════════════════════════════════

def _load_json(filename: str) -> Optional[dict]:
    """Load a single JSON result file, returning None on failure."""
    path = RESULTS_DIR / filename
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


def load_results() -> Dict[str, Any]:
    """Load all benchmark result JSON files."""
    return {
        "training_vision": _load_json("training_vision.json"),
        "training_nlp": _load_json("training_nlp.json"),
        "inference_vision": _load_json("inference_vision.json"),
        "inference_nlp": _load_json("inference_nlp.json"),
        "llm": _load_json("llm_tokens_per_sec.json"),
        "vram_limits": _load_json("vram_limits.json"),
        "mixed_precision": _load_json("mixed_precision.json"),
        "training_detection": _load_json("training_detection.json"),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  GPU info extraction
# ═════════════════════════════════════════════════════════════════════════════

def _gpu_meta(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract gpu name, vram, etc. from the first available result file."""
    for d in data.values():
        if d and "gpu" in d:
            return {
                "gpu_name": d.get("gpu", "Unknown GPU"),
                "vram_gb": d.get("vram_gb", 0),
                "compute_capability": d.get("compute_capability", "N/A"),
            }
    return {"gpu_name": "Unknown GPU", "vram_gb": 0, "compute_capability": "N/A"}


# ═════════════════════════════════════════════════════════════════════════════
#  Markdown report
# ═════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(data: Dict[str, Any]) -> str:
    meta = _gpu_meta(data)
    lines: List[str] = []

    lines.append("# GPU AI Benchmark Results\n")
    lines.append(f"**GPU**: {meta['gpu_name']}  ")
    lines.append(f"**VRAM**: {meta['vram_gb']:.1f} GB  ")
    lines.append(f"**Compute Capability**: {meta['compute_capability']}  ")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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

    # ── Mixed Precision ───────────────────────────────────────────────────
    mp = data.get("mixed_precision")
    if mp:
        lines.append("## Mixed Precision Speedup\n")
        lines.append("| Model | Precision | Throughput | Speedup vs FP32 |")
        lines.append("|-------|-----------|------------|-----------------|")
        for r in mp["results"]:
            unit = "img/s" if r["model_type"] == "vision" else "samples/s"
            lines.append(f"| {r['model']} | {r['precision']} | {r['throughput']:.1f} {unit} | {r['speedup_vs_fp32']:.2f}x |")
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

def generate_csv_summary(data: Dict[str, Any]) -> List[Dict[str, str]]:
    meta = _gpu_meta(data)
    gpu = meta["gpu_name"]
    vram = meta["vram_gb"]
    rows: List[Dict[str, str]] = []

    def _add(test_type: str, metric: str, value: str):
        rows.append({"gpu": gpu, "vram_gb": vram, "test_type": test_type, "metric": metric, "value": value})

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

    # LLM
    llm = data.get("llm")
    if llm:
        for r in llm["results"]:
            if r.get("tokens_per_second"):
                _add("LLM", r["model"], f"{r['tokens_per_second']:.1f} t/s")

    # Mixed Precision
    mp = data.get("mixed_precision")
    if mp:
        for r in mp["results"]:
            _add("Mixed Precision", f"{r['model']} {r['precision']}",
                 f"{r['speedup_vs_fp32']:.2f}x vs FP32")

    return rows


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("Generating Benchmark Report")
    print("=" * 70)
    print()

    data = load_results()
    RESULTS_DIR.mkdir(exist_ok=True)

    # Markdown
    print("Generating Markdown report...")
    md = generate_markdown_report(data)
    md_path = RESULTS_DIR / "benchmark_summary.md"
    md_path.write_text(md, encoding="utf-8")

    # CSV
    print("Generating CSV summary...")
    csv_rows = generate_csv_summary(data)
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    # JSON
    print("Generating JSON summary...")
    json_path = RESULTS_DIR / "benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nGenerated files:")
    print(f"  - {md_path}")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print()


if __name__ == "__main__":
    main()
