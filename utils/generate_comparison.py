#!/usr/bin/env python3
"""
utils/generate_comparison.py — Extract metrics + generate comparison charts

Scans all benchmark sessions under results/, extracts metrics into
comparison.csv/json, and generates side-by-side GPU comparison charts.

Usage:
  python utils/generate_comparison.py
  python utils/generate_comparison.py --results-dir /path/to/results
  python utils/generate_comparison.py --sessions 20260218_abc123
  python utils/generate_comparison.py --skip-charts
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ─── Project root ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# ─── Style ────────────────────────────────────────────────────────────────────
BG_COLOR      = "#0D1117"
CARD_COLOR    = "#161B22"
TEXT_COLOR    = "#E6EDF3"
SUBTEXT_COLOR = "#8B949E"
GRID_COLOR    = "#21262D"
ACCENT_GREEN  = "#3FB950"
ACCENT_BLUE   = "#58A6FF"
ACCENT_PURPLE = "#BC8CFF"
ACCENT_ORANGE = "#F0883E"
ACCENT_RED    = "#F85149"
ACCENT_CYAN   = "#39D2C0"
ACCENT_YELLOW = "#D29922"

GPU_COLORS = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE,
              ACCENT_ORANGE, ACCENT_CYAN, ACCENT_YELLOW]

DPI = 150
FIG_W, FIG_H = 16, 9

WATERMARK = "Crossfirelab GPU AI Benchmark Suite"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   CARD_COLOR,
    "axes.edgecolor":   GRID_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "grid.color":       GRID_COLOR,
    "grid.alpha":       0.4,
    "font.family":      "sans-serif",
    "font.size":        12,
})


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 — Metric extraction
# ═══════════════════════════════════════════════════════════════════════════════

MetricDict = Dict[str, Any]


def _best(rows: List[Dict], filter_fn, value_key: str) -> Optional[float]:
    hits = [r.get(value_key) for r in rows if filter_fn(r) and r.get(value_key) is not None]
    return max(hits) if hits else None


def _safe(v: Any) -> Any:
    if isinstance(v, float):
        import math as _m
        return None if (_m.isnan(v) or _m.isinf(v)) else round(v, 4)
    return v


# ── Extractors ────────────────────────────────────────────────────────────────

def extract_training_vision(data: Dict) -> MetricDict:
    rows = data.get("results", [])
    out: MetricDict = {}
    for model in ("resnet50", "resnet101"):
        for prec in ("FP32", "FP16", "BF16", "FP8"):
            v = _best(rows,
                      lambda r, m=model, p=prec: (r.get("model") == m
                                                   and r.get("precision") == p
                                                   and r.get("status") == "success"),
                      "throughput_img_per_sec")
            if v is not None:
                out[f"train_vision_{model}_{prec.lower()}_img_per_sec"] = _safe(v)
    return out


def extract_training_nlp(data: Dict) -> MetricDict:
    rows = data.get("results", [])
    out: MetricDict = {}
    for model in ("bert-base", "bert-large"):
        for prec in ("FP32", "FP16", "BF16", "FP8"):
            v = _best(rows,
                      lambda r, m=model, p=prec: (r.get("model") == m
                                                   and r.get("precision") == p
                                                   and r.get("status") == "success"),
                      "throughput_samples_per_sec")
            if v is not None:
                out[f"train_nlp_{model}_{prec.lower()}_samples_per_sec"] = _safe(v)
    return out


def extract_inference_vision(data: Dict) -> MetricDict:
    rows = data.get("results", [])
    out: MetricDict = {}
    for model in ("resnet50", "resnet101"):
        for prec in ("FP32", "FP16", "BF16", "FP8"):
            v = _best(rows,
                      lambda r, m=model, p=prec: (r.get("model") == m
                                                   and r.get("precision") == p
                                                   and r.get("status") == "success"),
                      "throughput_img_per_sec")
            if v is not None:
                out[f"infer_vision_{model}_{prec.lower()}_img_per_sec"] = _safe(v)
    return out


def extract_inference_nlp(data: Dict) -> MetricDict:
    rows = data.get("results", [])
    out: MetricDict = {}
    for model in ("bert-base", "bert-large"):
        for prec in ("FP32", "FP16", "BF16", "FP8"):
            v = _best(rows,
                      lambda r, m=model, p=prec: (r.get("model") == m
                                                   and r.get("precision") == p
                                                   and r.get("status") == "success"),
                      "throughput_samples_per_sec")
            if v is not None:
                out[f"infer_nlp_{model}_{prec.lower()}_samples_per_sec"] = _safe(v)
    return out


def extract_llm_tokens(data: Dict) -> MetricDict:
    rows = data.get("results", [])
    out: MetricDict = {}
    best_tps = 0.0
    best_model_desc = None
    for r in rows:
        if r.get("status") == "success" and r.get("tokens_per_second") is not None:
            key = r["model"].lower().replace(" ", "_").replace("-", "_").replace(".", "_")
            tps = r["tokens_per_second"]
            out[f"llm_{key}_tokens_per_sec"] = _safe(tps)
            if r.get("time_to_first_token_ms") is not None:
                out[f"llm_{key}_ttft_ms"] = _safe(r["time_to_first_token_ms"])
            if tps > best_tps:
                best_tps = tps
                quant = r.get("quantization", "")
                best_model_desc = f"{r['model']} {quant}".strip()
    if best_model_desc:
        out["llm_best_model"] = best_model_desc
        out["llm_best_tokens_per_sec"] = _safe(best_tps)
    return out


def extract_vram_limits(data: Dict) -> MetricDict:
    rows = data.get("results", [])
    out: MetricDict = {}
    loadable = [r for r in rows if r.get("loadable")]
    if loadable:
        largest = max(loadable, key=lambda r: r.get("approx_params_b", 0))
        out["vram_largest_loadable_params_b"] = _safe(largest.get("approx_params_b"))
        out["vram_largest_loadable_actual_gb"] = _safe(largest.get("actual_vram_gb"))
    untried = [r for r in rows if not r.get("loadable")]
    if untried:
        smallest_fail = min(untried, key=lambda r: r.get("approx_params_b", 1e9), default=None)
        if smallest_fail:
            out["vram_smallest_oom_params_b"] = _safe(smallest_fail.get("approx_params_b"))
    max_ctx = data.get("max_context_length")
    if max_ctx is not None:
        out["vram_max_context_length"] = _safe(max_ctx)
    max_label = data.get("max_model_size_label")
    if max_label is not None:
        out["vram_max_model_size_label"] = str(max_label)
    if loadable:
        largest = max(loadable, key=lambda r: r.get("approx_params_b", 0))
        lbl = largest.get("label", "")
        gb = largest.get("actual_vram_gb")
        if lbl:
            detail = f"{lbl} params"
            if gb:
                detail += f" ({gb:.1f} GB VRAM)"
            out["vram_max_model_detail"] = detail
    return out


def extract_gemm_stress(data: Dict) -> MetricDict:
    out: MetricDict = {}
    peak = data.get("peak_tflops", {})
    for prec, val in peak.items():
        out[f"gemm_{prec.lower()}_peak_tflops"] = _safe(val)
    if not peak:
        rows = data.get("results", [])
        by_prec: Dict[str, float] = {}
        for r in rows:
            p = r.get("precision", "")
            t = r.get("tflops")
            if p and t is not None:
                by_prec[p] = max(by_prec.get(p, 0.0), t)
        for prec, val in by_prec.items():
            out[f"gemm_{prec.lower()}_peak_tflops"] = _safe(val)
    return out


def extract_training_detection(data: Dict) -> MetricDict:
    rows = data.get("results", [])
    out: MetricDict = {}
    for model in ("faster-rcnn-resnet50", "mask-rcnn-resnet50"):
        for prec in ("FP32", "FP16", "BF16"):
            v = _best(rows,
                      lambda r, m=model, p=prec: (r.get("model") == m
                                                   and r.get("precision") == p
                                                   and r.get("status") == "success"),
                      "throughput_img_per_sec")
            if v is not None:
                safe_model = model.replace("-", "_")
                out[f"detect_{safe_model}_{prec.lower()}_img_per_sec"] = _safe(v)
    return out


def extract_gpu_fundamentals(data: Dict) -> MetricDict:
    rows = data.get("results", [])
    out: MetricDict = {}
    bw_rows = [r for r in rows if r.get("category") == "memory_bandwidth"]
    if bw_rows:
        out["fund_d2d_bw_peak_gb_s"] = _safe(max(r["value"] for r in bw_rows))
    h2d = [r for r in rows if r.get("category") == "pcie" and "h2d" in r.get("test", "")]
    if h2d:
        out["fund_pcie_h2d_gb_s"] = _safe(max(r["value"] for r in h2d))
    d2h = [r for r in rows if r.get("category") == "pcie" and "d2h" in r.get("test", "")]
    if d2h:
        out["fund_pcie_d2h_gb_s"] = _safe(max(r["value"] for r in d2h))
    for dtype in ("FP32", "FP64"):
        fft = [r for r in rows if r.get("category") == "fft" and r.get("dtype") == dtype]
        if fft:
            out[f"fund_fft_{dtype.lower()}_peak_gflops"] = _safe(max(r["value"] for r in fft))
    for dtype in ("FP32", "FP64"):
        nb = [r for r in rows if r.get("category") == "nbody" and r.get("dtype") == dtype]
        if nb:
            out[f"fund_nbody_{dtype.lower()}_m_steps_per_s"] = _safe(max(r["value"] for r in nb))
    spmm = [r for r in rows if r.get("category") == "spmm"]
    if spmm:
        out["fund_spmm_peak_gflops"] = _safe(max(r["value"] for r in spmm))
    red = [r for r in rows if r.get("category") == "reduction"]
    if red:
        out["fund_reduction_peak_gb_s"] = _safe(max(r["value"] for r in red))
    kl = [r for r in rows if r.get("category") == "kernel_launch"]
    if kl:
        out["fund_kernel_launch_latency_us"] = _safe(min(r["value"] for r in kl))
    return out


def extract_multi_gpu_scaling(data: Dict) -> MetricDict:
    rows = data.get("results", [])
    out: MetricDict = {}
    for r in rows:
        if r.get("status") != "success":
            continue
        model  = r.get("model", "").replace("-", "_")
        mode   = r.get("mode", "")
        n_gpus = r.get("n_gpus", 1)
        tput   = r.get("throughput_samples_per_sec")
        eff    = r.get("scaling_efficiency_pct")
        if tput is not None:
            out[f"multigpu_{model}_{mode}_{n_gpus}gpu_samples_per_sec"] = _safe(tput)
        if eff is not None and n_gpus > 1:
            out[f"multigpu_{model}_{mode}_{n_gpus}gpu_eff_pct"] = _safe(eff)
    return out


_EXTRACTORS = {
    "training_vision.json":    extract_training_vision,
    "training_nlp.json":       extract_training_nlp,
    "inference_vision.json":   extract_inference_vision,
    "inference_nlp.json":      extract_inference_nlp,
    "llm_tokens_per_sec.json": extract_llm_tokens,
    "vram_limits.json":        extract_vram_limits,
    "gemm_stress.json":        extract_gemm_stress,
    "training_detection.json": extract_training_detection,
    "gpu_fundamentals.json":   extract_gpu_fundamentals,
    "multi_gpu_scaling.json":  extract_multi_gpu_scaling,
}


# ── Session loader ────────────────────────────────────────────────────────────

def _load_session(session_dir: Path) -> Tuple[str, str, MetricDict]:
    meta_path = session_dir / "session_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"session_meta.json not found in {session_dir}")

    with meta_path.open() as f:
        meta = json.load(f)

    session_id = meta.get("session_id", session_dir.name)
    date_str = session_id[:8]

    gpu_name = "GPU"
    for p in session_dir.glob("*.json"):
        if p.name in ("session_meta.json", "benchmark_summary.json"):
            continue
        try:
            with p.open() as f:
                d = json.load(f)
            if "gpu" in d:
                gpu_name = d["gpu"].replace(" ", "_").replace("/", "_")
                break
        except Exception:
            continue

    col_id = f"{gpu_name}_{date_str}"
    col_label = f"{gpu_name.replace('_', ' ')} — {date_str}"

    combined: MetricDict = {}
    combined["session_id"] = session_id
    combined["gpu_name"] = gpu_name.replace("_", " ")
    combined["run_date"] = date_str
    combined["elapsed_sec"] = meta.get("elapsed_seconds")

    for filename, extractor in _EXTRACTORS.items():
        jpath = session_dir / filename
        if not jpath.exists():
            continue
        try:
            with jpath.open() as f:
                data = json.load(f)
            combined.update(extractor(data))
        except Exception as e:
            print(f"  WARNING: could not parse {jpath.name} — {e}", file=sys.stderr)

    # Aggregate hw_monitor across all benchmarks
    power_vals, temp_vals = [], []
    for jpath in session_dir.glob("*.json"):
        if jpath.name in ("session_meta.json", "benchmark_summary.json"):
            continue
        try:
            with jpath.open() as f:
                data = json.load(f)
            hw = data.get("hw_monitor")
            if hw:
                if hw.get("avg_power_w") is not None:
                    power_vals.append(hw["avg_power_w"])
                if hw.get("avg_temp_c") is not None:
                    temp_vals.append(hw["avg_temp_c"])
        except Exception:
            continue
    if power_vals:
        combined["hw_avg_power_w"] = round(sum(power_vals) / len(power_vals), 1)
        combined["hw_peak_power_w"] = round(max(power_vals), 1)
    if temp_vals:
        combined["hw_avg_temp_c"] = round(sum(temp_vals) / len(temp_vals), 1)
        combined["hw_peak_temp_c"] = round(max(temp_vals), 1)

    return col_id, col_label, combined


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(
    sessions: List[Tuple[str, str, MetricDict]],
) -> Tuple[List[str], List[str], List[Dict]]:
    meta_keys = ["session_id", "gpu_name", "run_date", "elapsed_sec"]
    seen = set(meta_keys)
    prefix_order = [
        "train_vision", "train_nlp", "infer_vision", "infer_nlp",
        "llm_", "vram_", "gemm_", "detect_", "fund_", "multigpu_", "hw_",
    ]

    all_keys: set = set()
    for _, _, metrics in sessions:
        all_keys.update(metrics.keys())
    all_keys -= seen

    def _prefix_rank(k: str) -> int:
        for i, pfx in enumerate(prefix_order):
            if k.startswith(pfx):
                return i
        return len(prefix_order)

    metric_keys = sorted(all_keys, key=lambda k: (_prefix_rank(k), k))
    col_ids = [c[0] for c in sessions]

    rows: List[Dict] = []
    for key in meta_keys + metric_keys:
        row: Dict = {"metric_id": key}
        for col_id, _, metrics in sessions:
            row[col_id] = metrics.get(key)
        rows.append(row)

    return meta_keys + metric_keys, col_ids, rows


# ── Output writers ────────────────────────────────────────────────────────────

def write_csv(rows: List[Dict], col_ids: List[str], output_path: Path) -> None:
    fieldnames = ["metric_id"] + col_ids
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: List[Dict], col_ids: List[str],
               sessions: List[Tuple[str, str, MetricDict]],
               output_path: Path) -> None:
    col_labels = {c[0]: c[1] for c in sessions}
    out = {
        "generated_at": datetime.now().isoformat(),
        "columns": [{"id": c, "label": col_labels.get(c, c)} for c in col_ids],
        "metrics": rows,
    }
    with output_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 — Chart generation
# ═══════════════════════════════════════════════════════════════════════════════

class ComparisonData:
    """Parsed comparison.json with convenient metric lookup."""

    def __init__(self, path: Path):
        with path.open() as f:
            raw = json.load(f)
        self.columns: List[Dict] = raw["columns"]
        self.col_ids: List[str] = [c["id"] for c in self.columns]
        self.col_labels: List[str] = [c["label"] for c in self.columns]
        self._metrics: Dict[str, Dict[str, Any]] = {}
        for row in raw["metrics"]:
            mid = row["metric_id"]
            self._metrics[mid] = {k: v for k, v in row.items() if k != "metric_id"}

    @property
    def n_gpus(self) -> int:
        return len(self.col_ids)

    def gpu_names(self) -> List[str]:
        names = []
        for cid in self.col_ids:
            v = self._metrics.get("gpu_name", {}).get(cid, cid)
            names.append(str(v))
        return names

    def get(self, metric_id: str) -> List[Optional[float]]:
        row = self._metrics.get(metric_id, {})
        return [row.get(cid) for cid in self.col_ids]

    def metric_ids(self) -> List[str]:
        return list(self._metrics.keys())

    def find_metrics(self, prefix: str) -> List[str]:
        return [m for m in self._metrics if m.startswith(prefix)]


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR,
                edgecolor="none", pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ {path.name}")


def _watermark(fig):
    fig.text(0.99, 0.01, WATERMARK,
             ha="right", va="bottom", fontsize=9, color=SUBTEXT_COLOR, alpha=0.5)


def _direction_text(fig, text: str, color: str):
    """Place direction indicator just below the suptitle, right-aligned."""
    fig.text(0.97, 0.935, text, ha="right", va="top",
             fontsize=10, color=color, alpha=0.9)


def _grouped_bar(ax, categories: List[str], gpu_names: List[str],
                 values_per_gpu: List[List[float]], ylabel: str,
                 title: str, title_color: str,
                 fmt: str = "{:.0f}", val_fontsize: int = 10):
    n_groups = len(categories)
    n_gpus = len(gpu_names)
    if n_groups == 0 or n_gpus == 0:
        return

    x = np.arange(n_groups)
    total_width = 0.75
    bar_w = total_width / n_gpus

    all_vals = []
    for gi, (gname, gvals) in enumerate(zip(gpu_names, values_per_gpu)):
        offset = (gi - (n_gpus - 1) / 2) * bar_w
        color = GPU_COLORS[gi % len(GPU_COLORS)]
        vals = [v if v is not None else 0 for v in gvals]
        all_vals.extend(vals)
        bars = ax.bar(x + offset, vals, bar_w * 0.88, color=color,
                      edgecolor=BG_COLOR, linewidth=1, zorder=3, label=gname)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt.format(v), ha="center", va="bottom",
                        fontsize=val_fontsize, color=TEXT_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, color=title_color, pad=10)
    ax.grid(axis="y", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    max_v = max(all_vals) if all_vals else 1
    ax.set_ylim(0, max_v * 1.18)
    ax.legend(fontsize=10, loc="upper left", facecolor=CARD_COLOR, edgecolor=GRID_COLOR)


def _horizontal_grouped_bar(ax, categories: List[str], gpu_names: List[str],
                            values_per_gpu: List[List[float]], xlabel: str,
                            title: str, title_color: str,
                            fmt: str = "{:.1f}"):
    n_groups = len(categories)
    n_gpus = len(gpu_names)
    if n_groups == 0 or n_gpus == 0:
        return

    y = np.arange(n_groups)
    total_height = 0.75
    bar_h = total_height / n_gpus

    all_vals = []
    for gi, (gname, gvals) in enumerate(zip(gpu_names, values_per_gpu)):
        offset = (gi - (n_gpus - 1) / 2) * bar_h
        color = GPU_COLORS[gi % len(GPU_COLORS)]
        vals = [v if v is not None else 0 for v in gvals]
        all_vals.extend(vals)
        bars = ax.barh(y + offset, vals, bar_h * 0.88, color=color,
                       edgecolor=BG_COLOR, linewidth=1, zorder=3, label=gname)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(v + max(all_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                        fmt.format(v), va="center", fontsize=10,
                        color=TEXT_COLOR, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_title(title, fontsize=14, color=title_color, pad=10)
    ax.grid(axis="x", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    max_v = max(all_vals) if all_vals else 1
    ax.set_xlim(0, max_v * 1.25)
    ax.invert_yaxis()
    ax.legend(fontsize=10, loc="lower right", facecolor=CARD_COLOR, edgecolor=GRID_COLOR)


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_training_vision(d: ComparisonData, out: Path):
    cats, keys = [], []
    for model in ["resnet50", "resnet101"]:
        for prec in ["fp32", "fp16", "bf16"]:
            mid = f"train_vision_{model}_{prec}_img_per_sec"
            if any(v is not None for v in d.get(mid)):
                cats.append(f"{model.upper()}\n{prec.upper()}")
                keys.append(mid)
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.suptitle("Training Throughput — Vision", fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)
    _direction_text(fig, "▲ Higher is better", ACCENT_GREEN)
    _grouped_bar(ax, cats, d.gpu_names(), vals, "Images / sec",
                 "ResNet-50 / ResNet-101", ACCENT_BLUE)
    _watermark(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_training_vision.png")


def chart_training_nlp(d: ComparisonData, out: Path):
    cats, keys = [], []
    for model in ["bert-base", "bert-large"]:
        for prec in ["fp32", "fp16", "bf16"]:
            mid = f"train_nlp_{model}_{prec}_samples_per_sec"
            if any(v is not None for v in d.get(mid)):
                cats.append(f"{model.upper()}\n{prec.upper()}")
                keys.append(mid)
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.suptitle("Training Throughput — NLP", fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)
    _direction_text(fig, "▲ Higher is better", ACCENT_GREEN)
    _grouped_bar(ax, cats, d.gpu_names(), vals, "Samples / sec",
                 "BERT-Base / BERT-Large", ACCENT_GREEN)
    _watermark(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_training_nlp.png")


def chart_inference_vision(d: ComparisonData, out: Path):
    cats, keys = [], []
    for model in ["resnet50", "resnet101"]:
        for prec in ["fp32", "fp16", "bf16"]:
            mid = f"infer_vision_{model}_{prec}_img_per_sec"
            if any(v is not None for v in d.get(mid)):
                cats.append(f"{model.upper()}\n{prec.upper()}")
                keys.append(mid)
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.suptitle("Inference Throughput — Vision", fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)
    _direction_text(fig, "▲ Higher is better", ACCENT_GREEN)
    _grouped_bar(ax, cats, d.gpu_names(), vals, "Images / sec",
                 "ResNet-50 / ResNet-101", ACCENT_BLUE)
    _watermark(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_inference_vision.png")


def chart_inference_nlp(d: ComparisonData, out: Path):
    cats, keys = [], []
    for model in ["bert-base", "bert-large"]:
        for prec in ["fp32", "fp16", "bf16"]:
            mid = f"infer_nlp_{model}_{prec}_samples_per_sec"
            if any(v is not None for v in d.get(mid)):
                cats.append(f"{model.upper()}\n{prec.upper()}")
                keys.append(mid)
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.suptitle("Inference Throughput — NLP", fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)
    _direction_text(fig, "▲ Higher is better", ACCENT_GREEN)
    _grouped_bar(ax, cats, d.gpu_names(), vals, "Samples / sec",
                 "BERT-Base / BERT-Large", ACCENT_GREEN)
    _watermark(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_inference_nlp.png")


def chart_llm(d: ComparisonData, out: Path):
    tps_keys = sorted([m for m in d.metric_ids()
                       if m.startswith("llm_") and m.endswith("_tokens_per_sec")])
    ttft_keys = sorted([m for m in d.metric_ids()
                        if m.startswith("llm_") and m.endswith("_ttft_ms")])
    if not tps_keys:
        return

    def _label(mid: str) -> str:
        name = mid.replace("llm_", "").replace("_tokens_per_sec", "").replace("_ttft_ms", "")
        return name.replace("_", " ").title()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                                    gridspec_kw={"width_ratios": [3, 2]})
    fig.suptitle("LLM Performance", fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)
    fig.text(0.35, 0.935, "▲ Higher is better", ha="center", va="top",
             fontsize=10, color=ACCENT_GREEN, alpha=0.9)
    fig.text(0.80, 0.935, "▼ Lower is better", ha="center", va="top",
             fontsize=10, color=ACCENT_ORANGE, alpha=0.9)

    cats_tps = [_label(m) for m in tps_keys]
    vals_tps = [[d.get(m)[gi] for m in tps_keys] for gi in range(d.n_gpus)]
    _horizontal_grouped_bar(ax1, cats_tps, d.gpu_names(), vals_tps,
                            "Tokens / sec", "Generation Speed", ACCENT_GREEN,
                            fmt="{:.1f}")

    if ttft_keys:
        cats_ttft = [_label(m) for m in ttft_keys]
        vals_ttft = [[d.get(m)[gi] for m in ttft_keys] for gi in range(d.n_gpus)]
        _horizontal_grouped_bar(ax2, cats_ttft, d.gpu_names(), vals_ttft,
                                "Time to First Token (ms)", "TTFT", ACCENT_ORANGE,
                                fmt="{:.0f}")
    _watermark(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_llm.png")


def chart_gemm(d: ComparisonData, out: Path):
    cats, keys = [], []
    for prec in ["fp64", "fp32", "fp16", "bf16", "fp8"]:
        mid = f"gemm_{prec}_peak_tflops"
        if any(v is not None for v in d.get(mid)):
            cats.append(prec.upper())
            keys.append(mid)
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.suptitle("GEMM Peak Compute", fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)
    _direction_text(fig, "▲ Higher is better", ACCENT_GREEN)
    _grouped_bar(ax, cats, d.gpu_names(), vals, "TFLOPS",
                 "Peak Sustained TFLOPS per Precision", ACCENT_PURPLE,
                 fmt="{:.1f}", val_fontsize=11)
    _watermark(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_gemm.png")


def chart_fundamentals(d: ComparisonData, out: Path):
    specs = [
        ("fund_d2d_bw_peak_gb_s",            "Mem BW\n(GB/s)",          "{:.0f}"),
        ("fund_pcie_h2d_gb_s",                "PCIe H2D\n(GB/s)",       "{:.1f}"),
        ("fund_pcie_d2h_gb_s",                "PCIe D2H\n(GB/s)",       "{:.1f}"),
        ("fund_kernel_launch_latency_us",     "Kernel\nLatency (μs)",   "{:.1f}"),
        ("fund_reduction_peak_gb_s",          "Reduction\n(GB/s)",      "{:.0f}"),
        ("fund_spmm_peak_gflops",             "SpMM\n(GFLOPS)",        "{:.0f}"),
        ("fund_fft_fp32_peak_gflops",         "FFT FP32\n(GFLOPS)",    "{:.0f}"),
    ]

    cats, metric_keys, fmts = [], [], []
    for mid, label, fmt in specs:
        if any(v is not None for v in d.get(mid)):
            cats.append(label)
            metric_keys.append(mid)
            fmts.append(fmt)
    if not cats:
        return

    vals = [[d.get(m)[gi] for m in metric_keys] for gi in range(d.n_gpus)]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.suptitle("GPU Fundamentals", fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)
    _direction_text(fig, "BW/Throughput ▲  Latency ▼", ACCENT_CYAN)

    n_groups = len(cats)
    n_gpus = d.n_gpus
    x = np.arange(n_groups)
    total_width = 0.75
    bar_w = total_width / n_gpus

    for gi, gname in enumerate(d.gpu_names()):
        offset = (gi - (n_gpus - 1) / 2) * bar_w
        color = GPU_COLORS[gi % len(GPU_COLORS)]
        gvals = [v if v is not None else 0 for v in vals[gi]]
        bars = ax.bar(x + offset, gvals, bar_w * 0.88, color=color,
                      edgecolor=BG_COLOR, linewidth=1, zorder=3, label=gname)
        for bar, v, fmt in zip(bars, gvals, fmts):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt.format(v), ha="center", va="bottom",
                        fontsize=9, color=TEXT_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_title("Compare bars within each group (different scales)",
                 fontsize=12, color=SUBTEXT_COLOR, pad=10)
    ax.grid(axis="y", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper right", facecolor=CARD_COLOR, edgecolor=GRID_COLOR)
    _watermark(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_fundamentals.png")


def chart_detection(d: ComparisonData, out: Path):
    cats, keys = [], []
    for model in ["faster_rcnn_resnet50", "mask_rcnn_resnet50"]:
        for prec in ["fp32", "fp16", "bf16"]:
            mid = f"detect_{model}_{prec}_img_per_sec"
            if any(v is not None for v in d.get(mid)):
                nice = model.replace("faster_rcnn_resnet50", "Faster R-CNN") \
                            .replace("mask_rcnn_resnet50", "Mask R-CNN")
                cats.append(f"{nice}\n{prec.upper()}")
                keys.append(mid)
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.7))
    fig.suptitle("Object Detection Training", fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, y=0.98)
    _direction_text(fig, "▲ Higher is better", ACCENT_GREEN)
    _grouped_bar(ax, cats, d.gpu_names(), vals, "Images / sec",
                 "Faster R-CNN / Mask R-CNN", ACCENT_ORANGE, fmt="{:.1f}")
    _watermark(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    _save(fig, out / "cmp_detection.png")


def chart_scorecard(d: ComparisonData, out: Path):
    headline_metrics = [
        ("train_vision_resnet50_fp16_img_per_sec", "ResNet-50 Train FP16", "img/s", "{:.0f}", None),
        ("infer_vision_resnet50_fp16_img_per_sec", "ResNet-50 Infer FP16", "img/s", "{:.0f}", None),
        ("train_nlp_bert-base_bf16_samples_per_sec", "BERT-Base Train BF16", "samples/s", "{:.0f}", None),
        ("gemm_bf16_peak_tflops", "GEMM BF16 Peak", "TFLOPS", "{:.1f}", None),
        ("fund_d2d_bw_peak_gb_s", "Memory Bandwidth", "GB/s", "{:.0f}", None),
    ]

    # Best LLM
    tps_keys = sorted([m for m in d.metric_ids()
                       if m.startswith("llm_") and m.endswith("_tokens_per_sec")
                       and m != "llm_best_tokens_per_sec"])
    if tps_keys:
        best_key = max(tps_keys, key=lambda m: sum(v or 0 for v in d.get(m)))
        best_descs = d.get("llm_best_model")
        if any(v is not None for v in best_descs):
            headline_metrics.append((best_key, "Best LLM Speed", "t/s", "{:.0f}", best_descs))
        else:
            name = best_key.replace("llm_", "").replace("_tokens_per_sec", "").replace("_", " ").title()
            headline_metrics.append((best_key, f"LLM {name}", "t/s", "{:.0f}", None))

    # VRAM
    vram_params = d.get("vram_largest_loadable_params_b")
    if any(v is not None for v in vram_params):
        vram_detail = d.get("vram_max_model_detail")
        detail = vram_detail if any(v is not None for v in vram_detail) else None
        headline_metrics.append(("vram_largest_loadable_params_b", "Max Loadable Model", "B params", "{:.0f}", detail))

    vram_ctx = d.get("vram_max_context_length")
    if any(v is not None for v in vram_ctx):
        headline_metrics.append(("vram_max_context_length", "Max Context Length", "tokens", "{:,.0f}", None))

    # Power & temp
    hw_power = d.get("hw_avg_power_w")
    if any(v is not None for v in hw_power):
        headline_metrics.append(("hw_avg_power_w", "Avg Power Draw", "W", "{:.0f}", None))
    hw_temp = d.get("hw_avg_temp_c")
    if any(v is not None for v in hw_temp):
        headline_metrics.append(("hw_avg_temp_c", "Avg Temperature", "°C", "{:.0f}", None))

    # DLPerf
    dlperf_vals = []
    fp32_train = d.get("train_vision_resnet50_fp32_img_per_sec")
    for v in fp32_train:
        dlperf_vals.append(round(v / 13.0, 1) if v else None)

    gpu_names = d.gpu_names()
    n_gpus = d.n_gpus

    rows = []
    for mid, label, unit, fmt, detail in headline_metrics:
        vals = d.get(mid)
        if any(v is not None for v in vals):
            rows.append((label, unit, fmt, vals, detail))
    if dlperf_vals and any(v is not None for v in dlperf_vals):
        rows.append(("DLPerf Score", "", "{:.1f}", dlperf_vals, None))

    if not rows:
        return

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    title_parts = " vs ".join(gpu_names)
    fig.suptitle(f"GPU Scorecard — {title_parts}", fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)

    n_rows = len(rows)
    n_cols = max(n_gpus + 1, 3)

    for ri, (label, unit, fmt, vals, detail) in enumerate(rows):
        ax = fig.add_subplot(n_rows, n_cols, ri * n_cols + 1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.95, 0.5, label, ha="right", va="center",
                fontsize=12, color=SUBTEXT_COLOR, fontweight="bold")

        valid = [(i, v) for i, v in enumerate(vals)
                 if v is not None and isinstance(v, (int, float))]
        lower_is_better = label in ("Avg Power Draw", "Avg Temperature")
        if valid:
            best_idx = min(valid, key=lambda x: x[1])[0] if lower_is_better \
                       else max(valid, key=lambda x: x[1])[0]
        else:
            best_idx = -1

        for gi in range(n_gpus):
            ax = fig.add_subplot(n_rows, n_cols, ri * n_cols + 2 + gi)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

            v = vals[gi]
            color = GPU_COLORS[gi % len(GPU_COLORS)]
            is_winner = (gi == best_idx and n_gpus > 1)

            ec = ACCENT_GREEN if is_winner else GRID_COLOR
            lw = 2.5 if is_winner else 1
            rect = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=CARD_COLOR, edgecolor=ec, linewidth=lw)
            ax.add_patch(rect)

            if v is not None:
                if isinstance(v, (int, float)):
                    val_str = fmt.format(v)
                else:
                    val_str = str(v)
                if unit:
                    val_str += f" {unit}"

                has_detail = detail and gi < len(detail) and detail[gi] is not None
                val_y = 0.62 if has_detail else 0.55
                ax.text(0.5, val_y, val_str, ha="center", va="center",
                        fontsize=15, fontweight="bold", color=color)
                if has_detail:
                    ax.text(0.5, 0.32, str(detail[gi]), ha="center", va="center",
                            fontsize=9, color=SUBTEXT_COLOR, style="italic")
                if is_winner and n_gpus > 1:
                    best_y = 0.15 if has_detail else 0.18
                    ax.text(0.5, best_y, "★ BEST", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=ACCENT_GREEN)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=13, color=SUBTEXT_COLOR)

            if ri == 0:
                ax.text(0.5, 1.05, gpu_names[gi], ha="center", va="bottom",
                        fontsize=12, fontweight="bold",
                        color=GPU_COLORS[gi % len(GPU_COLORS)])

    _watermark(fig)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_scorecard.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_sessions(results_dir: Path) -> List[Path]:
    return sorted(
        p for p in results_dir.iterdir()
        if p.is_dir() and not p.is_symlink() and (p / "session_meta.json").exists()
    )


def _run_extraction(results_dir: Path, session_dirs: List[Path],
                    output_dir: Path) -> Path:
    """Extract metrics and write comparison.csv/json. Returns path to JSON."""
    sessions: List[Tuple[str, str, MetricDict]] = []
    for d in session_dirs:
        try:
            col_id, col_label, metrics = _load_session(d)
            existing_ids = [s[0] for s in sessions]
            if col_id in existing_ids:
                short = d.name[-8:]
                col_id += f"_{short}"
                col_label += f" ({short})"
            sessions.append((col_id, col_label, metrics))
            print(f"  Loaded: {col_label} — {len(metrics)} metrics")
        except Exception as e:
            print(f"  WARNING: skipping {d.name} — {e}", file=sys.stderr)

    if not sessions:
        print("ERROR: no sessions could be loaded", file=sys.stderr)
        sys.exit(1)

    _, col_ids, rows = build_comparison_table(sessions)

    csv_path = output_dir / "comparison.csv"
    json_path = output_dir / "comparison.json"
    write_csv(rows, col_ids, csv_path)
    write_json(rows, col_ids, sessions, json_path)

    print(f"\n  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")
    print(f"  Metrics: {len(rows)}  |  Sessions: {len(sessions)}")
    return json_path


def _run_charts(json_path: Path, output_dir: Path):
    """Generate all comparison charts from comparison.json."""
    d = ComparisonData(json_path)
    chart_dir = output_dir / "comparison_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Charts → {chart_dir}")
    print(f"  GPUs   : {d.n_gpus} — {', '.join(d.gpu_names())}")

    count = 0
    for fn in [chart_training_vision, chart_training_nlp,
               chart_inference_vision, chart_inference_nlp,
               chart_llm, chart_gemm, chart_fundamentals,
               chart_detection, chart_scorecard]:
        try:
            fn(d, chart_dir)
            count += 1
        except Exception as e:
            print(f"  ⚠ {fn.__name__}: {e}", file=sys.stderr)

    print(f"\n  Generated {count} chart(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract benchmark metrics and generate comparison charts"
    )
    parser.add_argument("--results-dir", type=Path,
                        default=_PROJECT_ROOT / "results",
                        help="Root results directory (default: <project>/results)")
    parser.add_argument("--sessions", nargs="+", metavar="SESSION_ID",
                        help="Specific session IDs to include (default: all)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <results-dir>)")
    parser.add_argument("--skip-charts", action="store_true",
                        help="Only extract metrics, skip chart generation")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover sessions
    if args.sessions:
        session_dirs = [results_dir / s for s in args.sessions]
        missing = [d for d in session_dirs if not d.exists()]
        if missing:
            for m in missing:
                print(f"ERROR: session not found: {m}", file=sys.stderr)
            sys.exit(1)
    else:
        session_dirs = _discover_sessions(results_dir)
        if not session_dirs:
            print("ERROR: no benchmark sessions found in", results_dir, file=sys.stderr)
            sys.exit(1)

    print("=" * 50)
    print(f"Found {len(session_dirs)} session(s)")
    for d in session_dirs:
        print(f"  {d.name}")

    # Extract
    json_path = _run_extraction(results_dir, session_dirs, output_dir)

    # Charts
    if not args.skip_charts:
        _run_charts(json_path, output_dir)

    print("=" * 50)


if __name__ == "__main__":
    main()
