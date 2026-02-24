#!/usr/bin/env python3
"""
utils/generate_comparison.py — Extract metrics + generate comparison charts

Scans all benchmark sessions under results/, extracts metrics into
comparison.csv/json, and generates side-by-side GPU comparison charts.

Usage:
  python utils/generate_comparison.py                         # Turkish (default)
  python utils/generate_comparison.py --lang en               # English
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
from math import exp, log
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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
ACCENT_PINK   = "#DB61A2"
ACCENT_TEAL   = "#2EA043"

GPU_COLORS = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE,
              ACCENT_ORANGE, ACCENT_CYAN, ACCENT_YELLOW,
              ACCENT_PINK, ACCENT_TEAL]

DPI = 150
BASE_FIG_W, BASE_FIG_H = 16, 9

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   CARD_COLOR,
    "axes.edgecolor":   GRID_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "grid.color":       GRID_COLOR,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
    "font.size":        12,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


# ── Visual effects helpers ────────────────────────────────────────────────────

def _glow_text(ax_or_fig, x, y, text, fontsize=12, color=TEXT_COLOR,
               fontweight="bold", ha="center", va="center", **kwargs):
    """Draw text with a soft glow/shadow behind it."""
    txt = ax_or_fig.text(x, y, text, fontsize=fontsize, color=color,
                         fontweight=fontweight, ha=ha, va=va, **kwargs)
    txt.set_path_effects([
        path_effects.withStroke(linewidth=3, foreground=BG_COLOR, alpha=0.7),
        path_effects.Normal(),
    ])
    return txt


def _gradient_barh(ax, y: float, width: float, height: float,
                   color: str, alpha: float = 1.0, zorder: int = 3):
    """Draw a horizontal bar with a vertical gradient (lit from top)."""
    if width <= 0:
        return None
    rgb = mcolors.to_rgb(color)
    # Top highlight → base → darker bottom
    highlight = tuple(min(1.0, c * 1.35) for c in rgb)
    shadow = tuple(c * 0.6 for c in rgb)
    cmap = LinearSegmentedColormap.from_list(
        "bar_grad", [shadow, rgb, highlight])
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    extent = [0, width, y - height / 2, y + height / 2]
    ax.imshow(gradient, aspect="auto", cmap=cmap, extent=extent,
             zorder=zorder, alpha=alpha, interpolation="bicubic")
    # Thin bright edge on top for "3D" illusion
    ax.plot([0, width], [y + height / 2, y + height / 2],
            color=highlight, linewidth=0.8, alpha=0.5, zorder=zorder + 1)
    return extent


def _gradient_bar_v(ax, x: float, bottom: float, height: float,
                    width: float, color: str, zorder: int = 3):
    """Draw a vertical bar with a horizontal gradient (lit from left)."""
    if height <= 0:
        return None
    rgb = mcolors.to_rgb(color)
    highlight = tuple(min(1.0, c * 1.35) for c in rgb)
    shadow = tuple(c * 0.6 for c in rgb)
    cmap = LinearSegmentedColormap.from_list(
        "bar_grad_v", [shadow, rgb, highlight])
    gradient = np.linspace(0, 1, 256).reshape(1, 256)
    extent = [x - width / 2, x + width / 2, bottom, bottom + height]
    ax.imshow(gradient, aspect="auto", cmap=cmap, extent=extent,
             zorder=zorder, interpolation="bicubic")
    ax.plot([x - width / 2, x - width / 2], [bottom, bottom + height],
            color=highlight, linewidth=0.8, alpha=0.5, zorder=zorder + 1)
    return extent


# ═══════════════════════════════════════════════════════════════════════════════
#  LOCALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

_STRINGS: dict[str, dict[str, str]] = {
    "tr": {
        "watermark":                "Crossfirelab GPU Yapay Zeka Benchmark Paketi",
        "higher_better":            "\u25b2 Yüksek = Daha iyi",
        "lower_better":             "\u25bc Düşük = Daha iyi",
        "bw_high_latency_low":      "Bant genişliği \u25b2  Gecikme \u25bc",
        # — Training Vision
        "train_vision_title":       "Eğitim Verimi \u2014 Görüntü",
        "train_vision_subtitle":    "CNN (ResNet-50 / ResNet-101) — Maks. Verim",
        "train_vision_ylabel":      "Görüntü / sn",
        # — Training NLP
        "train_nlp_title":          "Eğitim Verimi \u2014 NLP (Doğal Dil İşleme)",
        "train_nlp_subtitle":       "Transformer (BERT-Base / BERT-Large) — Maks. Verim",
        "train_nlp_ylabel":         "Örnek / sn",
        # — Inference Vision
        "infer_vision_title":       "Çıkarım Verimi \u2014 Görüntü",
        "infer_vision_subtitle":    "CNN (ResNet-50 / ResNet-101) — Maks. Verim",
        "infer_vision_ylabel":      "Görüntü / sn",
        # — Inference NLP
        "infer_nlp_title":          "Çıkarım Verimi \u2014 NLP (Doğal Dil İşleme)",
        "infer_nlp_subtitle":       "Transformer (BERT-Base / BERT-Large) — Maks. Verim",
        "infer_nlp_ylabel":         "Örnek / sn",
        # — LLM
        "llm_title":                "LLM (Büyük Dil Modeli) Performansı",
        "llm_speed":                "Üretim Hızı",
        "llm_ttft":                 "İlk Token Süresi",
        "llm_xlabel_tps":           "Token / sn",
        "llm_xlabel_ttft":          "İlk Token Süresi (ms)",
        "llm_higher":               "\u25b2 Yüksek = Daha iyi",
        "llm_lower":                "\u25bc Düşük = Daha iyi",
        # — GEMM
        "gemm_title":               "GEMM Zirve Hesaplama Gücü",
        "gemm_subtitle":            "Hassasiyet Başına Zirve Sürdürülebilir TFLOPS",
        "gemm_ylabel":              "TFLOPS",
        # — Fundamentals
        "fund_title":               "GPU Temelleri",
        "fund_subtitle":            "Her grup kendi içinde karşılaştırılır (farklı ölçekler)",
        "fund_mem_bw":              "Bellek BG\n(GB/sn)",
        "fund_pcie_h2d":            "PCIe H2D\n(GB/sn)",
        "fund_pcie_d2h":            "PCIe D2H\n(GB/sn)",
        "fund_kernel_lat":          "Çekirdek\nGecikmesi (μs)",
        "fund_reduction":           "Azaltma\n(GB/sn)",
        "fund_spmm":                "SpMM\n(GFLOPS)",
        "fund_fft":                 "FFT FP32\n(GFLOPS)",
        # — Detection
        "detect_title":             "Nesne Algılama Eğitimi",
        "detect_subtitle":          "Faster R-CNN / Mask R-CNN",
        "detect_ylabel":            "Görüntü / sn",
        # — Same Batch Size comparison
        "batch_norm_suffix":        "(Aynı Batch Boyutu)",
        "batch_norm_ylabel":        "Örnek / sn",
        # — Scorecard
        "score_title":              "GPU Karşılaştırma Kartı",
        "score_best":               "\u2605 EN İYİ",
        "score_resnet_train":       "CNN (ResNet-50) Eğitim FP16",
        "score_bert_train":         "Transformer (BERT-Base) Eğitim BF16",
        "score_gemm":               "GEMM BF16 Zirve",
        "score_membw":              "Bellek Bant Genişliği",
        "score_llm_speed":          "En İyi LLM Hızı",
        "score_llm_label":          "LLM",
        "score_vram_model":         "Maks. Yüklenebilir Model",
        "score_vram_ctx":           "Maks. Bağlam Uzunluğu (3B)",
        "score_llmfit_best":        "En İyi Yüklenebilir LLM",
        "score_llmfit_largest":     "En Büyük Dense LLM",
        "score_llmfit_moe":         "En Büyük MoE LLM",
        "score_llmfit_disclaimer":  "* LLM önerileri teorik VRAM tahminlerine dayalıdır (llmfit veritabanı). Gerçek performans; quantization, bağlam uzunluğu ve sistem yapılandırmasına göre değişebilir.",
        "score_max_train_tput":     "Maks. Eğitim Verimi",
        "score_max_infer_tput":     "Maks. Çıkarım Verimi",
        "score_power":              "Ort. Güç Tüketimi\n(Test Sırasında)",
        "score_temp":               "Ort. Sıcaklık\n(Test Sırasında)",
        "score_dlperf_cnn":          "DLPerf (CNN)",
        "score_dlperf_transformer":  "DLPerf (Transformer)",
        "unit_img_s":               "görüntü/sn",
        "unit_samples_s":           "örnek/sn",
        "unit_tflops":              "TFLOPS",
        "unit_gb_s":                "GB/sn",
        "unit_tokens_s":            "t/sn",
        "unit_b_params":            "B param",
        "unit_tokens":              "token",
        "unit_watt":                "W",
        "unit_celsius":             "°C",
        # — Power Efficiency
        "power_eff_title":          "Watt Başına Performans",
        "power_eff_ylabel":         "Verim / Watt",
        "power_eff_subtitle":       "Yüksek = Daha verimli",
        "power_resnet_train":      "ResNet-50 Eğitim",
        "power_bert_train":        "BERT-base Eğitim",
        "power_llm_tokens":        "LLM Token/sn",
        # — Relative Performance
        "relative_title":           "Görece Performans Karşılaştırması",
        "relative_subtitle":        "En düşük puanlı GPU = 1.0x temel çizgi",
        "relative_xlabel":          "Kat (x)",
        # — Multi-GPU
        "multigpu_title":           "Çift GPU Hızlandırma",
        "multigpu_subtitle":        "1 GPU vs 2 GPU — DDP / FSDP / Tensor Parallelism",
        "multigpu_ylabel":          "Örnek / sn",
        "multigpu_llm_ylabel":      "Token / sn",
        "multigpu_1gpu":            "1 GPU",
        "multigpu_ddp":             "2 GPU DDP",
        "multigpu_fsdp":            "2 GPU FSDP",
        "multigpu_tp":              "2 GPU TP",
        "multigpu_efficiency":      "Ölçekleme Verimi",
        # — VRAM Limits
        "vram_title":               "VRAM Kapasite Karşılaştırması",
        "vram_subtitle":            "Yüklenebilir model boyutu, VRAM kullanımı ve bağlam uzunluğu",
        "vram_cat_params":          "En Büyük Model\n(Milyar Param)",
        "vram_cat_actual_gb":       "VRAM Kullanımı\n(GB)",
        "vram_cat_context":         "Maks. Bağlam\n(K token)",
        # — CNN vs Transformer
        "cnn_vs_tf_title":          "CNN & Transformer — Ortalama Performans",
        "cnn_vs_tf_subtitle":       "Geometrik ortalama (en düşük GPU = 1.0x)",
        "cnn_vs_tf_ylabel":         "Normalize Skor",
        "cnn_vs_tf_cnn":            "CNN Ortalama",
        "cnn_vs_tf_transformer":    "Transformer Ortalama",
        "cnn_vs_tf_cnn_ppw":        "CNN Perf/Watt",
        "cnn_vs_tf_transformer_ppw":"Transformer Perf/Watt",
        "cnn_vs_tf_score":          "Verimlilik Skoru",
        "cnn_vs_tf_perf_watt":      "Performans / Watt",
    },
    "en": {
        "watermark":                "Crossfirelab GPU AI Benchmark Suite",
        "higher_better":            "\u25b2 Higher is better",
        "lower_better":             "\u25bc Lower is better",
        "bw_high_latency_low":      "BW/Throughput \u25b2  Latency \u25bc",
        # — Training Vision
        "train_vision_title":       "Training Throughput \u2014 Vision",
        "train_vision_subtitle":    "CNN (ResNet-50 / ResNet-101) — Max Throughput",
        "train_vision_ylabel":      "Images / sec",
        # — Training NLP
        "train_nlp_title":          "Training Throughput \u2014 NLP",
        "train_nlp_subtitle":       "Transformer (BERT-Base / BERT-Large) — Max Throughput",
        "train_nlp_ylabel":         "Samples / sec",
        # — Inference Vision
        "infer_vision_title":       "Inference Throughput \u2014 Vision",
        "infer_vision_subtitle":    "CNN (ResNet-50 / ResNet-101) — Max Throughput",
        "infer_vision_ylabel":      "Images / sec",
        # — Inference NLP
        "infer_nlp_title":          "Inference Throughput \u2014 NLP",
        "infer_nlp_subtitle":       "Transformer (BERT-Base / BERT-Large) — Max Throughput",
        "infer_nlp_ylabel":         "Samples / sec",
        # — LLM
        "llm_title":                "LLM Performance",
        "llm_speed":                "Generation Speed",
        "llm_ttft":                 "Time to First Token",
        "llm_xlabel_tps":           "Tokens / sec",
        "llm_xlabel_ttft":          "Time to First Token (ms)",
        "llm_higher":               "\u25b2 Higher is better",
        "llm_lower":                "\u25bc Lower is better",
        # — GEMM
        "gemm_title":               "GEMM Peak Compute",
        "gemm_subtitle":            "Peak Sustained TFLOPS per Precision",
        "gemm_ylabel":              "TFLOPS",
        # — Fundamentals
        "fund_title":               "GPU Fundamentals",
        "fund_subtitle":            "Compare bars within each group (different scales)",
        "fund_mem_bw":              "Mem BW\n(GB/s)",
        "fund_pcie_h2d":            "PCIe H2D\n(GB/s)",
        "fund_pcie_d2h":            "PCIe D2H\n(GB/s)",
        "fund_kernel_lat":          "Kernel\nLatency (μs)",
        "fund_reduction":           "Reduction\n(GB/s)",
        "fund_spmm":                "SpMM\n(GFLOPS)",
        "fund_fft":                 "FFT FP32\n(GFLOPS)",
        # — Detection
        "detect_title":             "Object Detection Training",
        "detect_subtitle":          "Faster R-CNN / Mask R-CNN",
        "detect_ylabel":            "Images / sec",
        # — Same Batch Size comparison
        "batch_norm_suffix":        "(Same Batch Size)",
        "batch_norm_ylabel":        "Samples / sec",
        # — Scorecard
        "score_title":              "GPU Scorecard",
        "score_best":               "\u2605 BEST",
        "score_resnet_train":       "CNN (ResNet-50) Train FP16",
        "score_bert_train":         "Transformer (BERT-Base) Train BF16",
        "score_gemm":               "GEMM BF16 Peak",
        "score_membw":              "Memory Bandwidth",
        "score_llm_speed":          "Best LLM Speed",
        "score_llm_label":          "LLM",
        "score_vram_model":         "Max Loadable Model",
        "score_vram_ctx":           "Max Context Length (3B)",
        "score_llmfit_best":        "Best Loadable LLM",
        "score_llmfit_largest":     "Largest Dense LLM",
        "score_llmfit_moe":         "Largest MoE LLM",
        "score_llmfit_disclaimer":  "* LLM recommendations are theoretical VRAM estimates (llmfit database). Actual fit depends on quantization, context length, and system configuration.",
        "score_max_train_tput":     "Max Training Throughput",
        "score_max_infer_tput":     "Max Inference Throughput",
        "score_power":              "Avg Power Draw\n(During Test)",
        "score_temp":               "Avg Temperature\n(During Test)",
        "score_dlperf_cnn":          "DLPerf (CNN)",
        "score_dlperf_transformer":  "DLPerf (Transformer)",
        "unit_img_s":               "img/s",
        "unit_samples_s":           "samples/s",
        "unit_tflops":              "TFLOPS",
        "unit_gb_s":                "GB/s",
        "unit_tokens_s":            "t/s",
        "unit_b_params":            "B params",
        "unit_tokens":              "tokens",
        "unit_watt":                "W",
        "unit_celsius":             "°C",
        # — Power Efficiency
        "power_eff_title":          "Performance per Watt",
        "power_eff_ylabel":         "Throughput / Watt",
        "power_eff_subtitle":       "Higher = More efficient",
        "power_resnet_train":      "ResNet-50 Train",
        "power_bert_train":        "BERT-base Train",
        "power_llm_tokens":        "LLM Token/s",
        # — Relative Performance
        "relative_title":           "Relative Performance Comparison",
        "relative_subtitle":        "Lowest scoring GPU = 1.0x baseline",
        "relative_xlabel":          "Speedup (x)",
        # — Multi-GPU
        "multigpu_title":           "Dual-GPU Speedup",
        "multigpu_subtitle":        "1 GPU vs 2 GPU — DDP / FSDP / Tensor Parallelism",
        "multigpu_ylabel":          "Samples / sec",
        "multigpu_llm_ylabel":      "Tokens / sec",
        "multigpu_1gpu":            "1 GPU",
        "multigpu_ddp":             "2 GPU DDP",
        "multigpu_fsdp":            "2 GPU FSDP",
        "multigpu_tp":              "2 GPU TP",
        "multigpu_efficiency":      "Scaling Efficiency",
        # — VRAM Limits
        "vram_title":               "VRAM Capacity Comparison",
        "vram_subtitle":            "Max loadable model size, VRAM usage & context length",
        "vram_cat_params":          "Largest Model\n(B Params)",
        "vram_cat_actual_gb":       "VRAM Used\n(GB)",
        "vram_cat_context":         "Max Context\n(K tokens)",
        # — CNN vs Transformer
        "cnn_vs_tf_title":          "CNN & Transformer — Average Performance",
        "cnn_vs_tf_subtitle":       "Geometric mean (lowest GPU = 1.0x)",
        "cnn_vs_tf_ylabel":         "Normalized Score",
        "cnn_vs_tf_cnn":            "CNN Average",
        "cnn_vs_tf_transformer":    "Transformer Average",
        "cnn_vs_tf_cnn_ppw":        "CNN Perf/Watt",
        "cnn_vs_tf_transformer_ppw":"Transformer Perf/Watt",
        "cnn_vs_tf_score":          "Throughput Score",
        "cnn_vs_tf_perf_watt":      "Performance / Watt",
    },
}

# Active language — set by CLI, default Turkish
_LANG = "tr"


def S(key: str) -> str:
    """Return localised string for *key*."""
    return _STRINGS.get(_LANG, _STRINGS["tr"]).get(key, key)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 — Metric extraction
# ═══════════════════════════════════════════════════════════════════════════════

Metricdict = dict[str, Any]


def _best(rows: list[dict], filter_fn, value_key: str) -> float | None:
    hits = [r.get(value_key) for r in rows if filter_fn(r) and r.get(value_key) is not None]
    return max(hits) if hits else None


def _best_with_bs(rows: list[dict], filter_fn, value_key: str) -> tuple[float | None, int | None]:
    """Return (best_value, batch_size_that_achieved_it)."""
    candidates = [(r.get(value_key), r.get("batch_size"))
                  for r in rows if filter_fn(r) and r.get(value_key) is not None]
    if not candidates:
        return None, None
    best = max(candidates, key=lambda x: x[0])
    return best[0], best[1]


def _safe(v: Any) -> Any:
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 4)
    return v


# ── Extractors ────────────────────────────────────────────────────────────────

def extract_training_vision(data: dict) -> Metricdict:
    rows = data.get("results", [])
    out: Metricdict = {}
    for model in ("resnet50", "resnet101"):
        for prec in ("FP32", "FP16", "BF16", "FP8"):
            filt = lambda r, m=model, p=prec: (r.get("model") == m
                                                and r.get("precision") == p
                                                and r.get("status") == "success")
            v, bs = _best_with_bs(rows, filt, "throughput_img_per_sec")
            if v is not None:
                out[f"train_vision_{model}_{prec.lower()}_img_per_sec"] = _safe(v)
                if bs is not None:
                    out[f"train_vision_{model}_{prec.lower()}_best_bs"] = bs
                    out[f"train_vision_{model}_{prec.lower()}_norm_iter_per_sec"] = _safe(v / bs)
            # Store per-batch-size throughputs for same-BS comparison
            for r in rows:
                if filt(r) and r.get("throughput_img_per_sec") is not None and r.get("batch_size") is not None:
                    bsz = int(r["batch_size"])
                    out[f"train_vision_{model}_{prec.lower()}_bs{bsz}_img_per_sec"] = _safe(r["throughput_img_per_sec"])
    return out


def extract_training_nlp(data: dict) -> Metricdict:
    rows = data.get("results", [])
    out: Metricdict = {}
    for model in ("bert-base", "bert-large"):
        for prec in ("FP32", "FP16", "BF16", "FP8"):
            filt = lambda r, m=model, p=prec: (r.get("model") == m
                                                and r.get("precision") == p
                                                and r.get("status") == "success")
            v, bs = _best_with_bs(rows, filt, "throughput_samples_per_sec")
            if v is not None:
                out[f"train_nlp_{model}_{prec.lower()}_samples_per_sec"] = _safe(v)
                if bs is not None:
                    out[f"train_nlp_{model}_{prec.lower()}_best_bs"] = bs
                    out[f"train_nlp_{model}_{prec.lower()}_norm_iter_per_sec"] = _safe(v / bs)
            # Store per-batch-size throughputs for same-BS comparison
            for r in rows:
                if filt(r) and r.get("throughput_samples_per_sec") is not None and r.get("batch_size") is not None:
                    bsz = int(r["batch_size"])
                    out[f"train_nlp_{model}_{prec.lower()}_bs{bsz}_samples_per_sec"] = _safe(r["throughput_samples_per_sec"])
    return out


def extract_inference_vision(data: dict) -> Metricdict:
    rows = data.get("results", [])
    out: Metricdict = {}
    for model in ("resnet50", "resnet101"):
        for prec in ("FP32", "FP16", "BF16", "FP8"):
            filt = lambda r, m=model, p=prec: (r.get("model") == m
                                                and r.get("precision") == p
                                                and r.get("status") == "success")
            v, bs = _best_with_bs(rows, filt, "throughput_img_per_sec")
            if v is not None:
                out[f"infer_vision_{model}_{prec.lower()}_img_per_sec"] = _safe(v)
                if bs is not None:
                    out[f"infer_vision_{model}_{prec.lower()}_best_bs"] = bs
    return out


def extract_inference_nlp(data: dict) -> Metricdict:
    rows = data.get("results", [])
    out: Metricdict = {}
    for model in ("bert-base", "bert-large"):
        for prec in ("FP32", "FP16", "BF16", "FP8"):
            filt = lambda r, m=model, p=prec: (r.get("model") == m
                                                and r.get("precision") == p
                                                and r.get("status") == "success")
            v, bs = _best_with_bs(rows, filt, "throughput_samples_per_sec")
            if v is not None:
                out[f"infer_nlp_{model}_{prec.lower()}_samples_per_sec"] = _safe(v)
                if bs is not None:
                    out[f"infer_nlp_{model}_{prec.lower()}_best_bs"] = bs
    return out


def extract_llm_tokens(data: dict) -> Metricdict:
    rows = data.get("results", [])
    out: Metricdict = {}
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


def extract_vram_limits(data: dict) -> Metricdict:
    rows = data.get("results", [])
    out: Metricdict = {}
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


def extract_gemm_stress(data: dict) -> Metricdict:
    out: Metricdict = {}
    peak = data.get("peak_tflops", {})
    for prec, val in peak.items():
        out[f"gemm_{prec.lower()}_peak_tflops"] = _safe(val)
    if not peak:
        rows = data.get("results", [])
        by_prec: dict[str, float] = {}
        for r in rows:
            p = r.get("precision", "")
            t = r.get("tflops")
            if p and t is not None:
                by_prec[p] = max(by_prec.get(p, 0.0), t)
        for prec, val in by_prec.items():
            out[f"gemm_{prec.lower()}_peak_tflops"] = _safe(val)
    return out


def extract_training_detection(data: dict) -> Metricdict:
    rows = data.get("results", [])
    out: Metricdict = {}
    for model in ("faster-rcnn-resnet50", "mask-rcnn-resnet50"):
        for prec in ("FP32", "FP16", "BF16"):
            filt = lambda r, m=model, p=prec: (r.get("model") == m
                                                and r.get("precision") == p
                                                and r.get("status") == "success")
            safe_model = model.replace("-", "_")
            v, bs = _best_with_bs(rows, filt, "throughput_img_per_sec")
            if v is not None:
                out[f"detect_{safe_model}_{prec.lower()}_img_per_sec"] = _safe(v)
                if bs is not None:
                    out[f"detect_{safe_model}_{prec.lower()}_best_bs"] = bs
                    out[f"detect_{safe_model}_{prec.lower()}_norm_iter_per_sec"] = _safe(v / bs)
            # Store per-batch-size throughputs for same-BS comparison
            for r in rows:
                if filt(r) and r.get("throughput_img_per_sec") is not None and r.get("batch_size") is not None:
                    bsz = int(r["batch_size"])
                    out[f"detect_{safe_model}_{prec.lower()}_bs{bsz}_img_per_sec"] = _safe(r["throughput_img_per_sec"])
    return out


def extract_gpu_fundamentals(data: dict) -> Metricdict:
    rows = data.get("results", [])
    out: Metricdict = {}
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


def extract_multi_gpu_scaling(data: dict) -> Metricdict:
    rows = data.get("results", [])
    out: Metricdict = {}
    for r in rows:
        if r.get("status") != "success":
            continue
        model  = r.get("model", "").replace("-", "_")
        method = r.get("method", "single").lower()
        n_gpus = r.get("n_gpus", 1)
        tput   = r.get("throughput_samples_per_sec")
        eff    = r.get("scaling_efficiency_pct")
        spd    = r.get("speedup")
        if tput is not None:
            out[f"multigpu_{model}_{method}_{n_gpus}gpu_samples_per_sec"] = _safe(tput)
        if eff is not None and n_gpus > 1:
            out[f"multigpu_{model}_{method}_{n_gpus}gpu_eff_pct"] = _safe(eff)
        if spd is not None and n_gpus > 1:
            out[f"multigpu_{model}_{method}_{n_gpus}gpu_speedup"] = _safe(spd)
    return out


_EXTRACTORS = {
    "training_vision.json":    extract_training_vision,
    "training_nlp.json":       extract_training_nlp,
    "inference_vision.json":   extract_inference_vision,
    "inference_nlp.json":      extract_inference_nlp,
    "llm_tokens_per_sec.json": extract_llm_tokens,
    "vram_limits.json":        extract_vram_limits,
    "training_detection.json": extract_training_detection,
    "multi_gpu_scaling.json":  extract_multi_gpu_scaling,
}

# Files from which we extract per-benchmark power for efficiency charts
_POWER_BENCHMARKS = {
    "training_vision":    "training_vision.json",
    "training_nlp":       "training_nlp.json",
    "inference_vision":   "inference_vision.json",
    "inference_nlp":      "inference_nlp.json",
    "llm_tokens_per_sec": "llm_tokens_per_sec.json",
    "training_detection": "training_detection.json",
}


# ── Session loader ────────────────────────────────────────────────────────────

def _load_session(session_dir: Path) -> tuple[str, str, Metricdict]:
    meta_path = session_dir / "session_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"session_meta.json not found in {session_dir}")

    with meta_path.open() as f:
        meta = json.load(f)

    session_id = meta.get("session_id", session_dir.name)
    date_str = session_id[:8]

    gpu_name = "GPU"
    gpu_vram_gb = None
    for p in session_dir.glob("*.json"):
        if p.name in ("session_meta.json", "benchmark_summary.json"):
            continue
        try:
            with p.open() as f:
                d = json.load(f)
            if "gpu" in d:
                gpu_name = d["gpu"].replace(" ", "_").replace("/", "_")
                if "vram_gb" in d:
                    gpu_vram_gb = d["vram_gb"]
                break
        except Exception:
            continue

    col_id = f"{gpu_name}_{date_str}"
    col_label = f"{gpu_name.replace('_', ' ')} \u2014 {date_str}"

    combined: Metricdict = {}
    combined["session_id"] = session_id
    combined["gpu_name"] = gpu_name.replace("_", " ")
    combined["run_date"] = date_str
    combined["elapsed_sec"] = meta.get("elapsed_seconds")
    if gpu_vram_gb is not None:
        combined["gpu_vram_gb"] = gpu_vram_gb

    for filename, extractor in _EXTRACTORS.items():
        jpath = session_dir / filename
        if not jpath.exists():
            continue
        try:
            with jpath.open() as f:
                data = json.load(f)
            combined.update(extractor(data))
        except Exception as e:
            print(f"  WARNING: could not parse {jpath.name} \u2014 {e}", file=sys.stderr)

    # Per-benchmark power data
    for bench_key, bench_file in _POWER_BENCHMARKS.items():
        jpath = session_dir / bench_file
        if not jpath.exists():
            continue
        try:
            with jpath.open() as f:
                data = json.load(f)
            hw = data.get("hw_monitor")
            if hw and hw.get("avg_power_w") is not None:
                combined[f"hw_power_{bench_key}_avg_w"] = round(hw["avg_power_w"], 1)
        except Exception:
            continue

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
    sessions: list[tuple[str, str, Metricdict]],
) -> tuple[list[str], list[str], list[dict]]:
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

    rows: list[dict] = []
    for key in meta_keys + metric_keys:
        row: dict = {"metric_id": key}
        for col_id, _, metrics in sessions:
            row[col_id] = metrics.get(key)
        rows.append(row)

    return meta_keys + metric_keys, col_ids, rows


# ── Output writers ────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], col_ids: list[str], output_path: Path) -> None:
    fieldnames = ["metric_id"] + col_ids
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], col_ids: list[str],
               sessions: list[tuple[str, str, Metricdict]],
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

    # Key throughput metrics used to compute overall DL performance rank.
    _PERF_METRICS = [
        "train_vision_resnet50_fp16_img_per_sec",
        "train_vision_resnet101_fp16_img_per_sec",
        "train_nlp_bert-base_bf16_samples_per_sec",
        "train_nlp_bert-large_bf16_samples_per_sec",
        "infer_vision_resnet50_fp16_img_per_sec",
        "infer_nlp_bert-base_bf16_samples_per_sec",
    ]

    def __init__(self, path: Path):
        with path.open() as f:
            raw = json.load(f)
        self.columns: list[dict] = raw["columns"]
        self.col_ids: list[str] = [c["id"] for c in self.columns]
        self.col_labels: list[str] = [c["label"] for c in self.columns]
        self._metrics: dict[str, dict[str, Any]] = {}
        for row in raw["metrics"]:
            mid = row["metric_id"]
            self._metrics[mid] = {k: v for k, v in row.items() if k != "metric_id"}
        self._sort_by_performance()

    def _sort_by_performance(self) -> None:
        """Sort GPUs by geometric mean of key DL throughput metrics (best first)."""
        scores: list[tuple[float, int]] = []
        for gi, cid in enumerate(self.col_ids):
            log_sum, count = 0.0, 0
            for mid in self._PERF_METRICS:
                row = self._metrics.get(mid, {})
                v = row.get(cid)
                if v is not None and v > 0:
                    log_sum += log(v)
                    count += 1
            score = exp(log_sum / count) if count > 0 else 0.0
            scores.append((score, gi))
        # Sort descending by score (best GPU first)
        order = [gi for _, gi in sorted(scores, key=lambda x: -x[0])]
        self.col_ids = [self.col_ids[i] for i in order]
        self.columns = [self.columns[i] for i in order]
        self.col_labels = [self.col_labels[i] for i in order]

    @property
    def n_gpus(self) -> int:
        return len(self.col_ids)

    def gpu_names(self) -> list[str]:
        names = []
        for cid in self.col_ids:
            v = self._metrics.get("gpu_name", {}).get(cid, cid)
            names.append(str(v))
        return names

    def get(self, metric_id: str) -> list[float | None]:
        row = self._metrics.get(metric_id, {})
        return [row.get(cid) for cid in self.col_ids]

    def metric_ids(self) -> list[str]:
        return list(self._metrics.keys())

    def find_metrics(self, prefix: str) -> list[str]:
        return [m for m in self._metrics if m.startswith(prefix)]


# ── Adaptive sizing helpers ───────────────────────────────────────────────────

def _fig_w(n_gpus: int, n_categories: int = 6) -> float:
    """Dynamic figure width based on GPU count and category count."""
    base = max(BASE_FIG_W, 3.0 * n_gpus)
    if n_categories > 8:
        base = max(base, 1.8 * n_categories)
    return min(base, 32)


def _val_fontsize(n_gpus: int) -> int:
    if n_gpus <= 3:
        return 10
    if n_gpus <= 5:
        return 9
    return 7


def _tick_fontsize(n_gpus: int) -> int:
    if n_gpus <= 4:
        return 11
    return 9


def _legend_kwargs(n_gpus: int) -> dict:
    if n_gpus <= 4:
        return dict(fontsize=10, loc="lower right",
                    facecolor=CARD_COLOR, edgecolor=GRID_COLOR)
    return dict(fontsize=9, loc="lower right",
                facecolor=CARD_COLOR, edgecolor=GRID_COLOR)


def _gpu_name_fontsize(n_gpus: int) -> int:
    """Font size for GPU name labels rendered inside chart bars."""
    if n_gpus <= 3:
        return 8
    if n_gpus <= 5:
        return 7
    return 6


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR,
                edgecolor="none", pad_inches=0.3)
    plt.close(fig)
    print(f"  \u2713 {path.name}")


def _watermark(fig):
    txt = fig.text(0.99, 0.01, S("watermark"),
                   ha="right", va="bottom", fontsize=9, color=SUBTEXT_COLOR, alpha=0.5)
    txt.set_path_effects([
        path_effects.withStroke(linewidth=2, foreground=BG_COLOR, alpha=0.3),
    ])


def _standard_fig(title: str, direction_text: str, direction_color: str,
                  n_gpus: int, n_categories: int = 6,
                  height_ratio: float = 1.0,
                  n_axes: int = 1, width_ratios: list | None = None):
    """Create figure with standardized title/direction/watermark layout."""
    w = _fig_w(n_gpus, n_categories)
    h = BASE_FIG_H * height_ratio

    if n_axes == 1:
        fig, ax = plt.subplots(figsize=(w, h))
        axes = ax
    else:
        kw = {"width_ratios": width_ratios} if width_ratios else {}
        fig, axes = plt.subplots(1, n_axes, figsize=(w, h), gridspec_kw=kw)

    t = fig.suptitle(title, fontsize=24, fontweight="bold",
                     color=TEXT_COLOR, y=0.97)
    t.set_path_effects([
        path_effects.withStroke(linewidth=4, foreground=BG_COLOR, alpha=0.5),
        path_effects.Normal(),
    ])
    dt = fig.text(0.97, 0.935, direction_text, ha="right", va="top",
                  fontsize=10, color=direction_color, alpha=0.9)
    dt.set_path_effects([
        path_effects.withStroke(linewidth=2, foreground=BG_COLOR, alpha=0.5),
    ])
    _watermark(fig)
    return fig, axes


def _grouped_bar(ax, categories: list[str], gpu_names: list[str],
                 values_per_gpu: list[list[float]], ylabel: str,
                 subtitle: str,
                 fmt: str = "{:.0f}"):
    n_groups = len(categories)
    n_gpus = len(gpu_names)
    if n_groups == 0 or n_gpus == 0:
        return

    x = np.arange(n_groups)
    total_width = min(0.85, 0.15 * n_gpus + 0.25)
    bar_w = total_width / n_gpus
    vfs = _val_fontsize(n_gpus)
    gfs = _gpu_name_fontsize(n_gpus)

    all_vals = []
    for gi, (gname, gvals) in enumerate(zip(gpu_names, values_per_gpu)):
        offset = (gi - (n_gpus - 1) / 2) * bar_w
        color = GPU_COLORS[gi % len(GPU_COLORS)]
        vals = [v if v is not None else 0 for v in gvals]
        all_vals.extend(vals)
        # Invisible bar for legend only
        ax.bar([], [], color=color, label=gname)
        for ci, v in enumerate(vals):
            if v > 0:
                cx = x[ci] + offset
                _gradient_bar_v(ax, cx, 0, v, bar_w * 0.88, color, zorder=3)
                txt = ax.text(cx, v, fmt.format(v),
                              ha="center", va="bottom",
                              fontsize=vfs, color=TEXT_COLOR, fontweight="bold")
                txt.set_path_effects([
                    path_effects.withStroke(linewidth=2, foreground=BG_COLOR, alpha=0.6),
                ])
                ax.text(cx, v * 0.5, gname, ha="center", va="center",
                        fontsize=gfs, color=BG_COLOR, fontweight="bold",
                        alpha=0.85, rotation=90, clip_on=True)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=_tick_fontsize(n_gpus))
    ax.set_ylabel(ylabel, fontsize=13)
    st = ax.set_title(subtitle, fontsize=14, color=ACCENT_BLUE, pad=10)
    st.set_path_effects([path_effects.withStroke(linewidth=2, foreground=BG_COLOR, alpha=0.4)])
    ax.grid(axis="y", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    max_v = max(all_vals) if all_vals else 1
    ax.set_ylim(0, max_v * 1.18)


def _horizontal_grouped_bar(ax, categories: list[str], gpu_names: list[str],
                            values_per_gpu: list[list[float]], xlabel: str,
                            subtitle: str,
                            fmt: str = "{:.1f}",
                            annotations_per_gpu: list[list[str]] | None = None):
    """Draw horizontal grouped bar chart with gradient fill.

    *annotations_per_gpu*, if provided, is a list (one per GPU) of lists
    (one per category) holding short strings (e.g. "BS=64") appended to
    the value label on each bar.
    """
    n_groups = len(categories)
    n_gpus = len(gpu_names)
    if n_groups == 0 or n_gpus == 0:
        return

    y = np.arange(n_groups)
    total_height = min(0.85, 0.15 * n_gpus + 0.25)
    bar_h = total_height / n_gpus
    vfs = _val_fontsize(n_gpus)
    gfs = _gpu_name_fontsize(n_gpus)

    all_vals = []
    for gi, gvals in enumerate(values_per_gpu):
        vals = [v if v is not None else 0 for v in gvals]
        all_vals.extend(vals)
    max_v_pre = max(all_vals) if all_vals else 1

    for gi, (gname, gvals) in enumerate(zip(gpu_names, values_per_gpu)):
        offset = (gi - (n_gpus - 1) / 2) * bar_h
        color = GPU_COLORS[gi % len(GPU_COLORS)]
        vals = [v if v is not None else 0 for v in gvals]
        # Invisible bar for legend
        ax.barh([], [], color=color, label=gname)
        ann = annotations_per_gpu[gi] if annotations_per_gpu else [None] * len(vals)
        for ci, v in enumerate(vals):
            if v > 0:
                cy = y[ci] + offset
                _gradient_barh(ax, cy, v, bar_h * 0.88, color, zorder=3)
                label = fmt.format(v)
                a = ann[ci] if ci < len(ann) else None
                if a:
                    label += f"  ({a})"
                txt = ax.text(v + max_v_pre * 0.01, cy, label,
                              va="center", fontsize=vfs,
                              color=TEXT_COLOR, fontweight="bold")
                txt.set_path_effects([
                    path_effects.withStroke(linewidth=2, foreground=BG_COLOR, alpha=0.6),
                ])
                ax.text(max_v_pre * 0.01, cy, gname,
                        ha="left", va="center",
                        fontsize=gfs, color=BG_COLOR, fontweight="bold",
                        alpha=0.85, clip_on=True)

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=_tick_fontsize(n_gpus))
    ax.set_xlabel(xlabel, fontsize=13)
    st = ax.set_title(subtitle, fontsize=14, color=ACCENT_BLUE, pad=10)
    st.set_path_effects([path_effects.withStroke(linewidth=2, foreground=BG_COLOR, alpha=0.4)])
    ax.grid(axis="x", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_v_pre * 1.30)
    ax.invert_yaxis()


def _build_bs_annotations(d: "ComparisonData", bs_keys: list[str]) -> list[list[str]]:
    """Build per-GPU, per-category annotation strings like 'BS=64'.

    Returns a list (one per GPU) of lists (one per category) of strings.
    If a batch-size metric is missing for a particular cell the string is empty.
    """
    anns: list[list[str]] = []
    for gi in range(d.n_gpus):
        row: list[str] = []
        for bk in bs_keys:
            v = d.get(bk)[gi]
            row.append(f"BS={int(v)}" if v is not None else "")
        anns.append(row)
    return anns


# ═══════════════════════════════════════════════════════════════════════════════
#  Chart builders
# ═══════════════════════════════════════════════════════════════════════════════

def chart_training_vision(d: ComparisonData, out: Path):
    cats, keys, bs_keys = [], [], []
    _model_labels = {"resnet50": "CNN (ResNet-50)", "resnet101": "CNN (ResNet-101)"}
    for model in ["resnet50", "resnet101"]:
        for prec in ["fp32", "fp16", "bf16"]:
            mid = f"train_vision_{model}_{prec}_img_per_sec"
            if any(v is not None for v in d.get(mid)):
                cats.append(f"{_model_labels[model]} {prec.upper()}")
                keys.append(mid)
                bs_keys.append(f"train_vision_{model}_{prec}_best_bs")
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    anns = _build_bs_annotations(d, bs_keys)
    fig, ax = _standard_fig(S("train_vision_title"), S("higher_better"),
                            ACCENT_GREEN, d.n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.18))
    _horizontal_grouped_bar(ax, cats, d.gpu_names(), vals, S("train_vision_ylabel"),
                 S("train_vision_subtitle"), annotations_per_gpu=anns)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_training_vision.png")


def chart_training_nlp(d: ComparisonData, out: Path):
    cats, keys, bs_keys = [], [], []
    _model_labels = {"bert-base": "Transformer (BERT-Base)", "bert-large": "Transformer (BERT-Large)"}
    for model in ["bert-base", "bert-large"]:
        for prec in ["fp32", "fp16", "bf16"]:
            mid = f"train_nlp_{model}_{prec}_samples_per_sec"
            if any(v is not None for v in d.get(mid)):
                cats.append(f"{_model_labels[model]} {prec.upper()}")
                keys.append(mid)
                bs_keys.append(f"train_nlp_{model}_{prec}_best_bs")
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    anns = _build_bs_annotations(d, bs_keys)
    fig, ax = _standard_fig(S("train_nlp_title"), S("higher_better"),
                            ACCENT_GREEN, d.n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.18))
    _horizontal_grouped_bar(ax, cats, d.gpu_names(), vals, S("train_nlp_ylabel"),
                 S("train_nlp_subtitle"), annotations_per_gpu=anns)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_training_nlp.png")


def chart_inference_vision(d: ComparisonData, out: Path):
    cats, keys, bs_keys = [], [], []
    _model_labels = {"resnet50": "CNN (ResNet-50)", "resnet101": "CNN (ResNet-101)"}
    for model in ["resnet50", "resnet101"]:
        for prec in ["fp32", "fp16", "bf16"]:
            mid = f"infer_vision_{model}_{prec}_img_per_sec"
            if any(v is not None for v in d.get(mid)):
                cats.append(f"{_model_labels[model]} {prec.upper()}")
                keys.append(mid)
                bs_keys.append(f"infer_vision_{model}_{prec}_best_bs")
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    anns = _build_bs_annotations(d, bs_keys)
    fig, ax = _standard_fig(S("infer_vision_title"), S("higher_better"),
                            ACCENT_GREEN, d.n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.18))
    _horizontal_grouped_bar(ax, cats, d.gpu_names(), vals, S("infer_vision_ylabel"),
                 S("infer_vision_subtitle"), annotations_per_gpu=anns)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_inference_vision.png")


def chart_inference_nlp(d: ComparisonData, out: Path):
    cats, keys, bs_keys = [], [], []
    _model_labels = {"bert-base": "Transformer (BERT-Base)", "bert-large": "Transformer (BERT-Large)"}
    for model in ["bert-base", "bert-large"]:
        for prec in ["fp32", "fp16", "bf16"]:
            mid = f"infer_nlp_{model}_{prec}_samples_per_sec"
            if any(v is not None for v in d.get(mid)):
                cats.append(f"{_model_labels[model]} {prec.upper()}")
                keys.append(mid)
                bs_keys.append(f"infer_nlp_{model}_{prec}_best_bs")
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    anns = _build_bs_annotations(d, bs_keys)
    fig, ax = _standard_fig(S("infer_nlp_title"), S("higher_better"),
                            ACCENT_GREEN, d.n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.18))
    _horizontal_grouped_bar(ax, cats, d.gpu_names(), vals, S("infer_nlp_ylabel"),
                 S("infer_nlp_subtitle"), annotations_per_gpu=anns)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_inference_nlp.png")


def chart_llm(d: ComparisonData, out: Path):
    """LLM token generation chart — only shows bars for GPUs that have data
    for each model, eliminating empty gaps."""
    tps_keys = sorted([m for m in d.metric_ids()
                       if m.startswith("llm_") and m.endswith("_tokens_per_sec")
                       and m != "llm_best_tokens_per_sec"])
    if not tps_keys:
        return

    # Custom display order: DeepSeek 8B, Phi 4, Gemma 2 27B, QwQ 32B
    _LLM_ORDER = ["deepseek", "phi", "gemma", "qwq"]

    def _order_key(mid: str) -> int:
        low = mid.lower()
        for i, prefix in enumerate(_LLM_ORDER):
            if prefix in low:
                return i
        return len(_LLM_ORDER)

    tps_keys.sort(key=_order_key)

    def _label(mid: str) -> str:
        name = mid.replace("llm_", "").replace("_tokens_per_sec", "")
        pretty = name.replace("_", " ").title()
        # Add parameter counts for known models
        _PARAMS = {
            "deepseek r1 distill llama 8b": "8B",
            "phi 4": "14B",
            "gemma 2 27b": "27B",
            "qwq 32b": "32B",
        }
        size = _PARAMS.get(pretty.lower(), "")
        if size:
            return f"LLM ({pretty} — {size})"
        return "LLM (" + pretty + ")"

    all_gpu_names = d.gpu_names()
    n_gpus = d.n_gpus

    # Build compact model data: only GPUs with data per model
    model_data: list[tuple[str, list[tuple[int, str, float]]]] = []
    for mk in tps_keys:
        label = _label(mk)
        vals = d.get(mk)
        gpu_bars = []
        for gi in range(n_gpus):
            if vals[gi] is not None and vals[gi] > 0:
                gpu_bars.append((gi, all_gpu_names[gi], vals[gi]))
        if gpu_bars:
            model_data.append((label, gpu_bars))

    if not model_data:
        return

    # Calculate total number of bars for figure height
    total_bars = sum(len(bars) for _, bars in model_data)
    # Add spacing between model groups
    total_rows = total_bars + len(model_data) - 1

    fig_h = max(BASE_FIG_H, total_rows * 0.55 + 2)
    fig_w = _fig_w(n_gpus, total_rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    t = fig.suptitle(S("llm_title"), fontsize=24, fontweight="bold",
                     color=TEXT_COLOR, y=0.97)
    t.set_path_effects([
        path_effects.withStroke(linewidth=4, foreground=BG_COLOR, alpha=0.5),
        path_effects.Normal(),
    ])
    dt = fig.text(0.97, 0.935, S("llm_higher"), ha="right", va="top",
                  fontsize=10, color=ACCENT_GREEN, alpha=0.9)
    dt.set_path_effects([path_effects.withStroke(linewidth=2, foreground=BG_COLOR, alpha=0.5)])
    _watermark(fig)

    bar_h = 0.7
    vfs = _val_fontsize(n_gpus)
    gfs = _gpu_name_fontsize(n_gpus)
    y_positions: list[float] = []
    y_labels: list[str] = []
    bar_colors: list[str] = []
    bar_vals: list[float] = []

    y_pos = 0
    model_label_positions: list[tuple[float, float, str]] = []  # (y_start, y_end, label)

    for mi, (model_label, gpu_bars) in enumerate(model_data):
        if mi > 0:
            y_pos += 0.8  # gap between model groups

        group_start = y_pos
        for gi_orig, gname, val in gpu_bars:
            y_positions.append(y_pos)
            y_labels.append(gname)
            bar_colors.append(GPU_COLORS[gi_orig % len(GPU_COLORS)])
            bar_vals.append(val)
            y_pos += 1.0

        model_label_positions.append((group_start, y_pos - 1.0, model_label))

    # Draw bars
    all_vals = [v for v in bar_vals if v > 0]
    max_v = max(all_vals) if all_vals else 1

    for i, (yp, val, color, ylbl) in enumerate(zip(y_positions, bar_vals, bar_colors, y_labels)):
        _gradient_barh(ax, yp, val, bar_h, color, zorder=3)
        # GPU name inside the bar
        ax.text(max_v * 0.01, yp, ylbl, ha="left", va="center",
                fontsize=gfs, color=BG_COLOR, fontweight="bold",
                alpha=0.85, clip_on=True)
        # Value label after the bar
        txt = ax.text(val + max_v * 0.01, yp, f"{val:.1f} t/s",
                      va="center", fontsize=vfs, color=TEXT_COLOR, fontweight="bold")
        txt.set_path_effects([
            path_effects.withStroke(linewidth=2, foreground=BG_COLOR, alpha=0.6),
        ])

    # Y-axis: model group labels (centered for each group)
    model_ticks = []
    model_tick_labels = []
    for y_start, y_end, label in model_label_positions:
        center = (y_start + y_end) / 2
        model_ticks.append(center)
        model_tick_labels.append(label)

    ax.set_yticks(model_ticks)
    ax.set_yticklabels(model_tick_labels, fontsize=11, fontweight="bold",
                       color=ACCENT_BLUE)
    ax.invert_yaxis()

    ax.set_xlabel(S("llm_xlabel_tps"), fontsize=13)
    ax.set_title(S("llm_speed"), fontsize=14, color=ACCENT_BLUE, pad=10)
    ax.grid(axis="x", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_v * 1.20)

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
    fig, ax = _standard_fig(S("gemm_title"), S("higher_better"),
                            ACCENT_GREEN, d.n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.18))
    _horizontal_grouped_bar(ax, cats, d.gpu_names(), vals, S("gemm_ylabel"),
                 S("gemm_subtitle"), fmt="{:.1f}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_gemm.png")


def chart_fundamentals(d: ComparisonData, out: Path):
    specs = [
        ("fund_d2d_bw_peak_gb_s",            S("fund_mem_bw"),       "{:.0f}"),
        ("fund_kernel_launch_latency_us",     S("fund_kernel_lat"),   "{:.1f}"),
        ("fund_reduction_peak_gb_s",          S("fund_reduction"),    "{:.0f}"),
        ("fund_spmm_peak_gflops",             S("fund_spmm"),        "{:.0f}"),
        ("fund_fft_fp32_peak_gflops",         S("fund_fft"),         "{:.0f}"),
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
    n_gpus = d.n_gpus

    fig, ax = _standard_fig(S("fund_title"), S("bw_high_latency_low"),
                            ACCENT_CYAN, n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.18))

    y = np.arange(len(cats))
    total_height = min(0.85, 0.15 * n_gpus + 0.25)
    bar_h = total_height / n_gpus
    vfs = _val_fontsize(n_gpus)
    gfs = _gpu_name_fontsize(n_gpus)

    all_vals = []
    for gi, gname in enumerate(d.gpu_names()):
        offset = (gi - (n_gpus - 1) / 2) * bar_h
        color = GPU_COLORS[gi % len(GPU_COLORS)]
        gvals = [v if v is not None else 0 for v in vals[gi]]
        all_vals.extend(gvals)
        bars = ax.barh(y + offset, gvals, bar_h * 0.88, color=color,
                       edgecolor=BG_COLOR, linewidth=1, zorder=3, label=gname)
        for bar, v, fmt in zip(bars, gvals, fmts):
            if v > 0:
                ax.text(v + max(all_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                        fmt.format(v), va="center",
                        fontsize=vfs, color=TEXT_COLOR, fontweight="bold")
                ax.text(max(all_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                        gname, ha="left", va="center",
                        fontsize=gfs, color=BG_COLOR, fontweight="bold",
                        alpha=0.85, clip_on=True)

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=_tick_fontsize(n_gpus))
    ax.set_title(S("fund_subtitle"), fontsize=12, color=SUBTEXT_COLOR, pad=10)
    ax.grid(axis="x", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    max_v = max(all_vals) if all_vals else 1
    ax.set_xlim(0, max_v * 1.25)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_fundamentals.png")


def chart_detection(d: ComparisonData, out: Path):
    cats, keys, bs_keys = [], [], []
    _model_labels = {
        "faster_rcnn_resnet50": "CNN (Faster R-CNN)",
        "mask_rcnn_resnet50": "CNN (Mask R-CNN)",
    }
    for model in ["faster_rcnn_resnet50", "mask_rcnn_resnet50"]:
        for prec in ["fp32", "fp16", "bf16"]:
            mid = f"detect_{model}_{prec}_img_per_sec"
            if any(v is not None for v in d.get(mid)):
                cats.append(f"{_model_labels[model]}\n{prec.upper()}")
                keys.append(mid)
                bs_keys.append(f"detect_{model}_{prec}_best_bs")
    if not cats:
        return
    vals = [[d.get(m)[gi] for m in keys] for gi in range(d.n_gpus)]
    anns = _build_bs_annotations(d, bs_keys)
    fig, ax = _standard_fig(S("detect_title"), S("higher_better"),
                            ACCENT_GREEN, d.n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.18))
    _horizontal_grouped_bar(ax, cats, d.gpu_names(), vals, S("detect_ylabel"),
                 S("detect_subtitle"), fmt="{:.1f}", annotations_per_gpu=anns)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    _save(fig, out / "cmp_detection.png")


# ── Batch-Normalized Training Charts ─────────────────────────────────────────

def _batch_norm_chart(d: ComparisonData, out: Path,
                      models: list[str], precisions: list[str],
                      metric_prefix: str,
                      bs_prefix: str,
                      title_key: str, subtitle_key: str,
                      filename: str,
                      tput_suffix: str = "_img_per_sec"):
    """Chart comparing GPUs at the same (smallest common) batch size.

    For each model/precision combo, finds the smallest batch size that ALL
    GPUs ran, then shows raw throughput at that batch size.  This removes
    the VRAM advantage — a GPU with more VRAM can use larger batches, but
    here we compare pure compute speed at an equal workload.
    """
    import re as _re

    cats: list[str] = []
    val_keys: list[str] = []
    chosen_bs: list[int] = []

    for model in models:
        for prec in precisions:
            # Find all per-batch-size metric keys for this model/prec
            pattern = f"{metric_prefix}{model}_{prec}_bs"
            bs_metrics = sorted(d.find_metrics(pattern))
            if not bs_metrics:
                continue

            # Extract available batch sizes per GPU
            # Key format: prefix_model_prec_bs{N}_throughput_suffix
            bs_to_key: dict[int, str] = {}
            for mk in bs_metrics:
                m = _re.search(r"_bs(\d+)" + _re.escape(tput_suffix) + r"$", mk)
                if m:
                    bs_to_key[int(m.group(1))] = mk

            if not bs_to_key:
                continue

            # Find smallest batch size where ALL GPUs have data
            common_bs = None
            for bs_val in sorted(bs_to_key.keys()):
                key = bs_to_key[bs_val]
                vals_at_bs = d.get(key)
                if all(v is not None for v in vals_at_bs):
                    common_bs = bs_val
                    break

            if common_bs is None:
                # Fallback: use smallest BS with at least some data
                for bs_val in sorted(bs_to_key.keys()):
                    key = bs_to_key[bs_val]
                    vals_at_bs = d.get(key)
                    if any(v is not None for v in vals_at_bs):
                        common_bs = bs_val
                        break

            if common_bs is None:
                continue

            nice_model = model.replace("_", "-").upper() if "_" not in model else \
                         model.replace("faster_rcnn_resnet50", "Faster R-CNN") \
                              .replace("mask_rcnn_resnet50", "Mask R-CNN") \
                              .replace("resnet", "RESNET").replace("bert-", "BERT-")
            if nice_model == model:
                nice_model = model.upper()
            cats.append(f"{nice_model} {prec.upper()}")
            val_keys.append(bs_to_key[common_bs])
            chosen_bs.append(common_bs)

    if not cats:
        return

    vals = [[d.get(m)[gi] for m in val_keys] for gi in range(d.n_gpus)]

    # Build annotations showing the common BS used
    anns: list[list[str]] = []
    for gi in range(d.n_gpus):
        row: list[str] = []
        for ci in range(len(cats)):
            v = vals[gi][ci]
            if v is not None:
                row.append(f"BS={chosen_bs[ci]}")
            else:
                row.append("")
        anns.append(row)

    title = f"{S(title_key)} {S('batch_norm_suffix')}"
    fig, ax = _standard_fig(title, S("higher_better"),
                            ACCENT_BLUE, d.n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.18))
    _horizontal_grouped_bar(ax, cats, d.gpu_names(), vals, S("batch_norm_ylabel"),
                 S(subtitle_key), fmt="{:.1f}", annotations_per_gpu=anns)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / filename)


def chart_training_vision_batch_norm(d: ComparisonData, out: Path):
    _batch_norm_chart(d, out,
                      models=["resnet50", "resnet101"],
                      precisions=["fp32", "fp16", "bf16"],
                      metric_prefix="train_vision_",
                      bs_prefix="train_vision_",
                      title_key="train_vision_title",
                      subtitle_key="train_vision_subtitle",
                      filename="cmp_training_vision_batch_norm.png",
                      tput_suffix="_img_per_sec")


def chart_training_nlp_batch_norm(d: ComparisonData, out: Path):
    _batch_norm_chart(d, out,
                      models=["bert-base", "bert-large"],
                      precisions=["fp32", "fp16", "bf16"],
                      metric_prefix="train_nlp_",
                      bs_prefix="train_nlp_",
                      title_key="train_nlp_title",
                      subtitle_key="train_nlp_subtitle",
                      filename="cmp_training_nlp_batch_norm.png",
                      tput_suffix="_samples_per_sec")


def chart_detection_batch_norm(d: ComparisonData, out: Path):
    _batch_norm_chart(d, out,
                      models=["faster_rcnn_resnet50", "mask_rcnn_resnet50"],
                      precisions=["fp32", "fp16", "bf16"],
                      metric_prefix="detect_",
                      bs_prefix="detect_",
                      title_key="detect_title",
                      subtitle_key="detect_subtitle",
                      filename="cmp_detection_batch_norm.png",
                      tput_suffix="_img_per_sec")


# ── Power Efficiency Charts ──────────────────────────────────────────────────

def chart_power_efficiency(d: ComparisonData, out: Path):
    """Single chart: max throughput / avg watts for ResNet Train, BERT Train, LLM Token/s."""
    bench_specs = [
        {
            "power_key": "hw_power_training_vision_avg_w",
            "tput_prefix": "train_vision_",
            "tput_suffix": "_img_per_sec",
            "label": S("power_resnet_train"),
            "unit_suffix": S("unit_img_s") + "/W",
        },
        {
            "power_key": "hw_power_training_nlp_avg_w",
            "tput_prefix": "train_nlp_",
            "tput_suffix": "_samples_per_sec",
            "label": S("power_bert_train"),
            "unit_suffix": S("unit_samples_s") + "/W",
        },
        {
            "power_key": "hw_power_llm_tokens_per_sec_avg_w",
            "tput_prefix": "llm_",
            "tput_suffix": "_tokens_per_sec",
            "label": S("power_llm_tokens"),
            "unit_suffix": "token/s/W",
        },
    ]

    cats: list[str] = []
    eff_per_gpu: list[list[float | None]] = []

    for spec in bench_specs:
        power_vals = d.get(spec["power_key"])

        # Find all throughput keys for this benchmark and pick the max per GPU
        tput_keys = sorted([m for m in d.metric_ids()
                            if m.startswith(spec["tput_prefix"])
                            and m.endswith(spec["tput_suffix"])
                            and m != "llm_best_tokens_per_sec"])
        if not tput_keys:
            continue

        # For LLM: use only the best model that ALL GPUs ran (fair comparison).
        # Using max across all models would penalise GPUs with more VRAM since
        # their average power includes slow large-model runs.
        if spec["tput_prefix"] == "llm_":
            # Find the model key where ALL GPUs have data
            common_keys = [k for k in tput_keys
                           if all(v is not None for v in d.get(k))]
            if common_keys:
                # Pick the one with highest average throughput
                best_common = max(common_keys,
                                  key=lambda k: sum(v or 0 for v in d.get(k)))
                tput_keys = [best_common]

        # For each GPU, find the max throughput across all model variants
        max_tput: list[float | None] = [None] * d.n_gpus
        for mk in tput_keys:
            tvals = d.get(mk)
            for gi in range(d.n_gpus):
                if tvals[gi] is not None:
                    if max_tput[gi] is None or tvals[gi] > max_tput[gi]:
                        max_tput[gi] = tvals[gi]

        # Compute throughput / watts
        row: list[float | None] = []
        has_any = False
        for gi in range(d.n_gpus):
            t = max_tput[gi]
            p = power_vals[gi]
            if t is not None and p is not None and p > 0:
                row.append(round(t / p, 3))
                has_any = True
            else:
                row.append(None)

        if has_any:
            cats.append(spec["label"])
            eff_per_gpu.append(row)

    if not cats:
        return

    vals = [[eff_per_gpu[ci][gi] for ci in range(len(cats))]
            for gi in range(d.n_gpus)]

    fig, ax = _standard_fig(
        S("power_eff_title"),
        S("higher_better"), ACCENT_GREEN,
        d.n_gpus, len(cats),
        height_ratio=max(1.0, len(cats) * 0.18))
    _horizontal_grouped_bar(ax, cats, d.gpu_names(), vals,
                 S("power_eff_ylabel"), S("power_eff_subtitle"), fmt="{:.2f}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_power_efficiency.png")


# ── Relative Performance Chart ────────────────────────────────────────────────

def chart_relative_performance(d: ComparisonData, out: Path):
    """Lowest-scoring GPU = 1.0x baseline, show speedup for others."""
    metric_specs = [
        ("train_vision_resnet50_fp16_img_per_sec",     "CNN (ResNet-50) Train FP16"),
        ("train_nlp_bert-base_bf16_samples_per_sec",   "Transformer (BERT-Base) Train BF16"),
        ("infer_vision_resnet50_fp16_img_per_sec",     "CNN (ResNet-50) Infer FP16"),
    ]

    # Add best LLM metric
    tps_keys = sorted([m for m in d.metric_ids()
                       if m.startswith("llm_") and m.endswith("_tokens_per_sec")
                       and m != "llm_best_tokens_per_sec"])
    if tps_keys:
        best_key = max(tps_keys, key=lambda m: sum(v or 0 for v in d.get(m)))
        name = best_key.replace("llm_", "").replace("_tokens_per_sec", "").replace("_", " ").title()
        metric_specs.append((best_key, f"LLM ({name})"))

    cats, rel_per_gpu = [], []
    for mid, label in metric_specs:
        vals = d.get(mid)
        valid = [v for v in vals if v is not None and v > 0]
        if len(valid) < 1:
            continue
        baseline = min(valid)
        row = []
        for v in vals:
            if v is not None and baseline > 0:
                row.append(round(v / baseline, 2))
            else:
                row.append(None)
        cats.append(label)
        rel_per_gpu.append(row)

    if not cats:
        return

    vals = [[rel_per_gpu[ci][gi] for ci in range(len(cats))]
            for gi in range(d.n_gpus)]

    n_gpus = d.n_gpus
    fig, ax = _standard_fig(S("relative_title"), S("higher_better"),
                            ACCENT_GREEN, n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.18))

    y = np.arange(len(cats))
    total_height = min(0.85, 0.15 * n_gpus + 0.25)
    bar_h = total_height / n_gpus
    vfs = _val_fontsize(n_gpus)
    gfs = _gpu_name_fontsize(n_gpus)

    all_vals_flat = []
    for gi, gname in enumerate(d.gpu_names()):
        offset = (gi - (n_gpus - 1) / 2) * bar_h
        color = GPU_COLORS[gi % len(GPU_COLORS)]
        gvals = [v if v is not None else 0 for v in vals[gi]]
        all_vals_flat.extend(gvals)
        bars = ax.barh(y + offset, gvals, bar_h * 0.88, color=color,
                       edgecolor=BG_COLOR, linewidth=1, zorder=3, label=gname)
        for bar, v in zip(bars, gvals):
            if v > 0:
                ax.text(v + max(all_vals_flat) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{v:.2f}x", va="center",
                        fontsize=vfs, color=TEXT_COLOR, fontweight="bold")
                ax.text(max(all_vals_flat) * 0.01, bar.get_y() + bar.get_height() / 2,
                        gname, ha="left", va="center",
                        fontsize=gfs, color=BG_COLOR, fontweight="bold",
                        alpha=0.85, clip_on=True)

    max_v = max(all_vals_flat) if all_vals_flat else 2
    ax.axvline(x=1.0, color=ACCENT_RED, linestyle="--", linewidth=1.5, alpha=0.7, zorder=2)
    ax.text(1.0, y[-1] + 0.5, "1.0x", color=ACCENT_RED, fontsize=9, ha="center")

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=_tick_fontsize(n_gpus))
    ax.set_xlabel(S("relative_xlabel"), fontsize=13)
    ax.set_title(S("relative_subtitle"), fontsize=14, color=ACCENT_BLUE, pad=10)
    ax.grid(axis="x", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_v * 1.25)
    ax.invert_yaxis()
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_relative_performance.png")


# ── CNN vs Transformer Chart ──────────────────────────────────────────────────

def chart_cnn_vs_transformer(d: ComparisonData, out: Path):
    """Average normalised CNN vs Transformer throughput & perf/watt."""
    CNN_METRICS = [
        "train_vision_resnet50_bf16_img_per_sec",
        "train_vision_resnet101_bf16_img_per_sec",
        "infer_vision_resnet50_bf16_img_per_sec",
        "infer_vision_resnet101_bf16_img_per_sec",
        "detect_faster_rcnn_resnet50_bf16_img_per_sec",
        "detect_mask_rcnn_resnet50_bf16_img_per_sec",
    ]
    TF_METRICS = [
        "train_nlp_bert-base_bf16_samples_per_sec",
        "train_nlp_bert-large_bf16_samples_per_sec",
        "infer_nlp_bert-base_bf16_samples_per_sec",
        "infer_nlp_bert-large_bf16_samples_per_sec",
    ]
    CNN_POWER = [
        "hw_power_training_vision_avg_w",
        "hw_power_inference_vision_avg_w",
        "hw_power_training_detection_avg_w",
    ]
    TF_POWER = [
        "hw_power_training_nlp_avg_w",
        "hw_power_inference_nlp_avg_w",
    ]

    def _geomean_normalised(metric_keys: list[str]) -> list[float | None]:
        """Return per-GPU geometric mean of min-normalised metric values."""
        n_gpus = d.n_gpus
        log_sums = [0.0] * n_gpus
        counts = [0] * n_gpus
        for mk in metric_keys:
            vals = d.get(mk)
            valid = [v for v in vals if v is not None and v > 0]
            if not valid:
                continue
            baseline = min(valid)
            for gi, v in enumerate(vals):
                if v is not None and v > 0 and baseline > 0:
                    log_sums[gi] += log(v / baseline)
                    counts[gi] += 1
        result: list[float | None] = []
        for gi in range(n_gpus):
            if counts[gi] > 0:
                result.append(exp(log_sums[gi] / counts[gi]))
            else:
                result.append(None)
        # Re-normalise so the minimum score is exactly 1.0×
        valid_results = [v for v in result if v is not None and v > 0]
        if valid_results:
            base = min(valid_results)
            result = [round(v / base, 3) if v is not None else None for v in result]
        return result

    def _avg_power(power_keys: list[str]) -> list[float | None]:
        """Average power draw across benchmarks for each GPU."""
        n_gpus = d.n_gpus
        sums = [0.0] * n_gpus
        counts = [0] * n_gpus
        for pk in power_keys:
            vals = d.get(pk)
            for gi, v in enumerate(vals):
                if v is not None and v > 0:
                    sums[gi] += v
                    counts[gi] += 1
        return [round(sums[gi] / counts[gi], 1) if counts[gi] > 0 else None
                for gi in range(n_gpus)]

    cnn_score = _geomean_normalised(CNN_METRICS)
    tf_score = _geomean_normalised(TF_METRICS)

    # Need at least one valid score in each family
    if not any(v is not None for v in cnn_score) and \
       not any(v is not None for v in tf_score):
        return

    cnn_power = _avg_power(CNN_POWER)
    tf_power = _avg_power(TF_POWER)

    # Compute perf/watt (score / watts, then normalise to min)
    def _ppw(scores, powers):
        raw = []
        for s, p in zip(scores, powers):
            if s is not None and p is not None and p > 0:
                raw.append(s / p)
            else:
                raw.append(None)
        valid = [v for v in raw if v is not None and v > 0]
        if not valid:
            return [None] * len(raw)
        baseline = min(valid)
        return [round(v / baseline, 3) if v is not None else None for v in raw]

    cnn_ppw = _ppw(cnn_score, cnn_power)
    tf_ppw = _ppw(tf_score, tf_power)

    # Build categories
    cats = []
    vals_per_gpu: list[list[float | None]] = []  # [gpu_i][cat_j]
    n_gpus = d.n_gpus

    cat_data = [
        (S("cnn_vs_tf_cnn"),            cnn_score),
        (S("cnn_vs_tf_cnn_ppw"),         cnn_ppw),
        (S("cnn_vs_tf_transformer"),     tf_score),
        (S("cnn_vs_tf_transformer_ppw"), tf_ppw),
    ]

    for label, data in cat_data:
        if any(v is not None for v in data):
            cats.append(label)
            for gi in range(n_gpus):
                if gi >= len(vals_per_gpu):
                    vals_per_gpu.append([])
                vals_per_gpu[gi].append(data[gi] if data[gi] is not None else 0)

    if not cats:
        return

    gpu_names = d.gpu_names()
    fig, ax = _standard_fig(S("cnn_vs_tf_title"), S("higher_better"),
                            ACCENT_GREEN, n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.18))

    y = np.arange(len(cats))
    total_height = min(0.85, 0.15 * n_gpus + 0.25)
    bar_h = total_height / n_gpus
    vfs = _val_fontsize(n_gpus)
    gfs = _gpu_name_fontsize(n_gpus)

    all_v = []
    for gi, gname in enumerate(gpu_names):
        gvals = vals_per_gpu[gi]
        all_v.extend(gvals)
    max_v_pre = max(all_v) if all_v else 2

    for gi, gname in enumerate(gpu_names):
        offset = (gi - (n_gpus - 1) / 2) * bar_h
        color = GPU_COLORS[gi % len(GPU_COLORS)]
        gvals = vals_per_gpu[gi]
        ax.barh([], [], color=color, label=gname)  # legend
        for ci, v in enumerate(gvals):
            if v > 0:
                cy = y[ci] + offset
                _gradient_barh(ax, cy, v, bar_h * 0.88, color, zorder=3)
                txt = ax.text(v + max_v_pre * 0.01, cy,
                              f"{v:.2f}x", va="center",
                              fontsize=vfs, color=TEXT_COLOR, fontweight="bold")
                txt.set_path_effects([
                    path_effects.withStroke(linewidth=2, foreground=BG_COLOR, alpha=0.6),
                ])
                ax.text(max_v_pre * 0.01, cy, gname,
                        ha="left", va="center",
                        fontsize=gfs, color=BG_COLOR, fontweight="bold",
                        alpha=0.85, clip_on=True)

    max_v = max_v_pre
    ax.axvline(x=1.0, color=ACCENT_RED, linestyle="--", linewidth=1.5, alpha=0.7, zorder=2)
    ax.text(1.0, y[-1] + 0.5, "1.0x", color=ACCENT_RED, fontsize=9, ha="center")

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=_tick_fontsize(n_gpus))
    ax.set_xlabel(S("cnn_vs_tf_ylabel"), fontsize=13)
    ax.set_title(S("cnn_vs_tf_subtitle"), fontsize=14, color=ACCENT_BLUE, pad=10)
    ax.grid(axis="x", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_v * 1.25)
    ax.invert_yaxis()
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_cnn_vs_transformer.png")


# ── Enhanced Dual-GPU Charts (per model) ─────────────────────────────────────

def chart_dual_gpu(d: ComparisonData, out: Path):
    """Single combined horizontal chart for multi-GPU scaling.

    Training panels: 1-GPU vs 2-GPU DDP vs 2-GPU FSDP.
    LLM inference panels: 1-GPU vs 2-GPU tensor parallelism.
    Only includes GPUs that actually ran benchmark 10 (multi-GPU scaling).
    """
    # ── Training models ───────────────────────────────────────────────────
    training_model_specs = [
        ("resnet50",   "CNN (ResNet-50)"),
        ("bert_base",  "Transformer (BERT-Base)"),
        ("gpt2_large", "Transformer (GPT-2 Large)"),
    ]
    training_method_specs = [
        ("single", "1gpu", S("multigpu_1gpu"),  0.7),
        ("ddp",    "2gpu", S("multigpu_ddp"),    1.0),
        ("fsdp",   "2gpu", S("multigpu_fsdp"),   1.3),
    ]

    # ── LLM inference models (auto-discovered) ───────────────────────────
    llm_method_specs = [
        ("single",          "1gpu", S("multigpu_1gpu"),     0.7),
        ("tensor_parallel", "2gpu", S("multigpu_tp"),       1.0),
    ]

    # Discover LLM inference models from metric keys
    llm_model_specs: list[tuple[str, str]] = []
    for mid in d.metric_ids():
        if mid.startswith("multigpu_llm_") and mid.endswith("_single_1gpu_samples_per_sec"):
            mk = mid.replace("multigpu_", "").replace("_single_1gpu_samples_per_sec", "")
            nice = mk.replace("llm_", "").replace("_", " ").title()
            label = f"LLM ({nice})"
            llm_model_specs.append((mk, label))

    # Determine which GPUs actually have multi-GPU data
    all_gpu_names = d.gpu_names()
    has_multigpu = []
    for gi in range(d.n_gpus):
        has_any = False
        for mk, ml in training_model_specs:
            key = f"multigpu_{mk}_single_1gpu_samples_per_sec"
            v = d.get(key)[gi]
            if v is not None:
                has_any = True
                break
        if not has_any:
            for mk, ml in llm_model_specs:
                key = f"multigpu_{mk}_single_1gpu_samples_per_sec"
                v = d.get(key)[gi]
                if v is not None:
                    has_any = True
                    break
        has_multigpu.append(has_any)

    active_indices = [gi for gi, has in enumerate(has_multigpu) if has]
    if not active_indices:
        return

    n_active = len(active_indices)
    gpu_names = [all_gpu_names[gi] for gi in active_indices]

    # Build list of (model_key, model_label, method_specs, unit_label) panels
    panels: list[tuple[str, str, list, str]] = []

    # Training panels
    for model_key, model_label in training_model_specs:
        key_single = f"multigpu_{model_key}_single_1gpu_samples_per_sec"
        v_single = [d.get(key_single)[gi] for gi in active_indices]
        if any(v is not None for v in v_single):
            panels.append((model_key, model_label, training_method_specs,
                           S("multigpu_ylabel")))

    # LLM inference panels
    for model_key, model_label in llm_model_specs:
        key_single = f"multigpu_{model_key}_single_1gpu_samples_per_sec"
        v_single = [d.get(key_single)[gi] for gi in active_indices]
        if any(v is not None for v in v_single):
            panels.append((model_key, model_label, llm_method_specs,
                           S("multigpu_llm_ylabel")))

    if not panels:
        return

    n_panels = len(panels)
    max_methods = max(len(ms) for _, _, ms, _ in panels)
    fig_w = max(BASE_FIG_W, 6 * n_panels)
    fig_h = max(BASE_FIG_H, 1.2 * n_active * max_methods + 3)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(S("multigpu_title"), fontsize=24, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)
    fig.text(0.97, 0.935, S("higher_better"), ha="right", va="top",
             fontsize=10, color=ACCENT_GREEN, alpha=0.9)
    _watermark(fig)

    vfs = _val_fontsize(n_active * max_methods)
    gfs = _gpu_name_fontsize(n_active)

    for pi, (model_key, model_label, p_methods, unit_label) in enumerate(panels):
        ax = axes[pi]
        n_methods = len(p_methods)

        # Collect values
        cats = [loc_label for _, _, loc_label, _ in p_methods]
        y = np.arange(n_methods)
        total_height = min(0.85, 0.15 * n_active + 0.25)
        bar_h = total_height / n_active

        all_vals: list[float] = []
        for gi_local, gi_global in enumerate(active_indices):
            offset = (gi_local - (n_active - 1) / 2) * bar_h
            base_rgb = mcolors.to_rgb(GPU_COLORS[gi_global % len(GPU_COLORS)])

            vals: list[float] = []
            effs: list[str] = []
            for method, n_tag, _, shade in p_methods:
                key_t = f"multigpu_{model_key}_{method}_{n_tag}_samples_per_sec"
                key_e = f"multigpu_{model_key}_{method}_{n_tag}_eff_pct"
                v = d.get(key_t)[gi_global]
                e = d.get(key_e)[gi_global]
                vals.append(v if v is not None else 0)
                if e is not None and method != "single":
                    effs.append(f"{e:.0f}%")
                else:
                    effs.append("")

            all_vals.extend(vals)
            bar_color = GPU_COLORS[gi_global % len(GPU_COLORS)]

            # Invisible bar for legend
            if pi == 0:
                ax.barh([], [], color=bar_color, label=gpu_names[gi_local])

            for ci, (v, eff) in enumerate(zip(vals, effs)):
                if v > 0:
                    cy = y[ci] + offset
                    _gradient_barh(ax, cy, v, bar_h * 0.88, bar_color, zorder=3)
                    ann = f"{v:.0f}"
                    if eff:
                        ann += f" ({eff})"
                    ax.text(v + max(all_vals) * 0.01 if all_vals else v * 0.01,
                            cy,
                            ann, va="center", fontsize=vfs,
                            color=TEXT_COLOR, fontweight="bold")
                    # Use "2xGPUNAME" for multi-GPU bars
                    method_tag = p_methods[ci][0]
                    bar_label = (f"2x{gpu_names[gi_local]}"
                                 if method_tag != "single"
                                 else gpu_names[gi_local])
                    pad = max(all_vals) * 0.01 if all_vals else 0
                    ax.text(pad, cy,
                            bar_label, ha="left", va="center",
                            fontsize=gfs, color=BG_COLOR, fontweight="bold",
                            alpha=0.85, clip_on=True)

        ax.set_yticks(y)
        ax.set_yticklabels(cats, fontsize=_tick_fontsize(n_active))
        ax.set_xlabel(unit_label, fontsize=11)
        ax.set_title(model_label, fontsize=14, color=ACCENT_BLUE, pad=10)
        ax.grid(axis="x", linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        max_v = max(all_vals) if all_vals else 1
        ax.set_xlim(0, max_v * 1.35)
        ax.invert_yaxis()

    fig.text(0.5, 0.92, S("multigpu_subtitle"), ha="center",
             fontsize=13, color=SUBTEXT_COLOR)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    _save(fig, out / "cmp_dual_gpu.png")


# ── VRAM Limits ───────────────────────────────────────────────────────────────

def chart_vram_limits(d: ComparisonData, out: Path):
    """Horizontal grouped bar chart comparing VRAM capacity across GPUs.

    Three bars per GPU:
      1. Largest loadable model (billion params)
      2. Actual VRAM used for that model (GB)
      3. Max context length (shown in K-tokens for readability)
    """
    specs = [
        ("vram_largest_loadable_params_b",  S("vram_cat_params"),    "{:.1f}", 1.0),
        ("vram_largest_loadable_actual_gb", S("vram_cat_actual_gb"), "{:.1f}", 1.0),
        ("vram_max_context_length",         S("vram_cat_context"),   "{:.1f}", 1 / 1024),
    ]

    cats, metric_keys, fmts = [], [], []
    scale_factors: list[float] = []
    for mid, label, fmt, sf in specs:
        raw = d.get(mid)
        if any(v is not None for v in raw):
            cats.append(label)
            metric_keys.append(mid)
            fmts.append(fmt)
            scale_factors.append(sf)
    if not cats:
        return

    # Build values matrix — apply scale factor (context → K-tokens)
    vals: list[list[float]] = []
    for gi in range(d.n_gpus):
        row = []
        for mid, sf in zip(metric_keys, scale_factors):
            v = d.get(mid)[gi]
            row.append(v * sf if v is not None else None)
        vals.append(row)

    n_gpus = d.n_gpus
    fig, ax = _standard_fig(S("vram_title"), S("higher_better"),
                            ACCENT_PURPLE, n_gpus, len(cats),
                            height_ratio=max(1.0, len(cats) * 0.22))

    y = np.arange(len(cats))
    total_height = min(0.85, 0.15 * n_gpus + 0.25)
    bar_h = total_height / n_gpus
    vfs = _val_fontsize(n_gpus)
    gfs = _gpu_name_fontsize(n_gpus)

    all_vals: list[float] = []
    for gi, gname in enumerate(d.gpu_names()):
        offset = (gi - (n_gpus - 1) / 2) * bar_h
        color = GPU_COLORS[gi % len(GPU_COLORS)]
        gvals = [v if v is not None else 0 for v in vals[gi]]
        all_vals.extend(gvals)
        bars = ax.barh(y + offset, gvals, bar_h * 0.88, color=color,
                       edgecolor=BG_COLOR, linewidth=1, zorder=3, label=gname)
        for bar, v, fmt in zip(bars, gvals, fmts):
            if v > 0:
                ax.text(v + max(all_vals) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        fmt.format(v), va="center",
                        fontsize=vfs, color=TEXT_COLOR, fontweight="bold")
                ax.text(max(all_vals) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        gname, ha="left", va="center",
                        fontsize=gfs, color=BG_COLOR, fontweight="bold",
                        alpha=0.85, clip_on=True)

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=_tick_fontsize(n_gpus))
    ax.set_title(S("vram_subtitle"), fontsize=12, color=SUBTEXT_COLOR, pad=10)
    ax.grid(axis="x", linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    max_v = max(all_vals) if all_vals else 1
    ax.set_xlim(0, max_v * 1.25)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _save(fig, out / "cmp_vram_limits.png")


# ── Scorecard ─────────────────────────────────────────────────────────────────

def chart_scorecard(d: ComparisonData, out: Path):
    headline_metrics = []

    # VRAM
    vram_params = d.get("vram_largest_loadable_params_b")
    if any(v is not None for v in vram_params):
        vram_detail = d.get("vram_max_model_detail")
        detail = vram_detail if any(v is not None for v in vram_detail) else None
        headline_metrics.append(("vram_largest_loadable_params_b", S("score_vram_model"), S("unit_b_params"), "{:.0f}", detail))

    vram_ctx = d.get("vram_max_context_length")
    if any(v is not None for v in vram_ctx):
        headline_metrics.append(("vram_max_context_length", S("score_vram_ctx"), S("unit_tokens"), "{:,.0f}", None))

    # llmfit — Best Loadable LLM per VRAM tier
    gpu_vram_vals = d.get("gpu_vram_gb")
    llmfit_json_path = Path(__file__).resolve().parent.parent / "benchmarks" / "llmfit_vram_tiers.json"
    if any(v is not None for v in gpu_vram_vals) and llmfit_json_path.exists():
        try:
            with llmfit_json_path.open() as _f:
                llmfit_data = json.load(_f)
            tiers = llmfit_data.get("vram_tiers", {})
            tier_keys = sorted(tiers.keys(), key=lambda k: int(k))  # ["16","24","32","48"]

            llmfit_best_vals: list[str | None] = [None] * d.n_gpus
            llmfit_best_detail: list[str | None] = [None] * d.n_gpus
            llmfit_largest_vals: list[str | None] = [None] * d.n_gpus
            llmfit_largest_detail: list[str | None] = [None] * d.n_gpus
            llmfit_moe_vals: list[str | None] = [None] * d.n_gpus
            llmfit_moe_detail: list[str | None] = [None] * d.n_gpus

            for gi in range(d.n_gpus):
                vram = gpu_vram_vals[gi]
                if vram is None:
                    continue
                # Find the best matching tier (largest tier <= GPU VRAM)
                # Use 0.93× threshold to account for GiB vs GB difference
                # (e.g. a "24 GB" GPU reports ~23.5 GiB)
                matched_tier = None
                for tk in tier_keys:
                    if int(tk) * 0.93 <= vram:
                        matched_tier = tk
                # Fall back to smallest tier if GPU is below all tiers
                if matched_tier is None:
                    matched_tier = tier_keys[0] if tier_keys else None
                if matched_tier and matched_tier in tiers:
                    tier = tiers[matched_tier]
                    # Best recommended model (quality/speed/fit composite)
                    top = tier.get("top_models", [])
                    if top:
                        best = top[0]
                        llmfit_best_vals[gi] = best["name"]
                        q = best.get("best_quant", "Q4_K_M")
                        p = best.get("params_b", 0)
                        fit = best.get("fit", "")
                        llmfit_best_detail[gi] = f"{p}B {q} ({fit})"
                    # Largest dense model (most aggressive quantization)
                    largest = tier.get("largest_dense_model")
                    if largest:
                        llmfit_largest_vals[gi] = str(largest)
                    # Largest quantized model detail (first entry in largest_quantized)
                    lq = tier.get("largest_quantized", [])
                    if lq:
                        lq0 = lq[0]
                        note = lq0.get("note", "")
                        llmfit_largest_detail[gi] = note if note else None
                    # Largest MoE model
                    largest_moe = tier.get("largest_moe_model")
                    if largest_moe:
                        llmfit_moe_vals[gi] = str(largest_moe)

            if any(v is not None for v in llmfit_best_vals):
                headline_metrics.append(("_llmfit_best", S("score_llmfit_best"), "", "{}", llmfit_best_detail))
            if any(v is not None for v in llmfit_largest_vals):
                headline_metrics.append(("_llmfit_largest", S("score_llmfit_largest"), "", "{}", llmfit_largest_detail))
            if any(v is not None for v in llmfit_moe_vals):
                headline_metrics.append(("_llmfit_moe", S("score_llmfit_moe"), "", "{}", None))
        except Exception:
            pass  # Graceful fallback: skip llmfit rows if JSON is invalid


    # Power & temp
    hw_power = d.get("hw_avg_power_w")
    if any(v is not None for v in hw_power):
        headline_metrics.append(("hw_avg_power_w", S("score_power"), S("unit_watt"), "{:.0f}", None))
    hw_temp = d.get("hw_avg_temp_c")
    if any(v is not None for v in hw_temp):
        headline_metrics.append(("hw_avg_temp_c", S("score_temp"), S("unit_celsius"), "{:.0f}", None))

    # DLPerf (CNN) — based on ResNet-50 FP32
    dlperf_cnn_vals = []
    fp32_train = d.get("train_vision_resnet50_fp32_img_per_sec")
    for v in fp32_train:
        dlperf_cnn_vals.append(round(v / 13.0, 1) if v else None)

    # DLPerf (Transformer) — based on BERT-Base FP32
    dlperf_transformer_vals = []
    fp32_bert = d.get("train_nlp_bert-base_fp32_samples_per_sec")
    for v in fp32_bert:
        dlperf_transformer_vals.append(round(v / 13.0, 1) if v else None)

    gpu_names = d.gpu_names()
    n_gpus = d.n_gpus

    rows = []
    for mid, label, unit, fmt, detail in headline_metrics:
        if mid.startswith("_llmfit_"):
            # Synthetic llmfit metrics — string values
            if mid == "_llmfit_best":
                vals = llmfit_best_vals
            elif mid == "_llmfit_largest":
                vals = llmfit_largest_vals
            elif mid == "_llmfit_moe":
                vals = llmfit_moe_vals
            else:
                continue
            if any(v is not None for v in vals):
                rows.append((label, unit, fmt, vals, detail))
        else:
            vals = d.get(mid)
            if any(v is not None for v in vals):
                rows.append((label, unit, fmt, vals, detail))
    if dlperf_cnn_vals and any(v is not None for v in dlperf_cnn_vals):
        rows.append((S("score_dlperf_cnn"), "", "{:.1f}", dlperf_cnn_vals, None))
    if dlperf_transformer_vals and any(v is not None for v in dlperf_transformer_vals):
        rows.append((S("score_dlperf_transformer"), "", "{:.1f}", dlperf_transformer_vals, None))

    if not rows:
        return

    n_rows = len(rows)
    n_cols = n_gpus + 1

    fig_w = max(BASE_FIG_W, 4.5 * n_gpus + 3)
    fig_h = max(BASE_FIG_H, 1.1 * n_rows + 2)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(S('score_title'), fontsize=22, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)

    for ri, (label, unit, fmt, vals, detail) in enumerate(rows):
        ax = fig.add_subplot(n_rows, n_cols, ri * n_cols + 1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.95, 0.5, label, ha="right", va="center",
                fontsize=12, color=SUBTEXT_COLOR, fontweight="bold")

        for gi in range(n_gpus):
            ax = fig.add_subplot(n_rows, n_cols, ri * n_cols + 2 + gi)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

            v = vals[gi]
            color = GPU_COLORS[gi % len(GPU_COLORS)]

            rect = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=CARD_COLOR, edgecolor=GRID_COLOR, linewidth=1)
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
                # Auto-shrink font for long text (e.g. MoE LLM model names)
                val_fs = 15
                if len(val_str) > 20:
                    val_fs = 10
                elif len(val_str) > 14:
                    val_fs = 12
                ax.text(0.5, val_y, val_str, ha="center", va="center",
                        fontsize=val_fs, fontweight="bold", color=color)
                if has_detail:
                    ax.text(0.5, 0.32, str(detail[gi]), ha="center", va="center",
                            fontsize=9, color=SUBTEXT_COLOR, style="italic")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=13, color=SUBTEXT_COLOR)

            if ri == 0:
                ax.text(0.5, 1.05, gpu_names[gi], ha="center", va="bottom",
                        fontsize=12, fontweight="bold",
                        color=GPU_COLORS[gi % len(GPU_COLORS)])

    # Disclaimer for LLM fit recommendations
    fig.text(0.5, 0.01, S("score_llmfit_disclaimer"),
             ha="center", va="bottom", fontsize=8,
             color=SUBTEXT_COLOR, style="italic", alpha=0.8)

    _watermark(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    _save(fig, out / "cmp_scorecard.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_sessions(results_dir: Path) -> list[Path]:
    return sorted(
        p for p in results_dir.iterdir()
        if p.is_dir() and not p.is_symlink() and (p / "session_meta.json").exists()
    )


def _run_extraction(results_dir: Path, session_dirs: list[Path],
                    output_dir: Path) -> Path:
    """Extract metrics and write comparison.csv/json. Returns path to JSON."""
    sessions: list[tuple[str, str, Metricdict]] = []
    for d in session_dirs:
        try:
            col_id, col_label, metrics = _load_session(d)
            existing_ids = [s[0] for s in sessions]
            if col_id in existing_ids:
                short = d.name[-8:]
                col_id += f"_{short}"
                col_label += f" ({short})"
            sessions.append((col_id, col_label, metrics))
            print(f"  Loaded: {col_label} \u2014 {len(metrics)} metrics")
        except Exception as e:
            print(f"  WARNING: skipping {d.name} \u2014 {e}", file=sys.stderr)

    if not sessions:
        print("ERROR: no sessions could be loaded", file=sys.stderr)
        sys.exit(1)

    _, col_ids, rows = build_comparison_table(sessions)

    csv_path = output_dir / "comparison.csv"
    json_path = output_dir / "comparison.json"
    write_csv(rows, col_ids, csv_path)
    write_json(rows, col_ids, sessions, json_path)

    print(f"\n  CSV  \u2192 {csv_path}")
    print(f"  JSON \u2192 {json_path}")
    print(f"  Metrics: {len(rows)}  |  Sessions: {len(sessions)}")
    return json_path


def _run_charts(json_path: Path, output_dir: Path):
    """Generate all comparison charts from comparison.json."""
    d = ComparisonData(json_path)
    chart_dir = output_dir / "comparison_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Charts \u2192 {chart_dir}")
    print(f"  GPUs   : {d.n_gpus} \u2014 {', '.join(d.gpu_names())}")
    print(f"  Lang   : {_LANG}")

    chart_fns = [
        chart_training_vision,
        chart_training_nlp,
        chart_inference_vision,
        chart_inference_nlp,
        chart_llm,
        chart_power_efficiency,
        chart_cnn_vs_transformer,
        chart_dual_gpu,
        chart_scorecard,
    ]

    count = 0
    for fn in chart_fns:
        try:
            fn(d, chart_dir)
            count += 1
        except Exception as e:
            print(f"  \u26a0 {fn.__name__}: {e}", file=sys.stderr)

    print(f"\n  Generated {count} chart set(s)")


def main():
    global _LANG

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
    parser.add_argument("--lang", choices=["tr", "en"], default="en",
                        help="Chart language: tr (T\u00fcrk\u00e7e) or en (English). Default: en")
    args = parser.parse_args()

    _LANG = args.lang

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
