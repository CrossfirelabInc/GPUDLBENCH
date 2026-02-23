# GPU Deep Learning Benchmark Suite

**By Crossfirelab** — Vibe Coded 🎯

AI-focused benchmark suite for NVIDIA GPUs. Measures training throughput, inference speed, LLM token generation, VRAM capacity, and dual-GPU scaling in a single run. Built on PyTorch + llama.cpp — no Docker required.

---

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd GPUDLBENCH
python3 install.py
```

The installer automatically creates a virtual environment, installs the correct PyTorch for your GPU, builds llama.cpp, and downloads test models (~57 GB).

### 2. Activate the Environment

```bash
source venv/bin/activate
```

### 3. Run Benchmarks

```bash
python run_benchmarks.py
```

Runs all benchmarks sequentially (~2–4 hours). Results are saved to a timestamped folder under `results/`.

### 4. Generate Comparison Charts

After benchmarking multiple GPUs (in separate sessions), generate cross-GPU comparison charts:

```bash
python utils/generate_comparison.py
```

Charts are saved to `results/comparison_charts/`. English is the default language; use `--lang tr` for Turkish.

---

## What's Being Tested?

| # | Benchmark | What It Measures |
|---|-----------|-----------------|
| 1 | CNN Training (ResNet-50/101) | Training throughput (images/sec) |
| 2 | Transformer Training (BERT-Base/Large) | Training throughput (samples/sec) |
| 3 | CNN Inference (ResNet-50/101) | Inference latency & throughput |
| 4 | Transformer Inference (BERT-Base/Large) | NLP inference throughput |
| 5 | LLM Token Generation (llama.cpp) | Token generation speed (tokens/sec) |
| 6 | VRAM Capacity Test | Largest loadable model, max context length |
| 8 | Object Detection Training (Faster/Mask R-CNN) | Detection model training throughput |
| 10 | Dual-GPU Scaling (DDP + FSDP) | Scaling efficiency with 2 identical GPUs |

---

## Requirements

- **Linux** (Ubuntu 20.04+ recommended)
- **NVIDIA GPU** with CUDA 11.8+
- **Python 3.9+**
- **16 GB RAM** minimum
- **Internet connection** (for initial model downloads)

---

## Run Options

### Demo Mode

Quick ~5-minute smoke test to verify everything works:

```bash
python run_benchmarks.py --demo
```

### Skip Specific Benchmarks

```bash
python run_benchmarks.py --skip 5 6 10    # skip LLM, VRAM, dual-GPU
```

### Run a Single Benchmark

```bash
python benchmarks/1_training_vision.py
python benchmarks/5_llm_tokens_per_sec.py
```

### Install Options

```bash
python3 install.py --skip-llama        # skip llama.cpp build (benchmark 5 won't work)
python3 install.py --skip-models       # skip model downloads
python3 install.py --model-set popular # smaller model set (~38 GB)
```

---

## Results & Reports

Each run creates a timestamped folder under `results/`. The latest run is accessible via `results/latest`.

### Per-Session Summary Report

```bash
python utils/generate_report.py
```

### Cross-GPU Comparison Charts

```bash
python utils/generate_comparison.py              # English (default)
python utils/generate_comparison.py --lang tr     # Turkish
python utils/generate_comparison.py --skip-charts # metrics only, no PNGs
```

17 chart types are generated:

| Charts | Description |
|--------|-------------|
| CNN Training Throughput | ResNet-50/101 max & batch-normalized |
| Transformer Training Throughput | BERT-Base/Large max & batch-normalized |
| CNN Inference | ResNet-50/101 |
| Transformer Inference | BERT-Base/Large |
| LLM Performance | Tokens/sec + time-to-first-token |
| GEMM Peak TFLOPS | Raw compute (FP64/FP32/FP16/BF16/FP8) |
| GPU Fundamentals | Memory BW, PCIe, latency, FFT, N-body |
| Object Detection | Faster/Mask R-CNN max & batch-normalized |
| Power Efficiency | Throughput per watt |
| Relative Performance | GPU-to-GPU multiplier (weakest = 1.0×) |
| CNN vs Transformer | Architecture comparison |
| Dual-GPU Scaling | DDP/FSDP speedup |
| VRAM Capacity | Largest model, VRAM used, max context |
| Scorecard | Summary table with per-metric winners |

---

## Dual-GPU Setup

If your system has 2 identical GPUs, benchmark 10 (dual-GPU scaling) runs automatically. The runner auto-detects the display GPU and benchmarks on the other one.

If your GPUs are different, you'll be prompted to choose which one to benchmark and the dual-GPU test is skipped.

---

## Configuration

Batch sizes, iteration counts, model lists, and other tunables live in:

```
benchmarks/config.py
```

---

## Troubleshooting

### "no kernel image is available"

The installed PyTorch doesn't match your GPU architecture. Re-run the installer — it auto-detects your GPU and installs the correct build (including nightly for Blackwell GPUs):

```bash
python3 install.py
```

### Switched to a Different GPU

After physically swapping the GPU, re-run the installer. It will detect the new card, update PyTorch, and rebuild llama.cpp:

```bash
python3 install.py
```

---

## HuggingFace Token (Optional)

Some gated models require a HuggingFace token. If you need one, create a `.credentials` file in the project root:

```
HF_TOKEN=hf_yourTokenHere
```

Or pass it directly: `python3 install.py --hf-token hf_yourTokenHere`

Get a token at https://huggingface.co/settings/tokens (Read access is sufficient). The suite works without a token — gated model downloads will simply be skipped.

---

## License

MIT
