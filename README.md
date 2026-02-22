# GPU Deep Learning Benchmark Suite

Vibe Coded by Crossfirelab

Crossfirelab Benchmark suite for NVIDIA GPUs specially for Deep Learning. Measures training, inference, LLM generation, VRAM limits, compute stress, and multi-GPU scaling. PyTorch + llama.cpp. No Docker required.

Benchmarks:
1. Vision Training - ResNet-50/101 throughput (FP32/FP16/BF16/FP8)
2. NLP Training - BERT-base/large throughput (FP32/FP16/BF16/FP8)
3. Vision Inference - ResNet-50/101 latency + throughput
4. NLP Inference - BERT-base/large latency + throughput
5. LLM Tokens/sec - GGUF models via llama.cpp
6. VRAM Limits - Max model size + context length
7. GEMM Stress - Peak TFLOPS (FP64/FP32/FP16/BF16/FP8)
8. Detection Training - Faster/Mask R-CNN
9. GPU Fundamentals - Memory BW, PCIe, FFT, SpMM, attention
10. Multi-GPU Scaling - DDP/FSDP efficiency

Requirements: Linux, NVIDIA GPU with CUDA 11.8+, Python 3.9+, 16 GB RAM minimum.

## Setup

    git clone <repo-url> && cd GPUDLBENCH
    nano .credentials          # paste HF token from https://huggingface.co/settings/tokens
    python3 install.py
    source venv/bin/activate

The .credentials file holds your HuggingFace token (needed for gated model downloads):

    HF_TOKEN=hf_yourTokenHere

Install options:

    python3 install.py --skip-llama        # no llama.cpp (skips benchmark 5)
    python3 install.py --skip-models       # no model downloads
    python3 install.py --model-set popular # smaller model set (~38 GB vs ~57 GB)

## Running

    python run_benchmarks.py               # all benchmarks
    python run_benchmarks.py --skip 5 6 10 # skip specific ones
    python benchmarks/1_training_vision.py # single benchmark

## Results

Each run creates a timestamped folder in results/ with JSON files and a summary. A "latest" symlink points to the most recent run.

    python utils/generate_report.py        # per-session markdown report
    python utils/generate_comparison.py    # extract metrics + comparison charts

## Configuration

Edit benchmarks/config.py to change batch sizes, iteration counts, model lists, data paths, or precision modes.

## Troubleshooting

### `no kernel image is available for execution on the device`

This means your installed PyTorch does not include GPU kernels for your card's architecture.

**Blackwell GPUs (RTX 5090, RTX 5080, etc. — sm_120):** PyTorch 2.6.0+cu124 does NOT support Blackwell. Re-run the installer — it will auto-detect the architecture and install a compatible PyTorch:

    python3 install.py

If stable wheels are not yet available, the installer will try PyTorch nightly automatically.

### `_Float64x` / `_Float128` / `CMakeDetermineCUDACompiler` errors

Your host C++ compiler is too old for your glibc headers (common on Ubuntu 24.04+).

Quick fix:

    sudo apt-get update
    sudo apt-get install -y gcc-13 g++-13
    python3 install.py

If your distro `nvidia-cuda-toolkit` is outdated, install from NVIDIA's official CUDA repo and re-run `install.py`.

### GPU swap workflow

After physically swapping a GPU (or changing drivers), just re-run:

    python3 install.py

The installer detects GPU/driver/toolchain changes, picks the correct PyTorch version, rebuilds llama.cpp for the new architecture, and revalidates everything.

## License

MIT
