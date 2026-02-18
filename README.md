# GPU Deep Learning Benchmark Suite

Vide Coded by Crossfirelab

Crossfirelab Benchmark suite for NVIDIA GPUs. Measures training, inference, LLM generation, VRAM limits, compute stress, and multi-GPU scaling. PyTorch + llama.cpp. No Docker required.

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

## License

MIT
