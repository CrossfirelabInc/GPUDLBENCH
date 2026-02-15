# GPU Deep Learning Benchmark Suite

Benchmark suite for NVIDIA GPUs. Measures training throughput, inference latency, LLM token generation, VRAM limits, and mixed precision speedups. Works on any NVIDIA GPU with CUDA support.

Everything runs on PyTorch. No Docker, no TensorFlow, no weird dependencies.


## Benchmarks

1. **Vision training** -- ResNet-50/101, FP32/FP16/BF16, measures images/sec
2. **NLP training** -- BERT-base/large, FP32/FP16/BF16, measures samples/sec
3. **Vision inference** -- ResNet-50/101, latency and throughput
4. **NLP inference** -- BERT-base/large, latency and throughput
5. **LLM tokens/sec** -- 8 GGUF models (8B to 70B) via llama.cpp, tokens/sec and time-to-first-token
6. **VRAM limits** -- How large a model can you load? How long a context? How many models at once?
7. **Mixed precision** -- FP32 vs FP16 vs BF16 speedup comparison
8. **Detection training** -- Faster R-CNN with ResNet-50 FPN backbone
9. **Quick smoke tests** -- Matrix multiply TFLOPS, memory bandwidth, kernel launch latency, conv and attention throughput. Runs in under 2 minutes.


## Requirements

- Linux (tested on Ubuntu 22.04, should work on most distros)
- NVIDIA GPU with CUDA 11.8 or newer
- Python 3.9+
- 16 GB RAM minimum, 32+ recommended
- About 50 GB disk if you want to run the LLM benchmarks (model downloads)
- cmake and git (for building llama.cpp)


## Setup

```bash
git clone https://github.com/<your-username>/gpu-dl-benchmark.git
cd gpu-dl-benchmark

# This creates a venv, installs PyTorch with the right CUDA version,
# installs dependencies, and builds llama.cpp
python install.py

# Activate the virtual environment
source venv/bin/activate
```

If you don't need llama.cpp (skipping benchmark 5):

```bash
python install.py --skip-llama
```


## Running

Run everything:

```bash
python run_benchmarks.py
```

Skip specific benchmarks:

```bash
python run_benchmarks.py --skip 5 6    # skip LLM and VRAM tests
```

Run a single benchmark:

```bash
python benchmarks/1_training_vision.py
python benchmarks/9_quick_benchmarks.py
```

All benchmarks accept `--help` for options like `--precisions`, `--device`, `--output-dir`, `--seed`.


## Results

Everything goes into `results/`. After benchmarks finish, a report is generated automatically:

- `benchmark_summary.md` -- human-readable summary
- `benchmark_summary.csv` -- for spreadsheets
- `benchmark_summary.json` -- all data combined

Each benchmark also writes its own JSON file (e.g. `training_vision.json`).

You can regenerate the report anytime:

```bash
python utils/generate_report.py
```


## Project layout

```
benchmarks/
    config.py              -- all tunable parameters (batch sizes, models, etc.)
    benchmark_utils.py     -- shared code (GPU detection, AMP helpers, monitoring)
    1_training_vision.py
    2_training_nlp.py
    3_inference_vision.py
    4_inference_nlp.py
    5_llm_tokens_per_sec.py
    6_vram_limits.py
    7_mixed_precision.py
    8_training_detection.py
    9_quick_benchmarks.py
utils/
    generate_report.py     -- reads all result JSONs, writes summary
install.py                 -- sets up venv, PyTorch, dependencies, llama.cpp
run_benchmarks.py          -- runs all 9 benchmarks with pass/fail tracking
requirements.txt
LICENSE
```


## How it works

Training benchmarks use proper automatic mixed precision (`torch.amp.autocast` + `GradScaler`), not raw `.half()` calls. BF16 is automatically skipped on GPUs that don't support it (pre-Ampere). Batch sizes are adjusted based on available VRAM.

LLM benchmark downloads GGUF models from Hugging Face and runs them through llama.cpp with full GPU offloading. If you already have llama.cpp installed somewhere, set `LLAMA_CPP_PATH` to point at the binary.

VRAM limits test builds real transformer models with accurate parameter counts (not fake numbers) and loads progressively larger ones until the GPU runs out of memory.

A background monitoring thread polls nvidia-smi during each benchmark to record power draw, temperature, and clock speed.


## Troubleshooting

**Out of memory** -- Edit `benchmarks/config.py` and reduce the batch size lists. The suite tries to auto-adjust but some GPU + model combinations are tight.

**llama.cpp not found** -- Set `export LLAMA_CPP_PATH=/path/to/llama-cli` or re-run `python install.py`.

**Slow model downloads** -- Try `export HF_ENDPOINT=https://hf-mirror.com` before running benchmark 5.

**BF16 tests skipped** -- Normal on GTX 10xx/RTX 20xx GPUs. BF16 requires Ampere or newer.

**Can't detect GPU** -- Check that `nvidia-smi` works and `python -c "import torch; print(torch.cuda.is_available())"` returns True.


## License

MIT
