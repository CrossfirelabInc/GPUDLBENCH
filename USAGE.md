# Usage Reference

## install.py

```bash
python3 install.py
python3 install.py --skip-llama          # skip llama.cpp build
python3 install.py --skip-models         # skip model downloads
python3 install.py --model-set popular   # smaller model set (~38 GB vs ~57 GB)
python3 install.py --hf-token TOKEN      # pass HF token directly
python3 install.py --venv DIR            # custom venv directory
```

## run_benchmarks.py

```bash
python run_benchmarks.py
python run_benchmarks.py --skip 5 6 10
python run_benchmarks.py --model-set popular
```

## Individual benchmarks

```bash
python benchmarks/1_training_vision.py
python benchmarks/1_training_vision.py --device 0 --precisions fp32 fp16 bf16
python benchmarks/1_training_vision.py --no-monitor --seed 42 --output-dir /tmp
```

Benchmark 5 also accepts `--model-set`, `--models-dir`, `--llama-path`.

## Utilities

```bash
python utils/generate_report.py             # per-session markdown report
python utils/generate_comparison.py          # extract metrics + comparison charts
python utils/generate_comparison.py --skip-charts  # metrics only (no PNGs)
```

## .credentials

```
HF_TOKEN=hf_yourTokenHere
```

Get a token at https://huggingface.co/settings/tokens

Resolution order: `--hf-token` flag > `.credentials` file > `HF_TOKEN` env var > cached login.

## LLM model sets

- **default** (~57 GB): DeepSeek-R1-8B, Phi-4, Gemma-2-27B, QwQ-32B
- **popular** (~38 GB): Llama-3.1-8B, Mistral-7B, Qwen2.5-14B, Qwen2.5-32B

Use `--model-set popular` in both `install.py` and `run_benchmarks.py`.

## Environment variables

- `HF_TOKEN` — HuggingFace token (alternative to `.credentials`)
- `LLAMA_CPP_PATH` — path to llama-completion binary
- `CUDA_VISIBLE_DEVICES` — restrict visible GPUs
