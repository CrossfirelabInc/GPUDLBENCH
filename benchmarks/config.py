#!/usr/bin/env python3
"""Central configuration for GPU AI Benchmark Suite."""

import os
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
# ── Demo Mode ───────────────────────────────────────────────────────────────────────
# When --demo is used: only minimal batch size, fewer iterations, skip
# heavy benchmarks (6, 9, 10).  Useful for quick integration testing.
DEMO_TRAIN_WARMUP: int = 3
DEMO_TRAIN_ITERATIONS: int = 20
DEMO_INFER_WARMUP: int = 5
DEMO_INFER_ITERATIONS: int = 50
DEMO_GEMM_SIZES: list[int] = [1024, 4096]
DEMO_GEMM_WARMUP: int = 2
DEMO_GEMM_REPEATS: int = 5
DEMO_LLM_NUM_TOKENS: int = 128
DEMO_DETECTION_WARMUP: int = 2
DEMO_DETECTION_ITERATIONS: int = 15
# ── General ───────────────────────────────────────────────────────────────────
OUTPUT_DIR: str = "results"

# ── Data directories ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = DATA_DIR / "models"
HF_CACHE_DIR: Path = DATA_DIR / "hf_cache"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

# ── Precision modes ───────────────────────────────────────────────────────────
# FP8 is excluded here — PyTorch AMP has no native FP8 autocast, so
# training/inference benchmarks would just re-run FP16.  True FP8 is
# benchmarked in GEMM stress (benchmark 7) via torch._scaled_mm.
TRAINING_PRECISIONS: list[str] = ["fp32", "fp16", "bf16"]
INFERENCE_PRECISIONS: list[str] = ["fp32", "fp16", "bf16"]

# ── Vision (Benchmarks 1, 3) ─────────────────────────────────────────────────
VISION_MODELS: list[str] = ["resnet50", "resnet101"]
VISION_IMAGE_SIZE: int = 224
VISION_NUM_CLASSES: int = 1000
VISION_TRAIN_WARMUP: int = 10
VISION_TRAIN_ITERATIONS: int = 100
VISION_INFER_WARMUP: int = 20
VISION_INFER_ITERATIONS: int = 200

# ── NLP (Benchmarks 2, 4) ────────────────────────────────────────────────────
NLP_MODELS: dict[str, dict] = {
    "bert-base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
    "bert-large": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
    },
}
NLP_SEQ_LENGTH: int = 128
NLP_VOCAB_SIZE: int = 30522
NLP_NUM_LABELS: int = 2
NLP_TRAIN_WARMUP: int = 10
NLP_TRAIN_ITERATIONS: int = 50
NLP_INFER_WARMUP: int = 20
NLP_INFER_ITERATIONS: int = 200

# ── LLM Token Generation (Benchmark 5) ───────────────────────────────────────
LLM_PROMPT: str = (
    "Explain the concept of machine learning, including supervised learning, "
    "unsupervised learning, and deep learning. Provide examples of each and "
    "discuss their real-world applications."
)
LLM_NUM_TOKENS: int = 512
LLM_TIMEOUT_SECONDS: int = 600

LLM_MODEL_SETS: dict[str, list] = {
    "default": [
        {"name": "DeepSeek-R1-Distill-Llama-8B", "repo_id": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
         "filename": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf", "size_gb": 4.5, "quant": "Q4_K_M"},
        {"name": "Phi-4", "repo_id": "microsoft/phi-4-gguf",
         "filename": "phi-4-Q8_0.gguf", "size_gb": 15, "quant": "Q8_0"},
        {"name": "Gemma-2-27B", "repo_id": "bartowski/gemma-2-27b-it-GGUF",
         "filename": "gemma-2-27b-it-Q4_K_M.gguf", "size_gb": 17, "quant": "Q4_K_M"},
        {"name": "QwQ-32B", "repo_id": "bartowski/Qwen_QwQ-32B-GGUF",
         "filename": "Qwen_QwQ-32B-Q4_K_M.gguf", "size_gb": 20, "quant": "Q4_K_M"},
    ],
    "popular": [
        {"name": "Llama-3.1-8B", "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
         "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "size_gb": 4.9, "quant": "Q4_K_M"},
        {"name": "Mistral-7B-v0.3", "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
         "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "size_gb": 4.4, "quant": "Q4_K_M"},
        {"name": "Qwen2.5-14B", "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
         "filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf", "size_gb": 9.0, "quant": "Q4_K_M"},
        {"name": "Qwen2.5-32B", "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
         "filename": "Qwen2.5-32B-Instruct-Q4_K_M.gguf", "size_gb": 19.9, "quant": "Q4_K_M"},
    ],
}


def get_llm_models(model_set: str = "default") -> list:
    """Return the model list for the given set name."""
    if model_set not in LLM_MODEL_SETS:
        available = ", ".join(sorted(LLM_MODEL_SETS.keys()))
        raise ValueError(f"Unknown model set '{model_set}'. Available: {available}")
    return LLM_MODEL_SETS[model_set]

# ── VRAM Limits (Benchmark 6) ────────────────────────────────────────────────
VRAM_TEST_MODELS: list = [
    {"label": "1.5B", "config": {"num_hidden_layers": 24, "hidden_size": 1536, "num_attention_heads": 16, "intermediate_size": 6144, "vocab_size": 50257}, "approx_params_b": 1.5},
    {"label": "3B",   "config": {"num_hidden_layers": 32, "hidden_size": 2048, "num_attention_heads": 16, "intermediate_size": 8192, "vocab_size": 50257}, "approx_params_b": 3.0},
    {"label": "7B",   "config": {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32, "intermediate_size": 11008, "vocab_size": 32000}, "approx_params_b": 6.7},
    {"label": "13B",  "config": {"num_hidden_layers": 40, "hidden_size": 5120, "num_attention_heads": 40, "intermediate_size": 13824, "vocab_size": 32000}, "approx_params_b": 13.0},
    {"label": "20B",  "config": {"num_hidden_layers": 44, "hidden_size": 6144, "num_attention_heads": 48, "intermediate_size": 16384, "vocab_size": 50257}, "approx_params_b": 20.0},
    {"label": "30B",  "config": {"num_hidden_layers": 48, "hidden_size": 6656, "num_attention_heads": 52, "intermediate_size": 17920, "vocab_size": 32000}, "approx_params_b": 30.0},
    {"label": "70B",  "config": {"num_hidden_layers": 80, "hidden_size": 8192, "num_attention_heads": 64, "intermediate_size": 28672, "vocab_size": 32000}, "approx_params_b": 65.0},
]
VRAM_CONTEXT_LENGTHS: list[int] = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# ── Detection Training (Benchmark 8) ─────────────────────────────────────────
DETECTION_NUM_CLASSES: int = 91
DETECTION_IMAGE_SIZE: int = 800
DETECTION_WARMUP: int = 5
DETECTION_ITERATIONS: int = 50

# ── GEMM Compute Stress (Benchmark 7) ────────────────────────────────────────
GEMM_SIZES: list[int] = [1024, 2048, 4096, 8192]
GEMM_WARMUP: int = 5
GEMM_REPEATS: int = 15

# ── GPU Fundamentals (Benchmark 9) ───────────────────────────────────────────
FUND_BW_SIZES: list[int] = [
    16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024,
    1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024,
]
FUND_PCIE_SIZE: int = 256 * 1024 * 1024
FUND_FFT_SIZES: list[int] = [1024, 4096, 16384, 65536, 262144]
FUND_NBODY_N: int = 32768
FUND_NBODY_STEPS: int = 50
FUND_STENCIL_SIZE: int = 4096
FUND_STENCIL_STEPS: int = 100
FUND_WARMUP: int = 5
FUND_REPEATS: int = 20

# ── Multi-GPU Scaling (Benchmark 10) ─────────────────────────────────────────
MULTIGPU_VISION_BATCH_PER_GPU: int = 32
MULTIGPU_NLP_BATCH_PER_GPU: int = 16
MULTIGPU_LLM_BATCH_PER_GPU: int = 4
MULTIGPU_LLM_SEQ_LENGTH: int = 256
MULTIGPU_LLM_GEN_TOKENS: int = 64
MULTIGPU_WARMUP: int = 5
MULTIGPU_ITERATIONS: int = 50
MULTIGPU_GRAD_ACCUM_STEPS: int = 4   # gradient accumulation steps (improves PCIe scaling)

# GPT-2 Large architecture (nanoGPT-style)
# ~774M parameters — good compute-to-communication ratio for multi-GPU scaling.
MULTIGPU_LLM_CONFIG: dict = {
    "n_layer": 36,
    "n_head": 20,
    "n_embd": 1280,
    "vocab_size": 50257,
    "n_positions": 1024,
}

# ── Power / Thermal Monitoring ────────────────────────────────────────────────
MONITOR_INTERVAL_SEC: float = 0.5
