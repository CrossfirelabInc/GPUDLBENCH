#!/usr/bin/env python3
"""
Central configuration for GPU AI Benchmark Suite.

All tunable parameters live here. Benchmark scripts import from this module
so users have a single place to adjust settings.
"""

import os
from pathlib import Path
from typing import Dict, List

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ─── General ──────────────────────────────────────────────────────────────────
OUTPUT_DIR: str = "results"

# ─── Data directory ───────────────────────────────────────────────────────────
#  All downloaded models (GGUF, HuggingFace cache) are stored here.
#  Change this if you want to keep data on a different disk / mount point.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = DATA_DIR / "models"          # GGUF models for llama.cpp
HF_CACHE_DIR: Path = DATA_DIR / "hf_cache"      # HuggingFace transformers cache

# Create data directories on import so every benchmark gets them for free
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Point HuggingFace cache to our data directory (set once, used everywhere)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

# ─── Precision modes ─────────────────────────────────────────────────────────
#  "fp32"  — standard single-precision
#  "fp16"  — AMP with float16 (all NVIDIA GPUs with Tensor Cores)
#  "bf16"  — AMP with bfloat16 (Ampere+ GPUs, compute capability >= 8.0)
#  "tf32"  — FP32 math but with TF32 Tensor Core acceleration (Ampere+)
#  "fp8"   — float8_e4m3fn (Ada Lovelace / Hopper / Blackwell, CC >= 8.9)
TRAINING_PRECISIONS: List[str] = ["fp32", "fp16", "bf16", "fp8"]
INFERENCE_PRECISIONS: List[str] = ["fp32", "fp16", "bf16", "fp8"]

# ─── Vision Training (Benchmark 1) ───────────────────────────────────────────
VISION_MODELS: Dict[str, str] = {
    "resnet50": "resnet50",
    "resnet101": "resnet101",
}
VISION_IMAGE_SIZE: int = 224
VISION_NUM_CLASSES: int = 1000
VISION_TRAIN_WARMUP: int = 10
VISION_TRAIN_ITERATIONS: int = 100

# ─── NLP Training (Benchmark 2) ──────────────────────────────────────────────
NLP_MODELS: Dict[str, dict] = {
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
NLP_VOCAB_SIZE: int = 30522  # Standard BERT WordPiece vocabulary
NLP_NUM_LABELS: int = 2
NLP_TRAIN_WARMUP: int = 10
NLP_TRAIN_ITERATIONS: int = 50

# ─── Vision Inference (Benchmark 3) ──────────────────────────────────────────
VISION_INFER_WARMUP: int = 20
VISION_INFER_ITERATIONS: int = 200

# ─── NLP Inference (Benchmark 4) ─────────────────────────────────────────────
NLP_INFER_WARMUP: int = 20
NLP_INFER_ITERATIONS: int = 200

# ─── LLM Token Generation (Benchmark 5) ──────────────────────────────────────
LLM_PROMPT: str = (
    "Explain the concept of machine learning, including supervised learning, "
    "unsupervised learning, and deep learning. Provide examples of each and "
    "discuss their real-world applications."
)
LLM_NUM_TOKENS: int = 512
LLM_TIMEOUT_SECONDS: int = 600  # 10 minutes per model

# ─── LLM Model Sets ──────────────────────────────────────────────────────────
#  "default"  — curated niche models (DeepSeek, Phi-4, Gemma-2, QwQ)
#  "popular"  — mainstream models everyone recognises (Llama, Mistral, Qwen)
#
#  Select via:  python run_benchmarks.py --model-set popular
#               python benchmarks/5_llm_tokens_per_sec.py --model-set popular
#               python install.py --model-set popular

LLM_MODEL_SETS: Dict[str, list] = {
    "default": [
        {
            "name": "DeepSeek-R1-Distill-Llama-8B",
            "repo_id": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
            "filename": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
            "size_gb": 4.5,
            "quant": "Q4_K_M",
        },
        {
            "name": "Phi-4",
            "repo_id": "microsoft/phi-4-gguf",
            "filename": "phi-4-Q8_0.gguf",
            "size_gb": 15,
            "quant": "Q8_0",
        },
        {
            "name": "Gemma-2-27B",
            "repo_id": "bartowski/gemma-2-27b-it-GGUF",
            "filename": "gemma-2-27b-it-Q4_K_M.gguf",
            "size_gb": 17,
            "quant": "Q4_K_M",
        },
        {
            "name": "QwQ-32B",
            "repo_id": "bartowski/Qwen_QwQ-32B-GGUF",
            "filename": "Qwen_QwQ-32B-Q4_K_M.gguf",
            "size_gb": 20,
            "quant": "Q4_K_M",
        },
    ],
    "popular": [
        {
            "name": "Llama-3.1-8B",
            "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "size_gb": 4.9,
            "quant": "Q4_K_M",
        },
        {
            "name": "Mistral-7B-v0.3",
            "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            "size_gb": 4.4,
            "quant": "Q4_K_M",
        },
        {
            "name": "Qwen2.5-14B",
            "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
            "filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            "size_gb": 9.0,
            "quant": "Q4_K_M",
        },
        {
            "name": "Qwen2.5-32B",
            "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
            "filename": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
            "size_gb": 19.9,
            "quant": "Q4_K_M",
        },
    ],
}

# Backwards compatibility — default model list
LLM_MODELS: list = LLM_MODEL_SETS["default"]


def get_llm_models(model_set: str = "default") -> list:
    """Return the model list for the given set name."""
    if model_set not in LLM_MODEL_SETS:
        available = ", ".join(sorted(LLM_MODEL_SETS.keys()))
        raise ValueError(f"Unknown model set '{model_set}'. Available: {available}")
    return LLM_MODEL_SETS[model_set]

# ─── VRAM Limits (Benchmark 6) ───────────────────────────────────────────────
# Real model configs with accurate parameter counts
VRAM_TEST_MODELS: list = [
    # (label, transformers model id or config, approximate param billions)
    {"label": "1.5B", "config": {"num_hidden_layers": 24, "hidden_size": 1536, "num_attention_heads": 16, "intermediate_size": 6144, "vocab_size": 50257}, "approx_params_b": 1.5},
    {"label": "3B",   "config": {"num_hidden_layers": 32, "hidden_size": 2048, "num_attention_heads": 16, "intermediate_size": 8192, "vocab_size": 50257}, "approx_params_b": 3.0},
    {"label": "7B",   "config": {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32, "intermediate_size": 11008, "vocab_size": 32000}, "approx_params_b": 6.7},
    {"label": "13B",  "config": {"num_hidden_layers": 40, "hidden_size": 5120, "num_attention_heads": 40, "intermediate_size": 13824, "vocab_size": 32000}, "approx_params_b": 13.0},
    {"label": "20B",  "config": {"num_hidden_layers": 44, "hidden_size": 6144, "num_attention_heads": 48, "intermediate_size": 16384, "vocab_size": 50257}, "approx_params_b": 20.0},
    {"label": "30B",  "config": {"num_hidden_layers": 48, "hidden_size": 6656, "num_attention_heads": 52, "intermediate_size": 17920, "vocab_size": 32000}, "approx_params_b": 30.0},
    {"label": "70B",  "config": {"num_hidden_layers": 80, "hidden_size": 8192, "num_attention_heads": 64, "intermediate_size": 28672, "vocab_size": 32000}, "approx_params_b": 65.0},
]
VRAM_CONTEXT_LENGTHS: List[int] = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# ─── Detection Training (Benchmark 8) ────────────────────────────────────────
DETECTION_NUM_CLASSES: int = 91  # COCO
DETECTION_IMAGE_SIZE: int = 800
DETECTION_WARMUP: int = 5
DETECTION_ITERATIONS: int = 50

# ─── GEMM Compute Stress (Benchmark 7) ───────────────────────────────────────
GEMM_SIZES: List[int] = [1024, 2048, 4096, 8192]
GEMM_WARMUP: int = 2
GEMM_REPEATS: int = 5

# ─── GPU Fundamentals (Benchmark 9) ──────────────────────────────────────────
# Memory bandwidth sweep sizes in bytes
FUND_BW_SIZES: List[int] = [
    16 * 1024 * 1024,        # 16 MB
    64 * 1024 * 1024,        # 64 MB
    256 * 1024 * 1024,       # 256 MB
    1024 * 1024 * 1024,      # 1 GB
    2 * 1024 * 1024 * 1024,  # 2 GB
]
FUND_PCIE_SIZE: int = 256 * 1024 * 1024  # 256 MB per H2D/D2H transfer
FUND_FFT_SIZES: List[int] = [1024, 4096, 16384, 65536, 262144]
FUND_NBODY_N: int = 32768           # number of particles for N-body
FUND_NBODY_STEPS: int = 50          # timestep iterations
FUND_STENCIL_SIZE: int = 4096       # NxN grid for 2-D heat stencil
FUND_STENCIL_STEPS: int = 100       # stencil iterations
FUND_WARMUP: int = 5
FUND_REPEATS: int = 20

# ─── Multi-GPU Scaling (Benchmark 10) ────────────────────────────────────────
MULTIGPU_VISION_BATCH_PER_GPU: int = 32
MULTIGPU_NLP_BATCH_PER_GPU: int = 16
MULTIGPU_WARMUP: int = 5
MULTIGPU_ITERATIONS: int = 50

# ─── Power / Thermal Monitoring ──────────────────────────────────────────────
MONITOR_INTERVAL_SEC: float = 0.5  # nvidia-smi polling interval
