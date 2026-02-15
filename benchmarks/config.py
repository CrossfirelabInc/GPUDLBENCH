#!/usr/bin/env python3
"""
Central configuration for GPU AI Benchmark Suite.

All tunable parameters live here. Benchmark scripts import from this module
so users have a single place to adjust settings.
"""

from typing import Dict, List

# ─── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ─── General ──────────────────────────────────────────────────────────────────
OUTPUT_DIR: str = "results"

# ─── Precision modes ─────────────────────────────────────────────────────────
#  "fp32"  — standard single-precision
#  "fp16"  — AMP with float16 (all NVIDIA GPUs with Tensor Cores)
#  "bf16"  — AMP with bfloat16 (Ampere+ GPUs, compute capability >= 8.0)
#  "tf32"  — FP32 math but with TF32 Tensor Core acceleration (Ampere+)
TRAINING_PRECISIONS: List[str] = ["fp32", "fp16", "bf16"]
INFERENCE_PRECISIONS: List[str] = ["fp32", "fp16", "bf16"]

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

# HuggingFace GGUF model specifications (GamersNexus methodology)
LLM_MODELS: list = [
    {
        "name": "DeepSeek-R1-Distill-Llama-8B",
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "filename": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        "size_gb": 4.5,
        "quant": "Q4_K_M",
    },
    {
        "name": "Phi-4",
        "repo_id": "bartowski/Phi-4-GGUF",
        "filename": "Phi-4-Q8_0.gguf",
        "size_gb": 15,
        "quant": "Q8_0",
    },
    {
        "name": "Qwen2.5-14B",
        "repo_id": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "size_gb": 9,
        "quant": "Q4_K_M",
    },
    {
        "name": "InternLM2-Chat-20B",
        "repo_id": "TheBloke/internlm-chat-20b-GGUF",
        "filename": "internlm-chat-20b.Q4_K_M.gguf",
        "size_gb": 13,
        "quant": "Q4_K_M",
    },
    {
        "name": "Mistral-Small-22B",
        "repo_id": "bartowski/Mistral-Small-Instruct-2409-GGUF",
        "filename": "Mistral-Small-Instruct-2409-Q4_K_M.gguf",
        "size_gb": 14,
        "quant": "Q4_K_M",
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
        "repo_id": "Qwen/QwQ-32B-Preview-GGUF",
        "filename": "qwq-32b-preview-q4_k_m.gguf",
        "size_gb": 20,
        "quant": "Q4_K_M",
    },
    {
        "name": "Llama-3.3-70B",
        "repo_id": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "filename": "Llama-3.3-70B-Instruct-Q4_K_S.gguf",
        "size_gb": 42,
        "quant": "Q4_K_S",
    },
]

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

# ─── Mixed Precision (Benchmark 7) ───────────────────────────────────────────
MIXED_PREC_VISION_BATCH: int = 32
MIXED_PREC_NLP_BATCH: int = 16
MIXED_PREC_VISION_ITERS: int = 100
MIXED_PREC_NLP_ITERS: int = 50

# ─── Power / Thermal Monitoring ──────────────────────────────────────────────
MONITOR_INTERVAL_SEC: float = 0.5  # nvidia-smi polling interval
