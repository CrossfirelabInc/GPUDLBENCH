#!/usr/bin/env python3
"""
Benchmark 6 — VRAM Limitation Tests

Tests what different VRAM capacities can actually handle:
  1. Maximum model size that can be loaded (FP16)
  2. Maximum context length for a ~7B-scale model
  3. Simultaneous model deployment count

Uses realistically-sized GPT-2-style configs whose parameter counts
match the labels (1.5B, 3B, 7B, 13B, 20B, 30B, 70B).
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import VRAM_TEST_MODELS, VRAM_CONTEXT_LENGTHS
from benchmarks.benchmark_utils import (
    add_common_args,
    check_cuda_available,
    get_gpu_info,
    print_gpu_banner,
    save_results,
    set_reproducibility,
)


def _count_params(model: torch.nn.Module) -> float:
    """Return total parameter count in billions."""
    return sum(p.numel() for p in model.parameters()) / 1e9


def test_model_size_limit(vram_gb: float, device: torch.device, base_config) -> Tuple[List[Dict], str]:
    """Test maximum model size that can be loaded in FP16."""

    print("\n" + "=" * 70)
    print("Test 1: Maximum Model Size (FP16)")
    print("=" * 70)
    print()

    results: List[Dict] = []
    max_label = "none"

    for entry in VRAM_TEST_MODELS:
        label = entry["label"]
        cfg_dict = entry["config"]
        approx = entry["approx_params_b"]

        # Quick VRAM estimate: params_B * 2 bytes (FP16) * 1.2 overhead → GB
        estimated_gb = approx * 2 * 1.2

        if estimated_gb > vram_gb * 0.95:
            print(f"  {label:>4s} model (~{approx:.1f}B params, ~{estimated_gb:.1f}GB)... SKIP (exceeds {vram_gb:.1f}GB)")
            results.append({
                "label": label,
                "approx_params_b": approx,
                "actual_params_b": None,
                "estimated_vram_gb": round(estimated_gb, 1),
                "actual_vram_gb": None,
                "loadable": False,
                "reason": "insufficient_vram",
            })
            continue

        try:
            config = copy.deepcopy(base_config)
            for k, v in cfg_dict.items():
                setattr(config, k, v)


            print(f"  {label:>4s} model (~{approx:.1f}B params, ~{estimated_gb:.1f}GB)... ", end="", flush=True)
            model = AutoModelForCausalLM.from_config(config)
            actual_params = _count_params(model)
            model = model.half().to(device)

            actual_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"OK ({actual_params:.2f}B params, {actual_vram:.1f}GB VRAM)")

            results.append({
                "label": label,
                "approx_params_b": approx,
                "actual_params_b": round(actual_params, 2),
                "estimated_vram_gb": round(estimated_gb, 1),
                "actual_vram_gb": round(actual_vram, 1),
                "loadable": True,
                "reason": "success",
            })
            max_label = label

            del model
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM")
                results.append({
                    "label": label,
                    "approx_params_b": approx,
                    "actual_params_b": None,
                    "estimated_vram_gb": round(estimated_gb, 1),
                    "actual_vram_gb": None,
                    "loadable": False,
                    "reason": "oom",
                })
                torch.cuda.empty_cache()
                break
            raise

    return results, max_label


def test_context_length_limit(vram_gb: float, device: torch.device, base_config) -> Tuple[List[Dict], int]:
    """Test maximum context length for a ~7B-scale model."""

    print("\n" + "=" * 70)
    print("Test 2: Maximum Context Length (~7B Model)")
    print("=" * 70)
    print()

    # Use the 7B config from VRAM_TEST_MODELS
    cfg_7b = next((m for m in VRAM_TEST_MODELS if m["label"] == "7B"), None)
    if cfg_7b is None:
        print("  SKIP: 7B config not found")
        return [], 0

    results: List[Dict] = []
    max_context = 0

    for ctx_len in VRAM_CONTEXT_LENGTHS:
        # Estimate VRAM: model (~13GB for 7B FP16) + KV cache grows with context
        # Rough heuristic: skip if context would clearly exceed VRAM
        # Each layer needs ~2 * hidden_size * ctx_len * 2 bytes (K+V, FP16)
        n_layers = cfg_7b["config"]["num_hidden_layers"]
        hidden = cfg_7b["config"]["hidden_size"]
        kv_estimate_gb = (2 * n_layers * hidden * ctx_len * 2) / (1024 ** 3)
        if kv_estimate_gb + 13.0 > vram_gb * 0.95:
            print(f"  Context {ctx_len:>7,}... SKIP (estimated {kv_estimate_gb + 13.0:.1f}GB exceeds {vram_gb:.1f}GB)")
            results.append({"context_length": ctx_len, "vram_used_gb": None, "success": False})
            break

        try:
            # Build a fresh config for each context length so n_positions matches
            config = copy.deepcopy(base_config)
            for k, v in cfg_7b["config"].items():
                setattr(config, k, v)
            # Override max position embeddings to support the target context length
            config.n_positions = ctx_len
            config.max_position_embeddings = ctx_len


            print(f"  Context {ctx_len:,}... ", end="", flush=True)
            model = AutoModelForCausalLM.from_config(config).half().to(device)
            input_ids = torch.randint(0, config.vocab_size, (1, ctx_len), device=device)

            with torch.inference_mode():
                _ = model(input_ids)

            vram_used = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"OK ({vram_used:.1f}GB)")

            results.append({"context_length": ctx_len, "vram_used_gb": round(vram_used, 1), "success": True})
            max_context = ctx_len

            del input_ids, model
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("OOM")
                results.append({"context_length": ctx_len, "vram_used_gb": None, "success": False})
                torch.cuda.empty_cache()
                break
            else:
                # Other CUDA errors — log and stop gracefully
                print(f"ERROR ({type(e).__name__})")
                results.append({"context_length": ctx_len, "vram_used_gb": None, "success": False})
                torch.cuda.empty_cache()
                break

    return results, max_context


def test_multi_model_deployment(vram_gb: float, device: torch.device, base_config) -> int:
    """Test how many ~7B models can be loaded simultaneously."""

    print("\n" + "=" * 70)
    print("Test 3: Multi-Model Deployment (~7B Models)")
    print("=" * 70)
    print()

    cfg_7b = next((m for m in VRAM_TEST_MODELS if m["label"] == "7B"), None)
    if cfg_7b is None:
        return 0

    config = copy.deepcopy(base_config)
    for k, v in cfg_7b["config"].items():
        setattr(config, k, v)

    loaded_models: list = []
    max_models = 0

    for i in range(1, 10):
        print(f"  Loading model {i}... ", end="", flush=True)
        try:
            m = AutoModelForCausalLM.from_config(config).half().to(device)
            loaded_models.append(m)
            vram_used = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"OK ({vram_used:.1f}GB total)")
            max_models = i
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM")
                torch.cuda.empty_cache()
                break
            raise

    for m in loaded_models:
        del m
    torch.cuda.empty_cache()
    return max_models


def main() -> None:
    parser = argparse.ArgumentParser(description="VRAM Limitation Tests")
    add_common_args(parser)
    args = parser.parse_args()

    print("=" * 70)
    print("VRAM Limitation Tests")
    print("=" * 70)
    print()

    check_cuda_available()
    device = torch.device(f"cuda:{args.device}")
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)
    set_reproducibility(args.seed)

    vram_gb = gpu_info["vram_gb"]

    # Load GPT-2 base config once (pre-cached by install.py) and reuse it
    # as the base for all test configs.  Using local_files_only avoids any
    # network requests — the config must be cached by install.py step 8.
    try:
        base_config = AutoConfig.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        print("  GPT-2 config not cached. Downloading (one-time)...")
        base_config = AutoConfig.from_pretrained("gpt2")

    model_size_results, max_model_label = test_model_size_limit(vram_gb, device, base_config)
    context_results, max_context = test_context_length_limit(vram_gb, device, base_config)
    max_models = test_multi_model_deployment(vram_gb, device, base_config)

    summary = {
        "max_model_size_label": max_model_label,
        "max_context_length": max_context,
        "max_simultaneous_7b_models": max_models,
        "model_size_tests": model_size_results,
        "context_length_tests": context_results,
    }

    csv_path, json_path = save_results(
        model_size_results,
        "vram_limits",
        gpu_info,
        extra_meta=summary,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("VRAM Limitation Tests Complete!")
    print("=" * 70)
    print(f"\nResults saved to:\n  - {csv_path}\n  - {json_path}\n")

    print("Summary:")
    print("-" * 70)
    print(f"  Maximum loadable model:    ~{max_model_label} parameters")
    print(f"  Maximum context (7B):      {max_context:,} tokens")
    print(f"  Simultaneous 7B models:    {max_models}")


if __name__ == "__main__":
    main()
