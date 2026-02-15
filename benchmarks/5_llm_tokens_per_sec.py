#!/usr/bin/env python3
"""
Benchmark 5 — LLM Token Generation (llama.cpp)

Measures tokens/second and time-to-first-token for GGUF models.
Uses huggingface_hub for reliable downloads instead of wget.
Based on GamersNexus methodology.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import (
    LLM_MODELS,
    LLM_NUM_TOKENS,
    LLM_PROMPT,
    LLM_TIMEOUT_SECONDS,
)
from benchmarks.benchmark_utils import (
    add_common_args,
    check_cuda_available,
    get_gpu_info,
    GPUMonitor,
    print_gpu_banner,
    save_results,
    logger,
)


def find_llama_cli() -> Optional[Path]:
    """
    Locate the llama-cli binary.

    Search order:
      1. LLAMA_CPP_PATH env var
      2. ~/llama.cpp/build/bin/llama-cli  (CMake build)
      3. ~/llama.cpp/llama-cli            (legacy Makefile build)
      4. llama-cli on $PATH
    """
    # 1. Env var
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        # Maybe it's a directory
        for candidate in [p / "build" / "bin" / "llama-cli", p / "llama-cli"]:
            if candidate.exists():
                return candidate

    # 2/3. Home dir
    home = Path.home()
    for candidate in [
        home / "llama.cpp" / "build" / "bin" / "llama-cli",
        home / "llama.cpp" / "llama-cli",
    ]:
        if candidate.exists():
            return candidate

    # 4. On PATH
    try:
        result = subprocess.run(["which", "llama-cli"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except FileNotFoundError:
        pass

    return None


def download_model(model: dict, models_dir: Path) -> Optional[Path]:
    """
    Download a GGUF model using huggingface_hub (reliable, with retry).

    Falls back to wget/curl if huggingface_hub is unavailable.
    """
    model_path = models_dir / model["filename"]

    if model_path.exists():
        print(f"    Model already downloaded: {model_path.name}")
        return model_path

    print(f"    Downloading {model['name']} ({model['size_gb']:.1f}GB)...")

    # Prefer huggingface_hub
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        print(f"    Download complete: {Path(downloaded).name}")
        return Path(downloaded)
    except ImportError:
        logger.warning("huggingface_hub not installed; falling back to curl")
    except Exception as e:
        logger.warning(f"huggingface_hub download failed: {e}; falling back to curl")

    # Fallback: curl (more universal than wget)
    url = f"https://huggingface.co/{model['repo_id']}/resolve/main/{model['filename']}"
    try:
        subprocess.run(
            ["curl", "-L", "-o", str(model_path), "--progress-bar", url],
            check=True,
        )
        print(f"    Download complete")
        return model_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"    Download FAILED")
        return None


def parse_llama_output(stderr: str) -> Dict[str, Any]:
    """
    Parse llama.cpp timing output from stderr.

    Handles multiple output format variations across llama.cpp versions.
    """
    result: Dict[str, Any] = {
        "tokens_generated": None,
        "total_time_s": None,
        "tokens_per_second": None,
        "time_to_first_token_ms": None,
    }

    for line in stderr.split("\n"):
        # Prompt eval time → TTFT
        if "prompt eval time" in line.lower() or ("llama_print_timings" in line and "prompt eval" in line):
            match = re.search(r"([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
            if match:
                total_ms = float(match.group(1))
                result["time_to_first_token_ms"] = round(total_ms, 2)

        # Eval time → tokens/sec
        if ("eval time" in line and "prompt" not in line.lower()) or \
           ("llama_print_timings" in line and "eval time" in line and "prompt" not in line):
            # "eval time = 12345.67 ms / 512 tokens ( 24.13 ms per token, 41.44 tokens per second)"
            match = re.search(r"([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
            if match:
                total_ms = float(match.group(1))
                tokens = int(match.group(2))
                result["total_time_s"] = round(total_ms / 1000, 3)
                result["tokens_generated"] = tokens

            tps_match = re.search(r"([\d.]+)\s*tokens\s*per\s*second", line)
            if tps_match:
                result["tokens_per_second"] = round(float(tps_match.group(1)), 2)

    return result


def benchmark_model(model: dict, model_path: Optional[Path], llama_cli: Path) -> Dict[str, Any]:
    """Benchmark a single GGUF model."""

    print(f"  Running benchmark on {model['name']}...", end=" ", flush=True)

    base_result = {
        "model": model["name"],
        "size_gb": model["size_gb"],
        "quantization": model["quant"],
        "tokens_generated": None,
        "total_time_s": None,
        "tokens_per_second": None,
        "time_to_first_token_ms": None,
    }

    if not model_path or not model_path.exists():
        print("SKIP (download failed)")
        return {**base_result, "status": "download_failed"}

    try:
        result = subprocess.run(
            [
                str(llama_cli),
                "-m", str(model_path),
                "-n", str(LLM_NUM_TOKENS),
                "-p", LLM_PROMPT,
                "-ngl", "999",       # Offload all layers to GPU
                "--temp", "0.7",
                "-b", "512",
            ],
            capture_output=True,
            text=True,
            timeout=LLM_TIMEOUT_SECONDS,
        )

        parsed = parse_llama_output(result.stderr)

        if parsed["tokens_per_second"]:
            print(f"{parsed['tokens_per_second']:.1f} t/s (TTFT: {parsed.get('time_to_first_token_ms', 'N/A')} ms)")
            return {**base_result, **parsed, "status": "success"}
        else:
            # Check for OOM in llama.cpp output
            if "out of memory" in result.stderr.lower() or "cudaMalloc failed" in result.stderr:
                print("OOM")
                return {**base_result, "status": "oom"}
            print("ERROR (parse failed)")
            logger.debug(f"llama.cpp stderr:\n{result.stderr[-500:]}")
            return {**base_result, "status": "parse_failed"}

    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {**base_result, "status": "timeout"}
    except Exception as e:
        print(f"ERROR: {e}")
        return {**base_result, "status": "error"}


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Token Generation Benchmark")
    add_common_args(parser)
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Directory to store GGUF models (default: ~/llama.cpp/models)")
    parser.add_argument("--llama-path", type=str, default=None,
                        help="Path to llama-cli binary or llama.cpp dir")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM Token Generation Benchmark")
    print("GamersNexus Methodology")
    print("=" * 70)
    print()

    # Override env var if CLI arg provided
    if args.llama_path:
        os.environ["LLAMA_CPP_PATH"] = args.llama_path

    llama_cli = find_llama_cli()
    if not llama_cli:
        print("ERROR: llama-cli not found!")
        print("Set LLAMA_CPP_PATH env var or use --llama-path, or install llama.cpp:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release")
        sys.exit(1)

    print(f"llama-cli: {llama_cli}")

    check_cuda_available()
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)

    vram_gb = gpu_info["vram_gb"]

    # Models directory
    models_dir = Path(args.models_dir) if args.models_dir else (llama_cli.parent.parent.parent / "models")
    if not models_dir.exists():
        models_dir = Path.home() / "llama.cpp" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    monitor = None
    if not args.no_monitor:
        monitor = GPUMonitor(device_id=args.device)
        monitor.start()

    results: List[Dict[str, Any]] = []

    for i, model in enumerate(LLM_MODELS, 1):
        print(f"\n[{i}/{len(LLM_MODELS)}] {model['name']} ({model['size_gb']:.1f}GB, {model['quant']})")
        print("-" * 70)

        # VRAM check (leave 10% headroom)
        if model["size_gb"] > vram_gb * 0.9:
            print(f"  SKIP: Requires ~{model['size_gb']:.1f}GB, only {vram_gb:.1f}GB available")
            results.append({
                "model": model["name"],
                "size_gb": model["size_gb"],
                "quantization": model["quant"],
                "tokens_generated": None,
                "total_time_s": None,
                "tokens_per_second": None,
                "time_to_first_token_ms": None,
                "status": "insufficient_vram",
            })
            continue

        model_path = download_model(model, models_dir)
        result = benchmark_model(model, model_path, llama_cli)
        results.append(result)
        time.sleep(2)  # Cool-down between models

    hw_stats = monitor.stop() if monitor else {}

    csv_path, json_path = save_results(
        results, "llm_tokens_per_sec", gpu_info,
        extra_meta={
            "prompt": LLM_PROMPT,
            "num_tokens": LLM_NUM_TOKENS,
            "hw_monitor": hw_stats if hw_stats else None,
        },
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("LLM Benchmark Complete!")
    print("=" * 70)
    print(f"\nResults saved to:\n  - {csv_path}\n  - {json_path}\n")

    print("Summary:")
    print("-" * 70)
    for r in results:
        if r["status"] == "success":
            ttft = f", TTFT={r['time_to_first_token_ms']}ms" if r.get("time_to_first_token_ms") else ""
            print(f"  {r['model']:30s} {r['tokens_per_second']:6.1f} t/s{ttft}")
        else:
            print(f"  {r['model']:30s} {r['status'].upper()}")


if __name__ == "__main__":
    main()
