#!/usr/bin/env python3
"""
Benchmark 5 — LLM Token Generation (llama.cpp)

Measures tokens/second and time-to-first-token for GGUF models.
Models must be pre-downloaded via install.py.
Based on GamersNexus methodology.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import (
    LLM_MODEL_SETS,
    LLM_NUM_TOKENS,
    LLM_PROMPT,
    LLM_TIMEOUT_SECONDS,
    MODELS_DIR,
    get_llm_models,
)
from benchmarks.benchmark_utils import (
    add_common_args,
    check_cuda_available,
    get_gpu_info,
    GPUMonitor,
    print_gpu_banner,
    save_results,
)


def find_llama_cli() -> Path | None:
    """Locate the best llama.cpp binary (llama-completion or llama-cli)."""
    # Binary preference order: llama-completion is non-interactive by design,
    # llama-cli requires --no-conversation flag (newer builds default to chat mode)
    binaries = ["llama-completion", "llama-cli"]

    # 1. Env var
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        # Maybe it's a directory — search for preferred binaries
        for binary in binaries:
            for candidate in [p / "build" / "bin" / binary, p / binary]:
                if candidate.exists():
                    return candidate

    # 2/3. Project-local and home dir
    home = Path.home()
    project_dir = Path(__file__).resolve().parent.parent
    for binary in binaries:
        for candidate in [
            project_dir / "llama.cpp" / "build" / "bin" / binary,
            project_dir / "llama.cpp" / binary,
            home / "llama.cpp" / "build" / "bin" / binary,
            home / "llama.cpp" / binary,
        ]:
            if candidate.exists():
                return candidate

    # 4. On PATH
    for binary in binaries:
        found = shutil.which(binary)
        if found:
            return Path(found)

    return None


def find_model(model: dict, models_dir: Path) -> Path | None:
    """Check if a pre-downloaded GGUF model exists. Returns path or None."""
    model_path = models_dir / model["filename"]
    if model_path.exists():
        print(f"    Model ready: {model_path.name}")
        return model_path
    print(f"    Model not found: {model_path.name}")
    print(f"    Run 'python install.py' to download all models.")
    return None


def parse_llama_output(output: str) -> dict:
    """
    Parse llama.cpp timing output from stderr and/or stdout.

    Handles multiple output format variations across llama.cpp versions:
      - Old format: "eval time = 12345.67 ms / 512 tokens ( 24.13 ms per token, 41.44 tokens per second)"
      - New format: "[ Prompt: 652.8 t/s | Generation: 43.8 t/s ]"
      - perf timings: "llama_perf_context_print: eval time = ..."
    """
    result: dict = {
        "tokens_generated": None,
        "total_time_s": None,
        "tokens_per_second": None,
        "time_to_first_token_ms": None,
    }

    for line in output.split("\n"):
        # ── New compact format: [ Prompt: 652.8 t/s | Generation: 43.8 t/s ] ──
        gen_match = re.search(r"Generation:\s*([\.\d]+)\s*t/s", line)
        if gen_match:
            result["tokens_per_second"] = round(float(gen_match.group(1)), 2)

        prompt_match = re.search(r"Prompt:\s*([\.\d]+)\s*t/s", line)
        if prompt_match:
            # Approximate TTFT from prompt speed (not exact but useful)
            pass  # TTFT is better captured from the old format if available

        # ── Old format: prompt eval time ──
        if "prompt eval time" in line.lower() or ("llama_perf" in line and "prompt eval" in line):
            match = re.search(r"([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
            if match:
                total_ms = float(match.group(1))
                result["time_to_first_token_ms"] = round(total_ms, 2)

        # ── Old format: eval time (generation) ──
        if ("eval time" in line and "prompt" not in line.lower()) or \
           ("llama_perf" in line and "eval time" in line and "prompt" not in line):
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


def benchmark_model(model: dict, model_path: Path | None, llama_cli: Path) -> dict:
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
        # Build the command
        cmd = [
            str(llama_cli),
            "-m", str(model_path),
            "-n", str(LLM_NUM_TOKENS),
            "-p", LLM_PROMPT,
            "-ngl", "999",       # Offload all layers to GPU
            "--temp", "0.7",
            "-b", "512",
            "--perf",            # Enable perf timings in output
        ]

        # If using llama-cli (not llama-completion), disable conversation mode
        if llama_cli.name == "llama-cli":
            cmd.extend(["--no-conversation", "--single-turn"])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=LLM_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
        )

        # Parse from both stdout and stderr (different versions put timings in different streams)
        combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
        parsed = parse_llama_output(combined_output)

        if parsed["tokens_per_second"]:
            print(f"{parsed['tokens_per_second']:.1f} t/s (TTFT: {parsed.get('time_to_first_token_ms', 'N/A')} ms)")
            return {**base_result, **parsed, "status": "success"}
        else:
            # Check for OOM in llama.cpp output
            if "out of memory" in result.stderr.lower() or "cudaMalloc failed" in result.stderr:
                print("OOM")
                return {**base_result, "status": "oom"}
            print("ERROR (parse failed)")
            print(f"    llama.cpp stderr (last 500 chars):\n{result.stderr[-500:]}")
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
    parser.add_argument("--model-set", type=str, default="default",
                        choices=sorted(LLM_MODEL_SETS.keys()),
                        help="Which model set to benchmark (default: default)")
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
        print("ERROR: llama-completion / llama-cli not found!")
        print("Set LLAMA_CPP_PATH env var or use --llama-path, or install llama.cpp:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release")
        sys.exit(1)

    print(f"llama binary: {llama_cli} ({llama_cli.name})")

    check_cuda_available()
    gpu_info = get_gpu_info(args.device)
    print_gpu_banner(gpu_info)

    vram_gb = gpu_info["vram_gb"]

    # Models directory (created by config.py on import)
    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # Resolve model set
    models = get_llm_models(args.model_set)
    print(f"Model set: {args.model_set} ({len(models)} models)")

    monitor = None
    if not args.no_monitor:
        monitor = GPUMonitor(device_id=args.device)
        monitor.start()

    results: list[dict] = []

    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model['name']} ({model['size_gb']:.1f}GB, {model['quant']})")
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

        model_path = find_model(model, models_dir)
        result = benchmark_model(model, model_path, llama_cli)
        results.append(result)
        time.sleep(2)  # Cool-down between models

    hw_stats = monitor.stop() if monitor else {}

    csv_path, json_path = save_results(
        results, "llm_tokens_per_sec", gpu_info,
        extra_meta={
            "prompt": LLM_PROMPT,
            "num_tokens": LLM_NUM_TOKENS,
            "model_set": args.model_set,
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
