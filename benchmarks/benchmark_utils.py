#!/usr/bin/env python3
"""Shared utilities for GPU AI Benchmark Suite."""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from benchmarks.config import OUTPUT_DIR, RANDOM_SEED, MONITOR_INTERVAL_SEC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gpubench")


# ── GPU / CUDA helpers ────────────────────────────────────────────────────────

def check_cuda_available() -> None:
    """Exit if CUDA is not available."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        sys.exit(1)


def get_gpu_info(device_id: int = 0) -> dict:
    """Return dict with GPU name, VRAM, compute capability, SM count."""
    check_cuda_available()
    props = torch.cuda.get_device_properties(device_id)
    return {
        "gpu_name": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 1),
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "device_id": device_id,
    }


def get_system_info(device_id: int = 0) -> dict:
    """Collect environment info for reproducibility."""
    import platform as _platform

    info: dict = {
        "python_version": _platform.python_version(),
        "os": f"{_platform.system()} {_platform.release()}",
        "os_version": _platform.version(),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
    }

    # NVIDIA driver
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        info["nvidia_driver_version"] = r.stdout.strip().split("\n")[0] if r.returncode == 0 else None
    except Exception:
        info["nvidia_driver_version"] = None

    # CUDA toolkit (nvcc)
    try:
        import re as _re
        import shutil as _shutil
        nvcc = _shutil.which("nvcc")
        if nvcc:
            r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            m = _re.search(r"release (\d+\.\d+)", r.stdout)
            info["cuda_toolkit_version"] = m.group(1) if m else None
        else:
            info["cuda_toolkit_version"] = None
    except Exception:
        info["cuda_toolkit_version"] = None

    # GPU power info
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=power.limit,power.default_limit,power.max_limit,enforced.power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split("\n")[0].split(",")]
            _p = lambda i: float(parts[i]) if len(parts) > i and parts[i] not in ("[N/A]", "") else None
            info["gpu_power_limit_w"] = _p(0)
            info["gpu_power_default_w"] = _p(1)
            info["gpu_power_max_w"] = _p(2)
            info["gpu_power_enforced_w"] = _p(3)
        else:
            info["gpu_power_limit_w"] = None
            info["gpu_power_default_w"] = None
    except Exception:
        info["gpu_power_limit_w"] = None
        info["gpu_power_default_w"] = None

    # Optional library versions
    for pkg in ["transformers", "accelerate", "numpy", "pillow", "datasets"]:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = None

    return info


def supports_bf16(device_id: int = 0) -> bool:
    """BF16 requires compute capability >= 8.0 (Ampere+)."""
    return torch.cuda.get_device_properties(device_id).major >= 8


def supports_fp8(device_id: int = 0) -> bool:
    """FP8 requires compute capability >= 8.9 (Ada/Hopper/Blackwell)."""
    props = torch.cuda.get_device_properties(device_id)
    return (props.major > 8) or (props.major == 8 and props.minor >= 9)


def print_gpu_banner(info: dict) -> None:
    """Print GPU info at the start of a benchmark."""
    print(f"GPU: {info['gpu_name']}")
    print(f"VRAM: {info['vram_gb']} GB")
    print(f"Compute Capability: {info['compute_capability']}")
    print(f"SM Count: {info['sm_count']}")
    bf16 = "Yes" if supports_bf16(info['device_id']) else "No"
    fp8 = "Yes" if supports_fp8(info['device_id']) else "No"
    print(f"BF16 Support: {bf16}")
    print(f"FP8 Support: {fp8}")
    print()


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_reproducibility(seed: int = RANDOM_SEED, deterministic: bool = False) -> None:
    """Set seeds for torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ── Batch-size selection ──────────────────────────────────────────────────────

_VISION_TRAIN_BS = [(40, [16, 32, 64, 128, 256]), (24, [16, 32, 64, 128]),
                    (16, [8, 16, 32, 64]), (8, [4, 8, 16, 32]), (0, [2, 4, 8])]
_VISION_INFER_BS = [(40, [1, 8, 16, 32, 64, 128]), (24, [1, 8, 16, 32, 64]),
                    (16, [1, 8, 16, 32]), (8, [1, 4, 8, 16]), (0, [1, 2, 4, 8])]
_NLP_TRAIN_BASE  = [(40, [8, 16, 32, 64]), (24, [8, 16, 32]),
                    (16, [4, 8, 16]), (0, [2, 4, 8])]
_NLP_TRAIN_LARGE = [(40, [4, 8, 16, 32]), (24, [2, 4, 8]),
                    (16, [2, 4]), (0, [1, 2])]
_NLP_INFER_BASE  = [(40, [1, 8, 16, 32, 64]), (24, [1, 8, 16, 32]),
                    (16, [1, 4, 8, 16]), (0, [1, 2, 4, 8])]
_NLP_INFER_LARGE = [(40, [1, 4, 8, 16, 32]), (24, [1, 4, 8, 16]),
                    (16, [1, 2, 4, 8]), (0, [1, 2, 4])]
_DETECTION_BS    = [(40, [2, 4, 8]), (24, [1, 2, 4]),
                    (16, [1, 2]), (0, [1])]


def _pick(table: list, vram_gb: float) -> list[int]:
    for threshold, sizes in table:
        if vram_gb >= threshold:
            return sizes
    return table[-1][1]


def get_vision_train_batch_sizes(vram_gb: float) -> list[int]:
    return _pick(_VISION_TRAIN_BS, vram_gb)

def get_vision_infer_batch_sizes(vram_gb: float) -> list[int]:
    return _pick(_VISION_INFER_BS, vram_gb)

def get_nlp_train_batch_sizes(vram_gb: float, model_name: str) -> list[int]:
    table = _NLP_TRAIN_LARGE if "large" in model_name.lower() else _NLP_TRAIN_BASE
    return _pick(table, vram_gb)

def get_nlp_infer_batch_sizes(vram_gb: float, model_name: str) -> list[int]:
    table = _NLP_INFER_LARGE if "large" in model_name.lower() else _NLP_INFER_BASE
    return _pick(table, vram_gb)

def get_detection_batch_sizes(vram_gb: float) -> list[int]:
    return _pick(_DETECTION_BS, vram_gb)


# ── AMP helpers ───────────────────────────────────────────────────────────────

def get_amp_context(precision: str):
    """Return autocast context for the given precision. FP8 maps to FP16."""
    if precision in ("fp16", "fp8"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    elif precision == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        return contextlib.nullcontext()


def get_grad_scaler(precision: str):
    """Return GradScaler for FP16/FP8, None otherwise."""
    if precision in ("fp16", "fp8"):
        return torch.amp.GradScaler("cuda")
    return None


def filter_precisions_for_gpu(precisions: list[str], device_id: int = 0) -> list[str]:
    """Remove bf16/fp8 if the GPU doesn't support them."""
    result = list(precisions)
    if not supports_bf16(device_id):
        result = [p for p in result if p != "bf16"]
    if not supports_fp8(device_id):
        result = [p for p in result if p != "fp8"]
    return result


# ── TF32 control ──────────────────────────────────────────────────────────────

def set_tf32(enabled: bool = True) -> None:
    """Enable/disable TF32 for matmul and cuDNN (Ampere+)."""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


# ── Timing helpers (shared by GEMM stress and GPU fundamentals) ───────────────

def cuda_sync() -> None:
    """Synchronize CUDA device."""
    torch.cuda.synchronize()


def median_time(fn, warmup: int, repeats: int) -> float:
    """Return median wall-clock time (seconds) over *repeats* measured calls."""
    for _ in range(warmup):
        fn()
    cuda_sync()
    times: list[float] = []
    for _ in range(repeats):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ── Result I/O ────────────────────────────────────────────────────────────────

def ensure_output_dir(output_dir: str = OUTPUT_DIR) -> Path:
    """Create output directory if needed and return as Path."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_results(
    results: list[dict],
    filename_stem: str,
    gpu_info: dict,
    extra_meta: dict | None = None,
    output_dir: str = OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Save benchmark results to CSV + JSON. Returns (csv_path, json_path)."""
    out = ensure_output_dir(output_dir)

    # CSV
    csv_path = out / f"{filename_stem}.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # JSON
    json_data: dict = {
        "gpu": gpu_info["gpu_name"],
        "vram_gb": gpu_info["vram_gb"],
        "compute_capability": gpu_info["compute_capability"],
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(gpu_info.get("device_id", 0)),
        "results": results,
    }
    if extra_meta:
        json_data.update(extra_meta)

    json_path = out / f"{filename_stem}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return csv_path, json_path


# ── Power / Thermal Monitoring ────────────────────────────────────────────────

class GPUMonitor:
    """Background thread polling nvidia-smi for power, temperature, clocks."""

    def __init__(self, device_id: int = 0, interval: float = MONITOR_INTERVAL_SEC):
        self.device_id = device_id
        self.interval = interval
        self._thread = None
        self._stop_event = threading.Event()
        self._samples: list[dict] = []

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", f"--id={self.device_id}",
                     "--query-gpu=power.draw,temperature.gpu,clocks.current.sm",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) == 3:
                        self._samples.append({
                            "power_w": float(parts[0]),
                            "temp_c": float(parts[1]),
                            "clock_mhz": float(parts[2]),
                        })
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """Stop monitoring and return aggregated statistics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

        if not self._samples:
            return {}

        powers = [s["power_w"] for s in self._samples]
        temps = [s["temp_c"] for s in self._samples]
        clocks = [s["clock_mhz"] for s in self._samples]

        return {
            "samples": len(self._samples),
            "avg_power_w": round(sum(powers) / len(powers), 1),
            "max_power_w": round(max(powers), 1),
            "min_power_w": round(min(powers), 1),
            "avg_temp_c": round(sum(temps) / len(temps), 1),
            "max_temp_c": round(max(temps), 1),
            "avg_clock_mhz": round(sum(clocks) / len(clocks), 0),
            "max_clock_mhz": round(max(clocks), 0),
        }


# ── CLI argument helper ──────────────────────────────────────────────────────

def add_common_args(parser) -> None:
    """Add arguments shared by all benchmark scripts."""
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory for result files (default: {OUTPUT_DIR})")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device ID (default: 0)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Disable power/thermal monitoring")
    parser.add_argument("--precisions", nargs="+", default=None,
                        help="Precision modes to test, e.g. --precisions fp32 fp16 bf16")
