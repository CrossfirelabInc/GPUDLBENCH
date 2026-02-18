#!/usr/bin/env python3
"""
Shared utilities for GPU AI Benchmark Suite.

Consolidates common code used across all benchmark scripts:
  - GPU / CUDA detection and info
  - VRAM-aware batch-size selection
  - Reproducibility seed helpers
  - AMP (Automatic Mixed Precision) context managers
  - Result saving (CSV + JSON)
  - Power / thermal monitoring via nvidia-smi
  - Progress bar wrapper
"""

from __future__ import annotations

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from benchmarks.config import OUTPUT_DIR, RANDOM_SEED, MONITOR_INTERVAL_SEC

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gpubench")


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU / CUDA helpers
# ═══════════════════════════════════════════════════════════════════════════════

def check_cuda_available() -> None:
    """Exit with a clear message if CUDA is not available."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! Please install NVIDIA drivers and a CUDA-enabled PyTorch.")
        sys.exit(1)


def get_gpu_info(device_id: int = 0) -> Dict[str, Any]:
    """Return a dict with GPU name, VRAM (GB), compute capability, etc."""
    check_cuda_available()
    props = torch.cuda.get_device_properties(device_id)
    return {
        "gpu_name": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 1),
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "device_id": device_id,
    }


def get_system_info(device_id: int = 0) -> Dict[str, Any]:
    """
    Collect comprehensive environment version info for reproducibility.

    Returns a dict with driver, CUDA, cuDNN, PyTorch, Python, OS versions, etc.
    """
    import platform as _platform

    info: Dict[str, Any] = {}

    # ── Python ────────────────────────────────────────────────────────────
    info["python_version"] = _platform.python_version()

    # ── OS ────────────────────────────────────────────────────────────────
    info["os"] = f"{_platform.system()} {_platform.release()}"
    info["os_version"] = _platform.version()

    # ── PyTorch ───────────────────────────────────────────────────────────
    info["pytorch_version"] = torch.__version__
    info["pytorch_cuda_version"] = getattr(torch.version, "cuda", None)  # e.g. "12.4"

    # ── cuDNN ─────────────────────────────────────────────────────────────
    if torch.backends.cudnn.is_available():
        info["cudnn_version"] = str(torch.backends.cudnn.version())
    else:
        info["cudnn_version"] = None

    # ── NVIDIA driver version ─────────────────────────────────────────────
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            info["nvidia_driver_version"] = r.stdout.strip().split("\n")[0]
        else:
            info["nvidia_driver_version"] = None
    except Exception:
        info["nvidia_driver_version"] = None

    # ── CUDA toolkit (nvcc) version ───────────────────────────────────────
    try:
        import shutil as _shutil
        nvcc = _shutil.which("nvcc")
        if nvcc:
            r = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5,
            )
            import re as _re
            m = _re.search(r"release (\d+\.\d+)", r.stdout)
            info["cuda_toolkit_version"] = m.group(1) if m else None
        else:
            info["cuda_toolkit_version"] = None
    except Exception:
        info["cuda_toolkit_version"] = None

    # ── GPU power info ────────────────────────────────────────────────────
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=power.limit,power.default_limit,power.max_limit,enforced.power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split("\n")[0].split(",")]
            info["gpu_power_limit_w"] = float(parts[0]) if parts[0] not in ("[N/A]", "") else None
            info["gpu_power_default_w"] = float(parts[1]) if len(parts) > 1 and parts[1] not in ("[N/A]", "") else None
            info["gpu_power_max_w"] = float(parts[2]) if len(parts) > 2 and parts[2] not in ("[N/A]", "") else None
            info["gpu_power_enforced_w"] = float(parts[3]) if len(parts) > 3 and parts[3] not in ("[N/A]", "") else None
        else:
            info["gpu_power_limit_w"] = None
            info["gpu_power_default_w"] = None
    except Exception:
        info["gpu_power_limit_w"] = None
        info["gpu_power_default_w"] = None

    # ── Optional library versions ─────────────────────────────────────────
    for pkg in ["transformers", "accelerate", "numpy", "pillow", "datasets"]:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = None

    return info


def supports_bf16(device_id: int = 0) -> bool:
    """BF16 requires compute capability >= 8.0 (Ampere+)."""
    props = torch.cuda.get_device_properties(device_id)
    return props.major >= 8


def supports_fp8(device_id: int = 0) -> bool:
    """
    FP8 (float8_e4m3fn) requires compute capability >= 8.9 (Ada Lovelace / Hopper / Blackwell).

    - Ada Lovelace (RTX 40xx): CC 8.9  — FP8 inference
    - Hopper (H100):           CC 9.0  — FP8 training + inference
    - Blackwell (B-series):    CC 10.0 — native FP8 training + inference
    """
    props = torch.cuda.get_device_properties(device_id)
    # FP8 tensor core support starts at CC 8.9 (Ada) and above
    if props.major > 9:
        return True  # Blackwell+
    if props.major == 9:
        return True  # Hopper
    if props.major == 8 and props.minor >= 9:
        return True  # Ada Lovelace
    return False


def supports_tf32(device_id: int = 0) -> bool:
    """TF32 Tensor Core acceleration requires compute capability >= 8.0."""
    return supports_bf16(device_id)


def print_gpu_banner(info: Dict[str, Any]) -> None:
    """Pretty-print GPU info at the start of a benchmark."""
    print(f"GPU: {info['gpu_name']}")
    print(f"VRAM: {info['vram_gb']} GB")
    print(f"Compute Capability: {info['compute_capability']}")
    print(f"SM Count: {info['sm_count']}")
    bf16 = "Yes" if supports_bf16(info['device_id']) else "No"
    fp8 = "Yes" if supports_fp8(info['device_id']) else "No"
    print(f"BF16 Support: {bf16}")
    print(f"FP8 Support: {fp8}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def set_reproducibility(seed: int = RANDOM_SEED, deterministic: bool = False) -> None:
    """
    Set seeds for reproducibility across runs.

    Parameters
    ----------
    seed : int
        Random seed for torch, numpy, and python random.
    deterministic : bool
        If True, force cuDNN deterministic mode (slower but bit-exact).
        Default False because benchmarks care about throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch-size selection based on VRAM
# ═══════════════════════════════════════════════════════════════════════════════

def get_vision_train_batch_sizes(vram_gb: float) -> List[int]:
    """Return training batch sizes for vision models based on VRAM."""
    if vram_gb >= 40:
        return [16, 32, 64, 128, 256]
    elif vram_gb >= 24:
        return [16, 32, 64, 128]
    elif vram_gb >= 16:
        return [8, 16, 32, 64]
    elif vram_gb >= 8:
        return [4, 8, 16, 32]
    else:
        return [2, 4, 8]


def get_vision_infer_batch_sizes(vram_gb: float) -> List[int]:
    """Return inference batch sizes for vision models."""
    if vram_gb >= 40:
        return [1, 8, 16, 32, 64, 128]
    elif vram_gb >= 24:
        return [1, 8, 16, 32, 64]
    elif vram_gb >= 16:
        return [1, 8, 16, 32]
    elif vram_gb >= 8:
        return [1, 4, 8, 16]
    else:
        return [1, 2, 4, 8]


def get_nlp_train_batch_sizes(vram_gb: float, model_name: str) -> List[int]:
    """Return training batch sizes for NLP models, accounting for model size."""
    is_large = "large" in model_name.lower()
    if is_large:
        if vram_gb >= 40:
            return [4, 8, 16, 32]
        elif vram_gb >= 24:
            return [2, 4, 8]
        elif vram_gb >= 16:
            return [2, 4]
        else:
            return [1, 2]
    else:
        if vram_gb >= 40:
            return [8, 16, 32, 64]
        elif vram_gb >= 24:
            return [8, 16, 32]
        elif vram_gb >= 16:
            return [4, 8, 16]
        else:
            return [2, 4, 8]


def get_nlp_infer_batch_sizes(vram_gb: float, model_name: str) -> List[int]:
    """Return inference batch sizes for NLP models."""
    is_large = "large" in model_name.lower()
    if is_large:
        if vram_gb >= 40:
            return [1, 4, 8, 16, 32]
        elif vram_gb >= 24:
            return [1, 4, 8, 16]
        elif vram_gb >= 16:
            return [1, 2, 4, 8]
        else:
            return [1, 2, 4]
    else:
        if vram_gb >= 40:
            return [1, 8, 16, 32, 64]
        elif vram_gb >= 24:
            return [1, 8, 16, 32]
        elif vram_gb >= 16:
            return [1, 4, 8, 16]
        else:
            return [1, 2, 4, 8]


def get_detection_batch_sizes(vram_gb: float) -> List[int]:
    """Return training batch sizes for detection models."""
    if vram_gb >= 40:
        return [2, 4, 8]
    elif vram_gb >= 24:
        return [1, 2, 4]
    elif vram_gb >= 16:
        return [1, 2]
    else:
        return [1]


# ═══════════════════════════════════════════════════════════════════════════════
#  AMP (Automatic Mixed Precision) helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_amp_context(precision: str, device_type: str = "cuda"):
    """
    Return an appropriate autocast context manager for the given precision.

    Parameters
    ----------
    precision : str
        One of "fp32", "fp16", "bf16", "fp8".
    device_type : str
        Typically "cuda".

    Returns
    -------
    torch.amp.autocast context manager (or nullcontext for fp32/fp8).

    Note: FP8 does not use autocast; callers handle float8 casting manually.
    """
    if precision == "fp16":
        return torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    elif precision == "bf16":
        return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    else:
        # FP32 and FP8 — no autocasting (FP8 uses manual dtype conversion)
        import contextlib
        return contextlib.nullcontext()


def get_grad_scaler(precision: str) -> Optional[torch.amp.GradScaler]:
    """
    Return a GradScaler for FP16 training, or None for FP32/BF16.

    BF16 does not need loss scaling because its dynamic range matches FP32.
    """
    if precision == "fp16":
        return torch.amp.GradScaler("cuda")
    return None


def filter_precisions_for_gpu(precisions: List[str], device_id: int = 0) -> List[str]:
    """Remove bf16/fp8 from the list if the GPU does not support them."""
    result = list(precisions)
    if not supports_bf16(device_id):
        result = [p for p in result if p != "bf16"]
    if not supports_fp8(device_id):
        result = [p for p in result if p != "fp8"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  TF32 control
# ═══════════════════════════════════════════════════════════════════════════════

def set_tf32(enabled: bool = True) -> None:
    """
    Enable or disable TF32 for matmul and cuDNN on Ampere+ GPUs.

    TF32 is ON by default in PyTorch >= 1.12 on Ampere+ GPUs.
    Disabling it gives true FP32 precision at the cost of speed.
    """
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


# ═══════════════════════════════════════════════════════════════════════════════
#  Result I/O
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_output_dir(output_dir: str = OUTPUT_DIR) -> Path:
    """Create the output directory if it doesn't exist and return as Path."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_results(
    results: List[Dict[str, Any]],
    filename_stem: str,
    gpu_info: Dict[str, Any],
    extra_meta: Optional[Dict[str, Any]] = None,
    output_dir: str = OUTPUT_DIR,
) -> Tuple[Path, Path]:
    """
    Save benchmark results to both CSV and JSON.

    Parameters
    ----------
    results : list of dicts
        Each dict is one benchmark row.
    filename_stem : str
        Base name without extension, e.g. "training_vision".
    gpu_info : dict
        Output of get_gpu_info().
    extra_meta : dict, optional
        Additional metadata to include in the JSON top-level.
    output_dir : str
        Directory for output files.

    Returns
    -------
    (csv_path, json_path)
    """
    out = ensure_output_dir(output_dir)

    # CSV
    csv_path = out / f"{filename_stem}.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # JSON
    json_data: Dict[str, Any] = {
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Power / Thermal Monitoring
# ═══════════════════════════════════════════════════════════════════════════════

class GPUMonitor:
    """
    Background thread that polls nvidia-smi for power draw, temperature,
    and clock speeds at a configurable interval.

    Usage
    -----
    >>> monitor = GPUMonitor(device_id=0)
    >>> monitor.start()
    >>> # ... run benchmark ...
    >>> stats = monitor.stop()
    >>> print(stats)  # {'avg_power_w': ..., 'max_power_w': ..., ...}
    """

    def __init__(self, device_id: int = 0, interval: float = MONITOR_INTERVAL_SEC):
        self.device_id = device_id
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: List[Dict[str, float]] = []

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.device_id}",
                        "--query-gpu=power.draw,temperature.gpu,clocks.current.sm",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
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
                pass  # nvidia-smi may not be available; silently skip
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        """Start background monitoring."""
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Progress helpers
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_iterator(total: int, desc: str = "Benchmarking", warmup: int = 0, show_progress: bool = True):
    """
    Yield iteration indices with an optional tqdm progress bar.

    Usage
    -----
    >>> for i in benchmark_iterator(100, desc="ResNet-50 FP32"):
    ...     # run one iteration
    ...     pass
    """
    total_iters = warmup + total
    if show_progress:
        pbar = tqdm(range(total_iters), desc=desc, leave=False, ncols=80)
    else:
        pbar = range(total_iters)

    for i in pbar:
        yield i, (i < warmup)  # (index, is_warmup)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI argument parsing helper
# ═══════════════════════════════════════════════════════════════════════════════

def add_common_args(parser) -> None:
    """
    Add arguments shared by all benchmark scripts.

    Call this from each script's argparse setup.
    """
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Directory for result files (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="CUDA device ID (default: 0)"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--no-monitor", action="store_true",
        help="Disable power/thermal monitoring"
    )
    parser.add_argument(
        "--precisions", nargs="+", default=None,
        help="Precision modes to test, e.g. --precisions fp32 fp16 bf16"
    )
