#!/usr/bin/env python3
"""
Install prerequisites for the GPU DL Benchmark Suite.

Sets up a Python virtual environment, installs PyTorch with the right CUDA
version, installs Python dependencies, and optionally builds llama.cpp.

Usage:
    python install.py                 # full install
    python install.py --skip-llama    # skip llama.cpp build
    python install.py --venv myenv    # custom venv directory
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, check=True, capture=False, **kwargs):
    """Run a shell command, printing it first."""
    if isinstance(cmd, list):
        display = " ".join(cmd)
    else:
        display = cmd
    print(f"  $ {display}")
    return subprocess.run(
        cmd, check=check,
        capture_output=capture, text=True,
        **kwargs,
    )


def detect_cuda_version():
    """Try to figure out which CUDA version is available."""
    # Try nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        r = run(["nvcc", "--version"], capture=True, check=False)
        m = re.search(r"release (\d+\.\d+)", r.stdout)
        if m:
            return m.group(1)

    # Fallback: nvidia-smi header line
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        r = run(["nvidia-smi"], capture=True, check=False)
        m = re.search(r"CUDA Version:\s*(\d+\.\d+)", r.stdout)
        if m:
            return m.group(1)

    return None


def cuda_version_to_index(ver_string):
    """Map a CUDA version like '12.4' to a PyTorch wheel index suffix."""
    if ver_string is None:
        print("  WARNING: Could not detect CUDA. Defaulting to cu124.")
        return "cu124"

    major, minor = (int(x) for x in ver_string.split(".")[:2])
    if major >= 13 or (major == 12 and minor >= 4):
        return "cu124"
    if major == 12 and minor >= 1:
        return "cu121"
    if major == 11 and minor >= 8:
        return "cu118"

    print(f"  WARNING: CUDA {ver_string} may be too old. Trying cu118.")
    return "cu118"


def get_gpu_name():
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return "unknown"
    r = run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
        capture=True, check=False,
    )
    return r.stdout.strip().split("\n")[0] if r.returncode == 0 else "unknown"


def create_venv(venv_dir):
    if venv_dir.exists():
        print(f"  Virtual environment already exists at {venv_dir}")
    else:
        run([sys.executable, "-m", "venv", str(venv_dir)])
        print(f"  Created virtual environment at {venv_dir}")


def venv_python(venv_dir):
    """Return the path to the python binary inside the venv."""
    if platform.system() == "Windows":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def venv_pip(venv_dir):
    if platform.system() == "Windows":
        return str(venv_dir / "Scripts" / "pip.exe")
    return str(venv_dir / "bin" / "pip")


def install_pytorch(venv_dir, cuda_suffix):
    pip = venv_pip(venv_dir)
    run([pip, "install", "--upgrade", "pip", "setuptools", "wheel", "-q"])
    run([
        pip, "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_suffix}",
        "-q",
    ])


def verify_pytorch(venv_dir):
    py = venv_python(venv_dir)
    code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA not available in PyTorch'; "
        "print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, "
        "GPU: {torch.cuda.get_device_name(0)}')"
    )
    run([py, "-c", code])


def install_python_deps(venv_dir):
    pip = venv_pip(venv_dir)
    run([
        pip, "install",
        "transformers", "datasets", "accelerate",
        "pandas", "numpy", "psutil", "tqdm",
        "pillow", "huggingface-hub",
        "-q",
    ])


def build_llama_cpp(venv_dir):
    if not shutil.which("cmake"):
        print("  cmake not found, installing via pip...")
        pip = venv_pip(venv_dir)
        run([pip, "install", "cmake", "-q"])

    if not shutil.which("git"):
        print("  ERROR: git is required to clone llama.cpp")
        return False

    llama_dir = Path("llama.cpp")
    if not llama_dir.exists():
        print("  Cloning llama.cpp...")
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llama_dir)])
    else:
        print("  llama.cpp directory exists, pulling latest...")
        run(["git", "-C", str(llama_dir), "pull", "--ff-only"], check=False)

    print("  Building with CMake (GGML_CUDA=ON)...")
    build_dir = llama_dir / "build"
    run([
        "cmake", "-B", str(build_dir), "-S", str(llama_dir),
        "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
    ])

    # nproc equivalent
    jobs = str(os.cpu_count() or 4)
    run(["cmake", "--build", str(build_dir), "--config", "Release", "-j", jobs])

    # Find the binary
    for candidate in [
        build_dir / "bin" / "llama-cli",
        build_dir / "bin" / "main",
        build_dir / "llama-cli",
    ]:
        if candidate.exists():
            print(f"  llama.cpp binary found: {candidate}")
            print(f"  TIP: export LLAMA_CPP_PATH={candidate}")
            return True

    print("  WARNING: build completed but binary not found. Check llama.cpp/build/")
    return True


def main():
    parser = argparse.ArgumentParser(description="Install prerequisites for GPU DL Benchmark Suite")
    parser.add_argument("--skip-llama", action="store_true", help="Skip building llama.cpp")
    parser.add_argument("--venv", type=str, default="venv", help="Virtual environment directory (default: venv)")
    args = parser.parse_args()

    venv_dir = Path(args.venv)

    print("=" * 60)
    print("GPU DL Benchmark Suite - Installer")
    print("=" * 60)
    print()

    # 1. System info
    print("[1/6] System info")
    print(f"  Python: {platform.python_version()}")
    print(f"  OS:     {platform.system()} {platform.release()}")
    gpu = get_gpu_name()
    print(f"  GPU:    {gpu}")
    print()

    # 2. CUDA detection
    print("[2/6] Detecting CUDA version...")
    cuda_ver = detect_cuda_version()
    if cuda_ver:
        print(f"  Detected CUDA: {cuda_ver}")
    else:
        print("  Could not detect CUDA version")
    cuda_suffix = cuda_version_to_index(cuda_ver)
    print(f"  PyTorch wheel index: {cuda_suffix}")
    print()

    # 3. Virtual environment
    print("[3/6] Setting up virtual environment...")
    create_venv(venv_dir)
    print()

    # 4. PyTorch
    print("[4/6] Installing PyTorch...")
    install_pytorch(venv_dir, cuda_suffix)
    print("  Verifying...")
    verify_pytorch(venv_dir)
    print()

    # 5. Python dependencies
    print("[5/6] Installing Python dependencies...")
    install_python_deps(venv_dir)
    print()

    # 6. llama.cpp
    if args.skip_llama:
        print("[6/6] Skipping llama.cpp (--skip-llama)")
    else:
        print("[6/6] Building llama.cpp with CUDA support...")
        build_llama_cpp(venv_dir)
    print()

    # Done
    print("=" * 60)
    print("Installation complete.")
    print()
    if platform.system() == "Windows":
        activate = f"{venv_dir}\\Scripts\\activate"
    else:
        activate = f"source {venv_dir}/bin/activate"
    print(f"  Activate the environment:  {activate}")
    print(f"  Run all benchmarks:        python run_benchmarks.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
