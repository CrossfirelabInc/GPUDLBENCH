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


def ensure_system_prerequisites():
    """Install system-level packages required before venv creation."""
    system = platform.system()
    if system != "Linux":
        return  # Only Linux needs special handling

    needs_venv = False
    needs_build_tools = False

    # Check if venv module is usable
    r = run([sys.executable, "-m", "ensurepip", "--version"], capture=True, check=False)
    if r.returncode != 0:
        needs_venv = True

    # Check if build tools (make, gcc, g++) are available
    for tool in ("make", "gcc", "g++"):
        if not shutil.which(tool):
            needs_build_tools = True
            break

    if not needs_venv and not needs_build_tools:
        print("  All system prerequisites already satisfied.")
        return

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    if shutil.which("apt-get"):
        pkgs = ["build-essential"]
        if needs_venv:
            pkgs += [f"python{py_ver}-venv", f"python{py_ver}-dev"]
        print(f"  Installing system packages: {' '.join(pkgs)}")
        _run_sudo(["apt-get", "update", "-qq"])
        _run_sudo(["apt-get", "install", "-y", "-qq"] + pkgs)
    elif shutil.which("dnf"):
        pkgs = ["gcc", "gcc-c++", "make"]
        if needs_venv:
            pkgs.append(f"python{py_ver}-devel")
        _run_sudo(["dnf", "install", "-y"] + pkgs)
    elif shutil.which("yum"):
        pkgs = ["gcc", "gcc-c++", "make"]
        if needs_venv:
            pkgs.append(f"python{py_ver}-devel")
        _run_sudo(["yum", "install", "-y"] + pkgs)
    elif shutil.which("pacman"):
        _run_sudo(["pacman", "-Sy", "--noconfirm", "base-devel", "python"])
    elif shutil.which("zypper"):
        pkgs = ["gcc", "gcc-c++", "make"]
        if needs_venv:
            pkgs.append(f"python{py_ver}-devel")
        _run_sudo(["zypper", "install", "-y"] + pkgs)
    else:
        print(f"  ERROR: Could not detect package manager.")
        print(f"  Please install build-essential (make, gcc, g++) and python{py_ver}-venv manually.")
        sys.exit(1)

    # Verify venv works
    if needs_venv:
        r2 = run([sys.executable, "-m", "ensurepip", "--version"], capture=True, check=False)
        if r2.returncode != 0:
            print(f"  ERROR: Still cannot use ensurepip after installing packages.")
            print(f"  Please install python{py_ver}-venv manually and re-run this script.")
            sys.exit(1)

    print("  System prerequisites installed successfully.")


def _run_sudo(cmd):
    """Run a command with sudo if not already root."""
    if os.geteuid() == 0:
        run(cmd)
    else:
        run(["sudo"] + cmd)


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
    if platform.system() == "Windows":
        activate = venv_dir / "Scripts" / "activate.bat"
    else:
        activate = venv_dir / "bin" / "activate"

    if venv_dir.exists():
        if activate.exists():
            print(f"  Virtual environment already exists at {venv_dir}")
        else:
            # Broken venv (e.g. created before python3-venv was installed)
            print("  Existing venv is broken (activate script missing).")
            print("  Removing and recreating...")
            shutil.rmtree(str(venv_dir))
            run([sys.executable, "-m", "venv", str(venv_dir)])
            print(f"  Recreated virtual environment at {venv_dir}")
    else:
        run([sys.executable, "-m", "venv", str(venv_dir)])
        print(f"  Created virtual environment at {venv_dir}")

    # Ensure activate scripts are executable (some distros don't set this)
    if platform.system() != "Windows":
        for script in (venv_dir / "bin").glob("activate*"):
            script.chmod(script.stat().st_mode | 0o755)

    # Ensure pip is available inside the venv (some distros skip it)
    py = venv_python(venv_dir)
    r = run([py, "-m", "pip", "--version"], capture=True, check=False)
    if r.returncode != 0:
        print("  pip not found in venv, bootstrapping via ensurepip...")
        run([py, "-m", "ensurepip", "--upgrade"])


def venv_python(venv_dir):
    """Return the path to the python binary inside the venv."""
    if platform.system() == "Windows":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def pip_install(venv_dir, args):
    """Run pip via 'python -m pip' (more reliable than calling pip binary)."""
    py = venv_python(venv_dir)
    run([py, "-m", "pip"] + args)


# ─── Pinned versions for reproducibility ─────────────────────────────────────
TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"
TORCHAUDIO_VERSION = "2.6.0"
RECOMMENDED_CUDA_TOOLKIT = "12.4"  # nvcc version that matches pinned PyTorch wheels


def install_pytorch(venv_dir, cuda_suffix):
    pip_install(venv_dir, ["install", "--upgrade", "pip", "setuptools", "wheel", "-q"])
    pip_install(venv_dir, [
        "install",
        f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}", f"torchaudio=={TORCHAUDIO_VERSION}",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_suffix}",
        "-q",
    ])


def verify_pytorch(venv_dir):
    py = venv_python(venv_dir)
    code = (
        "import torch; "
        "print(f'  PyTorch {torch.__version__}'); "
        "cuda_ok = torch.cuda.is_available(); "
        "print(f'  CUDA available: {cuda_ok}'); "
        "exit(0) if not cuda_ok else None; "
        "print(f'  CUDA version:   {torch.version.cuda}'); "
        "print(f'  GPU:            {torch.cuda.get_device_name(0)}')"
    )
    r = run([py, "-c", code], check=False)
    if r.returncode != 0:
        print("  WARNING: PyTorch installed but could not verify CUDA support.")
        print("           Benchmarks requiring CUDA may fail.")
        print("           This can happen if the NVIDIA driver is not loaded.")
        return False
    return True


def install_python_deps(venv_dir):
    # Use requirements.txt for pinned versions
    req_file = Path(__file__).resolve().parent / "requirements.txt"
    if req_file.exists():
        pip_install(venv_dir, ["install", "-r", str(req_file), "-q"])
    else:
        # Fallback if requirements.txt is missing
        pip_install(venv_dir, [
            "install",
            "transformers==4.48.3", "datasets==3.3.2", "accelerate==1.4.0",
            "pandas==2.2.3", "numpy==2.2.3", "psutil==6.1.1", "tqdm==4.67.1",
            "pillow==11.1.0", "huggingface-hub==0.28.1",
            "-q",
        ])


def _find_cmake(venv_dir):
    """Find cmake, checking both system PATH and the venv bin directory."""
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    # Check inside the venv (pip-installed cmake lands here)
    venv_cmake = Path(venv_python(venv_dir)).parent / "cmake"
    if venv_cmake.exists():
        return str(venv_cmake)
    return None


def _check_cuda_toolkit_version():
    """Warn if installed nvcc version differs from the recommended one."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return
    r = run(["nvcc", "--version"], capture=True, check=False)
    m = re.search(r"release (\d+\.\d+)", r.stdout)
    if m:
        installed = m.group(1)
        if installed != RECOMMENDED_CUDA_TOOLKIT:
            print(f"  WARNING: nvcc reports CUDA {installed}, but pinned PyTorch"
                  f" wheels target CUDA {RECOMMENDED_CUDA_TOOLKIT}.")
            print(f"           This may cause mismatches. Consider installing"
                  f" CUDA toolkit {RECOMMENDED_CUDA_TOOLKIT}.")
        else:
            print(f"  CUDA toolkit {installed} matches recommended version. Good.")


def ensure_cuda_toolkit():
    """Install the CUDA toolkit (nvcc) if not already available."""
    if shutil.which("nvcc"):
        _check_cuda_toolkit_version()
        return True  # Already available

    system = platform.system()
    if system != "Linux":
        print("  WARNING: nvcc not found. Please install the CUDA toolkit manually.")
        return False

    print("  nvcc not found. Installing CUDA toolkit...")

    if shutil.which("apt-get"):
        # Debian / Ubuntu — install nvidia-cuda-toolkit
        _run_sudo(["apt-get", "update", "-qq"])
        _run_sudo(["apt-get", "install", "-y", "-qq", "nvidia-cuda-toolkit"])
    elif shutil.which("dnf"):
        _run_sudo(["dnf", "install", "-y", "cuda-compiler"])
    elif shutil.which("yum"):
        _run_sudo(["yum", "install", "-y", "cuda-compiler"])
    elif shutil.which("pacman"):
        _run_sudo(["pacman", "-Sy", "--noconfirm", "cuda"])
    elif shutil.which("zypper"):
        _run_sudo(["zypper", "install", "-y", "cuda-compiler"])
    else:
        print("  ERROR: Cannot detect package manager. Install CUDA toolkit manually.")
        return False

    if shutil.which("nvcc"):
        r = run(["nvcc", "--version"], capture=True, check=False)
        print(f"  CUDA toolkit installed: {r.stdout.strip().splitlines()[-1] if r.returncode == 0 else 'unknown version'}")
        return True
    else:
        print("  WARNING: CUDA toolkit installed but nvcc not found on PATH.")
        print("           You may need to add /usr/local/cuda/bin to your PATH.")
        # Try the common install location
        if Path("/usr/local/cuda/bin/nvcc").exists():
            os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}" 
            print("           Found nvcc at /usr/local/cuda/bin/nvcc — added to PATH.")
            return True
        return False


def build_llama_cpp(venv_dir):
    # --- Ensure cmake ---
    cmake = _find_cmake(venv_dir)
    if cmake is None:
        print("  cmake not found, installing via pip...")
        pip_install(venv_dir, ["install", "cmake", "-q"])
        cmake = _find_cmake(venv_dir)
    if cmake is None:
        print("  ERROR: cmake still not found after install. Skipping llama.cpp build.")
        return False
    print(f"  Using cmake: {cmake}")

    # --- Ensure CUDA toolkit (nvcc) for CUDA build ---
    use_cuda = ensure_cuda_toolkit()
    if not use_cuda:
        print("  WARNING: Building llama.cpp WITHOUT CUDA support (nvcc not available).")
        print("           LLM token/s benchmark will run on CPU only.")

    if not shutil.which("git"):
        print("  ERROR: git is required to clone llama.cpp")
        return False

    llama_dir = Path("llama.cpp")
    if llama_dir.exists() and not (llama_dir / "CMakeLists.txt").exists():
        # Directory exists but is empty/incomplete (e.g. stale submodule
        # reference or failed previous clone).  Remove and re-clone.
        print("  llama.cpp directory exists but appears empty — removing and re-cloning...")
        shutil.rmtree(llama_dir)

    if not llama_dir.exists():
        print("  Cloning llama.cpp...")
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llama_dir)])
    else:
        print("  llama.cpp directory exists, pulling latest...")
        run(["git", "-C", str(llama_dir), "pull", "--ff-only"], check=False)

    cuda_flag = "-DGGML_CUDA=ON" if use_cuda else "-DGGML_CUDA=OFF"
    print(f"  Building with CMake ({cuda_flag})...")
    build_dir = llama_dir / "build"
    run([
        cmake, "-B", str(build_dir), "-S", str(llama_dir),
        cuda_flag, "-DCMAKE_BUILD_TYPE=Release",
    ])

    # nproc equivalent
    jobs = str(os.cpu_count() or 4)
    run([cmake, "--build", str(build_dir), "--config", "Release", "-j", jobs])

    # Find the binary (prefer llama-completion for benchmarking)
    for candidate in [
        build_dir / "bin" / "llama-completion",
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


# ─── HuggingFace token setup ─────────────────────────────────────────────────

def _read_credentials_file():
    """Read HF_TOKEN from .credentials file in project root."""
    cred_path = Path(__file__).resolve().parent / ".credentials"
    if not cred_path.exists():
        return None
    for line in cred_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith("HF_TOKEN="):
            val = line.split("=", 1)[1].strip()
            if val and val != "hf_xxxxxxxxxxxxxxxxxxxx":
                return val
    return None


def setup_hf_token(venv_dir, cli_token=None):
    """
    Configure HuggingFace authentication.

    Priority: --hf-token arg > .credentials file > HF_TOKEN env var > cached login > skip.
    """
    token = cli_token
    source = "--hf-token flag"

    if not token:
        token = _read_credentials_file()
        source = ".credentials file"

    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        source = "environment variable"

    if not token:
        # Check if already logged in
        py = venv_python(venv_dir)
        r = run([py, "-c",
                 "from huggingface_hub import HfFolder; "
                 "t = HfFolder.get_token(); "
                 "print('OK' if t else 'NONE')"],
                capture=True, check=False)
        if r.returncode == 0 and "OK" in r.stdout:
            print("  Already logged in to HuggingFace Hub.")
            return

        print("  No HF token found. To authenticate:")
        print("    1. Get a token at https://huggingface.co/settings/tokens")
        print("    2. Paste it in .credentials (HF_TOKEN=hf_...)")
        print("  Skipping — some gated model downloads may fail.")
        return

    # Save token via huggingface-cli
    py = venv_python(venv_dir)
    r = run([py, "-c",
             f"from huggingface_hub import login; login(token='{token}', add_to_git_credential=False)"],
            check=False)
    if r.returncode == 0:
        print(f"  HuggingFace token loaded from {source}.")
        os.environ["HF_TOKEN"] = token
    else:
        print("  WARNING: Could not save HF token. Check your token value.")


# ─── Model pre-download ──────────────────────────────────────────────────────

def predownload_models(venv_dir, model_set="default"):
    """
    Pre-download all GGUF models and the GPT-2 config needed by benchmarks.

    Runs inside the venv so huggingface_hub and transformers are available.
    Uses config.py for model list, paths, and HF_HOME.
    """
    py = venv_python(venv_dir)
    project_dir = Path(__file__).resolve().parent

    # Get VRAM to decide which models to skip
    gpu_vram = None
    try:
        r = run([py, "-c",
                 "import torch; "
                 "print(f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}')"],
                capture=True, check=False)
        if r.returncode == 0:
            gpu_vram = float(r.stdout.strip())
            print(f"  GPU VRAM: {gpu_vram:.1f} GB")
    except Exception:
        pass

    # The download script imports everything from config.py (paths, model list, HF_HOME)
    download_script = '''
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# config.py creates data dirs and sets HF_HOME on import
from benchmarks.config import LLM_MODEL_SETS, MODELS_DIR, get_llm_models

vram_gb = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "None" else None
model_set = sys.argv[2] if len(sys.argv) > 2 else "default"
LLM_MODELS = get_llm_models(model_set)

print(f"  Models directory: {MODELS_DIR}")
print(f"  Model set: {model_set} ({len(LLM_MODELS)} models)")

# Resolve HF token (env var or cached login)
import os
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
    except Exception:
        pass

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("  ERROR: huggingface_hub not installed")
    sys.exit(1)

succeeded = 0
failed = 0
skipped = 0

for i, model in enumerate(LLM_MODELS, 1):
    name = model["name"]
    size = model["size_gb"]
    path = MODELS_DIR / model["filename"]

    print(f"  [{i}/{len(LLM_MODELS)}] {name} ({size:.1f} GB, {model['quant']})", end=" ... ", flush=True)

    if path.exists():
        print("already downloaded")
        succeeded += 1
        continue

    if vram_gb and size > vram_gb * 0.9:
        print(f"SKIP (needs {size:.1f}GB, have {vram_gb:.1f}GB VRAM)")
        skipped += 1
        continue

    try:
        hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            local_dir=str(MODELS_DIR),
            token=token,
        )
        print("OK")
        succeeded += 1
    except Exception as e:
        short = str(e).split("\\n")[0][:100]
        print(f"FAILED ({short})")
        failed += 1

# Cache GPT-2 config (used by benchmark 6 — tiny download)
print(f"  Caching GPT-2 config (for VRAM tests)...", end=" ", flush=True)
try:
    from transformers import AutoConfig
    AutoConfig.from_pretrained("gpt2")
    print("OK")
except Exception as e:
    print(f"FAILED ({e})")

print(f"\\n  Summary: {succeeded} downloaded, {failed} failed, {skipped} skipped (VRAM)")
if failed:
    print("  TIP: Set HF_TOKEN or run 'huggingface-cli login' for gated repos.")
'''

    # Write temp script and run it inside venv
    script_path = project_dir / "_predownload_models.py"
    script_path.write_text(download_script)
    try:
        vram_str = str(gpu_vram) if gpu_vram else "None"
        run([py, str(script_path), vram_str, model_set], check=False)
    finally:
        script_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Install prerequisites for GPU DL Benchmark Suite")
    parser.add_argument("--skip-llama", action="store_true", help="Skip building llama.cpp")
    parser.add_argument("--skip-models", action="store_true", help="Skip pre-downloading GGUF models")
    parser.add_argument("--model-set", type=str, default="default",
                        choices=["default", "popular"],
                        help="Which LLM model set to download (default: default)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--venv", type=str, default="venv", help="Virtual environment directory (default: venv)")
    args = parser.parse_args()

    venv_dir = Path(args.venv).resolve()

    print("=" * 60)
    print("GPU DL Benchmark Suite - Installer")
    print("=" * 60)
    print()

    # 0. System prerequisites (python3-venv, etc.)
    print("[0/8] Checking system prerequisites...")
    ensure_system_prerequisites()
    print()

    # 1. System info
    print("[1/8] System info")
    print(f"  Python: {platform.python_version()}")
    print(f"  OS:     {platform.system()} {platform.release()}")
    gpu = get_gpu_name()
    print(f"  GPU:    {gpu}")
    print()

    # 2. CUDA detection
    print("[2/8] Detecting CUDA version...")
    cuda_ver = detect_cuda_version()
    if cuda_ver:
        print(f"  Detected CUDA: {cuda_ver}")
    else:
        print("  Could not detect CUDA version")
    cuda_suffix = cuda_version_to_index(cuda_ver)
    print(f"  PyTorch wheel index: {cuda_suffix}")
    print()

    # 3. Virtual environment
    print("[3/8] Setting up virtual environment...")
    create_venv(venv_dir)
    print()

    # 4. PyTorch
    print("[4/8] Installing PyTorch...")
    install_pytorch(venv_dir, cuda_suffix)
    print("  Verifying...")
    verify_pytorch(venv_dir)
    print()

    # 5. Python dependencies
    print("[5/8] Installing Python dependencies...")
    install_python_deps(venv_dir)
    print()

    # 6. llama.cpp
    if args.skip_llama:
        print("[6/8] Skipping llama.cpp (--skip-llama)")
    else:
        print("[6/8] Building llama.cpp with CUDA support...")
        build_llama_cpp(venv_dir)
    print()

    # 7. HuggingFace token
    print("[7/8] HuggingFace authentication...")
    setup_hf_token(venv_dir, args.hf_token)
    print()

    # 8. Pre-download models
    if args.skip_models:
        print("[8/8] Skipping model pre-download (--skip-models)")
    else:
        print(f"[8/8] Pre-downloading benchmark models (set: {args.model_set})...")
        predownload_models(venv_dir, args.model_set)
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
