#!/usr/bin/env python3
"""
GPU DL Benchmark Suite — Clean Installer.

Philosophy:
  - The NVIDIA **driver** is the only thing assumed to be pre-installed.
  - Everything else (CUDA toolkit, venv, PyTorch, deps, llama.cpp) is
    owned by this script.
  - Every run purges stale state first, detects the GPU from the driver,
    then installs the correct versions of everything.

Usage:
    python install2.py                  # full clean install
    python install2.py --skip-llama     # skip llama.cpp build
    python install2.py --skip-models    # skip GGUF model downloads
    python install2.py --venv myenv     # custom venv directory
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run(cmd, check=True, capture=False, **kwargs):
    """Run a shell command, printing it first."""
    display = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"  $ {display}")
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, **kwargs)


def _sudo(cmd):
    """Run a command with sudo if not already root."""
    if os.geteuid() == 0:
        run(cmd)
    else:
        run(["sudo"] + cmd)


def _venv_python(venv_dir: Path) -> str:
    if platform.system() == "Windows":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def _pip(venv_dir: Path, args: list[str]):
    run([_venv_python(venv_dir), "-m", "pip"] + args)


def _system_python() -> str:
    """Return a path to the system python (not a venv python that may have been deleted)."""
    # If sys.executable is inside a venv that was purged, it won't exist
    if Path(sys.executable).exists():
        return sys.executable
    # Fall back to finding python3 on PATH
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p:
            return p
    return "python3"  # last resort


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 0 — Detect GPU (the only given)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_gpu() -> dict:
    """Query the NVIDIA driver for GPU info.  Returns a dict or exits."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        print("ERROR: nvidia-smi not found.  Install the NVIDIA driver first.")
        sys.exit(1)

    info = {"gpus": [], "driver_version": None, "cuda_driver_version": None}

    # GPU names, compute caps, VRAM
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0 or not r.stdout.strip():
        print("ERROR: nvidia-smi found but returned no GPU info.")
        sys.exit(1)

    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            info["gpus"].append({
                "name": parts[0],
                "compute_cap": parts[1],
                "vram_mib": parts[2],
            })

    # Driver version
    r2 = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if r2.returncode == 0 and r2.stdout.strip():
        info["driver_version"] = r2.stdout.strip().splitlines()[0].strip()

    # Max CUDA version the driver supports
    r3 = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", r3.stdout)
    if m:
        info["cuda_driver_version"] = m.group(1)

    # Derive SM codes
    sm_list = []
    for gpu in info["gpus"]:
        cc = gpu["compute_cap"].replace(".", "")
        if cc.isdigit():
            sm_list.append(int(cc))
    info["max_sm"] = max(sm_list) if sm_list else 0

    return info


# ═══════════════════════════════════════════════════════════════════════════════
#  Version mapping tables (SM → CUDA toolkit, SM → PyTorch index)
# ═══════════════════════════════════════════════════════════════════════════════

def cuda_toolkit_version_for_sm(sm: int) -> str:
    """Recommended CUDA toolkit version to install for a given SM.

    This is just the *preferred* version. install_cuda_toolkit() will
    fall back to whatever the NVIDIA repo actually has via
    _find_best_cuda_toolkit_pkg().
    """
    if sm >= 100:  return "12.8"    # Blackwell minimum
    return "12.6"                   # safe default; fallback picks latest


def pytorch_cuda_suffix(cuda_driver_ver: str | None, sm: int) -> str:
    """Pick the PyTorch CUDA wheel index (cu118, cu121, cu124, cu126, cu128).

    Blackwell (sm >= 100) forces cu128 because only those wheels ship
    sm_100/sm_120 kernels.
    """
    if sm >= 100:
        return "cu128"

    if cuda_driver_ver is None:
        return "cu124"

    major, minor = (int(x) for x in cuda_driver_ver.split(".")[:2])
    if major >= 13 or (major == 12 and minor >= 8):
        return "cu128"
    if major == 12 and minor >= 6:
        return "cu126"
    if major == 12 and minor >= 4:
        return "cu124"
    if major == 12:
        return "cu121"
    if major == 11 and minor >= 8:
        return "cu118"
    return "cu118"


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Purge everything we own
# ═══════════════════════════════════════════════════════════════════════════════

def purge_venv(venv_dir: Path):
    """Delete the entire venv so we get a clean Python environment."""
    if venv_dir.exists():
        print(f"  Removing venv: {venv_dir}")
        shutil.rmtree(venv_dir)


def purge_llama_build():
    """Remove llama.cpp/build (not the source — avoids re-clone)."""
    build_dir = Path("llama.cpp/build")
    if build_dir.exists():
        print("  Removing llama.cpp/build")
        shutil.rmtree(build_dir)


def purge_cuda_toolkit():
    """Remove distro and NVIDIA CUDA toolkit packages (keeps the driver)."""
    if platform.system() != "Linux":
        return
    if not shutil.which("apt-get"):
        return  # only handle apt-based distros for now

    # Packages that belong to the CUDA *toolkit* (NOT the driver).
    # nvidia-cuda-toolkit = distro package; cuda-toolkit-* = NVIDIA repo.
    pkgs_to_check = [
        "nvidia-cuda-toolkit",
        "nvidia-cuda-dev",
    ]

    # Also find any NVIDIA-repo cuda-toolkit-X-Y packages
    r = subprocess.run(
        ["dpkg", "-l", "cuda-toolkit-*"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if line.startswith("ii"):
                pkg = line.split()[1]
                pkgs_to_check.append(pkg)

    # Filter to only installed packages
    installed = []
    for pkg in pkgs_to_check:
        r = subprocess.run(
            ["dpkg", "-s", pkg], capture_output=True, text=True,
        )
        if r.returncode == 0 and "install ok installed" in r.stdout:
            installed.append(pkg)

    if installed:
        print(f"  Purging CUDA toolkit packages: {' '.join(installed)}")
        _sudo(["apt-get", "remove", "--purge", "-y"] + installed)
        _sudo(["apt-get", "autoremove", "-y", "-qq"])
    else:
        print("  No distro CUDA toolkit packages to remove.")

    # Clean stale nvcc from /usr/bin that might confuse cmake
    stale_nvcc = Path("/usr/bin/nvcc")
    if stale_nvcc.exists() and not stale_nvcc.is_symlink():
        # Only remove if it's not a symlink to /usr/local/cuda (NVIDIA repo)
        print(f"  Removing stale {stale_nvcc}")
        _sudo(["rm", "-f", str(stale_nvcc)])


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2 — Install system prerequisites
# ═══════════════════════════════════════════════════════════════════════════════

def install_system_prereqs():
    """Ensure build tools, python3-venv, git are present."""
    if platform.system() != "Linux":
        return

    needed: list[str] = []

    # python venv
    syspy = _system_python()
    r = run([syspy, "-m", "ensurepip", "--version"], capture=True, check=False)
    if r.returncode != 0:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        needed += [f"python{py_ver}-venv", f"python{py_ver}-dev"]

    # build tools
    for tool in ("make", "gcc", "g++"):
        if not shutil.which(tool):
            needed.append("build-essential")
            break

    if not shutil.which("git"):
        needed.append("git")

    if not needed:
        print("  All system prerequisites OK.")
        return

    if shutil.which("apt-get"):
        _sudo(["apt-get", "update", "-qq"])
        _sudo(["apt-get", "install", "-y", "-qq"] + needed)
    elif shutil.which("dnf"):
        _sudo(["dnf", "install", "-y"] + needed)
    elif shutil.which("pacman"):
        _sudo(["pacman", "-Sy", "--noconfirm", "base-devel", "python", "git"])
    else:
        print(f"  ERROR: Unknown package manager.  Please install: {' '.join(needed)}")
        sys.exit(1)


def install_gcc13():
    """Ensure gcc-13 / g++-13 are available (needed for glibc >= 2.38 + CUDA)."""
    if platform.system() != "Linux":
        return

    # Check if we need it
    try:
        _, ver = platform.libc_ver()
        m = re.match(r"(\d+)\.(\d+)", ver or "")
        if not m:
            return
        glibc = (int(m.group(1)), int(m.group(2)))
    except Exception:
        return

    if glibc < (2, 38):
        return  # Old glibc is fine with any gcc

    # Need gcc >= 13
    for v in range(13, 18):
        if shutil.which(f"gcc-{v}") and shutil.which(f"g++-{v}"):
            return  # Already have a suitable pair

    print("  glibc >= 2.38 detected — installing gcc-13/g++-13 for CUDA compatibility...")
    if shutil.which("apt-get"):
        _sudo(["apt-get", "install", "-y", "-qq", "gcc-13", "g++-13"])


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Install CUDA toolkit from NVIDIA repo
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_os_for_nvidia_repo() -> tuple[str, str] | None:
    os_rel = Path("/etc/os-release")
    if not os_rel.exists():
        return None
    text = os_rel.read_text()
    id_m  = re.search(r'^ID=["]?([\w]+)', text, re.MULTILINE)
    ver_m = re.search(r'^VERSION_ID=["\']?(\d+\.\d+)', text, re.MULTILINE)
    if not id_m or not ver_m:
        return None
    os_id  = id_m.group(1)
    os_ver = ver_m.group(1).replace(".", "")
    if os_id in ("ubuntu", "debian"):
        return (os_id, os_ver)
    return None


def install_cuda_toolkit(target_ver: str) -> str | None:
    """Install the NVIDIA CUDA toolkit and return the path to nvcc, or None.

    Installs from NVIDIA's official apt repo (not the broken distro packages).
    """
    major, minor = (int(x) for x in target_ver.split("."))

    # Check if we already have a working nvcc at the right version
    cuda_bin = Path(f"/usr/local/cuda-{target_ver}/bin")
    nvcc = cuda_bin / "nvcc" if cuda_bin.exists() else None
    if nvcc and nvcc.exists():
        ver = _nvcc_version(str(nvcc))
        if ver and ver >= (major, minor):
            print(f"  CUDA toolkit {target_ver} already installed at {nvcc}")
            _add_to_path(str(cuda_bin))
            return str(nvcc)

    # Also check /usr/local/cuda symlink
    for candidate in [Path("/usr/local/cuda/bin/nvcc")] + sorted(
        Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True
    ):
        if candidate.exists():
            ver = _nvcc_version(str(candidate))
            if ver and ver >= (major, minor):
                print(f"  Suitable nvcc already present: {candidate} (CUDA {ver[0]}.{ver[1]})")
                _add_to_path(str(candidate.parent))
                return str(candidate)

    if platform.system() != "Linux":
        print(f"  Please install CUDA toolkit {target_ver} manually.")
        return None

    os_info = _detect_os_for_nvidia_repo()
    if not os_info:
        print("  Cannot determine OS for NVIDIA repo (only Ubuntu/Debian supported).")
        print(f"  Please install CUDA toolkit {target_ver} manually:")
        print(f"    https://developer.nvidia.com/cuda-downloads")
        return None

    os_id, os_ver = os_info
    arch = platform.machine()
    repo_arch = {"x86_64": "x86_64", "aarch64": "sbsa"}.get(arch)
    if not repo_arch:
        print(f"  Unsupported CPU architecture: {arch}")
        return None

    keyring_url = (
        f"https://developer.download.nvidia.com/compute/cuda/repos/"
        f"{os_id}{os_ver}/{repo_arch}/cuda-keyring_1.1-1_all.deb"
    )
    pkg = f"cuda-toolkit-{major}-{minor}"

    print(f"  Installing {pkg} from NVIDIA official repo...")
    print(f"  (This can be ~3-5 GB; progress shown below)")
    try:
        deb = "/tmp/cuda-keyring.deb"
        urllib.request.urlretrieve(keyring_url, deb)
        _sudo(["dpkg", "-i", deb])
        _sudo(["apt-get", "update", "-qq"])
        # Check if the exact package exists; if not, find the best available
        r = subprocess.run(
            ["apt-cache", "show", pkg],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  Package {pkg} not found in repo — searching for alternatives...")
            pkg = _find_best_cuda_toolkit_pkg(major)
            if not pkg:
                print(f"  ERROR: No cuda-toolkit package found for CUDA {major}.x")
                return None
            print(f"  Using: {pkg}")
        _sudo(["apt-get", "install", "-y", pkg])
    except Exception as e:
        print(f"  CUDA toolkit install failed: {e}")
        print(f"  Install manually: https://developer.nvidia.com/cuda-downloads")
        return None

    nvcc_path = Path(f"/usr/local/cuda-{target_ver}/bin/nvcc")
    if nvcc_path.exists():
        _add_to_path(str(nvcc_path.parent))
        return str(nvcc_path)

    # The actual installed version may differ from target (e.g. requested 12.4
    # but got 12.6).  Scan /usr/local/cuda-*/bin/nvcc for the newest one.
    for candidate in sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True):
        if candidate.exists():
            _add_to_path(str(candidate.parent))
            print(f"  Found nvcc at {candidate}")
            return str(candidate)

    # Maybe installed as /usr/local/cuda
    fallback = Path("/usr/local/cuda/bin/nvcc")
    if fallback.exists():
        _add_to_path(str(fallback.parent))
        return str(fallback)

    print("  ERROR: CUDA toolkit installed but nvcc not found at expected path.")
    return None

def _find_best_cuda_toolkit_pkg(major: int) -> str | None:
    """Query apt-cache for the newest cuda-toolkit-MAJOR-MINOR package available."""
    r = subprocess.run(
        ["apt-cache", "search", f"^cuda-toolkit-{major}-"],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    # Lines like: "cuda-toolkit-12-6 - CUDA Toolkit 12.6 meta-package"
    # Pick the highest minor version
    best_pkg = None
    best_minor = -1
    for line in r.stdout.strip().splitlines():
        pkg_name = line.split()[0]
        m = re.match(rf"cuda-toolkit-{major}-(\d+)$", pkg_name)
        if m:
            mn = int(m.group(1))
            if mn > best_minor:
                best_minor = mn
                best_pkg = pkg_name
    return best_pkg


def _nvcc_version(nvcc_path: str) -> tuple[int, int] | None:
    try:
        r = subprocess.run([nvcc_path, "--version"],
                           capture_output=True, text=True, timeout=10)
        m = re.search(r"release (\d+)\.(\d+)", r.stdout)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass
    return None


def _add_to_path(directory: str):
    if directory not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{directory}:{os.environ.get('PATH', '')}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Create venv
# ═══════════════════════════════════════════════════════════════════════════════

def create_venv(venv_dir: Path):
    run([_system_python(), "-m", "venv", str(venv_dir)])
    print(f"  Created virtual environment at {venv_dir}")

    # Ensure pip
    py = _venv_python(venv_dir)
    r = run([py, "-m", "pip", "--version"], capture=True, check=False)
    if r.returncode != 0:
        run([py, "-m", "ensurepip", "--upgrade"])


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Install PyTorch
# ═══════════════════════════════════════════════════════════════════════════════

def install_pytorch(venv_dir: Path, cuda_suffix: str) -> bool:
    """Install latest PyTorch from the correct CUDA wheel index.

    Tries stable first, falls back to nightly (for brand-new architectures).
    """
    _pip(venv_dir, ["install", "--upgrade", "pip", "setuptools", "wheel", "-q"])

    index = f"https://download.pytorch.org/whl/{cuda_suffix}"
    print(f"  Installing latest PyTorch ({cuda_suffix})...")
    py = _venv_python(venv_dir)
    r = subprocess.run(
        [py, "-m", "pip", "install",
         "torch", "torchvision", "torchaudio",
         "--index-url", index],
        text=True,
    )
    if r.returncode == 0:
        return True

    # Nightly fallback
    print(f"  Stable {cuda_suffix} wheels not available — trying nightly...")
    nightly = f"https://download.pytorch.org/whl/nightly/{cuda_suffix}"
    r = subprocess.run(
        [py, "-m", "pip", "install", "--pre",
         "torch", "torchvision", "torchaudio",
         "--index-url", nightly],
        text=True,
    )
    if r.returncode == 0:
        return True

    print(f"  ERROR: Could not install PyTorch for {cuda_suffix}.")
    if r.stderr:
        print(f"  {r.stderr[-300:]}")
    return False


def verify_pytorch(venv_dir: Path) -> bool:
    """Smoke-test: can PyTorch see and run kernels on the GPU?"""
    py = _venv_python(venv_dir)
    code = (
        "import torch, sys\n"
        "print(f'  PyTorch {torch.__version__}')\n"
        "if not torch.cuda.is_available():\n"
        "    print('  CUDA not available'); sys.exit(1)\n"
        "print(f'  CUDA:  {torch.version.cuda}')\n"
        "print(f'  GPU:   {torch.cuda.get_device_name(0)}')\n"
        "try:\n"
        "    torch.zeros(1, device='cuda')\n"
        "except RuntimeError as e:\n"
        "    if 'no kernel image' in str(e):\n"
        "        maj, mn = torch.cuda.get_device_capability(0)\n"
        "        print(f'  FAIL: no kernels for sm_{maj}{mn}')\n"
        "        sys.exit(1)\n"
        "    raise\n"
        "print('  GPU smoke-test: OK')"
    )
    r = run([py, "-c", code], check=False)
    return r.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 6 — Install Python dependencies
# ═══════════════════════════════════════════════════════════════════════════════

def install_python_deps(venv_dir: Path):
    req = Path(__file__).resolve().parent / "requirements.txt"
    if not req.exists():
        print("  ERROR: requirements.txt not found.")
        return
    _pip(venv_dir, ["install", "--upgrade", "-r", str(req)])


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 7 — Build llama.cpp
# ═══════════════════════════════════════════════════════════════════════════════

def _find_host_compilers() -> tuple[str, str]:
    """Find a gcc/g++ pair >= 13 (for glibc 2.38+ compat).  Returns (gcc, g++)."""
    for v in range(13, 18):
        gcc = shutil.which(f"gcc-{v}")
        gxx = shutil.which(f"g++-{v}")
        if gcc and gxx:
            return (gcc, gxx)
    # Fall back to default
    gcc = shutil.which("gcc") or "gcc"
    gxx = shutil.which("g++") or "g++"
    return (gcc, gxx)


def _find_cmake(venv_dir: Path) -> str | None:
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    venv_cmake = Path(_venv_python(venv_dir)).parent / "cmake"
    return str(venv_cmake) if venv_cmake.exists() else None


def build_llama_cpp(venv_dir: Path, nvcc_path: str | None) -> bool:
    # cmake
    cmake = _find_cmake(venv_dir)
    if not cmake:
        print("  cmake not found, installing via pip...")
        _pip(venv_dir, ["install", "cmake", "-q"])
        cmake = _find_cmake(venv_dir)
    if not cmake:
        print("  ERROR: cmake still not found.  Skipping llama.cpp.")
        return False

    if not nvcc_path:
        print("  ERROR: nvcc not available — cannot build llama.cpp with CUDA.")
        return False

    if not shutil.which("git"):
        print("  ERROR: git not found.")
        return False

    llama_dir = Path("llama.cpp")
    if llama_dir.exists() and not (llama_dir / "CMakeLists.txt").exists():
        shutil.rmtree(llama_dir)

    if not llama_dir.exists():
        print("  Cloning llama.cpp...")
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llama_dir)])
    else:
        print("  Updating llama.cpp...")
        run(["git", "-C", str(llama_dir), "pull", "--ff-only"], check=False)

    build_dir = llama_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    gcc, gxx = _find_host_compilers()
    print(f"  Compilers: CC={gcc}  CXX={gxx}  NVCC={nvcc_path}")

    os.environ.update({"CC": gcc, "CXX": gxx, "CUDAHOSTCXX": gxx, "CUDACXX": nvcc_path})

    jobs = str(os.cpu_count() or 4)
    try:
        run([
            cmake, "-B", str(build_dir), "-S", str(llama_dir),
            "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_C_COMPILER={gcc}",
            f"-DCMAKE_CXX_COMPILER={gxx}",
            f"-DCMAKE_CUDA_COMPILER={nvcc_path}",
            f"-DCMAKE_CUDA_HOST_COMPILER={gxx}",
        ])
        run([cmake, "--build", str(build_dir), "--config", "Release", "-j", jobs])
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: llama.cpp build failed: {e}")
        return False

    for name in ("llama-completion", "llama-cli", "main"):
        for loc in (build_dir / "bin" / name, build_dir / name):
            if loc.exists():
                print(f"  llama.cpp binary: {loc}")
                return True

    print("  WARNING: build completed but binary not found.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 8 — HuggingFace token
# ═══════════════════════════════════════════════════════════════════════════════

def setup_hf_token(venv_dir: Path, cli_token: str | None = None):
    token = cli_token
    source = "--hf-token"

    if not token:
        cred = Path(__file__).resolve().parent / ".credentials"
        if cred.exists():
            for line in cred.read_text().splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN=") and "xxxx" not in line:
                    token = line.split("=", 1)[1].strip()
                    source = ".credentials"
                    break

    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        source = "env var"

    if not token:
        py = _venv_python(venv_dir)
        r = run([py, "-c",
                 "from huggingface_hub import HfFolder; "
                 "t = HfFolder.get_token(); print('OK' if t else 'NONE')"],
                capture=True, check=False)
        if r.returncode == 0 and "OK" in r.stdout:
            print("  Already logged in to HuggingFace Hub.")
            return
        print("  No HF token found.  Set HF_TOKEN in .credentials to download gated models.")
        return

    py = _venv_python(venv_dir)
    r = run([py, "-c",
             f"from huggingface_hub import login; login(token='{token}', add_to_git_credential=False)"],
            check=False)
    if r.returncode == 0:
        print(f"  HF token loaded ({source}).")
        os.environ["HF_TOKEN"] = token
    else:
        print("  WARNING: Could not save HF token.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 9 — Pre-download models
# ═══════════════════════════════════════════════════════════════════════════════

def predownload_models(venv_dir: Path, model_set: str = "default"):
    py = _venv_python(venv_dir)
    project_dir = Path(__file__).resolve().parent

    # Get VRAM
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

    download_script = '''
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarks.config import MODELS_DIR, get_llm_models

vram_gb = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "None" else None
model_set = sys.argv[2] if len(sys.argv) > 2 else "default"
models = get_llm_models(model_set)
print(f"  Models dir: {MODELS_DIR}  ({len(models)} models)")

token = os.environ.get("HF_TOKEN")
if not token:
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
    except Exception:
        pass

from huggingface_hub import hf_hub_download
ok = fail = skip = 0
for i, m in enumerate(models, 1):
    path = MODELS_DIR / m["filename"]
    print(f"  [{i}/{len(models)}] {m['name']} ({m['size_gb']:.1f} GB)", end=" ... ", flush=True)
    if path.exists():
        print("exists"); ok += 1; continue
    if vram_gb and m["size_gb"] > vram_gb * 0.9:
        print(f"SKIP (too large)"); skip += 1; continue
    try:
        hf_hub_download(repo_id=m["repo_id"], filename=m["filename"],
                        local_dir=str(MODELS_DIR), token=token)
        print("OK"); ok += 1
    except Exception as e:
        print(f"FAIL ({str(e)[:80]})"); fail += 1

print(f"  Caching GPT-2 config...", end=" ")
try:
    from transformers import AutoConfig
    AutoConfig.from_pretrained("gpt2"); print("OK")
except Exception as e:
    print(f"FAIL ({e})")

print(f"\\n  Done: {ok} ok, {fail} failed, {skip} skipped")
'''
    script_path = project_dir / "_predownload.py"
    script_path.write_text(download_script)
    try:
        run([py, str(script_path), str(gpu_vram) if gpu_vram else "None", model_set],
            check=False)
    finally:
        script_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPU DL Benchmark Suite — Clean Installer")
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--model-set", default="default", choices=["default", "popular"])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--venv", default="venv")
    args = parser.parse_args()

    venv_dir = Path(args.venv).resolve()

    print("=" * 60)
    print("GPU DL Benchmark Suite — Clean Installer")
    print("=" * 60)
    print()

    # ── 0. Detect GPU (the only given) ────────────────────────────────────
    print("[0] Detecting GPU from driver...")
    gpu = detect_gpu()
    for g in gpu["gpus"]:
        print(f"  GPU:    {g['name']}  (sm_{g['compute_cap'].replace('.','')}, "
              f"{int(g['vram_mib'])//1024} GB)")
    print(f"  Driver: {gpu['driver_version']}")
    print(f"  CUDA:   {gpu['cuda_driver_version']} (driver capability)")
    sm = gpu["max_sm"]
    print(f"  Max SM: {sm}")
    print()

    # Derive what to install
    target_cuda = cuda_toolkit_version_for_sm(sm)
    cu_suffix = pytorch_cuda_suffix(gpu["cuda_driver_version"], sm)
    print(f"  → CUDA toolkit to install: {target_cuda}")
    print(f"  → PyTorch wheel index:     {cu_suffix}")
    print()

    # ── 1. Purge everything we own ────────────────────────────────────────
    print("[1] Purging stale installations...")
    purge_venv(venv_dir)
    purge_llama_build()
    purge_cuda_toolkit()
    print()

    # ── 2. System prerequisites ───────────────────────────────────────────
    print("[2] System prerequisites...")
    install_system_prereqs()
    install_gcc13()
    print()

    # ── 3. CUDA toolkit ──────────────────────────────────────────────────
    print(f"[3] Installing CUDA toolkit {target_cuda}...")
    nvcc_path = install_cuda_toolkit(target_cuda)
    if nvcc_path:
        print(f"  nvcc: {nvcc_path}")
    else:
        print("  WARNING: nvcc not available — llama.cpp will fail.")
    print()

    # ── 4. Virtual environment ────────────────────────────────────────────
    print("[4] Creating venv...")
    create_venv(venv_dir)
    print()

    # ── 5. PyTorch ────────────────────────────────────────────────────────
    print(f"[5] Installing PyTorch ({cu_suffix})...")
    pt_ok = install_pytorch(venv_dir, cu_suffix)
    if pt_ok:
        pt_ok = verify_pytorch(venv_dir)
    if not pt_ok:
        print("  WARNING: PyTorch not working — benchmarks 1-4, 6-10 will fail.")
    print()

    # ── 6. Python deps ────────────────────────────────────────────────────
    print("[6] Installing Python dependencies...")
    install_python_deps(venv_dir)
    print()

    # ── 7. llama.cpp ─────────────────────────────────────────────────────
    if args.skip_llama:
        print("[7] Skipping llama.cpp (--skip-llama)")
    else:
        print("[7] Building llama.cpp...")
        build_llama_cpp(venv_dir, nvcc_path)
    print()

    # ── 8. HuggingFace token ─────────────────────────────────────────────
    print("[8] HuggingFace authentication...")
    setup_hf_token(venv_dir, args.hf_token)
    print()

    # ── 9. Models ─────────────────────────────────────────────────────────
    if args.skip_models:
        print("[9] Skipping model downloads (--skip-models)")
    else:
        print(f"[9] Downloading models ({args.model_set})...")
        predownload_models(venv_dir, args.model_set)
    print()

    # ── Save state ────────────────────────────────────────────────────────
    state = gpu.copy()
    state["cuda_toolkit"] = target_cuda
    state["pytorch_index"] = cu_suffix
    state["nvcc_path"] = nvcc_path
    Path(".gpu_state.json").write_text(json.dumps(state, indent=2) + "\n")

    # ── Done ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Installation complete.")
    print()
    activate = f"source {venv_dir}/bin/activate"
    print(f"  Activate:  {activate}")
    print(f"  Run:       python run_benchmarks.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
