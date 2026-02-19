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
import json
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
    """Detect the CUDA version supported by the driver.

    PyTorch uses the *driver* CUDA version at runtime (not nvcc).
    nvidia-smi reports the max CUDA version the driver supports,
    which is what determines which PyTorch cu* wheel to install.
    Falls back to nvcc if nvidia-smi is unavailable.
    """
    # Prefer nvidia-smi — this is the driver's CUDA capability
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        r = run(["nvidia-smi"], capture=True, check=False)
        m = re.search(r"CUDA Version:\s*(\d+\.\d+)", r.stdout)
        if m:
            return m.group(1)

    # Fallback: nvcc (compile-time toolkit)
    nvcc = shutil.which("nvcc")
    if nvcc:
        r = run(["nvcc", "--version"], capture=True, check=False)
        m = re.search(r"release (\d+\.\d+)", r.stdout)
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
    if major == 12:
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
GPU_STATE_FILE = ".gpu_state.json"  # tracks GPU info across installs for swap detection


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


# ─── GPU architecture → CUDA version mapping ─────────────────────────────────

def _min_cuda_for_sm(sm: int) -> tuple[int, int]:
    """Return (major, minor) minimum CUDA toolkit version for a compute capability.

    The *sm* parameter is a condensed integer, e.g. 86 for sm_8.6.
    """
    if sm >= 100:  return (12, 8)   # Blackwell  (sm_100, sm_120)
    if sm >= 90:   return (12, 0)   # Hopper     (sm_90)
    if sm >= 89:   return (11, 8)   # Ada Lovelace (sm_89)
    if sm >= 80:   return (11, 0)   # Ampere     (sm_80-87)
    if sm >= 75:   return (10, 0)   # Turing     (sm_75)
    if sm >= 70:   return (9, 0)    # Volta      (sm_70-72)
    if sm >= 60:   return (8, 0)    # Pascal     (sm_60-62)
    return (7, 0)                   # Maxwell / Kepler


def _recommended_cuda_for_sm(sm: int) -> str:
    """Return recommended CUDA toolkit version string to install."""
    if sm >= 100:  return "12.8"
    return "12.4"   # Covers Pascal through Hopper


def _parse_cuda_ver(ver_str: str) -> tuple[int, int]:
    """Parse '12.4' → (12, 4)."""
    parts = ver_str.split(".")
    return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def _nvcc_version(nvcc_path: str) -> tuple[int, int] | None:
    """Get CUDA version from an nvcc binary.  Returns (major, minor) or None."""
    try:
        r = subprocess.run([nvcc_path, "--version"],
                           capture_output=True, text=True, timeout=10)
        m = re.search(r"release (\d+)\.(\d+)", r.stdout)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass
    return None


def _find_suitable_nvcc(min_cuda: tuple[int, int]) -> str | None:
    """Scan the system for an nvcc that meets *min_cuda*.  Returns path or None.

    Search order:
      1. nvcc on $PATH
      2. /usr/local/cuda/bin/nvcc  (default symlink)
      3. /usr/local/cuda-*/bin/nvcc  (versioned installs, newest first)
    """
    seen: set[str] = set()
    candidates: list[str] = []

    def _add(p: str):
        rp = str(Path(p).resolve())
        if rp not in seen:
            seen.add(rp)
            candidates.append(p)

    system_nvcc = shutil.which("nvcc")
    if system_nvcc:
        _add(system_nvcc)

    default = Path("/usr/local/cuda/bin/nvcc")
    if default.exists():
        _add(str(default))

    for p in sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True):
        _add(str(p))

    for nvcc_path in candidates:
        ver = _nvcc_version(nvcc_path)
        if ver and ver >= min_cuda:
            return nvcc_path
    return None


def _detect_os_for_nvidia_repo() -> tuple[str | None, str | None]:
    """Read /etc/os-release to determine repo URL components."""
    os_rel = Path("/etc/os-release")
    if not os_rel.exists():
        return None, None
    text = os_rel.read_text()
    id_m  = re.search(r'^ID=["]?([\w]+)', text, re.MULTILINE)
    ver_m = re.search(r'^VERSION_ID=["\']?(\d+\.\d+)', text, re.MULTILINE)
    if not id_m or not ver_m:
        return None, None
    os_id  = id_m.group(1)   # "ubuntu"
    os_ver = ver_m.group(1).replace(".", "")  # "22.04" → "2204"
    return (os_id, os_ver) if os_id in ("ubuntu", "debian") else (None, None)


def _install_cuda_via_nvidia_repo(target_ver: str) -> bool:
    """Install CUDA toolkit from NVIDIA's official apt repo (Ubuntu/Debian)."""
    os_id, os_ver = _detect_os_for_nvidia_repo()
    if not os_id:
        return False

    arch = platform.machine()
    repo_arch = {"x86_64": "x86_64", "aarch64": "sbsa"}.get(arch)
    if not repo_arch:
        print(f"  Unsupported CPU architecture for NVIDIA repo: {arch}")
        return False

    keyring_url = (
        f"https://developer.download.nvidia.com/compute/cuda/repos/"
        f"{os_id}{os_ver}/{repo_arch}/cuda-keyring_1.1-1_all.deb"
    )
    maj, mn = _parse_cuda_ver(target_ver)
    pkg = f"cuda-toolkit-{maj}-{mn}"

    print(f"  Installing {pkg} from NVIDIA repo...")
    print(f"  NOTE: This is a large download (~3-5 GB). Progress shown below.")
    print(f"  Keyring URL: {keyring_url}")
    try:
        import urllib.request
        keyring_deb = "/tmp/cuda-keyring.deb"
        urllib.request.urlretrieve(keyring_url, keyring_deb)
        _run_sudo(["dpkg", "-i", keyring_deb])
        _run_sudo(["apt-get", "update", "-qq"])
        _run_sudo(["apt-get", "install", "-y", pkg])  # no -qq: show progress
    except Exception as e:
        print(f"  Failed to install from NVIDIA repo: {e}")
        return False

    # The package installs into /usr/local/cuda-X.Y
    cuda_bin = Path(f"/usr/local/cuda-{maj}.{mn}/bin")
    if cuda_bin.exists():
        os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
        print(f"  Added {cuda_bin} to PATH")
    return True


# ─── CUDA toolkit: detect / install ──────────────────────────────────────────

def ensure_cuda_toolkit() -> str | None:
    """Ensure we have an nvcc that can compile for the installed GPU(s).

    Returns the absolute path to a suitable nvcc, or None on failure.

    Steps:
      1. Detect GPU compute capabilities  →  determine minimum CUDA needed.
      2. Scan system for a suitable nvcc.
      3. If not found, try to install via the distro package manager or
         NVIDIA's official repo.
    """
    # --- What does our GPU need? ---
    archs_str = _detect_cuda_architectures()
    if archs_str:
        sm_list = [int(x) for x in archs_str.split(";") if x.isdigit()]
        max_sm = max(sm_list) if sm_list else 0
    else:
        max_sm = 0

    if max_sm > 0:
        min_cuda = _min_cuda_for_sm(max_sm)
        print(f"  GPU arch: sm_{max_sm}  →  requires CUDA ≥ {min_cuda[0]}.{min_cuda[1]}")
    else:
        min_cuda = (11, 0)  # safe fallback
        print(f"  Could not detect GPU arch — assuming CUDA ≥ {min_cuda[0]}.{min_cuda[1]}")

    def _accept_nvcc(nvcc: str) -> str | None:
        ver = _nvcc_version(nvcc)
        if not ver:
            return None
        print(f"  Found suitable nvcc: {nvcc}  (CUDA {ver[0]}.{ver[1]})")
        nvcc_dir = str(Path(nvcc).parent)
        if nvcc_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{nvcc_dir}:{os.environ.get('PATH', '')}"
            print(f"  Added {nvcc_dir} to PATH")
        return nvcc

    # --- Already have a good-enough nvcc? ---
    nvcc_path = _find_suitable_nvcc(min_cuda)
    if nvcc_path:
        return _accept_nvcc(nvcc_path)

    # --- Need to install ---
    print(f"  No nvcc found that supports CUDA ≥ {min_cuda[0]}.{min_cuda[1]}")

    if platform.system() != "Linux":
        print(f"  Please install CUDA toolkit ≥ {min_cuda[0]}.{min_cuda[1]} manually.")
        return None

    target_ver = _recommended_cuda_for_sm(max_sm) if max_sm else RECOMMENDED_CUDA_TOOLKIT

    # Try 1: distro package manager (quick, may be too old)
    installed_distro = False
    if shutil.which("apt-get"):
        print(f"  Trying distro nvidia-cuda-toolkit first...")
        _run_sudo(["apt-get", "update", "-qq"])
        _run_sudo(["apt-get", "install", "-y", "-qq", "nvidia-cuda-toolkit"])
        installed_distro = True
    elif shutil.which("dnf"):
        _run_sudo(["dnf", "install", "-y", "cuda-compiler"])
        installed_distro = True
    elif shutil.which("yum"):
        _run_sudo(["yum", "install", "-y", "cuda-compiler"])
        installed_distro = True
    elif shutil.which("pacman"):
        _run_sudo(["pacman", "-Sy", "--noconfirm", "cuda"])
        installed_distro = True
    elif shutil.which("zypper"):
        _run_sudo(["zypper", "install", "-y", "cuda-compiler"])
        installed_distro = True

    # Check if the distro package was sufficient
    nvcc_path = _find_suitable_nvcc(min_cuda)
    if nvcc_path:
        return _accept_nvcc(nvcc_path)

    # Try 2: NVIDIA's official repo (apt-based only)
    if installed_distro and shutil.which("apt-get"):
        print(f"  Distro nvcc too old — trying NVIDIA's official repo for CUDA {target_ver}...")
        _install_cuda_via_nvidia_repo(target_ver)
        nvcc_path = _find_suitable_nvcc(min_cuda)
        if nvcc_path:
            return _accept_nvcc(nvcc_path)

    if not installed_distro:
        print(f"  ERROR: Cannot detect package manager.")

    print(f"  ERROR: Could not install CUDA toolkit ≥ {min_cuda[0]}.{min_cuda[1]}.")
    print(f"  Please install CUDA toolkit {target_ver} manually:")
    print(f"    https://developer.nvidia.com/cuda-{target_ver.replace('.', '-')}-download-archive")
    return None


def _detect_cuda_architectures() -> str:
    """Return a semicolon-separated list of compute capabilities for all GPUs.

    Tries three methods in order:
      1. nvidia-smi --query-gpu=compute_cap
      2. Python torch.cuda (if importable)
      3. Falls back to empty string (let CMake decide)
    """
    # Method 1: nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                caps = set()
                for line in r.stdout.strip().splitlines():
                    cc = line.strip().replace(".", "")  # "8.6" -> "86"
                    if cc.isdigit():
                        caps.add(cc)
                if caps:
                    return ";".join(sorted(caps))
        except Exception:
            pass

    # Method 2: torch (may already be installed)
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            caps = set()
            for i in range(_torch.cuda.device_count()):
                major, minor = _torch.cuda.get_device_capability(i)
                caps.add(f"{major}{minor}")
            if caps:
                return ";".join(sorted(caps))
    except Exception:
        pass

    return ""


def _gcc_major_version(gcc_path: str) -> int | None:
    """Return the major version of a gcc binary, or None."""
    try:
        r = subprocess.run([gcc_path, "-dumpversion"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return int(r.stdout.strip().split(".")[0])
    except Exception:
        pass
    return None


def _find_compatible_host_compiler() -> tuple[str, str]:
    """Resolve a gcc/g++ pair suitable as CUDA host compiler.

    Ubuntu 24.04+ glibc headers use _Float64x / _Float128 which require
    GCC >= 13.  CUDA toolkit packages can pull in GCC 12, and on some
    systems ``gcc`` points to 13 while ``g++`` still points to 12 (or
    vice-versa) because they are managed by separate alternatives entries.

    We therefore:
      1. Prefer an explicit versioned pair (gcc-13 / g++-13) — most reliable.
      2. Fall back to the default ``gcc`` / ``g++`` IF *both* are >= 13.
      3. If nothing suitable exists, try to ``apt-get install`` gcc-13.
      4. Ultimate fallback: return the system default even if too old
         (caller should still try; the build will produce a clear error).

    Always returns (gcc_path, g++_path) so the caller can be explicit.
    """
    MIN_GCC = 13

    # ---- diagnostics (printed so remote failures are debuggable) ----
    default_gcc = shutil.which("gcc")
    default_gxx = shutil.which("g++")
    gcc_ver = _gcc_major_version(default_gcc) if default_gcc else None
    gxx_ver = _gcc_major_version(default_gxx) if default_gxx else None
    print(f"  Default compilers: gcc={default_gcc} (v{gcc_ver})  "
          f"g++={default_gxx} (v{gxx_ver})")

    # ---- 1. Prefer explicit versioned pair (most reliable) ----
    for v in range(MIN_GCC, MIN_GCC + 5):
        gcc = shutil.which(f"gcc-{v}")
        gxx = shutil.which(f"g++-{v}")
        if gcc and gxx:
            print(f"  Using explicit gcc-{v} / g++-{v}")
            return (gcc, gxx)

    # ---- 2. Default pair, but ONLY if both are new enough ----
    if (default_gcc and default_gxx
            and gcc_ver and gcc_ver >= MIN_GCC
            and gxx_ver and gxx_ver >= MIN_GCC):
        print(f"  Default gcc/g++ are both >= {MIN_GCC} — using them")
        return (default_gcc, default_gxx)

    # ---- 3. Try to install gcc-13 ----
    if platform.system() == "Linux" and shutil.which("apt-get"):
        print(f"  No GCC >= {MIN_GCC} pair found — installing gcc-{MIN_GCC} / g++-{MIN_GCC}...")
        try:
            _run_sudo(["apt-get", "install", "-y", "-qq",
                        f"gcc-{MIN_GCC}", f"g++-{MIN_GCC}"])
            gcc = shutil.which(f"gcc-{MIN_GCC}")
            gxx = shutil.which(f"g++-{MIN_GCC}")
            if gcc and gxx:
                return (gcc, gxx)
        except Exception as e:
            print(f"  Failed to install gcc-{MIN_GCC}: {e}")

    # ---- 4. Fallback: return whatever we have (build will likely fail) ----
    if default_gcc and default_gxx:
        print(f"  WARNING: Using gcc={gcc_ver} / g++={gxx_ver} — "
              f"build may fail with _Float128 errors")
        return (default_gcc, default_gxx)

    # Nothing at all — should not happen after ensure_system_prerequisites
    print("  ERROR: No gcc/g++ found on system!")
    return ("gcc", "g++")  # last resort, let cmake error out clearly


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
    nvcc_path = ensure_cuda_toolkit()  # returns path or None
    if not nvcc_path:
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

    print(f"  Building llama.cpp with CUDA...")
    build_dir = llama_dir / "build"
    jobs = str(os.cpu_count() or 4)

    if not nvcc_path:
        print("  ERROR: CUDA not available — skipping llama.cpp (GPU benchmarks need CUDA).")
        return False

    # Clean stale build
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # --- Resolve compilers BEFORE cmake ---
    # We ALWAYS set compilers explicitly (env vars + cmake flags) because:
    #   1. nvcc uses g++ as host compiler — on Ubuntu 24.04, ``gcc`` can be
    #      v13 while ``g++`` is still v12 (CUDA toolkit pulls in g++-12).
    #   2. CMake's CUDA compiler-ID test (enable_language(CUDA)) runs nvcc
    #      which picks up the system default g++ — not CMAKE_CXX_COMPILER.
    #   3. CUDAHOSTCXX env var is respected by nvcc *and* by CMake's CUDA
    #      module during compiler identification — it's the most reliable
    #      way to control the host compiler at every stage.
    gcc, gxx = _find_compatible_host_compiler()
    print(f"  Host compilers: CC={gcc}  CXX={gxx}")
    print(f"  CUDA compiler:  {nvcc_path}")

    # Environment variables — belt-and-suspenders.  These are checked by
    # CMake *and* by nvcc before any -D flags or CMakeLists.txt logic.
    os.environ["CC"] = gcc
    os.environ["CXX"] = gxx
    os.environ["CUDAHOSTCXX"] = gxx   # nvcc host compiler
    os.environ["CUDACXX"] = nvcc_path  # nvcc itself

    # Let llama.cpp's own CMake logic pick CUDA architectures — it already
    # checks CUDAToolkit_VERSION and only adds arches that nvcc supports.
    configure_cmd = [
        cmake, "-B", str(build_dir), "-S", str(llama_dir),
        "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
        # Explicit compilers — matches the env vars above.
        f"-DCMAKE_C_COMPILER={gcc}",
        f"-DCMAKE_CXX_COMPILER={gxx}",
        f"-DCMAKE_CUDA_COMPILER={nvcc_path}",
        f"-DCMAKE_CUDA_HOST_COMPILER={gxx}",
    ]

    try:
        run(configure_cmd)
        run([cmake, "--build", str(build_dir), "--config", "Release", "-j", jobs])
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: llama.cpp build failed: {e}")
        return False

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


# ─── GPU state tracking (detect GPU swaps between installs) ──────────────────

def _current_gpu_state() -> dict:
    """Snapshot of GPUs currently in the machine."""
    state: dict = {"gpus": []}
    if not shutil.which("nvidia-smi"):
        return state
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    state["gpus"].append({
                        "name": parts[0],
                        "compute_cap": parts[1],
                        "vram_mib": parts[2],
                    })
    except Exception:
        pass
    return state


def _load_gpu_state() -> dict | None:
    p = Path(GPU_STATE_FILE)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_gpu_state(state: dict) -> None:
    Path(GPU_STATE_FILE).write_text(json.dumps(state, indent=2) + "\n")


def _gpu_changed(old: dict | None, new: dict) -> bool:
    """Return True if the GPU(s) changed since last install."""
    if old is None:
        return False  # First run — nothing to compare
    return old.get("gpus") != new.get("gpus")


def _clean_llama_build() -> None:
    """Remove llama.cpp/build so it can be rebuilt for the new GPU."""
    build_dir = Path("llama.cpp/build")
    if build_dir.exists():
        print("  Removing stale llama.cpp/build...")
        shutil.rmtree(build_dir)


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

    # 1. System info + GPU-swap detection
    print("[1/8] System info")
    print(f"  Python: {platform.python_version()}")
    print(f"  OS:     {platform.system()} {platform.release()}")
    gpu = get_gpu_name()
    print(f"  GPU:    {gpu}")

    gpu_state_now = _current_gpu_state()
    gpu_state_prev = _load_gpu_state()
    gpu_swapped = _gpu_changed(gpu_state_prev, gpu_state_now)
    if gpu_swapped:
        print()
        print("  *** GPU CHANGE DETECTED ***")
        if gpu_state_prev and gpu_state_prev.get("gpus"):
            old_names = ", ".join(g["name"] for g in gpu_state_prev["gpus"])
            print(f"  Previous: {old_names}")
        new_names = ", ".join(g["name"] for g in gpu_state_now.get("gpus", []))
        print(f"  Current:  {new_names}")
        print("  Will clean llama.cpp build and re-detect CUDA requirements.")
        _clean_llama_build()
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

    # Save GPU state for future swap detection
    _save_gpu_state(gpu_state_now)

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
