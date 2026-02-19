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
    """Map a CUDA driver version like '12.4' to a PyTorch wheel index suffix."""
    if ver_string is None:
        print("  WARNING: Could not detect CUDA. Defaulting to cu124.")
        return "cu124"

    major, minor = (int(x) for x in ver_string.split(".")[:2])
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

    print(f"  WARNING: CUDA {ver_string} may be too old. Trying cu118.")
    return "cu118"


def _best_cuda_suffix(cuda_ver, gpu_sm):
    """Pick the PyTorch CUDA wheel index for this GPU.

    Blackwell+ (sm >= 100) forces cu128 because only those wheels
    contain sm_100 / sm_120 kernels.
    """
    if gpu_sm >= 100:
        print(f"  Blackwell+ GPU (sm_{gpu_sm}) — forcing cu128 index.")
        return "cu128"
    return cuda_version_to_index(cuda_ver)


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


GPU_STATE_FILE = ".gpu_state.json"  # tracks GPU/toolchain info across installs


def _get_max_gpu_sm() -> int:
    """Return the highest compute capability (as condensed int) across all GPUs.

    E.g. RTX 5090 (sm_12.0) → 120, RTX 4090 (sm_8.9) → 89, A100 (sm_8.0) → 80.
    Returns 0 if detection fails.
    """
    archs = _detect_cuda_architectures()
    if archs:
        sm_list = [int(x) for x in archs.split(";") if x.isdigit()]
        return max(sm_list) if sm_list else 0
    return 0


def install_pytorch(venv_dir, cuda_suffix):
    """Clean-install latest PyTorch from the correct CUDA wheel index.

    Always removes old torch packages first to avoid stale-kernel mismatches
    after a GPU swap.  Tries stable wheels first, falls back to nightly.
    """
    py = venv_python(venv_dir)
    pip_install(venv_dir, ["install", "--upgrade", "pip", "setuptools", "wheel", "-q"])

    # Clean existing PyTorch (avoids leftover builds compiled for a different arch)
    print("  Removing old PyTorch packages (if any)...")
    run([py, "-m", "pip", "uninstall", "-y",
         "torch", "torchvision", "torchaudio"], check=False)

    # Try stable wheels
    index = f"https://download.pytorch.org/whl/{cuda_suffix}"
    print(f"  Installing latest PyTorch ({cuda_suffix})...")
    r = subprocess.run(
        [py, "-m", "pip", "install",
         "torch", "torchvision", "torchaudio",
         "--index-url", index, "-q"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        return True

    # Stable failed — try nightly (common for brand-new architectures)
    print(f"  Stable {cuda_suffix} wheels not available — trying nightly...")
    nightly_index = f"https://download.pytorch.org/whl/nightly/{cuda_suffix}"
    r = subprocess.run(
        [py, "-m", "pip", "install", "--pre",
         "torch", "torchvision", "torchaudio",
         "--index-url", nightly_index, "-q"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        return True

    print(f"  ERROR: Could not install PyTorch for {cuda_suffix}.")
    print(f"  {r.stderr[-300:] if r.stderr else '(no details)'}")
    return False


def verify_pytorch(venv_dir):
    """Verify PyTorch detects the GPU and can run kernels on it."""
    py = venv_python(venv_dir)
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
        "print('  GPU compatibility: OK')"
    )
    r = run([py, "-c", code], check=False)
    if r.returncode != 0:
        print("  WARNING: PyTorch cannot use this GPU.")
        print("           Re-run install.py to fix, or check driver/CUDA.")
        return False
    return True


def install_python_deps(venv_dir):
    req_file = Path(__file__).resolve().parent / "requirements.txt"
    if not req_file.exists():
        print("  ERROR: requirements.txt not found.")
        return
    pip_install(venv_dir, ["install", "--upgrade", "-r", str(req_file), "-q"])


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

    target_ver = _recommended_cuda_for_sm(max_sm) if max_sm else "12.4"

    # Try 1: package manager install
    installed_distro = False
    if shutil.which("apt-get"):
        # On Debian/Ubuntu, distro nvidia-cuda-toolkit is often older.
        # Prefer NVIDIA's official repo first to reduce nvcc/header mismatches.
        print(f"  Trying NVIDIA official repo first (target CUDA {target_ver})...")
        if _install_cuda_via_nvidia_repo(target_ver):
            nvcc_path = _find_suitable_nvcc(min_cuda)
            if nvcc_path:
                return _accept_nvcc(nvcc_path)

        print(f"  Falling back to distro nvidia-cuda-toolkit...")
        try:
            _run_sudo(["apt-get", "update", "-qq"])
            _run_sudo(["apt-get", "install", "-y", "-qq", "nvidia-cuda-toolkit"])
            installed_distro = True
        except Exception as e:
            print(f"  Distro CUDA toolkit install failed: {e}")
    elif shutil.which("dnf"):
        try:
            _run_sudo(["dnf", "install", "-y", "cuda-compiler"])
            installed_distro = True
        except Exception as e:
            print(f"  DNF CUDA install failed: {e}")
    elif shutil.which("yum"):
        try:
            _run_sudo(["yum", "install", "-y", "cuda-compiler"])
            installed_distro = True
        except Exception as e:
            print(f"  YUM CUDA install failed: {e}")
    elif shutil.which("pacman"):
        try:
            _run_sudo(["pacman", "-Sy", "--noconfirm", "cuda"])
            installed_distro = True
        except Exception as e:
            print(f"  Pacman CUDA install failed: {e}")
    elif shutil.which("zypper"):
        try:
            _run_sudo(["zypper", "install", "-y", "cuda-compiler"])
            installed_distro = True
        except Exception as e:
            print(f"  Zypper CUDA install failed: {e}")

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


def _glibc_version() -> tuple[int, int] | None:
    """Return glibc version as (major, minor), or None if unavailable."""
    try:
        _, ver = platform.libc_ver()
        m = re.match(r"(\d+)\.(\d+)", ver or "")
        if m:
            return (int(m.group(1)), int(m.group(2)))
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


def build_llama_cpp(venv_dir, nvcc_path: str | None = None):
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
    if not nvcc_path:
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
    gxx_ver = _gcc_major_version(gxx)
    glibc_ver = _glibc_version()

    # Guardrail for the common Ubuntu 24.04+/glibc 2.38+ failure mode:
    # old g++ with newer glibc headers produces _Float64x/_Float128 errors.
    if glibc_ver and glibc_ver >= (2, 38) and (not gxx_ver or gxx_ver < 13):
        print("  ERROR: Detected glibc >= 2.38 with g++ < 13.")
        print("         This commonly fails with _Float64x/_Float128 errors during CUDA compiler detection.")
        print("         Install newer host compilers and retry:")
        print("           sudo apt-get update && sudo apt-get install -y gcc-13 g++-13")
        print("         Then re-run: python install.py")
        return False

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
    """Snapshot of GPU + CUDA toolchain state for drift detection."""
    nvcc = shutil.which("nvcc")
    nvcc_ver = _nvcc_version(nvcc) if nvcc else None
    gcc = shutil.which("gcc")
    gxx = shutil.which("g++")

    state: dict = {
        "gpus": [],
        "driver_version": None,
        "cuda_driver_version": detect_cuda_version(),
        "nvcc_path": nvcc,
        "nvcc_version": f"{nvcc_ver[0]}.{nvcc_ver[1]}" if nvcc_ver else None,
        "gcc_version": _gcc_major_version(gcc) if gcc else None,
        "gxx_version": _gcc_major_version(gxx) if gxx else None,
    }
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

        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if drv.returncode == 0 and drv.stdout.strip():
            state["driver_version"] = drv.stdout.strip().splitlines()[0].strip()
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
    """Return True if GPU/runtime/toolchain changed since last install."""
    if old is None:
        return False  # First run — nothing to compare
    keys = [
        "gpus",
        "driver_version",
        "cuda_driver_version",
        "nvcc_version",
        "gcc_version",
        "gxx_version",
    ]
    return any(old.get(k) != new.get(k) for k in keys)


def _gpu_change_reasons(old: dict | None, new: dict) -> list[str]:
    """Return human-readable reasons describing why repair/rebuild is needed."""
    if old is None:
        return []

    reasons: list[str] = []
    if old.get("gpus") != new.get("gpus"):
        reasons.append("GPU hardware (name/compute capability/VRAM)")
    if old.get("driver_version") != new.get("driver_version"):
        reasons.append("NVIDIA driver version")
    if old.get("cuda_driver_version") != new.get("cuda_driver_version"):
        reasons.append("CUDA driver capability")
    if old.get("nvcc_version") != new.get("nvcc_version"):
        reasons.append("CUDA toolkit (nvcc)")
    if old.get("gcc_version") != new.get("gcc_version") or old.get("gxx_version") != new.get("gxx_version"):
        reasons.append("host compiler (gcc/g++)")
    return reasons


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
    change_reasons = _gpu_change_reasons(gpu_state_prev, gpu_state_now)
    if gpu_swapped:
        print()
        print("  *** ENVIRONMENT CHANGE DETECTED ***")
        if gpu_state_prev and gpu_state_prev.get("gpus"):
            old_names = ", ".join(g["name"] for g in gpu_state_prev["gpus"])
            print(f"  Previous: {old_names}")
        new_names = ", ".join(g["name"] for g in gpu_state_now.get("gpus", []))
        print(f"  Current:  {new_names}")
        if change_reasons:
            print("  Reasons:")
            for reason in change_reasons:
                print(f"    - {reason}")
        print("  Will clean llama.cpp build and re-detect CUDA requirements.")
        _clean_llama_build()
    print()

    # 2. Detect GPU → pick CUDA suffix
    print("[2/8] Detecting GPU and CUDA environment...")
    cuda_ver = detect_cuda_version()
    gpu_sm = _get_max_gpu_sm()
    cuda_suffix = _best_cuda_suffix(cuda_ver, gpu_sm)
    print(f"  CUDA driver: {cuda_ver or 'not detected'}")
    print(f"  GPU arch:    {'sm_' + str(gpu_sm) if gpu_sm else 'unknown'}")
    print(f"  Wheel index: {cuda_suffix}")

    nvcc_path = ensure_cuda_toolkit()
    if nvcc_path:
        print(f"  CUDA toolkit: {nvcc_path}")
    else:
        print("  WARNING: nvcc not found — llama.cpp CUDA build may fail.")
    print()

    # 3. Virtual environment
    print("[3/8] Setting up virtual environment...")
    create_venv(venv_dir)
    print()

    # 4. PyTorch (clean install — always latest for this GPU)
    print("[4/8] Installing PyTorch...")
    pytorch_ok = install_pytorch(venv_dir, cuda_suffix)
    if pytorch_ok:
        print("  Verifying...")
        pytorch_ok = verify_pytorch(venv_dir)
    if not pytorch_ok:
        print("  WARNING: PyTorch not working for this GPU.")
        print("           PyTorch-based benchmarks (1-4, 6-10) will fail.")
        print("           Check https://pytorch.org/get-started/locally/")
    print()

    # 5. Python dependencies
    print("[5/8] Installing Python dependencies...")
    install_python_deps(venv_dir)
    print()

    # 6. llama.cpp
    if args.skip_llama:
        print("[6/8] Skipping llama.cpp (--skip-llama)")
        if gpu_swapped:
            print("  NOTE: Environment changed; run install.py without --skip-llama to rebuild llama.cpp for this GPU.")
    else:
        print("[6/8] Building llama.cpp with CUDA support...")
        build_llama_cpp(venv_dir, nvcc_path=nvcc_path)
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
    gpu_state_now["gpu_sm"] = gpu_sm
    gpu_state_now["pytorch_cuda_suffix"] = cuda_suffix
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
