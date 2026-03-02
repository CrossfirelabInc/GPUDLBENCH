#!/usr/bin/env python3
"""
GPU DL Benchmark Suite — Uninstaller

Removes everything installed by install.py without touching:
  - NVIDIA drivers
  - Your source code / benchmark scripts
  - Your results (unless --include-results is passed)
  - CUDA toolkit (unless --include-cuda is passed)

Usage:
    python uninstall.py                     # interactive confirmation
    python uninstall.py --force             # skip confirmation
    python uninstall.py --include-results   # also delete results/
    python uninstall.py --include-cuda      # also purge CUDA toolkit (Linux)
    python uninstall.py --dry-run           # show what would be deleted
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Items to remove ───────────────────────────────────────────────────────────
# Each entry: (path relative to project root, description)
_CLEANUP_TARGETS = [
    ("venv",              "Python virtual environment"),
    ("llama.cpp",         "llama.cpp repository & build"),
    ("data/models",       "Downloaded GGUF model files (~50+ GB)"),
    ("data/hf_cache",     "HuggingFace cache"),
    ("temp",              "Temporary files"),
    (".credentials",      "HuggingFace token file"),
    (".gpu_state.json",   "GPU state cache (created by install.py)"),
]

_OPTIONAL_TARGETS = [
    ("results",           "All benchmark results & charts"),
]


def _find_pycache(root: Path) -> list[Path]:
    """Recursively find all __pycache__ directories."""
    return sorted(root.rglob("__pycache__"))


def _sizeof_fmt(num_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} PB"


def _dir_size(path: Path) -> int:
    """Total size of a directory in bytes."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _remove(path: Path, dry_run: bool = False) -> bool:
    """Remove a file or directory. Returns True if removed."""
    if not path.exists():
        return False
    if dry_run:
        size = _dir_size(path)
        kind = "dir " if path.is_dir() else "file"
        print(f"  [DRY RUN] Would remove {kind}: {path}  ({_sizeof_fmt(size)})")
        return True
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except OSError as e:
        print(f"  ⚠ Could not remove {path}: {e}")
        return False


def _find_cuda_packages() -> list[str]:
    """Find installed CUDA toolkit packages (Linux apt-based only)."""
    if platform.system() != "Linux" or not shutil.which("apt-get"):
        print("  ℹ  CUDA purge is only supported on apt-based Linux distributions.")
        return []

    pkgs_to_check = ["nvidia-cuda-toolkit", "nvidia-cuda-dev"]

    # Also find NVIDIA-repo cuda-toolkit-X-Y packages
    try:
        r = subprocess.run(["dpkg", "-l", "cuda-toolkit-*"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if line.startswith("ii"):
                    pkgs_to_check.append(line.split()[1])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Filter to only installed packages
    installed: list[str] = []
    for pkg in pkgs_to_check:
        try:
            r = subprocess.run(["dpkg", "-s", pkg],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and "install ok installed" in r.stdout:
                installed.append(pkg)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return installed


def _purge_cuda_packages(packages: list[str], *, dry_run: bool = False) -> None:
    """Purge CUDA toolkit packages via apt (requires sudo)."""
    if not packages:
        return
    if dry_run:
        print(f"  [DRY RUN] Would purge: {' '.join(packages)}")
        return
    print(f"\n  Purging CUDA toolkit: {' '.join(packages)}")
    try:
        subprocess.run(["sudo", "apt-get", "remove", "--purge", "-y"] + packages,
                       check=True)
        subprocess.run(["sudo", "apt-get", "autoremove", "-y", "-qq"],
                       check=True)
        # Clean stale nvcc from /usr/bin
        stale_nvcc = Path("/usr/bin/nvcc")
        if stale_nvcc.exists() and not stale_nvcc.is_symlink():
            subprocess.run(["sudo", "rm", "-f", str(stale_nvcc)])
        print("  ✓ CUDA toolkit packages purged")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠ Failed to purge CUDA packages: {e}")
    except FileNotFoundError:
        print("  ⚠ sudo not found — cannot purge CUDA packages")


def uninstall(*, force: bool = False, include_results: bool = False,
              include_cuda: bool = False, dry_run: bool = False) -> None:
    """Main uninstall logic."""
    print("=" * 60)
    print("GPU DL Benchmark Suite — Uninstaller")
    print("=" * 60)

    # Build target list
    targets = list(_CLEANUP_TARGETS)
    if include_results:
        targets.extend(_OPTIONAL_TARGETS)

    # Resolve paths and check existence
    found: list[tuple[Path, str]] = []
    for rel, desc in targets:
        p = PROJECT_ROOT / rel
        if p.exists():
            found.append((p, desc))

    # Find __pycache__ dirs
    pycaches = _find_pycache(PROJECT_ROOT)

    if not found and not pycaches:
        print("\n  Nothing to clean — already uninstalled.")
        return

    # Show what will be removed
    print("\nThe following will be PERMANENTLY DELETED:\n")
    total_size = 0
    for p, desc in found:
        size = _dir_size(p)
        total_size += size
        marker = "📁" if p.is_dir() else "📄"
        print(f"  {marker} {str(p.relative_to(PROJECT_ROOT)):<25s}  {_sizeof_fmt(size):>10s}  — {desc}")

    if pycaches:
        pc_size = sum(_dir_size(pc) for pc in pycaches)
        total_size += pc_size
        print(f"  📁 __pycache__ (×{len(pycaches)})          {_sizeof_fmt(pc_size):>10s}  — Python bytecode cache")

    print(f"\n  Total: {_sizeof_fmt(total_size)}")

    if not include_results:
        print("\n  ℹ  results/ is preserved. Use --include-results to also delete it.")
    cuda_packages: list[str] = []
    if include_cuda:
        cuda_packages = _find_cuda_packages()
        if cuda_packages:
            print(f"\n  \u26a0  CUDA toolkit packages to purge: {' '.join(cuda_packages)}")
        else:
            print("\n  \u2139  No CUDA toolkit packages found to purge.")
    print("\n  ⚠  This does NOT touch NVIDIA drivers or your benchmark source code.")

    if dry_run:
        print("\n  [DRY RUN] No files will be deleted.\n")
        for p, _ in found:
            _remove(p, dry_run=True)
        for pc in pycaches:
            _remove(pc, dry_run=True)
        return

    # Confirmation
    if not force:
        try:
            answer = input("\n  Proceed? (yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return
        if answer not in ("yes", "y"):
            print("  Cancelled.")
            return

    # Execute removal
    print()
    removed = 0
    for p, desc in found:
        size = _dir_size(p)
        if _remove(p):
            print(f"  ✓ Removed {p.relative_to(PROJECT_ROOT)}  ({_sizeof_fmt(size)})")
            removed += 1

    for pc in pycaches:
        if _remove(pc):
            removed += 1

    if pycaches:
        print(f"  \u2713 Removed {len(pycaches)} __pycache__ directories")

    # Purge CUDA toolkit if requested
    if include_cuda and cuda_packages:
        _purge_cuda_packages(cuda_packages, dry_run=dry_run)

    # Clean up empty data/ directory
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists() and not any(data_dir.iterdir()):
        data_dir.rmdir()
        print("  ✓ Removed empty data/")

    print(f"\n  Done — {removed} item(s) cleaned up.")
    print("  To reinstall: python install.py\n")


def main():
    parser = argparse.ArgumentParser(
        description="Uninstall GPU DL Benchmark Suite — removes venv, models, "
                    "llama.cpp, caches, and temporary files.",
        epilog="WARNING: This permanently deletes downloaded models (~50+ GB), "
               "the virtual environment, and llama.cpp build. "
               "Your benchmark source code and results are preserved by default."
    )
    parser.add_argument("--force", "-f", action="store_true",
                        help="Skip interactive confirmation")
    parser.add_argument("--include-results", action="store_true",
                        help="Also delete results/ directory (benchmark outputs & charts)")
    parser.add_argument("--include-cuda", action="store_true",
                        help="Also purge CUDA toolkit packages installed by install.py "
                             "(Linux only, requires sudo)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be deleted without actually removing anything")
    args = parser.parse_args()

    uninstall(force=args.force, include_results=args.include_results,
              include_cuda=args.include_cuda, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
