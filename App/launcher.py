"""
launcher.py — First-run setup and application launcher.

Checks that all required packages are installed, installs missing ones,
then launches app.py.  Run via launcher.bat (double-click) or directly:
    python launcher.py
    python launcher.py --cpu        force CPU-only torch
    python launcher.py --skip-setup skip dependency checks (faster cold start)
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import os
import re
from pathlib import Path

APP_DIR     = Path(__file__).parent.resolve()
MARKER      = APP_DIR / ".setup_complete"
ASSETS_DIR  = APP_DIR / "assets"
ICON_PATH   = ASSETS_DIR / "icon.ico"
SAVED_DIR   = APP_DIR / "saved_kgs"
MIN_TORCH_VERSION = (2, 6)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _pip(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "pip", *args],
                          capture_output=False).returncode


def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _torch_ok() -> bool:
    try:
        import torch
        m = re.match(r"^(\d+)\.(\d+)", torch.__version__)
        if not m:
            return False
        current = (int(m.group(1)), int(m.group(2)))
        if current < MIN_TORCH_VERSION:
            print(
                f"  Found torch {torch.__version__}; need >= "
                f"{MIN_TORCH_VERSION[0]}.{MIN_TORCH_VERSION[1]} for security fixes.",
                flush=True,
            )
            return False
        return True
    except ImportError:
        return False


def _warn_if_python_too_new_for_cuda_torch() -> None:
    """
    Warn when running on a Python version that commonly lacks CUDA wheels.
    This prevents silent fallback to CPU-only torch.
    """
    if sys.version_info < (3, 13):
        return
    try:
        import torch
        is_cpu_build = "+cpu" in torch.__version__ or torch.version.cuda is None
    except Exception:
        is_cpu_build = True
    if not is_cpu_build:
        return
    print(
        "\n  WARNING: Python 3.13 detected with CPU-only PyTorch.\n"
        "  Your NVIDIA GPU is available, but this environment may not have CUDA torch wheels.\n"
        "  Recommended fix:\n"
        "    1) Install/use Python 3.12\n"
        "    2) Create a fresh venv\n"
        "    3) Install torch CUDA build, e.g.:\n"
        "       pip install torch --index-url https://download.pytorch.org/whl/cu121\n",
        flush=True,
    )


def _python_may_block_cuda_wheels() -> bool:
    """True when this Python version is likely too new for CUDA torch wheels."""
    return sys.version_info >= (3, 13)


def _install_torch(cpu_only: bool) -> None:
    print("\n  Installing PyTorch ...", flush=True)
    if cpu_only:
        print("  (CPU-only build requested)", flush=True)
        _pip("install", "torch", "--index-url",
             "https://download.pytorch.org/whl/cpu")
        return

    if sys.version_info >= (3, 13):
        print(
            "  Note: Python 3.13 may not have CUDA PyTorch wheels yet; "
            "GPU acceleration may require Python 3.12.",
            flush=True,
        )

    # Detect CUDA version from nvidia-smi
    cuda_ver = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi available; pick a known-good index url
            cuda_ver = "cu121"   # CUDA 12.1 index works for 12.x drivers
    except Exception:
        pass

    if cuda_ver:
        print(f"  GPU detected — installing torch with {cuda_ver} support.",
              flush=True)
        _pip("install", "torch",
             "--index-url",
             f"https://download.pytorch.org/whl/{cuda_ver}")
    else:
        print("  No GPU detected — installing CPU torch.", flush=True)
        _pip("install", "torch", "--index-url",
             "https://download.pytorch.org/whl/cpu")


def _install_requirements() -> None:
    req = APP_DIR / "requirements.txt"
    if req.exists():
        print("  Installing Python packages from requirements.txt …", flush=True)
        _pip("install", "--upgrade", "-r", str(req))


def _download_nltk() -> None:
    """Download WordNet silently if not already present."""
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Icon generation
# ---------------------------------------------------------------------------

def _create_icon() -> None:
    """Generate a simple ICO file using Pillow if it doesn't exist."""
    if ICON_PATH.exists():
        return
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageFont
        sizes = [16, 32, 48, 64, 128, 256]
        frames: list[Image.Image] = []
        for s in sizes:
            img = Image.new("RGBA", (s, s), (28, 28, 36, 255))
            draw = ImageDraw.Draw(img)
            m = s // 8
            # Outer circle
            draw.ellipse([m, m, s - m, s - m],
                         fill=(74, 144, 226, 255),
                         outline=(255, 255, 255, 180),
                         width=max(1, s // 24))
            # Inner "KG" dots
            cx, cy = s // 2, s // 2
            r = max(1, s // 10)
            draw.ellipse([cx - r - s//6, cy - r, cx - s//6 + r, cy + r],
                         fill=(255, 255, 255, 220))
            draw.ellipse([cx + s//6 - r, cy - r, cx + s//6 + r, cy + r],
                         fill=(255, 220, 80, 220))
            frames.append(img)

        frames[0].save(
            ICON_PATH, format="ICO",
            sizes=[(s, s) for s in sizes],
            append_images=frames[1:],
        )
    except Exception as e:
        print(f"  Note: could not generate icon ({e}). Default icon will be used.",
              flush=True)


# ---------------------------------------------------------------------------
# Setup entry-point
# ---------------------------------------------------------------------------

def setup(cpu_only: bool = False) -> None:
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    print("  Checking dependencies …", flush=True)

    if not _torch_ok():
        _install_torch(cpu_only)

    _install_requirements()
    _download_nltk()
    _create_icon()

    MARKER.write_text("ok")
    print("  Setup complete.\n", flush=True)


_REQUIRED_IMPORTS = [
    "PyQt6",
    "fitz",
    "pdfplumber",
    "networkx",
    "sentence_transformers",
    "transformers",
    "PIL",
    "nltk",
]


def _deps_look_ok() -> bool:
    return all(_try_import(mod) for mod in _REQUIRED_IMPORTS) and _torch_ok()


def launch(detach: bool = False) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    app_script = APP_DIR / "app.py"
    if detach and os.name == "nt":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            [sys.executable, str(app_script)],
            creationflags=creationflags,
            close_fds=True,
        )
        sys.exit(0)
    result = subprocess.run([sys.executable, str(app_script)])
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLP for PDFs launcher / first-run setup")
    parser.add_argument("--cpu",         action="store_true",
                        help="Install CPU-only PyTorch (no CUDA)")
    parser.add_argument("--skip-setup",  action="store_true",
                        help="Skip dependency checks and launch immediately")
    parser.add_argument("--setup-only",  action="store_true",
                        help="Run setup without launching the app")
    parser.add_argument("--detach", action="store_true",
                        help="Launch app and exit this terminal immediately")
    args = parser.parse_args()
    _warn_if_python_too_new_for_cuda_torch()

    # Keep this console open when Python is likely the root cause of
    # CPU-only torch, so users can read the remediation steps.
    if args.detach and _python_may_block_cuda_wheels():
        print(
            "  Detach disabled because this Python version may not support "
            "CUDA PyTorch wheels yet.\n"
            "  Keeping this window open so you can review the fix steps above.\n",
            flush=True,
        )
        args.detach = False

    if not args.skip_setup:
        if not MARKER.exists() or args.cpu:
            print("\n  First-run setup …", flush=True)
            setup(cpu_only=args.cpu)
        else:
            # Self-heal if core dependencies are missing.
            if not _deps_look_ok():
                print("\n  Detected missing packages — re-running setup …",
                      flush=True)
                setup(cpu_only=args.cpu)

    if not args.setup_only:
        launch(detach=args.detach)


if __name__ == "__main__":
    main()
