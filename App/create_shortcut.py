"""
create_shortcut.py — Creates a desktop shortcut for the application.

Run once after installation:
    python create_shortcut.py

Uses VBScript (available on all Windows systems — no extra packages needed).
"""

from __future__ import annotations
import ctypes
import ctypes.wintypes
import os
import sys
import tempfile
from pathlib import Path

APP_DIR  = Path(__file__).parent.resolve()
PROJECT_DIR = APP_DIR.parent
LAUNCHER = PROJECT_DIR / "launcher.bat"
ICON     = APP_DIR / "assets" / "icon.ico"


def _get_desktop() -> Path:
    """Return the real Desktop path (respects OneDrive redirection)."""
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    CSIDL_DESKTOPDIRECTORY = 0x10
    ctypes.windll.shell32.SHGetFolderPathW(
        None, CSIDL_DESKTOPDIRECTORY, None, 0, buf)
    p = Path(buf.value)
    return p if p.is_dir() else Path.home() / "Desktop"


def create_shortcut() -> None:
    if sys.platform != "win32":
        print("Shortcut creation is only supported on Windows.")
        return

    desktop  = _get_desktop()
    lnk_path = desktop / "NLP for PDFs.lnk"
    icon_str = str(ICON) if ICON.exists() else ""

    vbs = (
        'Set oWS = WScript.CreateObject("WScript.Shell")\n'
        f'sLinkFile = "{lnk_path}"\n'
        'Set oLink = oWS.CreateShortcut(sLinkFile)\n'
        f'oLink.TargetPath = "{LAUNCHER}"\n'
        f'oLink.WorkingDirectory = "{PROJECT_DIR}"\n'
    )
    if icon_str:
        vbs += f'oLink.IconLocation = "{icon_str}"\n'
    vbs += (
        'oLink.Description = "NLP for PDFs"\n'
        'oLink.Save\n'
    )

    tf = Path(tempfile.mktemp(suffix=".vbs"))
    tf.write_text(vbs, encoding="utf-8")
    try:
        ret = os.system(f'cscript //nologo "{tf}"')
        if ret == 0:
            print(f"Shortcut created on Desktop: {lnk_path}")
        else:
            print(f"Shortcut creation failed (exit code {ret}).")
    finally:
        tf.unlink(missing_ok=True)


if __name__ == "__main__":
    create_shortcut()
