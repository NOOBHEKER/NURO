"""Per-terminal JetBrains Mono font loader for NURO (Windows Console).

Uses the JetBrains Mono Nerd Font Mono TTF pre-bundled inside `assets/fonts/`
(or PyInstaller `_MEIPASS/assets/fonts/`) and applies it ONLY to the active
NURO console window via Win32 `AddFontResourceExW` + `SetCurrentConsoleFontEx`.
Safe no-op on non-Windows platforms, piped stdout, or terminals that manage
their own font profile (e.g., Windows Terminal).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def get_bundled_font_path() -> Optional[Path]:
    """Locate the pre-bundled JetBrains Mono Regular TTF inside the NURO folder."""
    candidates = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "assets" / "fonts" / "JetBrainsMonoNerdFontMono-Regular.ttf")
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates.append(repo_root / "assets" / "fonts" / "JetBrainsMonoNerdFontMono-Regular.ttf")
    candidates.append(repo_root / "assets" / "fonts" / "JetBrainsMono-Regular.ttf")
    for p in candidates:
        if p.exists():
            return p
    return None


def apply_bundled_console_font(preferred_height: int = 18) -> bool:
    """Apply bundled JetBrains Mono to the current Windows console window only.

    Returns True if SetCurrentConsoleFontEx succeeded, False otherwise.
    """
    if os.name != "nt" or not sys.stdout.isatty():
        return False

    try:
        import ctypes
        from ctypes import wintypes

        font_path = get_bundled_font_path()
        if font_path is not None:
            gdi32 = ctypes.windll.gdi32
            # FR_PRIVATE (0x10) loads font privately for this session without touching system settings
            gdi32.AddFontResourceExW(str(font_path), 0x10, 0)

        LF_FACESIZE = 32

        class COORD(ctypes.Structure):
            _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

        class CONSOLE_FONT_INFOEX(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.ULONG),
                ("nFont", wintypes.DWORD),
                ("dwFontSize", COORD),
                ("FontFamily", wintypes.UINT),
                ("FontWeight", wintypes.UINT),
                ("FaceName", wintypes.WCHAR * LF_FACESIZE),
            ]

        k32 = ctypes.windll.kernel32
        # Open active console screen buffer directly so redirection doesn't break lookup
        GENERIC_READ = 0x80000000
        GENERIC_WRITE = 0x40000000
        FILE_SHARE_READ = 0x00000001
        FILE_SHARE_WRITE = 0x00000002
        OPEN_EXISTING = 3
        h_out = k32.CreateFileW(
            "CONOUT$",
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None,
            OPEN_EXISTING,
            0,
            None,
        )
        if not h_out or h_out == wintypes.HANDLE(-1).value:
            h_out = k32.GetStdHandle(-11)

        cur = CONSOLE_FONT_INFOEX()
        cur.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)
        height = preferred_height
        if k32.GetCurrentConsoleFontEx(h_out, False, ctypes.byref(cur)) and cur.dwFontSize.Y > 0:
            # Preserve the user's comfortable font height (clamp sensible range)
            height = max(14, min(24, int(cur.dwFontSize.Y)))

        # Try exact bundled face name first, then standard JetBrains Mono fallback
        for face in ("JetBrainsMono NFM", "JetBrains Mono", "Cascadia Mono"):
            req = CONSOLE_FONT_INFOEX()
            req.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)
            req.nFont = 0
            req.dwFontSize.X = 0
            req.dwFontSize.Y = height
            req.FontFamily = 54  # FF_MODERN | FIXED_PITCH
            req.FontWeight = 400
            req.FaceName = face
            if k32.SetCurrentConsoleFontEx(h_out, False, ctypes.byref(req)):
                verify = CONSOLE_FONT_INFOEX()
                verify.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)
                if k32.GetCurrentConsoleFontEx(h_out, False, ctypes.byref(verify)):
                    if face.lower().split()[0] in verify.FaceName.lower():
                        return True
        return False
    except Exception:
        return False
