"""Shared helpers for safe, read-only-first ESP32-S3 board operations.

The scripts in this folder never touch a GPIO, never erase flash by accident,
and always verify what they read or write. They talk to the ROM bootloader
through esptool v5 (`python -m esptool`).

Board safety contract:
- Only the serial bootloader is used.
- `read-flash` is non-destructive.
- `write-flash` is only called by board_restore.py after an explicit
  `--confirm RESTORE` and a SHA-256 + MAC check.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from host.board import (  # noqa: E402
    DEFAULT_PORT,
    describe_ports,
    normalize_mac,
    parse_flash_id,
    resolve_port,
)

DEFAULT_BAUD = 460800
DEFAULT_FLASH_BYTES = 16 * 1024 * 1024  # ESP32-S3-WROOM-1-N16R8
BACKUP_ROOT = REPO_ROOT / "backups"


def esptool_cmd() -> List[str]:
    """Return the command prefix for esptool v5."""
    return [sys.executable, "-m", "esptool"]


def force_utf8_stdout() -> None:
    """Make Windows consoles safe for the board's Unicode boot banner.

    PowerShell 5.1 defaults redirected output to cp1252, which cannot encode
    characters such as the lightning bolt in the firmware banner. Reconfigure
    both streams instead of stripping the firmware's own output.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def run_esptool(
    args: Sequence[str],
    *,
    capture: bool = True,
    quiet: bool = False,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Run esptool and return the completed process.

    Raises RuntimeError with the combined output when the command fails, so a
    failed read never looks like a successful backup.
    """
    cmd = [*esptool_cmd(), *args]
    if not quiet:
        print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
        timeout=timeout,
    )
    if capture and not quiet:
        if proc.stdout:
            print(proc.stdout, end="", flush=True)
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr, flush=True)
    if proc.returncode != 0:
        detail = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(
            f"esptool failed with exit code {proc.returncode}:\n{detail}"
        )
    return proc


def flash_size_bytes(label: Optional[str]) -> int:
    """Convert '16MB' / '8 MB' to bytes; fall back to the N16R8 size."""
    if not label:
        return DEFAULT_FLASH_BYTES
    match = re.match(r"([0-9]+)\s*([KMG])B", label.strip(), re.IGNORECASE)
    if not match:
        return DEFAULT_FLASH_BYTES
    value = int(match.group(1))
    unit = match.group(2).upper()
    factor = {"K": 1024, "M": 1024**2, "G": 1024**3}[unit]
    return value * factor


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> Dict[str, str]:
    """Return branch, commit and describe tag without failing outside git."""
    out: Dict[str, str] = {}
    for key, args in {
        "branch": ["rev-parse", "--abbrev-ref", "HEAD"],
        "commit": ["rev-parse", "HEAD"],
        "describe": ["describe", "--tags", "--always", "--dirty"],
    }.items():
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                timeout=10,
            )
            if proc.returncode == 0:
                out[key] = proc.stdout.strip()
        except Exception:
            pass
    return out


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest_backup(root: Path = BACKUP_ROOT) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()],
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def human_bytes(value: int) -> str:
    value_f = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if value_f < 1024.0 or unit == "GB":
            return f"{value_f:.1f} {unit}"
        value_f /= 1024.0
    return f"{value_f:.1f} GB"
