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
DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 460800
DEFAULT_FLASH_BYTES = 16 * 1024 * 1024  # ESP32-S3-WROOM-1-N16R8
BACKUP_ROOT = REPO_ROOT / "backups"

# USB bridges commonly used on ESP32 dev boards.
_KNOWN_USB_IDS = {
    "1A86:55D3": "WCH CH343 (ESP32-S3 DevKitC-1 UART port)",
    "1A86:7523": "WCH CH340",
    "10C4:EA60": "Silicon Labs CP210x",
    "303A:1001": "Espressif native USB (USB-Serial-JTAG)",
    "303A:4001": "Espressif native USB CDC",
}


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


def resolve_port(preferred: Optional[str] = None) -> str:
    """Resolve a serial port without guessing blindly.

    Priority: explicit argument -> NL_PORT -> CH343 -> any known ESP32 bridge
    -> COM6 fallback. The fallback is reported so the caller can warn.
    """
    if preferred:
        return preferred
    import os

    env_port = os.environ.get("NL_PORT")
    if env_port:
        return env_port

    try:
        from serial.tools import list_ports
    except Exception:  # pragma: no cover - pyserial is a hard dependency in practice
        return DEFAULT_PORT

    candidates = list(list_ports.comports())
    for wanted in ("1A86:55D3",):
        for p in candidates:
            hwid = (p.hwid or "").upper()
            if wanted in hwid:
                return p.device
    for p in candidates:
        hwid = (p.hwid or "").upper()
        if any(key in hwid for key in _KNOWN_USB_IDS):
            return p.device
    return DEFAULT_PORT


def describe_ports() -> List[Dict[str, str]]:
    """Return a JSON-friendly description of every serial port."""
    try:
        from serial.tools import list_ports
    except Exception:  # pragma: no cover
        return []
    out: List[Dict[str, str]] = []
    for p in list_ports.comports():
        hwid = (p.hwid or "").upper()
        note = next((v for k, v in _KNOWN_USB_IDS.items() if k in hwid), "")
        out.append(
            {
                "device": p.device,
                "description": p.description or "",
                "hwid": p.hwid or "",
                "note": note,
            }
        )
    return out


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


def parse_flash_id(output: str) -> Dict[str, Any]:
    """Extract the useful facts from `esptool flash-id` output."""
    facts: Dict[str, Any] = {}
    patterns = {
        "chip": r"Chip type:\s*(ESP32-[A-Z0-9]+)",
        "chip_revision": r"revision (v?[0-9.]+)",
        "mac": r"MAC:\s*([0-9a-fA-F:]{17})",
        "manufacturer": r"Manufacturer:\s*(?:0x)?([0-9a-fA-F]+)",
        "device": r"Device:\s*(?:0x)?([0-9a-fA-F]+)",
        "flash_size": r"Detected flash size:\s*([0-9]+ ?[KMG]B)",
        "flash_type": r"Flash type set in eFuse:\s*([^\r\n]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            facts[key] = match.group(1).strip()
    return facts


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


def normalize_mac(value: Optional[str]) -> str:
    return (value or "").strip().lower().replace("-", ":")


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
