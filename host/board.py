"""Board discovery, identity probing, and the single serial-port policy.

Every host-side entry point (CLI, TUI, web cockpit, matplotlib GUI, HIL
transport, calculator hardware queries) resolves its port through this module,
so there is one place to change the default and one place that knows how to
find the board when it moves off COM6.

Resolution order:

1. an explicit ``--port`` / function argument,
2. ``NURO_PORT`` (legacy alias: ``NL_PORT``) from the environment,
3. an auto-detected ESP32-S3 / CH343 port,
4. the historical ``COM6`` fallback.

``probe_identity`` only runs esptool's read-only ``flash-id`` and ``read-mac``
commands. It never writes flash and never drives a GPIO.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 2_000_000

# USB bridges commonly used on ESP32 dev boards. VID:PID -> human note.
KNOWN_USB_IDS: Dict[str, str] = {
    "1A86:55D3": "WCH CH343 (ESP32-S3 DevKitC-1 UART port)",
    "1A86:7523": "WCH CH340",
    "10C4:EA60": "Silicon Labs CP210x",
    "303A:1001": "Espressif native USB (USB-Serial-JTAG)",
    "303A:4001": "Espressif native USB CDC",
}

# Higher score wins during auto-detection.
_PORT_SCORES = {
    "1A86:55D3": 100,  # CH343 on this exact DevKitC-1
    "303A:1001": 90,  # native ESP32-S3 USB
    "303A:4001": 90,
    "10C4:EA60": 60,  # CP210x
    "1A86:7523": 50,  # CH340
}


@dataclass(frozen=True)
class SerialPort:
    """A JSON-friendly snapshot of one serial port."""

    device: str
    description: str = ""
    hwid: str = ""
    vid: Optional[int] = None
    pid: Optional[int] = None
    manufacturer: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _usb_key(vid: Optional[int], pid: Optional[int]) -> str:
    if vid is None or pid is None:
        return ""
    return f"{vid:04X}:{pid:04X}"


def list_serial_ports() -> List[SerialPort]:
    """Return every serial port pyserial can see, annotated when recognised."""
    try:
        from serial.tools import list_ports
    except Exception:  # pragma: no cover - pyserial is a hard dependency in practice
        return []

    out: List[SerialPort] = []
    for p in list_ports.comports():
        key = _usb_key(getattr(p, "vid", None), getattr(p, "pid", None))
        hwid = (p.hwid or "").upper()
        if not key:
            key = next((k for k in KNOWN_USB_IDS if k in hwid), "")
        out.append(
            SerialPort(
                device=p.device,
                description=p.description or "",
                hwid=p.hwid or "",
                vid=getattr(p, "vid", None),
                pid=getattr(p, "pid", None),
                manufacturer=getattr(p, "manufacturer", "") or "",
                note=KNOWN_USB_IDS.get(key, ""),
            )
        )
    return out


def describe_ports() -> List[Dict[str, Any]]:
    """Compatibility wrapper used by the board tools."""
    return [p.to_dict() for p in list_serial_ports()]


def detect_board_port() -> Optional[str]:
    """Pick the most likely ESP32-S3 port, or ``None`` when nothing matches."""
    best: Optional[SerialPort] = None
    best_score = 0
    for p in list_serial_ports():
        key = _usb_key(p.vid, p.pid)
        if not key:
            hwid = p.hwid.upper()
            key = next((k for k in KNOWN_USB_IDS if k in hwid), "")
        score = _PORT_SCORES.get(key, 0)
        if score > best_score:
            best, best_score = p, score
    return best.device if best else None


def resolve_port(preferred: Optional[str] = None) -> str:
    """Resolve the serial port using the documented priority order."""
    if preferred:
        return preferred

    env_port = os.environ.get("NURO_PORT") or os.environ.get("NL_PORT")
    if env_port:
        return env_port

    detected = detect_board_port()
    if detected:
        return detected

    return DEFAULT_PORT


def port_source(preferred: Optional[str] = None) -> str:
    """Explain where the resolved port came from (useful for `nuro info`)."""
    if preferred:
        return "explicit --port"
    if os.environ.get("NURO_PORT") or os.environ.get("NL_PORT"):
        return "NURO_PORT environment variable"
    if detect_board_port():
        return "auto-detected USB bridge"
    return f"fallback default ({DEFAULT_PORT})"


def esptool_available() -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "esptool", "version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        return proc.returncode == 0
    except Exception:
        return False


def parse_flash_id(output: str) -> Dict[str, Any]:
    """Extract useful facts from `esptool flash-id` output (v5 format)."""
    facts: Dict[str, Any] = {}
    patterns = {
        "chip": r"Chip type:\s*(ESP32-[A-Z0-9]+)",
        "chip_revision": r"revision (v?[0-9.]+)",
        "mac": r"MAC:\s*([0-9a-fA-F:]{17})",
        "manufacturer": r"Manufacturer:\s*(?:0x)?([0-9a-fA-F]+)",
        "device": r"Device:\s*(?:0x)?([0-9a-fA-F]+)",
        "flash_size": r"Detected flash size:\s*([0-9]+ ?[KMG]B)",
        "flash_type": r"Flash type set in eFuse:\s*([^\r\n]+)",
        "features": r"Features:\s*([^\r\n]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            facts[key] = match.group(1).strip()
    return facts


def normalize_mac(value: Optional[str]) -> str:
    return (value or "").strip().lower().replace("-", ":")


def probe_identity(
    port: Optional[str] = None,
    *,
    baud: int = 460_800,
    timeout: int = 90,
) -> Dict[str, Any]:
    """Read-only identity probe using esptool's flash-id / read-mac.

    Returns a dict with ``ok`` plus whatever facts could be read. Never raises
    for an ordinary connection failure: the caller gets ``ok=False`` and an
    ``error`` string instead.
    """
    resolved = resolve_port(port)
    result: Dict[str, Any] = {
        "ok": False,
        "port": resolved,
        "port_source": port_source(port),
    }
    if not esptool_available():
        result["error"] = (
            "esptool is not installed. Run `pip install -r requirements.txt` "
            "or `pip install esptool>=5.0`."
        )
        return result

    base = [
        sys.executable,
        "-m",
        "esptool",
        "--chip",
        "esp32s3",
        "--port",
        resolved,
        "--baud",
        str(baud),
    ]
    try:
        id_proc = subprocess.run(
            [*base, "flash-id"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        id_text = (id_proc.stdout or "") + (id_proc.stderr or "")
        if id_proc.returncode != 0:
            result["error"] = _last_error(id_text)
            return result

        facts = parse_flash_id(id_text)
        mac_proc = subprocess.run(
            [*base, "read-mac"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        mac_text = (mac_proc.stdout or "") + (mac_proc.stderr or "")
        if "MAC:" in mac_text:
            facts["mac"] = normalize_mac(mac_text.split("MAC:", 1)[1].split()[0])

        result.update(facts)
        result["ok"] = True
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def _last_error(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        low = line.lower()
        if any(word in low for word in ("error", "failed", "could not", "no such")):
            return line
    return lines[-1] if lines else "esptool failed without output"


def identity_summary(facts: Dict[str, Any]) -> List[str]:
    """Format a probe result as the lines `nuro info` prints."""
    if not facts.get("ok"):
        return [
            f"  Port       : {facts.get('port', DEFAULT_PORT)}",
            f"  Status     : NOT RESPONDING",
            f"  Detail     : {facts.get('error', 'unknown error')}",
            "  Hint       : hold BOOT, tap RESET, release BOOT, then retry.",
        ]

    flash = facts.get("flash_size", "?")
    features = facts.get("features", "")
    psram = (
        "8 MB octal"
        if "PSRAM 8MB" in features
        else ("detected" if "PSRAM" in features else "not reported")
    )
    lines = [
        f"  Port       : {facts.get('port')} ({facts.get('port_source')})",
        f"  Chip       : {facts.get('chip', 'ESP32-S3')} rev {facts.get('chip_revision', '?')}",
        f"  MAC        : {facts.get('mac', '?')}",
        f"  Flash      : {flash} (manuf 0x{facts.get('manufacturer', '?')}, "
        f"device 0x{facts.get('device', '?')})",
        f"  PSRAM      : {psram}",
        f"  Flash mode : {facts.get('flash_type', '?')}",
    ]
    if features:
        lines.append(f"  Features   : {features}")
    return lines
