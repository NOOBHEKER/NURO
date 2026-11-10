"""Read-only ESP32-S3 identity probe.

Usage:
    python tools/board_info.py
    python tools/board_info.py --port COM6 --json
    python tools/board_info.py --banner

This script never writes flash and never drives a GPIO. It only asks the ROM
bootloader for its identity and, optionally, listens to the application's
serial banner.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from board_common import (  # noqa: E402
    describe_ports,
    force_utf8_stdout,
    normalize_mac,
    parse_flash_id,
    resolve_port,
    run_esptool,
)


def read_banner(port: str, baud: int = 2_000_000, seconds: float = 1.5) -> str:
    """Listen for the existing serial banner without sending a byte."""
    try:
        import serial
    except Exception as exc:  # pragma: no cover
        return f"[banner unavailable: {exc}]"
    try:
        with serial.Serial(port, baud, timeout=0.2) as ser:
            # Never reset the application while probing.
            ser.dtr = False
            ser.rts = False
            deadline = time.time() + seconds
            chunks = []
            while time.time() < deadline:
                data = ser.read(4096)
                if data:
                    chunks.append(data)
            return b"".join(chunks).decode("utf-8", errors="replace")
    except Exception as exc:
        return f"[banner unavailable: {exc}]"


def main() -> int:
    force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Read-only ESP32-S3 identity probe")
    parser.add_argument(
        "--port", default=None, help="Serial port (auto-detect by default)"
    )
    parser.add_argument(
        "--baud", type=int, default=460800, help="esptool baud (default 460800)"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument(
        "--banner", action="store_true", help="Also listen to the app banner"
    )
    parser.add_argument("--banner-seconds", type=float, default=1.5)
    args = parser.parse_args()

    port = resolve_port(args.port)
    result = {
        "port": port,
        "ports": describe_ports(),
        "flash_id": {},
        "mac": None,
        "banner": None,
        "ok": False,
    }

    try:
        flash_proc = run_esptool(
            ["--chip", "esp32s3", "--port", port, "--baud", str(args.baud), "flash-id"],
            quiet=args.json,
        )
        result["flash_id"] = parse_flash_id(
            (flash_proc.stdout or "") + (flash_proc.stderr or "")
        )

        mac_proc = run_esptool(
            ["--chip", "esp32s3", "--port", port, "--baud", str(args.baud), "read-mac"],
            quiet=args.json,
        )
        mac_text = (mac_proc.stdout or "") + (mac_proc.stderr or "")
        for line in mac_text.splitlines():
            if "MAC:" in line:
                result["mac"] = line.split("MAC:", 1)[1].strip()
                break
        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)

    if args.banner and result["ok"]:
        result["banner"] = read_banner(port, seconds=args.banner_seconds)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    print("\n=== ESP32-S3 BOARD IDENTITY (READ-ONLY) ===")
    print(f"  Port          : {port}")
    if result.get("error"):
        print(f"  Status        : NOT RESPONDING")
        print(f"  Detail        : {result['error'].splitlines()[0]}")
        print("\n  Ports seen:")
        for p in result["ports"]:
            print(f"    {p['device']:<8} {p['description']}  {p['note']}")
        return 1

    facts = result["flash_id"]
    print(
        f"  Chip          : {facts.get('chip', 'ESP32-S3')} rev {facts.get('chip_revision', '?')}"
    )
    print(f"  MAC           : {normalize_mac(result.get('mac'))}")
    print(
        f"  Flash         : {facts.get('flash_size', '?')} "
        f"(manuf 0x{facts.get('manufacturer', '?')}, device 0x{facts.get('device', '?')})"
    )
    print(f"  Flash type    : {facts.get('flash_type', '?')}")
    print("  Status        : RESPONDING (bootloader)")
    if result.get("banner"):
        banner = result["banner"].strip()
        print("\n  App banner (passive listen):")
        for line in banner.splitlines()[-12:]:
            print(f"    {line}")
    print("\n  Nothing was written to flash and no GPIO was driven.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
