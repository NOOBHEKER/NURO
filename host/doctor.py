"""`nuro doctor` — one-screen environment and board readiness check.

Exit code 0 when nothing is broken, 1 when a FAIL is present. A missing board
is a WARN, not a FAIL: the testbed is designed to run in-silico without it.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def _check_imports() -> List[Tuple[str, str, str]]:
    checks: List[Tuple[str, str, str]] = []
    required = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "serial": "pyserial",
        "sympy": "sympy",
        "requests": "requests",
        "pytest": "pytest",
    }
    for module, package in required.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "?")
            checks.append((PASS, f"Python package {package}", f"{module} {version}"))
        except Exception as exc:
            checks.append((FAIL, f"Python package {package}", str(exc)))
    return checks


def _check_repo() -> List[Tuple[str, str, str]]:
    checks: List[Tuple[str, str, str]] = []
    required = [
        "firmware/platformio.ini",
        "firmware/boards/esp32s3_n16r8.json",
        "host/board.py",
        "tests/test_suite.py",
        "tests/golden/crc16_sample.json",
        "docs/RECOVERY.md",
    ]
    for rel in required:
        path = REPO_ROOT / rel
        checks.append(
            (
                PASS if path.exists() else FAIL,
                f"Repository file {rel}",
                "present" if path.exists() else "MISSING",
            )
        )

    firmware_bin = REPO_ROOT / "firmware/.pio/build/esp32s3_n16r8/firmware.bin"
    if firmware_bin.exists():
        checks.append(
            (PASS, "Firmware build artifact", f"{firmware_bin.stat().st_size} bytes")
        )
    else:
        checks.append(
            (
                WARN,
                "Firmware build artifact",
                "not built yet (run: pio run -d firmware)",
            )
        )

    backup = REPO_ROOT / "backups"
    manifests = list(backup.glob("*/manifest.json")) if backup.exists() else []
    if manifests:
        checks.append(
            (
                PASS,
                "Full-flash recovery backup",
                f"{len(manifests)} manifest(s) in backups/",
            )
        )
    else:
        checks.append(
            (WARN, "Full-flash recovery backup", "none found (see docs/RECOVERY.md)")
        )
    return checks


def _check_board(port: Optional[str], probe: bool) -> List[Tuple[str, str, str]]:
    from host.board import (
        describe_ports,
        detect_board_port,
        esptool_available,
        probe_identity,
        resolve_port,
    )

    checks: List[Tuple[str, str, str]] = []
    ports = describe_ports()
    checks.append(
        (
            PASS if ports else WARN,
            "Serial ports visible",
            ", ".join(p["device"] for p in ports) if ports else "none",
        )
    )

    detected = detect_board_port()
    resolved = resolve_port(port)
    if detected:
        checks.append(
            (
                PASS,
                "ESP32-S3 bridge auto-detect",
                f"{detected} ({next((p['note'] for p in ports if p['device'] == detected), '')})",
            )
        )
    else:
        checks.append(
            (WARN, "ESP32-S3 bridge auto-detect", f"none; will use {resolved}")
        )

    checks.append(
        (
            PASS if esptool_available() else WARN,
            "esptool",
            "installed"
            if esptool_available()
            else "missing (pip install esptool>=5.0)",
        )
    )

    if probe:
        facts = probe_identity(port)
        if facts.get("ok"):
            checks.append(
                (
                    PASS,
                    "Read-only board probe",
                    f"{facts.get('chip', 'ESP32-S3')} rev {facts.get('chip_revision', '?')}, "
                    f"{facts.get('flash_size', '?')} flash, MAC {facts.get('mac', '?')}",
                )
            )
        else:
            checks.append(
                (WARN, "Read-only board probe", facts.get("error", "no response"))
            )
    return checks


def _check_git() -> List[Tuple[str, str, str]]:
    if not shutil.which("git"):
        return [(WARN, "git", "not installed")]
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return [(WARN, "git repository", "not a git checkout")]
        dirty = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        return [
            (
                PASS if not dirty else WARN,
                "git working tree",
                "clean" if not dirty else f"{len(dirty)} modified/untracked path(s)",
            )
        ]
    except Exception as exc:
        return [(WARN, "git", str(exc))]


def run_doctor(port: Optional[str] = None, probe: bool = True) -> int:
    print("=" * 63)
    print("  NEURAL LOOP — ENVIRONMENT & BOARD DOCTOR")
    print("=" * 63)

    checks: List[Tuple[str, str, str]] = []
    py = sys.version_info
    checks.append(
        (
            PASS if py >= (3, 10) else FAIL,
            "Python version",
            f"{py.major}.{py.minor}.{py.micro}",
        )
    )
    checks.extend(_check_imports())
    checks.extend(_check_repo())
    checks.extend(_check_board(port, probe))
    checks.extend(_check_git())

    for status, name, detail in checks:
        print(f"  [{status:<4}] {name:<38} {detail}")

    failures = sum(1 for status, _, _ in checks if status == FAIL)
    warnings = sum(1 for status, _, _ in checks if status == WARN)
    print("-" * 63)
    print(
        f"  {len(checks) - failures - warnings} pass · {warnings} warn · {failures} fail"
    )
    if failures:
        print("  Result: NOT READY — fix the FAIL lines above.")
        return 1
    print("  Result: READY" + (" (with warnings)" if warnings else ""))
    return 0
