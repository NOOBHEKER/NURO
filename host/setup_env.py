"""`nuro setup` — bootstrap a working environment on a fresh computer.

The goal is plug-and-play: copy or clone the repository, run one command, and
have `nuro` (plus every dashboard, the board probe and the tests) work without
editing a COM port or a path.

What it does
------------
1. checks the Python version (3.10+),
2. creates ``.venv`` in the repository root unless it already exists,
3. installs the project in editable mode (``pip install -e .[dev]``),
4. runs ``nuro doctor`` with the new interpreter.

It never touches the ESP32-S3, never opens the serial port, and never writes
flash. ``--recreate`` deletes and rebuilds ``.venv`` and requires the literal
confirmation token ``RECREATE``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / ".venv"


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(cmd: List[str], *, check: bool = True) -> int:
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if check and proc.returncode != 0:
        raise SystemExit(f"[setup] command failed with exit code {proc.returncode}")
    return proc.returncode


def _check_python() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(
            f"[setup] Python 3.10+ is required, found "
            f"{sys.version_info.major}.{sys.version_info.minor}. "
            "Install it from https://www.python.org/downloads/"
        )
    print(f"  Python {sys.version.split()[0]} at {sys.executable}")


def _running_from_venv() -> bool:
    try:
        return VENV_DIR.resolve() in Path(sys.executable).resolve().parents
    except Exception:  # pragma: no cover - defensive
        return False


def _recreate(confirm: Optional[str]) -> None:
    if confirm != "RECREATE":
        raise SystemExit(
            "[setup] --recreate deletes .venv. Re-run with "
            "`nuro setup --recreate --confirm RECREATE` if that is what you want."
        )
    if _running_from_venv():
        raise SystemExit(
            "[setup] this process is running from .venv, and Windows cannot delete a "
            "running interpreter. Run `setup.cmd` (or `python -m host.cli setup "
            "--recreate --confirm RECREATE`) with the system Python instead."
        )
    if VENV_DIR.exists():
        print(f"  Removing {VENV_DIR}")
        shutil.rmtree(VENV_DIR)


def run_setup(
    recreate: bool = False,
    confirm: Optional[str] = None,
    with_dev: bool = True,
    skip_doctor: bool = False,
) -> int:
    try:
        sys.stdout.reconfigure(
            line_buffering=True
        )  # keep progress visible when redirected
    except Exception:
        pass
    print("=" * 63)
    print("  NEURAL LOOP — PLUG-AND-PLAY SETUP")
    print("=" * 63)
    _check_python()

    if recreate:
        _recreate(confirm)

    python = _venv_python(VENV_DIR)
    if python.exists():
        print(f"  Reusing existing virtual environment: {VENV_DIR}")
    else:
        print(f"  Creating virtual environment: {VENV_DIR}")
        try:
            _run([sys.executable, "-m", "venv", str(VENV_DIR)])
        except SystemExit:
            raise SystemExit(
                "[setup] could not create .venv. On Debian/Ubuntu install the venv "
                "module first:  sudo apt install python3-venv  then re-run."
            )

    if not python.exists():
        raise SystemExit(
            f"[setup] could not find the interpreter inside {VENV_DIR}. "
            "Delete .venv and retry."
        )

    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    target = ".[dev]" if with_dev else "."
    _run([str(python), "-m", "pip", "install", "-e", target])

    console = VENV_DIR / ("Scripts/nuro.exe" if os.name == "nt" else "bin/nuro")
    print("-" * 63)
    if console.exists():
        print(f"  Installed console script: {console}")
    else:
        print("  Console script not found; use `python -m host.cli` instead.")

    if not skip_doctor:
        print()
        _run([str(python), "-m", "host.cli", "doctor"], check=False)

    print("-" * 63)
    if os.name == "nt":
        print("  Put nuro on PATH for this user (once):")
        print("    .\\install.ps1 -Apply")
        print("  Then open a new terminal and run:  nuro demo")
    else:
        print("  Add nuro to PATH (once), e.g. in ~/.bashrc:")
        print(f'    export PATH="{VENV_DIR / "bin"}:$PATH"')
        print("  Then run:  nuro demo")
    print("=" * 63)
    return 0
