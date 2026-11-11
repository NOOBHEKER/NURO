"""Plug & Play Hardware Auto-Launcher for Neural Loop.

Listens for the arrival of the ESP32-S3 neuromodulation board on any USB port
and automatically opens the live 60 FPS interactive TUI console window.

Usage:
    nuro watch              # Run interactive watcher in current terminal
    nuro watch --install    # Install into Windows Startup (runs silently on login)
    nuro watch --uninstall  # Remove from Windows Startup
    nuro watch --once       # Check if board is plugged in right now; launch if so
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\033[38;5;244m[{now}]\033[0m {msg}", flush=True)


def is_tui_running() -> bool:
    """Return True if an existing Neural Loop TUI process is running."""
    try:
        import psutil

        cur_pid = os.getpid()
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            if p.info["pid"] == cur_pid:
                continue
            cmd = " ".join(p.info["cmdline"] or []).lower()
            if ("host.cli" in cmd and "tui" in cmd) or "host.dash.tui" in cmd:
                return True
    except Exception:
        pass
    return False


def launch_tui(port: str) -> bool:
    """Launch the interactive TUI in a new console window."""
    repo_dir = str(Path(__file__).resolve().parents[1])
    nuro_cmd = os.path.join(repo_dir, "nuro.cmd")

    if os.name == "nt":
        try:
            if os.path.exists(nuro_cmd):
                subprocess.Popen(
                    [
                        "cmd.exe",
                        "/c",
                        "start",
                        "Neural Loop TUI (Plug & Play)",
                        nuro_cmd,
                        "tui",
                        "--port",
                        port,
                    ],
                    cwd=repo_dir,
                    shell=True,
                )
            else:
                subprocess.Popen(
                    [
                        "cmd.exe",
                        "/c",
                        "start",
                        "Neural Loop TUI (Plug & Play)",
                        sys.executable,
                        "-m",
                        "host.cli",
                        "tui",
                        "--port",
                        port,
                    ],
                    cwd=repo_dir,
                    shell=True,
                )
            return True
        except Exception as e:
            _log(f"\033[38;5;203m[ERROR]\033[0m Failed to spawn TUI window: {e}")
            return False
    else:
        try:
            subprocess.Popen(
                [sys.executable, "-m", "host.cli", "tui", "--port", port],
                cwd=repo_dir,
            )
            return True
        except Exception as e:
            _log(f"\033[38;5;203m[ERROR]\033[0m Failed to spawn TUI: {e}")
            return False


def get_startup_dir() -> Optional[Path]:
    """Return Windows user Startup folder path."""
    if os.name != "nt":
        return None
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    p = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    return p if p.exists() else None


def install_startup() -> bool:
    """Install silent background watcher into Windows Startup."""
    sdir = get_startup_dir()
    if not sdir:
        print("\033[38;5;203m[FAIL]\033[0m Windows Startup folder not found.")
        return False

    repo_dir = Path(__file__).resolve().parents[1]
    nuro_cmd = repo_dir / "nuro.cmd"
    vbs_path = sdir / "NeuralLoop_PlugAndPlay.vbs"

    # Silent runner script (windowstyle 0 = hidden)
    vbs_content = (
        'Set WshShell = CreateObject("WScript.Shell")\n'
        f'WshShell.CurrentDirectory = "{repo_dir}"\n'
        f'WshShell.Run """{nuro_cmd}"" watch --silent", 0, False\n'
        "Set WshShell = Nothing\n"
    )

    try:
        vbs_path.write_text(vbs_content, encoding="utf-8")
        print(f"\033[38;5;82m[SUCCESS]\033[0m Installed Plug & Play auto-launcher:")
        print(f"  Startup Script : {vbs_path}")
        print("  Whenever you log in, nuro watch will run silently in the background.")
        print("  Plugging in your ESP32-S3 will instantly open the live TUI.")
        return True
    except Exception as e:
        print(f"\033[38;5;203m[FAIL]\033[0m Failed to write startup script: {e}")
        return False


def uninstall_startup() -> bool:
    """Remove background watcher from Windows Startup."""
    sdir = get_startup_dir()
    if not sdir:
        return False

    vbs_path = sdir / "NeuralLoop_PlugAndPlay.vbs"
    if vbs_path.exists():
        try:
            vbs_path.unlink()
            print(f"\033[38;5;82m[SUCCESS]\033[0m Removed {vbs_path}")
            return True
        except Exception as e:
            print(f"\033[38;5;203m[FAIL]\033[0m Could not remove {vbs_path}: {e}")
            return False
    else:
        print("\033[38;5;214m[INFO]\033[0m No startup script was found.")
        return True


def is_startup_installed() -> bool:
    sdir = get_startup_dir()
    if not sdir:
        return False
    return (sdir / "NeuralLoop_PlugAndPlay.vbs").exists()


def run_watcher(
    poll_interval: float = 1.0,
    install: bool = False,
    uninstall: bool = False,
    silent: bool = False,
    once: bool = False,
) -> int:
    """Main watcher loop."""
    if install:
        return 0 if install_startup() else 1
    if uninstall:
        return 0 if uninstall_startup() else 1

    from host.board import detect_board_port, list_serial_ports

    if not silent:
        print("\n\033[1m\033[38;5;51m╔═══════════════════════════════════════════════════════════════╗\033[0m")
        print("\033[1m\033[38;5;51m║    ⚡ nuro watch — Plug & Play Hardware Auto-Launcher         ║\033[0m")
        print("\033[1m\033[38;5;51m╚═══════════════════════════════════════════════════════════════╝\033[0m")
        print(f"\033[38;5;244mStatus:\033[0m Windows Startup Auto-Launch is \033[1m{'ENABLED' if is_startup_installed() else 'DISABLED'}\033[0m (run 'nuro watch --install' to enable)")
        print(f"\033[38;5;244mPolling USB ports every {poll_interval:.1f}s...\033[0m")
        print("\033[38;5;82mPlug your ESP32-S3 into ANY USB port to automatically launch the TUI.\033[0m\n")

    last_detected: Optional[str] = None
    initial_check = True

    try:
        while True:
            cur_port = detect_board_port()

            if cur_port is not None:
                if last_detected is None:
                    # New board insertion detected!
                    last_detected = cur_port
                    if not silent:
                        _log(
                            f"\033[1m\033[38;5;82m⚡ [PLUG & PLAY]\033[0m ESP32-S3 detected on \033[1m{cur_port}\033[0m!"
                        )

                    if is_tui_running():
                        if not silent:
                            _log(
                                f"\033[38;5;214m[INFO]\033[0m TUI is already running — dynamic hotplug in TUI will connect to {cur_port} directly."
                            )
                    else:
                        if not silent:
                            _log(
                                f"\033[1m\033[38;5;51m[LAUNCH]\033[0m Spawning Neural Loop TUI window on {cur_port}..."
                            )
                        launch_tui(cur_port)

                    if once:
                        return 0
            else:
                if last_detected is not None:
                    if not silent:
                        _log(
                            f"\033[38;5;203m[HOT-UNPLUG]\033[0m ESP32-S3 disconnected from \033[1m{last_detected}\033[0m. Waiting for next plug-in..."
                        )
                    last_detected = None
                elif initial_check and not silent:
                    _log(
                        "\033[38;5;244mNo ESP32-S3 currently connected. Waiting for USB insertion...\033[0m"
                    )

            initial_check = False
            if once:
                if not silent:
                    _log("No board detected on this check.")
                return 1

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        if not silent:
            print("\n\033[38;5;244mWatcher stopped.\033[0m")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plug & Play Hardware Auto-Launcher for Neural Loop"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install silent watcher into Windows Startup folder",
    )
    parser.add_argument(
        "--uninstall", action="store_true", help="Remove from Windows Startup folder"
    )
    parser.add_argument(
        "--silent", action="store_true", help="Run without printing console logs"
    )
    parser.add_argument(
        "--once", action="store_true", help="Check once and exit immediately"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Poll interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    sys.exit(
        run_watcher(
            poll_interval=args.interval,
            install=args.install,
            uninstall=args.uninstall,
            silent=args.silent,
            once=args.once,
        )
    )
