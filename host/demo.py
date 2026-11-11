"""`nuro demo` — one command, all three views.

What it starts
--------------
1. **Web cockpit** (live): a localhost server plus the default browser. This is
   the view that owns the ESP32-S3 serial port.
2. **Terminal TUI** (in-silico mirror): runs in the current terminal. Windows
   serial ports are exclusive, so the TUI deliberately renders a synthetic
   mirror instead of fighting the web cockpit for the port.
3. **Matplotlib GUI** (in-silico mirror): opens in its own console/window.

The mirror flag is passed through ``NURO_DEMO_MIRROR=1`` and is shown in each
view's status line. Nothing here writes to the board or drives a GPIO.

Press ``q`` in the TUI to end the demo; the web server and GUI are then shut
down cleanly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spawn_gui(port: str) -> Optional[subprocess.Popen]:
    """Open the matplotlib cockpit in its own console when the OS allows it."""
    cmd = [sys.executable, "-m", "host.cli", "gui", "--port", port]
    env = dict(os.environ)
    env["NURO_DEMO_MIRROR"] = "1"
    kwargs = {"cwd": str(REPO_ROOT), "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    else:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    try:
        return subprocess.Popen(cmd, **kwargs)
    except Exception as exc:
        print(f"[demo] Could not open the matplotlib GUI: {exc}")
        return None


def _stop(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_demo(
    serial_port: Optional[str] = None,
    web_port: int = 8765,
    frames: Optional[int] = None,
    open_browser: bool = True,
    arm: bool = False,
) -> None:
    from host.board import resolve_port
    from host.dash.web_cockpit import create_web_cockpit

    port = resolve_port(serial_port)
    print("=" * 63)
    print("  NEURAL LOOP — THREE-VIEW DEMO")
    print("=" * 63)
    print(f"  Live serial port : {port}")
    print(f"  Web cockpit      : http://127.0.0.1:{web_port}")
    print("  TUI              : this terminal (in-silico mirror)")
    print("  Matplotlib GUI   : separate window (in-silico mirror)")
    print(
        "  Stimulation gate : "
        + ("ARM for this session" if arm else "disarmed (default)")
    )
    print("  Note             : serial ports are exclusive, so only the web")
    print("                     cockpit owns the board; the mirrors are labelled.")
    print("=" * 63)

    engine, srv = create_web_cockpit(port=web_port, serial_port=port, arm=arm)
    server_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    server_thread.start()
    print(
        f"[demo] Web cockpit live. Hardware: {'LIVE' if engine.hw_live else 'IN-SILICO FALLBACK'}"
    )

    if open_browser:

        def _open() -> None:
            time.sleep(0.6)
            try:
                webbrowser.open(f"http://127.0.0.1:{web_port}")
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    gui_proc = _spawn_gui(port)

    os.environ["NURO_DEMO_MIRROR"] = "1"
    try:
        from host.dash.tui import run_tui

        run_tui(port=port, max_frames=frames)
    except KeyboardInterrupt:
        pass
    finally:
        os.environ.pop("NURO_DEMO_MIRROR", None)
        engine.running = False
        try:
            srv.shutdown()
        except Exception:
            pass
        srv.server_close()
        engine.disable_stim_gate()
        engine.close()
        _stop(gui_proc)
        if arm:
            print("\n[demo] All three views stopped. Stimulation gate disarmed.")
        else:
            print("\n[demo] All three views stopped. Board was never written to.")
