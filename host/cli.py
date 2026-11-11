"""nuro — Unified CLI for Neural Loop ($0 ESP32-S3 HIL Closed-Loop Testbed).

Usage:
  nuro                  # Launch the interactive scientific calculator shell
  nuro setup            # One-command environment bootstrap (fresh computer)
  nuro tui              # Launch interactive Native Terminal TUI
  nuro web              # Launch 60 FPS Localhost TUI Workstation & open browser
  nuro gui              # Launch live matplotlib cockpit
  nuro demo             # Web cockpit + TUI + matplotlib GUI in one command
  nuro bench            # Run 1000-frame live HIL benchmark (auto-detected port)
  nuro eval             # Run Gates G5 & G6 on SYNTHETIC surrogates (not CHB-MIT)
  nuro test             # Run the automated verification suite
  nuro figures          # Regenerate all publication figures (PNG + PDF)
  nuro info             # Probe the real board (read-only) and print its facts
  nuro doctor           # Check the local environment and board readiness

Port selection is centralised in host.board: --port > NURO_PORT > detected
ESP32-S3/USB bridge > COM6 on Windows (no default elsewhere).
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from typing import List


def _version() -> str:
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("neural-loop")
    except Exception:
        return "0.3.0"


def _announce_port(port: str) -> None:
    """Print the resolved port, flagging the native USB port when it is chosen."""
    from host.board import is_native_usb

    note = ""
    if is_native_usb(port):
        note = "  (native USB — the HIL protocol needs the UART port)"
    print(f"[nuro] serial port: {port}{note}")


def handle_session_command(args: List[str]) -> str:
    """Manage named calculation sessions."""
    from host.session_manager import (
        get_active_session,
        set_active_session,
        list_sessions,
    )
    from host.calc import C_BOLD, C_GREEN, C_CYAN, C_AMBER, C_RESET, C_WHITE, C_GRAY

    if not args or args[0].lower() in ("list", "status", "show", "current"):
        active = get_active_session()
        sessions = list_sessions()
        out = [
            f"{C_BOLD}{C_GREEN}⚡ NURO CALCULATION SESSIONS:{C_RESET}",
            f"  {C_GRAY}Active Session  :{C_RESET} {C_BOLD}{C_CYAN}{active}{C_RESET}",
            f"\n  {C_GRAY}All Available Sessions:{C_RESET}",
        ]
        for name, is_act, count, desc in sessions:
            marker = (
                f"{C_BOLD}{C_GREEN}▶ [ACTIVE]{C_RESET}"
                if is_act
                else f"  {C_GRAY}[      ]{C_RESET}"
            )
            out.append(
                f"  {marker} {C_WHITE}{name:<22}{C_RESET} ({count} calcs) {C_GRAY}{desc}{C_RESET}"
            )
        out.append(
            f"\n  {C_GRAY}Switch Session :{C_RESET} nuro session switch <name>  │  session <name>"
        )
        return "\n".join(out)

    subcmd = args[0].lower()
    if subcmd in ("switch", "use", "select", "new", "create", "set"):
        if len(args) < 2:
            return f"{C_AMBER}[WARN]: Please specify session name: session switch <name>{C_RESET}"
        name = " ".join(args[1:])
    else:
        name = " ".join(args)

    new_sess = set_active_session(name)
    return (
        f"{C_BOLD}{C_GREEN}⚡ ACTIVE SESSION SWITCHED TO:{C_RESET} {C_BOLD}{C_CYAN}{new_sess}{C_RESET}\n"
        f"  {C_GRAY}Subsequent calculations saved to Obsidian will be linked under [[Session - {new_sess}]]!{C_RESET}"
    )


def main() -> None:
    # Windows consoles default to cp1252 when redirected; the CLI uses Unicode
    # symbols and the firmware banner uses a lightning bolt.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        prog="nuro",
        description="⚡ nuro — Unified CLI for Neural Loop ($0 ESP32-S3 Closed-Loop Testbed)",
        epilog="Examples:\n  nuro setup    (bootstrap this computer once)\n  nuro demo     (web cockpit + TUI + GUI)\n  nuro info     (read-only board probe)\n  nuro bench    (1000-frame live benchmark on the auto-detected port)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"nuro {_version()}")
    sub = parser.add_subparsers(dest="command")

    # nuro setup
    p_setup = sub.add_parser(
        "setup",
        help="Create .venv, install the project and run the doctor (fresh computer)",
    )
    p_setup.add_argument(
        "--recreate", action="store_true", help="Delete and rebuild .venv"
    )
    p_setup.add_argument(
        "--confirm", default=None, help="Required token for --recreate: RECREATE"
    )
    p_setup.add_argument(
        "--no-dev", action="store_true", help="Install runtime deps only"
    )
    p_setup.add_argument(
        "--skip-doctor", action="store_true", help="Don't run the doctor afterwards"
    )

    # nuro tui
    p_tui = sub.add_parser(
        "tui",
        aliases=["neural", "mneuro", "live"],
        help="Launch interactive native terminal TUI (default)",
    )
    p_tui.add_argument(
        "--port", default=None, help="Serial port (default: auto-detect)"
    )
    p_tui.add_argument("--frames", type=int, default=None, help="Batch frame limit")

    # nuro demo
    p_demo = sub.add_parser(
        "demo",
        help="One-command showcase: web cockpit + TUI + matplotlib GUI",
    )
    p_demo.add_argument(
        "--serial-port", default=None, help="Live serial port (default: auto-detect)"
    )
    p_demo.add_argument(
        "--web-port", type=int, default=8765, help="Localhost web port (default: 8765)"
    )
    p_demo.add_argument(
        "--frames", type=int, default=None, help="Stop the TUI after N batches"
    )
    p_demo.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open the browser"
    )
    p_demo.add_argument(
        "--arm",
        action="store_true",
        help="Enable the GPIO4 stimulation gate while the demo runs (disarmed on exit)",
    )

    # nuro web
    p_web = sub.add_parser(
        "web", help="Launch 60 FPS localhost TUI cockpit & open browser"
    )
    p_web.add_argument(
        "--port", default=8765, type=int, help="Web server port (default: 8765)"
    )
    p_web.add_argument(
        "--serial-port",
        default=None,
        help="ESP32-S3 serial port (default: auto-detect)",
    )
    p_web.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open browser"
    )

    # nuro gui
    p_gui = sub.add_parser(
        "gui", help="Launch live animated 4-panel interactive graphical window"
    )
    p_gui.add_argument(
        "--port", default=None, help="Serial port (default: auto-detect)"
    )

    # nuro bench
    p_bench = sub.add_parser("bench", help="Run 1000-frame live hardware benchmark")
    p_bench.add_argument(
        "--frames", default=1000, type=int, help="Frame count (default: 1000)"
    )
    p_bench.add_argument(
        "--port", default=None, help="Serial port (default: auto-detect)"
    )
    p_bench.add_argument(
        "--arm",
        action="store_true",
        help="Enable the GPIO4 stimulation gate for this run (disarmed afterwards)",
    )

    # nuro eval
    sub.add_parser(
        "eval", help="Run Gates G5 & G6 on synthetic surrogates (not CHB-MIT)"
    )

    # nuro test
    sub.add_parser("test", help="Run the automated pytest verification suite")

    # nuro figures
    sub.add_parser("figures", help="Regenerate all publication figures (PNG + PDF)")

    # nuro info
    p_info = sub.add_parser(
        "info", help="Probe the real ESP32-S3 (read-only) and print its facts"
    )
    p_info.add_argument(
        "--port", default=None, help="Serial port (default: auto-detect)"
    )
    p_info.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    # nuro doctor
    p_doctor = sub.add_parser(
        "doctor",
        help="Check Python deps, board presence, firmware build and repo state",
    )
    p_doctor.add_argument(
        "--port", default=None, help="Serial port (default: auto-detect)"
    )

    # nuro watch / pnp
    p_watch = sub.add_parser(
        "watch",
        aliases=["pnp", "plug-and-play", "autolaunch"],
        help="Background plug-and-play watcher: auto-launch TUI when ESP32-S3 is plugged in",
    )
    p_watch.add_argument(
        "--install", action="store_true", help="Install silent watcher into Windows Startup"
    )
    p_watch.add_argument(
        "--uninstall", action="store_true", help="Remove silent watcher from Windows Startup"
    )
    p_watch.add_argument(
        "--silent", action="store_true", help="Run without printing console logs"
    )
    p_watch.add_argument(
        "--once", action="store_true", help="Check once and exit immediately"
    )
    p_watch.add_argument(
        "--interval", type=float, default=1.0, help="Poll interval in seconds (default: 1.0s)"
    )

    # nuro calc (Interactive Scientific Graphing Calculator Shell or One-liner)
    p_calc = sub.add_parser(
        "calc", help="Whole Scientific Graphing Calculator & Calculus Engine"
    )
    p_calc.add_argument(
        "query",
        nargs="*",
        help="Optional math expression, equation, or calculus question",
    )

    # nuro chem (Advanced Chemistry Balancer & Numericals)
    p_chem = sub.add_parser(
        "chem", help="Chemistry Reaction Balancer, Molar Mass, pH & Kinetics"
    )
    p_chem.add_argument(
        "query",
        nargs="*",
        help="Optional chemical equation, molar mass, pH, or kinetics query",
    )

    # nuro phy (Over-the-Top Physics Simulation & Formulas)
    p_phy = sub.add_parser(
        "phy", help="Over-the-Top Physics Engine (Mechanics, RLC, Relativity, Quantum)"
    )
    p_phy.add_argument(
        "query",
        nargs="*",
        help="Optional physics query (e.g. 'projectile v=50 angle=45')",
    )

    # nuro session (Manage & Switch Named Calculation Sessions)
    p_session = sub.add_parser(
        "session", help="Manage, list, and switch named calculation sessions"
    )
    p_session.add_argument(
        "args", nargs="*", help="Subcommands: switch <name>, new <name>, list"
    )

    # nuro obsidian / graph (Open Obsidian Color-Coded Graph View of Session Calculations)
    p_obs = sub.add_parser(
        "obsidian", help="Sync & launch Obsidian interactive Color-Coded Graph View"
    )
    p_obs.add_argument(
        "vault",
        nargs="*",
        help="Optional vault name (e.g. 'ICSE Class 10'; default: Neural Loop Calculations)",
    )
    p_graph = sub.add_parser(
        "graph", help="Sync & launch Obsidian interactive Color-Coded Graph View"
    )
    p_graph.add_argument(
        "vault",
        nargs="*",
        help="Optional vault name (default: Neural Loop Calculations)",
    )

    # nuro solve / math
    p_solve = sub.add_parser(
        "solve", help="Natural-language formula solver with Unicode graph"
    )
    p_solve.add_argument(
        "query",
        nargs="+",
        help="Formula or question (e.g. 'sqrt of 16 times pi', 'sin(x)*exp(-x/5)')",
    )

    # nuro term / serial (Direct interactive serial terminal to ESP32-S3 hardware)
    p_term = sub.add_parser(
        "term", help="Interactive direct serial console to the ESP32-S3"
    )
    p_term.add_argument(
        "--port", default=None, help="Serial port (default: auto-detect)"
    )
    p_term.add_argument(
        "--baud", default=2000000, type=int, help="Baud rate (default: 2000000)"
    )

    args, unknown = parser.parse_known_args()

    # If first arg is not a known subcommand -> smart route based on query contents!
    if not args.command and unknown:
        q_raw = " ".join(unknown)
        q_low = q_raw.lower()
        if q_low.startswith("session"):
            print(handle_session_command(unknown[1:]))
            return
        elif (
            "->" in q_raw
            or "➔" in q_raw
            or any(w in q_low for w in ["molar mass", "titration", "kinetics", "kmno4"])
        ):
            from host.chem import solve_chem_numerical

            print(solve_chem_numerical(q_raw))
            return
        elif any(
            w in q_low
            for w in [
                "projectile",
                "rlc",
                "relativity",
                "bohr",
                "blackbody",
                "photoelectric",
                "lorentz",
                "damping",
                "pascal",
                "hydraulic",
                "archimedes",
            ]
        ):
            from host.phy import solve_physics

            print(solve_physics(q_raw))
            return
        elif "obsidian" in q_low or "graph" in q_low or "vault" in q_low:
            from host.obsidian_sync import launch_obsidian_vault

            print(launch_obsidian_vault())
            return
        else:
            from host.calc import calculate_and_graph

            print(calculate_and_graph(q_raw))
            return

    cmd = args.command or "calc"

    if cmd == "setup":
        from host.setup_env import run_setup

        raise SystemExit(
            run_setup(
                recreate=getattr(args, "recreate", False),
                confirm=getattr(args, "confirm", None),
                with_dev=not getattr(args, "no_dev", False),
                skip_doctor=getattr(args, "skip_doctor", False),
            )
        )

    if cmd in ("tui", "neural", "mneuro", "live"):
        from host.board import resolve_port
        from host.dash.tui import run_tui

        port = resolve_port(getattr(args, "port", None))
        frames = getattr(args, "frames", None)
        _announce_port(port)
        run_tui(port=port, max_frames=frames)

    elif cmd == "demo":
        from host.demo import run_demo

        run_demo(
            serial_port=getattr(args, "serial_port", None),
            web_port=getattr(args, "web_port", 8765),
            frames=getattr(args, "frames", None),
            open_browser=not getattr(args, "no_browser", False),
            arm=bool(getattr(args, "arm", False)),
        )

    elif cmd == "web":
        from host.board import resolve_port
        from host.dash.web_cockpit import run_web_cockpit
        import threading

        web_port = getattr(args, "port", 8765)
        ser_port = resolve_port(getattr(args, "serial_port", None))
        no_browse = getattr(args, "no_browser", False)
        _announce_port(ser_port)

        if not no_browse:

            def open_tab():
                import time

                time.sleep(0.8)
                webbrowser.open(f"http://127.0.0.1:{web_port}")

            threading.Thread(target=open_tab, daemon=True).start()

        run_web_cockpit(port=web_port, serial_port=ser_port)

    elif cmd == "gui":
        from host.board import resolve_port
        from host.dash.app import run_live_interactive_cockpit

        port = resolve_port(getattr(args, "port", None))
        _announce_port(port)
        run_live_interactive_cockpit(port=port)

    elif cmd == "bench":
        from host.board import resolve_port
        from host.hil.transport import run_hil_benchmark

        port = resolve_port(getattr(args, "port", None))
        n_frames = getattr(args, "frames", 1000)
        arm = bool(getattr(args, "arm", False))
        _announce_port(port)
        try:
            res = run_hil_benchmark(port=port, n_frames=n_frames, arm=arm)
        except RuntimeError as exc:
            print(f"[nuro] {exc}", file=sys.stderr)
            return 2
        import json

        print(json.dumps(res, indent=2))

    elif cmd == "eval":
        from host.replay.predictor_eval import (
            evaluate_gate_g5_detector,
            evaluate_gate_g6_predictor,
        )
        import json

        g5 = evaluate_gate_g5_detector()
        g6 = evaluate_gate_g6_predictor()
        print("\n=== SYNTHETIC SURROGATE EVALUATION — NOT CHB-MIT ===")
        print("  These numbers come from signals generated inside the repo.")
        print("  Real CHB-MIT EDF replay is Phase 2; do not cite as clinical data.")
        print("\n=== GATE G5: INT8 LIF BURST DETECTOR (SYNTHETIC) ===")
        print(f"  Int8 LIF SNN AUROC       : {g5['auroc_int8_lif_snn']:.4f}")
        print(f"  Raw Amplitude Baseline   : {g5['auroc_amplitude_baseline']:.4f}")
        print(f"  Envelope Z-Score Baseline: {g5['auroc_envelope_z_baseline']:.4f}")
        print(
            f"  Resting Spike Sparsity   : {g5['resting_spike_sparsity_pct']:.2f}% zero-spike"
        )
        print("\n=== GATE G6: PRE-ICTAL PREDICTOR (SYNTHETIC SURROGATE) ===")
        print(
            f"  Mean Test AUROC (synthetic chb20-23) : {g6['summary']['mean_test_auroc']:.4f}"
        )
        print(
            f"  Median Lead Time Before Onset        : {g6['summary']['median_lead_time_s']:.1f} s"
        )
        print(
            f"  False Alarms per Hour                : {g6['summary']['mean_fa_per_hour']:.3f} / hr"
        )
        print(f"  Model                                : {g6['model_architecture']}")

    elif cmd == "test":
        import pytest
        from pathlib import Path

        tests_dir = str(Path(__file__).resolve().parents[1] / "tests")
        sys.exit(pytest.main(["-v", tests_dir]))

    elif cmd == "figures":
        from host.analysis.figures import generate_all_figures

        files = generate_all_figures()
        print(f"Successfully generated {len(files)} figure files (PNG + PDF):")
        for f in files:
            print(f"  - {f}")

    elif cmd == "calc":
        from host.calc import calculate_and_graph, run_calc_repl

        q_list = getattr(args, "query", [])
        if q_list:
            print(calculate_and_graph(" ".join(q_list)))
        else:
            run_calc_repl()

    elif cmd == "chem":
        from host.chem import solve_chem_numerical, run_chem_repl

        q_list = getattr(args, "query", [])
        if q_list:
            print(solve_chem_numerical(" ".join(q_list)))
        else:
            run_chem_repl()

    elif cmd == "phy":
        from host.phy import solve_physics, run_phy_repl

        q_list = getattr(args, "query", [])
        if q_list:
            print(solve_physics(" ".join(q_list)))
        else:
            run_phy_repl()

    elif cmd == "session":
        s_args = getattr(args, "args", [])
        print(handle_session_command(s_args))

    elif cmd in ("obsidian", "graph"):
        from host.obsidian_sync import launch_obsidian_vault

        v_list = getattr(args, "vault", [])
        v_name = " ".join(v_list) if v_list else None
        print(launch_obsidian_vault(v_name))

    elif cmd in ("term", "serial"):
        from host.board import resolve_port

        port = resolve_port(getattr(args, "port", None))
        baud = getattr(args, "baud", 2000000)
        import serial
        from serial.tools.miniterm import Miniterm

        print(
            f"\033[1;36m[TERMINAL]\033[0m Connecting to ESP32-S3 on {port} @ {baud} baud (Press Ctrl+] to exit)..."
        )
        ser = serial.Serial(port, baud, timeout=0.1)
        term = Miniterm(ser, echo=False)
        term.set_rx_encoding("utf-8")
        term.set_tx_encoding("utf-8")
        ser.write(b"\r\n")
        term.start()
        try:
            term.join()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            term.close()
            ser.close()

    elif cmd == "solve":
        from host.calc import calculate_and_graph

        query_str = " ".join(getattr(args, "query", []))
        print(calculate_and_graph(query_str))

    elif cmd == "info":
        import json as _json

        from host.board import identity_summary, probe_identity

        port = getattr(args, "port", None)
        print(f"=== PROBING ESP32-S3 ON {port or 'AUTO-DETECTED PORT'} (READ-ONLY) ===")
        facts = probe_identity(port)
        if getattr(args, "json", False):
            print(_json.dumps(facts, indent=2))
        else:
            print("\n".join(identity_summary(facts)))

    elif cmd == "doctor":
        from host.doctor import run_doctor

        raise SystemExit(run_doctor(port=getattr(args, "port", None)))

    elif cmd in ("watch", "pnp", "plug-and-play", "autolaunch"):
        from host.watch import run_watcher

        sys.exit(
            run_watcher(
                poll_interval=getattr(args, "interval", 1.0),
                install=getattr(args, "install", False),
                uninstall=getattr(args, "uninstall", False),
                silent=getattr(args, "silent", False),
                once=getattr(args, "once", False),
            )
        )


if __name__ == "__main__":
    main()
