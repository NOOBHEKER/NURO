"""nuro — Unified CLI for Neural Loop ($0 ESP32-S3 HIL Closed-Loop Testbed).

Usage:
  nuro                  # Show all options, then open the interactive shell
  nuro chat [n]         # PC-only offline AI chatbot quiz (n questions, default 3)
  nuro ask "question"   # Ask the local AI anything (all subjects, PC only)
  nuro llm status       # Show the local model / runtime
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
import os
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


def print_main_menu() -> None:
    """Print the Warp-CLI-style starting screen shown when the user types `nuro`."""
    from host.dash.warp_fx import print_warp_cli_menu

    print_warp_cli_menu(version=_version(), animate=True)


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


def _looks_like_math_or_sci(q: str) -> bool:
    """Return True if a free-form query is a math/chem/physics expression rather than natural chat."""
    import re

    ql = q.strip().lower()
    if not ql:
        return False
    if "->" in q or "➔" in q:
        return True
    sci_kw = (
        "molar mass", "titration", "kinetics", "kmno4", "projectile", "rlc",
        "relativity", "bohr", "blackbody", "photoelectric", "lorentz", "damping",
        "pascal", "hydraulic", "archimedes", "d/dx", "diff(", "integrate",
        "integral", "derivative", "solve(", "factor(", "limit(", "sin(", "cos(",
        "tan(", "log(", "ln(", "sqrt(", "plot ", "graph ",
    )
    if any(k in ql for k in sci_kw):
        return True
    # Pure arithmetic/algebraic expressions (contain digits + math operators, no long prose)
    if re.search(r"\d", ql) and re.search(r"[+\-*/^=]", ql) and len(ql.split()) <= 6:
        return True
    return False


def _instant_fast_reply(q: str) -> str | None:
    """Zero-latency (<1 ms) reply for simple greetings/small-talk while local LLM stays warm."""
    ql = q.strip().lower().rstrip("!?., ")
    if ql in ("hi", "hey", "hello", "yo", "sup", "hii", "heyy", "hola", "namaste"):
        return "Hey! I'm NURO — ask me anything, type math/science, or press Enter for /tui & commands."
    if ql in ("good morning", "gm", "good afternoon", "good evening"):
        return f"{q.strip().capitalize()}! What are we working on today?"
    if ql in ("how are you", "how r u", "hru", "whats up", "what's up"):
        return "Running fast and ready! Ask any question, solve an equation, or launch /tui."
    if ql in ("thanks", "thank you", "thx", "ty"):
        return "Anytime! Let me know if you need anything else."
    if ql in ("who are you", "what are you", "what can you do"):
        return (
            "I'm NURO — your offline closed-loop ESP32-S3 SNN cockpit + local Qwen3 AI + "
            "math/chem/physics solver. Ask anything, or press Enter to return to the 3D menu."
        )
    return None


def _inline_ai_reply(first_query: str) -> None:
    """Answer any natural-language message directly via local Qwen3 (auto-starting Ollama) with follow-up chat."""
    from host import llm_backend

    history: List[dict] = []
    q = first_query.strip()
    print()
    while q:
        if q.lower() in ("back", "menu", "q", "quit", "exit"):
            return
        fast = _instant_fast_reply(q)
        if fast is not None:
            # Respond in <1 ms while quietly warming Ollama in the background for the next question!
            _start_llm_background_warmup()
            print(f"\033[1;36m[nuro]\033[0m {fast}\n")
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": fast})
        else:
            if not llm_backend.ensure_server(wait_seconds=18.0):
                try:
                    from host.chatbot import answer_question

                    ans = answer_question(q)
                    print(f"\033[1;36mNURO ❯\033[0m {ans}\n")
                except Exception:
                    print("\033[33m[nuro] Local AI not available. Run `nuro llm pull qwen3:0.6b`.\033[0m\n")
                return
            backend = llm_backend.detect()
            tag = backend.model if backend else llm_backend.DEFAULT_MODEL
            print(f"\033[1;36m[{tag}]\033[0m ", end="", flush=True)
            chunks: List[str] = []
            for piece in llm_backend.ask_stream(q, history=history, max_tokens=200):
                sys.stdout.write(piece)
                sys.stdout.flush()
                chunks.append(piece)
            print("\n")
            reply = "".join(chunks).strip()
            if reply:
                history.append({"role": "user", "content": q})
                history.append({"role": "assistant", "content": reply})
            else:
                print("\033[33m[nuro] No response from local model.\033[0m\n")
        if not sys.stdin.isatty():
            return
        try:
            nxt = input("\033[1;36mYou ❯\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not nxt:
            return
        q = nxt


def _start_llm_background_warmup() -> None:
    """Quietly boot Ollama in the background on `nuro` launch so 'hey' responds fast."""
    import threading

    def _boot() -> None:
        try:
            from host import llm_backend

            if llm_backend.ensure_server(wait_seconds=15.0):
                llm_backend.warmup(timeout=30.0)
        except Exception:
            pass

    threading.Thread(target=_boot, daemon=True).start()


def main() -> None:
    # Windows classic Command Prompt defaults to CP437; switch console to UTF-8 + VT
    # and reconfigure streams so Unicode box/block glyphs never turn into mojibake.
    if os.name == "nt":
        try:
            import ctypes

            k32 = ctypes.windll.kernel32
            k32.SetConsoleOutputCP(65001)
            k32.SetConsoleCP(65001)
            h_out = k32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k32.GetConsoleMode(h_out, ctypes.byref(mode)):
                k32.SetConsoleMode(h_out, mode.value | 0x0004)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    # Apply bundled JetBrains Mono font to the active terminal window (no-op if piped/unsupported)
    try:
        from host.dash.console_font import apply_bundled_console_font

        apply_bundled_console_font()
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        prog="nuro",
        description="⚡ nuro — Unified CLI for Neural Loop ($0 ESP32-S3 Closed-Loop Testbed)",
        epilog="Examples:\n  nuro          (show options + open the shell; type test for the AI chatbot)\n  nuro chat 5   (PC-only AI chatbot with 5 questions)\n  nuro ask \"explain photosynthesis in detail\"  (local AI, any subject)\n  nuro llm status   (show the local model)\n  nuro setup    (bootstrap this computer once)\n  nuro demo     (web cockpit + TUI + GUI)\n  nuro info     (read-only board probe)\n  nuro bench    (1000-frame live benchmark on the auto-detected port)",
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

    # nuro treat
    p_treat = sub.add_parser(
        "treat",
        aliases=["patient", "clinic"],
        help="Active clinical closed-loop epilepsy treatment on PhysioNet CHB-MIT patient EEG",
    )
    p_treat.add_argument(
        "--port", default=None, help="Serial port (default: auto-detect)"
    )
    p_treat.add_argument("--frames", type=int, default=None, help="Batch frame limit")

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

    # nuro chat (PC-only offline AI chatbot — never runs on the ESP32)
    p_chat = sub.add_parser(
        "chat",
        aliases=["chatbot", "ai"],
        help="Start the offline AI chatbot quiz (PC only, never on ESP32)",
    )
    p_chat.add_argument(
        "questions",
        nargs="?",
        type=int,
        default=3,
        help="Number of test questions to ask (default: 3)",
    )

    # nuro ask "question" (one-shot local LLM answer, PC only)
    p_ask = sub.add_parser(
        "ask",
        aliases=["query"],
        help="Ask the local AI anything, any subject (PC only)",
    )
    p_ask.add_argument("question", nargs="+", help="Your question")
    p_ask.add_argument(
        "--detail", action="store_true", help="Ask for a longer, more detailed answer"
    )
    p_ask.add_argument(
        "--model", default=None, help="Override the model (default: qwen3:0.6b)"
    )
    p_ask.add_argument(
        "--fast",
        action="store_true",
        help="Use the fast local model (qwen3:0.6b) for quicker answers",
    )

    # nuro llm [status|models|pull <model>|start]
    p_llm = sub.add_parser(
        "llm", help="Manage the local AI model (status, models, pull, start)"
    )
    p_llm.add_argument("action", nargs="?", default="status")
    p_llm.add_argument("model", nargs="?", default=None)

    # nuro symptoms "fever, cough" (educational symptom checker, PC only)
    p_sym = sub.add_parser(
        "symptoms",
        aliases=["symptom", "health"],
        help="Describe symptoms for an educational estimate (not a diagnosis)",
    )
    p_sym.add_argument(
        "text", nargs="*", help="Your symptoms, e.g. \"fever, cough, body ache\""
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

    _sub_choices = set(sub.choices.keys())
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in _sub_choices:
        args = argparse.Namespace(command=None)
        unknown = sys.argv[1:]
    else:
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
        elif _looks_like_math_or_sci(q_raw):
            from host.calc import calculate_and_graph

            print(calculate_and_graph(q_raw))
            return
        else:
            _inline_ai_reply(q_raw)
            return

    if args.command is None:
        from host.dash.warp_fx import run_warp_start_screen
        import shlex

        # Warm up local qwen3:1.7b silently in the background while the 3D screen spins
        if sys.stdin.isatty() and sys.stdout.isatty():
            _start_llm_background_warmup()

        known_top = {
            "setup", "tui", "neural", "mneuro", "live", "treat", "patient", "clinic",
            "demo", "web", "gui", "bench", "eval", "test", "figures", "chat", "chatbot",
            "quiz", "ask", "ai", "symptoms", "health", "check", "llm", "calc", "chem",
            "phy", "session", "obsidian", "graph", "term", "serial", "solve", "info",
            "doctor", "watch", "pnp", "plug-and-play", "autolaunch",
        }
        # Interactive views that take over the full screen: return straight to the
        # NURO start screen when exited (instead of exiting the whole CLI).
        fullscreen_cmds = {"tui", "neural", "mneuro", "live", "treat", "patient", "clinic", "demo", "gui"}
        persistent_chat: List[tuple] = []

        while True:
            first_cmd = run_warp_start_screen(version=_version(), chat_log=persistent_chat)
            if not first_cmd or first_cmd.lower() in ("q", "quit", "exit"):
                return
            try:
                tokens = shlex.split(first_cmd)
            except ValueError:
                tokens = first_cmd.split()
            if tokens and tokens[0].lower() in known_top:
                saved_argv = sys.argv[:]
                try:
                    sys.argv = [sys.argv[0]] + tokens
                    main()
                except SystemExit:
                    pass
                except KeyboardInterrupt:
                    pass
                finally:
                    sys.argv = saved_argv
                if tokens[0].lower() not in fullscreen_cmds and sys.stdin.isatty():
                    try:
                        input("\n  Press Enter to return to NURO...")
                    except (EOFError, KeyboardInterrupt):
                        pass
                continue
            if first_cmd.lower() in ("help", "?", "commands"):
                from host.calc import _print_calc_help

                _print_calc_help()
                if sys.stdin.isatty():
                    try:
                        input("\n  Press Enter to return to NURO...")
                    except (EOFError, KeyboardInterrupt):
                        pass
                continue
            if _looks_like_math_or_sci(first_cmd):
                from host.calc import calculate_and_graph, run_calc_repl

                print()
                print(calculate_and_graph(first_cmd))
                print()
                run_calc_repl(show_banner=False)
                continue
            # Conversational input (e.g., "hey", "explain black holes") -> answer directly with local AI!
            _inline_ai_reply(first_cmd)
            continue

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

    if cmd in ("tui", "neural", "mneuro", "live", "treat", "patient", "clinic"):
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

    elif cmd in ("chat", "chatbot", "ai"):
        from host.chatbot import run_chatbot_session

        run_chatbot_session(getattr(args, "questions", 3) or 3)

    elif cmd in ("symptoms", "symptom", "health"):
        from host import health

        described = " ".join(getattr(args, "text", [])).strip()
        if not described:
            print('Usage: nuro symptoms "fever, cough, body ache"')
            print(f"Conditions covered: {health.condition_options()}")
            raise SystemExit(2)
        answer_text = health.answer(described)
        if answer_text is None:
            print("I could not spot any symptoms I recognise in that sentence.")
            print('Try: nuro symptoms "fever, cough, loss of smell"')
            print(f"Conditions covered: {health.condition_options()}")
            raise SystemExit(2)
        print(answer_text)

    elif cmd in ("ask", "query"):
        import sys as _sys

        from host import llm_backend

        question = " ".join(getattr(args, "question", [])).strip()
        if not question:
            print("Usage: nuro ask \"your question\"")
            raise SystemExit(2)

        if getattr(args, "model", None):
            os.environ["NURO_LLM_MODEL"] = args.model
            llm_backend.set_model(args.model)

        if not llm_backend.ensure_server(wait_seconds=20.0):
            print(llm_backend.status_text())
            print("\nRun `nuro llm pull qwen3:0.6b` once, then try again.")
            raise SystemExit(2)

        if getattr(args, "fast", False):
            fast = llm_backend.fast_backend()
            if fast is None:
                print(
                    f"Fast model not installed. Run: nuro llm pull {llm_backend.FAST_MODEL}"
                )
                raise SystemExit(2)
            llm_backend.set_model(fast.model)

        backend = llm_backend.detect()
        print(f"[{backend.model}] ", end="", flush=True)
        got = False
        for piece in llm_backend.ask_stream(
            question,
            max_tokens=600 if getattr(args, "detail", False) else 220,
        ):
            _sys.stdout.write(piece)
            _sys.stdout.flush()
            got = True
        print()
        if not got:
            print("The local model did not return an answer.", file=_sys.stderr)
            raise SystemExit(1)

    elif cmd == "llm":
        from host import llm_backend

        action = (getattr(args, "action", None) or "status").lower()
        if action == "status":
            print(llm_backend.status_text())
        elif action in ("models", "list"):
            models = llm_backend.ollama_models()
            print("\n".join(models) if models else "No models installed.")
        elif action == "start":
            print("Started." if llm_backend.ensure_server() else llm_backend.status_text())
        elif action == "pull":
            model = getattr(args, "model", None) or llm_backend.DEFAULT_MODEL
            print(f"Pulling {model} ...")
            raise SystemExit(llm_backend.pull_model(model))
        else:
            print("Usage: nuro llm [status|models|start|pull <model>]")
            raise SystemExit(2)

    elif cmd == "calc":
        from host.calc import calculate_and_graph, run_calc_repl

        q_list = getattr(args, "query", [])
        if q_list:
            print(calculate_and_graph(" ".join(q_list)))
        else:
            run_calc_repl(show_banner=(args.command is not None))

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
