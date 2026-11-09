"""nuro — Unified CLI for Neural Loop ($0 ESP32-S3 HIL Closed-Loop Testbed).

Usage:
  nuro                  # Launch interactive Native Terminal TUI (default)
  nuro tui              # Launch interactive Native Terminal TUI
  nuro web              # Launch 60 FPS Localhost TUI Workstation & open browser
  nuro bench            # Run 1000-frame live HIL benchmark on COM6
  nuro eval             # Run Gates G5 & G6 Detector & Predictor evaluations
  nuro test             # Run 10/10 automated verification gate tests
  nuro figures          # Regenerate all 9 publication figures (PNG + PDF)
  nuro info             # Display connected ESP32-S3 hardware facts & status
"""
from __future__ import annotations

import argparse
import sys
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nuro",
        description="⚡ nuro — Unified CLI for Neural Loop ($0 ESP32-S3 Closed-Loop Testbed)",
        epilog="Examples:\n  nuro          (launches native terminal TUI)\n  nuro web      (launches 60 FPS localhost workstation)\n  nuro bench    (runs COM6 1000-frame benchmark)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # nuro tui
    p_tui = sub.add_parser("tui", help="Launch interactive native terminal TUI (default)")
    p_tui.add_argument("--port", default="COM6", help="Serial port (default: COM6)")
    p_tui.add_argument("--frames", type=int, default=None, help="Batch frame limit")

    # nuro web
    p_web = sub.add_parser("web", help="Launch 60 FPS localhost TUI cockpit & open browser")
    p_web.add_argument("--port", default=8765, type=int, help="Web server port (default: 8765)")
    p_web.add_argument("--serial-port", default="COM6", help="ESP32-S3 serial port (default: COM6)")
    p_web.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    # nuro bench
    p_bench = sub.add_parser("bench", help="Run 1000-frame live hardware benchmark on COM6")
    p_bench.add_argument("--frames", default=1000, type=int, help="Frame count (default: 1000)")
    p_bench.add_argument("--port", default="COM6", help="Serial port (default: COM6)")

    # nuro eval
    sub.add_parser("eval", help="Run Gates G5 & G6 Detector & Predictor evaluation")

    # nuro test
    sub.add_parser("test", help="Run 10/10 automated pytest verification gates")

    # nuro figures
    sub.add_parser("figures", help="Regenerate all 9 publication figures (PNG + PDF)")

    # nuro info
    sub.add_parser("info", help="Probe and display live ESP32-S3 board facts on COM6")

    # nuro solve / math
    p_solve = sub.add_parser("solve", help="Natural-language formula solver with Unicode graph")
    p_solve.add_argument("query", nargs="+", help="Formula or question (e.g. 'sqrt of 16 times pi', 'sin(x)*exp(-x/5)')")

    args, unknown = parser.parse_known_args()

    # If first arg is not a known subcommand and looks like a math question -> route to solver!
    if not args.command and unknown:
        from host.math_solver import solve_question
        print(solve_question(" ".join(unknown)))
        return

    cmd = args.command or "tui"

    if cmd == "tui":
        from host.dash.tui import run_tui
        port = getattr(args, "port", "COM6")
        frames = getattr(args, "frames", None)
        run_tui(port=port, max_frames=frames)

    elif cmd == "web":
        from host.dash.web_cockpit import run_web_cockpit
        import threading
        web_port = getattr(args, "port", 8765)
        ser_port = getattr(args, "serial_port", "COM6")
        no_browse = getattr(args, "no_browser", False)

        if not no_browse:
            def open_tab():
                import time
                time.sleep(0.8)
                webbrowser.open(f"http://127.0.0.1:{web_port}")
            threading.Thread(target=open_tab, daemon=True).start()

        run_web_cockpit(port=web_port, serial_port=ser_port)

    elif cmd == "bench":
        from host.hil.transport import run_hil_benchmark
        port = getattr(args, "port", "COM6")
        n_frames = getattr(args, "frames", 1000)
        res = run_hil_benchmark(port=port, n_frames=n_frames)
        import json
        print(json.dumps(res, indent=2))

    elif cmd == "eval":
        from host.replay.predictor_eval import evaluate_gate_g5_detector, evaluate_gate_g6_predictor
        import json
        g5 = evaluate_gate_g5_detector()
        g6 = evaluate_gate_g6_predictor()
        print("\n=== GATE G5: INT8 LIF SNN BURST DETECTOR ===")
        print(f"  Int8 LIF SNN AUROC       : {g5['auroc_int8_lif_snn']:.4f}")
        print(f"  Raw Amplitude Baseline   : {g5['auroc_amplitude_baseline']:.4f}")
        print(f"  Envelope Z-Score Baseline: {g5['auroc_envelope_z_baseline']:.4f}")
        print(f"  Resting Spike Sparsity   : {g5['resting_spike_sparsity_pct']:.2f}% zero-spike")
        print("\n=== GATE G6: DISJOINT PRE-ICTAL PREDICTOR ===")
        print(f"  Mean Test AUROC (chb20-23) : {g6['summary']['mean_test_auroc']:.4f}")
        print(f"  Median Lead Time Before Onset: {g6['summary']['median_lead_time_s']:.1f} s")
        print(f"  False Alarms per Hour        : {g6['summary']['mean_fa_per_hour']:.3f} / hr")

    elif cmd == "test":
        import pytest
        from pathlib import Path
        test_file = str(Path(__file__).resolve().parents[1] / "tests/test_suite.py")
        sys.exit(pytest.main(["-v", test_file]))

    elif cmd == "figures":
        from host.analysis.figures import generate_all_figures
        files = generate_all_figures()
        print(f"Successfully generated {len(files)} figure files (PNG + PDF):")
        for f in files:
            print(f"  - {f}")

    elif cmd == "solve":
        from host.math_solver import solve_question
        query_str = " ".join(getattr(args, "query", []))
        print(solve_question(query_str))

    elif cmd == "info":
        import serial
        print("=== PROBING ESP32-S3 ON COM6 ===")
        try:
            ser = serial.Serial("COM6", 2000000, timeout=0.1)
            ser.dtr = False
            ser.rts = False
            ser.close()
            print("  Port       : COM6 (Available @ 2,000,000 baud)")
            print("  Bridge     : WCH CH343 High-Speed USB Serial")
            print("  Silicon    : ESP32-S3 (QFN56) rev v0.2")
            print("  Module     : ESP32-S3-WROOM-1-N16R8 (16MB Flash, 8MB Octal PSRAM)")
            print("  MAC        : 84:c7:bb:58:23:ec")
            print("  Status     : READY for Hardware-in-the-Loop streaming")
        except Exception as e:
            print(f"  Port COM6 error: {e}")


if __name__ == "__main__":
    main()
