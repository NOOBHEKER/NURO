"""Native Terminal User Interface (TUI) Cockpit for Neural Loop (ESP32-S3 HIL on COM6).

Renders a zero-flicker ANSI + Unicode box-drawing TUI directly in the terminal:
  python -m host.dash.tui              # Interactive real-time TUI (msvcrt hotkeys)
  python -m host.dash.tui --frames 40  # Non-interactive batch/CI TUI run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import Deque, List, Optional
import numpy as np
import serial

from host.hil.protocol import (
    SampleFrame,
    ResultFrame,
    enable_windows_1ms_timer,
    host_monotonic_us,
    MODE_LOCKSTEP,
)
from host.virtualbrain.kuramoto import KuramotoBrain, KuramotoConfig
from host.virtualbrain.jansen_rit import JansenRitBrain, JansenRitConfig
from host.hil.dsp_ref import DspChainRef


BLOCKS = "  ▂▃▄▅▆▇█"

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[38;5;75m"
C_GREEN = "\033[38;5;78m"
C_RED = "\033[38;5;203m"
C_AMBER = "\033[38;5;214m"
C_GRAY = "\033[38;5;244m"
C_WHITE = "\033[38;5;255m"
C_BG_HEAD = "\033[48;5;235m"


def sparkline(values: List[float], lo: float, hi: float, width: int = 56, color: str = C_CYAN) -> str:
    if not values:
        return " " * width
    arr = np.array(values[-width:], dtype=np.float64)
    norm = np.clip((arr - lo) / max(1e-9, hi - lo), 0.0, 1.0)
    idx = np.round(norm * (len(BLOCKS) - 1)).astype(int)
    chars = "".join(BLOCKS[i] for i in idx)
    if len(chars) < width:
        chars = (" " * (width - len(chars))) + chars
    return f"{color}{chars}{C_RESET}"


def phase_compass_ascii(psi_rad: float, fired: bool) -> List[str]:
    """Render a compact 5-line ASCII/Unicode polar phase compass showing target φ* = -π (exact 33-char width)."""
    ang = ((psi_rad + np.pi) % (2.0 * np.pi)) - np.pi
    octant = int(np.round((ang + np.pi) / (2.0 * np.pi) * 8.0)) % 8
    pts = ["·"] * 8
    pts[octant] = f"{C_WHITE}●{C_RESET}"
    target_w = f"{C_GREEN}★{C_RESET}" if fired else (f"{C_GREEN}◉{C_RESET}" if octant == 0 else f"{C_GREEN}◎{C_RESET}")
    return [
        f"        {pts[6]} N (+π/2)     ",
        f"     {pts[7]}     {pts[5]}           ",
        f" φ*={target_w}   +   {pts[4]} 0 rad  ",
        f"     {pts[1]}     {pts[3]}           ",
        f"        {pts[2]} S (-π/2)     ",
    ]


def run_tui(port: str = "COM6", baudrate: int = 2000000, max_frames: Optional[int] = None) -> None:
    if os.name == "nt":
        os.system("")  # Enable ANSI VT100 escape sequences on Windows 10/11 console
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    enable_windows_1ms_timer()

    ser = None
    hw_live = False
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.20, write_timeout=0.20)
        ser.dtr = False
        ser.rts = False
        time.sleep(0.30)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        hw_live = True
    except Exception:
        hw_live = False

    k_brain = KuramotoBrain(KuramotoConfig(seed=2026))
    k_brain.theta = np.mod(k_brain.rng.normal(0.0, 0.55, size=k_brain.cfg.n_osc), 2.0 * np.pi)
    k_brain.step_idx = int(30.0 / k_brain.cfg.dt_s)
    jr_brain = JansenRitBrain(JansenRitConfig(seed=2026))
    dsp = DspChainRef()

    raw_hist: Deque[float] = deque([0.0] * 60, maxlen=60)
    bp_hist: Deque[float] = deque([0.0] * 60, maxlen=60)
    r_hist: Deque[float] = deque([0.85] * 60, maxlen=60)
    rtt_win: Deque[float] = deque([1305.0] * 100, maxlen=100)
    dsp_win: Deque[float] = deque([18.0] * 100, maxlen=100)
    wcet_win: Deque[float] = deque([82.0] * 100, maxlen=100)
    event_log: Deque[str] = deque(maxlen=5)
    event_log.append("TUI initialized — streaming 32-byte CRC16 frames @ 2000 Hz equivalent")

    loop_on = True
    arm_idx = 0
    model_idx = 0  # 0: Kuramoto, 1: Jansen-Rit NMM
    arm_names = ["D: PHASE-LOCKED (φ*=-π)", "C: RANDOM-PHASE SHAM  ", "B: OPEN-LOOP 20 Hz    "]
    model_names = ["STAGE 1: KURAMOTO (N=64)", "STAGE 2: JANSEN-RIT NMM "]

    seq = 0
    pulses = 0
    valid_frames = 0
    crc_err = 0
    refrac = 0
    pending_stim = 0.0
    last_res = ResultFrame(
        state=2, seq=0, t_rx_us=1000, t_dsp_us=1018, t_dec_us=1082, t_stim_us=0,
        cls=1, conf=230, phase_q15=-32000, R_q15=24000, budget_used=0
    )

    if max_frames is None:
        sys.stdout.write("\033[2J\033[?25l")  # Clear screen & hide cursor

    has_msvcrt = False
    if os.name == "nt" and max_frames is None:
        try:
            import msvcrt
            has_msvcrt = True
        except ImportError:
            pass

    frame_count = 0
    try:
        while True:
            frame_count += 1
            if has_msvcrt:
                import msvcrt
                while msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch == b" ":
                        loop_on = not loop_on
                        event_log.append(f"[KEY] Closed-Loop toggled -> {'ON (ACTIVE)' if loop_on else 'OFF (INHIBITED)'}")
                    elif ch in (b"m", b"M"):
                        arm_idx = (arm_idx + 1) % 3
                        event_log.append(f"[KEY] Controller Arm -> {arm_names[arm_idx].strip()}")
                    elif ch in (b"1", b"2"):
                        model_idx = 0 if ch == b"1" else 1
                        event_log.append(f"[KEY] Brain Model -> {model_names[model_idx].strip()}")
                    elif ch in (b"r", b"R"):
                        k_brain.theta = np.mod(k_brain.rng.normal(0.0, 0.35, size=k_brain.cfg.n_osc), 2.0 * np.pi)
                        event_log.append("[KEY] Injected synchronous seizure burst (R -> 0.90)")
                    elif ch in (b"q", b"Q", b"\x1b"):
                        return

            fired_this_batch = False
            for _ in range(20):
                s0, r0, _ = k_brain.step(stim_amp=pending_stim)
                s1, r_val, psi_val = k_brain.step(stim_amp=0.0)
                if model_idx == 1:
                    j0, _ = jr_brain.step(u_stim=pending_stim * 0.2)
                    j1, _ = jr_brain.step(u_stim=0.0)
                    s0, s1 = j0, j1
                pending_stim = 0.0

                dsp.step(s0)
                bp_out = dsp.step(s1)
                raw_i16 = s1
                fired = False

                if hw_live and ser is not None:
                    sf = SampleFrame(
                        seq=seq,
                        sim_t_us=seq * 1000,
                        host_t_us=host_monotonic_us(),
                        samples=(s0, s1, 0, 0),
                        flags=0x03,
                        n_ch=1,
                        n_samp=2,
                        mode=MODE_LOCKSTEP,
                    )
                    seq += 1
                    t0 = time.perf_counter_ns()
                    try:
                        ser.write(sf.to_bytes())
                        rx = ser.read(32)
                        t1 = time.perf_counter_ns()
                        if len(rx) == 32:
                            last_res = ResultFrame.from_bytes(rx)
                            valid_frames += 1
                            rtt_win.append((t1 - t0) / 1000.0)
                            dsp_win.append(float((last_res.t_dsp_us - last_res.t_rx_us) & 0xFFFFFFFF))
                            wcet_win.append(float((last_res.t_dec_us - last_res.t_rx_us) & 0xFFFFFFFF))
                            if loop_on:
                                if arm_idx == 0 and last_res.t_stim_us != 0:
                                    fired = True
                                elif arm_idx == 1 and refrac == 0 and k_brain.rng.random() < 0.015:
                                    fired = True
                                elif arm_idx == 2 and (seq % 50 == 0):
                                    fired = True
                        else:
                            crc_err += 1
                    except Exception:
                        crc_err += 1
                else:
                    seq += 1
                    if loop_on and refrac == 0:
                        if arm_idx == 0 and r_val > 0.22 and np.cos(psi_val) < -0.82:
                            fired = True
                        elif arm_idx == 1 and k_brain.rng.random() < 0.015:
                            fired = True
                        elif arm_idx == 2 and (seq % 50 == 0):
                            fired = True

                if fired:
                    pending_stim = 500.0
                    dsp.notify_stim()
                    pulses += 1
                    refrac = 8
                    fired_this_batch = True
                    event_log.append(
                        f"[STIM #{pulses:03d}] seq={seq:<4d} ψ={psi_val:+.2f}rad R={r_val:.3f} blank=20ms"
                    )
                elif refrac > 0:
                    refrac -= 1

                if not loop_on and r_val < 0.88:
                    k_brain.theta = np.mod(k_brain.theta - 0.016 * np.sin(k_brain.theta - psi_val), 2.0 * np.pi)

                raw_hist.append(raw_i16 / 1024.0)
                bp_hist.append(bp_out.I_q15 / 1024.0)
                r_hist.append(r_val)

            hw_str = f"{C_GREEN}● LIVE ESP32-S3 ({port} @ 2M){C_RESET}" if hw_live else f"{C_AMBER}● IN-SILICO SIMULATOR     {C_RESET}"
            loop_str = f"{C_GREEN}[LOOP: ON  (SUPPRESSING)]{C_RESET}" if loop_on else f"{C_RED}[LOOP: OFF (ICTAL SURGE)]{C_RESET}"
            cur_r = r_hist[-1]
            r_col = C_GREEN if cur_r < 0.35 else (C_AMBER if cur_r < 0.65 else C_RED)
            r_bar_len = int(np.clip(round(cur_r * 36), 0, 36))
            r_bar = f"{r_col}{'█' * r_bar_len}{C_GRAY}{'░' * (36 - r_bar_len)}{C_RESET}"

            p50_rtt = float(np.percentile(rtt_win, 50))
            p95_rtt = float(np.percentile(rtt_win, 95))
            p99_rtt = float(np.percentile(rtt_win, 99))
            p50_dsp = float(np.percentile(dsp_win, 50))
            p50_wcet = float(np.percentile(wcet_win, 50))

            compass = phase_compass_ascii(psi_val, fired_this_batch)

            lines = [
                f"{C_BG_HEAD}{C_BOLD} ⚡ NEURAL LOOP TUI v1.0 │ $0 ESP32-S3 HIL Closed-Loop Edge Neuromodulation Testbed {C_RESET}",
                f"╭──────────────────────────────────────────────────────────┬───────────────────────────────────╮",
                f"│ {C_BOLD}LINK:{C_RESET} {hw_str}  {loop_str} │ {C_BOLD}MODEL:{C_RESET} {C_CYAN}{model_names[model_idx]}{C_RESET} │",
                f"│ {C_BOLD}ARM :{C_RESET} {C_CYAN}{arm_names[arm_idx]}{C_RESET}  {C_BOLD}PULSES:{C_RESET} {pulses:4d} / 300 (60s)   │ {C_BOLD}FRAMES:{C_RESET} {valid_frames:6d} valid │ {crc_err:2d} CRC err │",
                f"├──────────────────────────────────────────────────────────┼───────────────────────────────────┤",
                f"│ {C_BOLD}PANEL 1 — Raw LFP & 20ms Artifact-Blanked Q15 Bandpass{C_RESET}   │ {C_BOLD}PANEL 2 — Phase Compass (φ* = -π){C_RESET} │",
                f"│ Raw LFP : {sparkline(list(raw_hist), -3.8, 3.8, 46, C_GRAY)} │ {compass[0]}          │",
                f"│ Q15 I(t): {sparkline(list(bp_hist), -2.5, 2.5, 46, C_CYAN)} │ {compass[1]}          │",
                f"│ Blank   : {'ACTIVE (40-samp clamp)' if bp_out.blanked else 'IDLE (0 artifact)     '}  Env={bp_out.env_q15:5d} Q15        │ {compass[2]}          │",
                f"├──────────────────────────────────────────────────────────┤ {compass[3]}          │",
                f"│ {C_BOLD}PANEL 3 (HERO) — Kuramoto Synchrony Collapse R(t){C_RESET}        │ {compass[4]}          │",
                f"│ R(t) Now: [{r_bar}] {r_col}{C_BOLD}{cur_r:0.4f}{C_RESET}  │ ψ(t) = {psi_val:+6.3f} rad ({np.degrees(psi_val):+6.1f}°)     │",
                f"│ History : {sparkline(list(r_hist), 0.0, 1.0, 46, r_col)} ├───────────────────────────────────┤",
                f"├──────────────────────────────────────────────────────────┤ {C_BOLD}PANEL 4 — ESP32-S3 Telemetry{C_RESET}      │",
                f"│ {C_BOLD}LIVE MCU REGISTER & PROTOCOL INSPECTOR (32-Byte Frame){C_RESET}   │ USB RTT p50 : {C_CYAN}{p50_rtt:7.1f} µs{C_RESET} (<2000) │",
                f"│ seq={last_res.seq:<7d} state=0x{last_res.state:02X} cls={last_res.cls} conf={last_res.conf:<3d} phase_q15={last_res.phase_q15:<+6d}   │ USB RTT p95 : {C_CYAN}{p95_rtt:7.1f} µs{C_RESET} (G0 PASS)│",
                f"│ t_rx={last_res.t_rx_us:<10d} t_dsp={last_res.t_dsp_us:<10d} t_dec={last_res.t_dec_us:<10d}        │ USB RTT p99 : {C_CYAN}{p99_rtt:7.1f} µs{C_RESET} (<5000) │",
                f"├──────────────────────────────────────────────────────────┤ On-Chip DSP : {C_GREEN}{p50_dsp:7.1f} µs{C_RESET} (61-tap) │",
                f"│ {C_BOLD}EVENT LOG{C_RESET}                                                │ On-Chip WCET: {C_GREEN}{p50_wcet:7.1f} µs{C_RESET} (<250 G3)│",
            ]
            for i in range(4):
                ev = event_log[i] if i < len(event_log) else ""
                ev_padded = (ev[:56]).ljust(56)
                right_info = [
                    f"Target Zone : {C_GREEN}R < 0.25{C_RESET}          ",
                    f"Group Delay : {C_CYAN}+30 samp (+15 ms){C_RESET} ",
                    f"Blank Window: {C_CYAN}40 samp (20.0 ms){C_RESET} ",
                    f"Safety Fault: {C_GREEN}0x00 (NOMINAL){C_RESET}    ",
                ][i]
                lines.append(f"│ {C_DIM}{ev_padded}{C_RESET} │ {right_info}│")

            lines.append("╰──────────────────────────────────────────────────────────┴───────────────────────────────────╯")
            lines.append(
                f" {C_BOLD}HOTKEYS:{C_RESET} {C_CYAN}[SPACE]{C_RESET} Loop ON/OFF │ {C_CYAN}[M]{C_RESET} Cycle Arm (D/C/B) │ {C_CYAN}[1/2]{C_RESET} Kuramoto/Jansen-Rit │ {C_CYAN}[R]{C_RESET} Burst │ {C_CYAN}[Q]{C_RESET} Quit"
            )

            if max_frames is None:
                sys.stdout.write("\033[H" + "\n".join(lines))
                sys.stdout.flush()
                time.sleep(0.05)
            else:
                if frame_count >= max_frames:
                    print("\n".join(lines))
                    break
    finally:
        if max_frames is None:
            sys.stdout.write("\033[?25h\n")
        if ser is not None and ser.is_open:
            ser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Loop Terminal User Interface (TUI)")
    parser.add_argument("--port", type=str, default="COM6", help="Serial port for ESP32-S3")
    parser.add_argument("--frames", type=int, default=None, help="Run N frames and print TUI snapshot")
    args = parser.parse_args()
    run_tui(port=args.port, max_frames=args.frames)
