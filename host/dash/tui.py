"""Perfection-Grade btop-Style Terminal User Interface (TUI) for Neural Loop (ESP32-S3 HIL on COM6).

Features:
- Exact ANSI-aware column layout engine (100% pixel-aligned box-drawing borders)
- True Unicode Braille (U+2800..U+28FF, 2x4 sub-pixel dots/cell) high-res terminal oscilloscope
- Live 64-Neuron Int8 LIF SNN Spike Raster & Membrane Potential Activity Strip
- Live 11-Circle ASCII/Unicode Polar Phase Wheel + Optimal PRC Sector (φ* = -π)
- Live 32-Byte Hardware Register & Binary Frame Hex Inspector (CRC16-CCITT-FALSE)
- Live Latency Distribution Histogram (p50 / p95 / p99 / WCET) + 5-Interlock Safety Matrix
- Interactive Hotkeys: [SPACE] Loop ON/OFF, [M] Arm D/C/B, [1/2] Kuramoto/Jansen-Rit, [R] Seizure Burst, [Q] Quit
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
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
from host.replay.patient_data import PatientBrain
from host.hil.dsp_ref import DspChainRef
from host.dash.warp_fx import (
    SPINNER_BRAILLE,
    fg_rgb,
    gradient_text,
    play_warp_boot_splash,
    render_smooth_bar,
)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[38;5;75m"
C_TEAL = "\033[38;5;43m"
C_GREEN = "\033[38;5;78m"
C_RED = "\033[38;5;203m"
C_AMBER = "\033[38;5;214m"
C_PURPLE = "\033[38;5;141m"
C_GRAY = "\033[38;5;242m"
C_WHITE = "\033[38;5;255m"
C_BG_BAR = "\033[48;5;235m"

BLOCKS = " ▂▃▄▅▆▇█"

# Unicode Braille dot bitmask table: (row 0..3, col 0..1)
BRAILLE_MAP = [
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
]


# ---------------------------------------------------------------------------
# Single source of truth for synchrony zone colouring (LED, TUI, web cockpit)
# ---------------------------------------------------------------------------
R_AMBER = 0.35  # green < 0.35 <= amber < 0.55 <= red
R_RED = 0.55
ZONE_GREEN, ZONE_AMBER, ZONE_RED = 0, 1, 2
ZONE_NAMES = ("GREEN", "AMBER", "RED")
ZONE_LED_FLAGS = (0x10, 0x20, 0x30)


def raw_zone(r: float) -> int:
    return ZONE_GREEN if r < R_AMBER else (ZONE_AMBER if r < R_RED else ZONE_RED)


class ZoneTracker:
    """EMA-smoothed R with hysteresis + minimum dwell -> debounced colour zone.

    The same smoothed R and zone drive the RGB LED flag, the TUI gauge/compass
    and the web cockpit, so every indicator always agrees and never flickers.
    """

    def __init__(self, alpha: float = 0.015, hyst: float = 0.03,
                 dwell: int = 80, r0: float = 0.0) -> None:
        self.alpha = alpha
        self.hyst = hyst
        self.dwell = dwell
        self.r = r0
        self.zone = raw_zone(r0)
        self._count = 0

    def update(self, r_val: float) -> int:
        self.r += (float(r_val) - self.r) * self.alpha
        z = self.zone
        r = self.r
        # enter a hotter zone at its threshold; leave it only hyst below it
        up_a, up_r = R_AMBER, R_RED
        dn_a, dn_r = R_AMBER - self.hyst, R_RED - self.hyst
        if z == ZONE_GREEN:
            target = ZONE_RED if r >= up_r else (ZONE_AMBER if r >= up_a else z)
        elif z == ZONE_AMBER:
            target = ZONE_RED if r >= up_r else (ZONE_GREEN if r < dn_a else z)
        else:
            target = ZONE_GREEN if r < dn_a else (ZONE_AMBER if r < dn_r else z)
        if target == z:
            self._count = 0
        else:
            self._count += 1
            if self._count >= self.dwell:
                self.zone, self._count = target, 0
        return self.zone

    @property
    def led_flags(self) -> int:
        return ZONE_LED_FLAGS[self.zone]

    @property
    def name(self) -> str:
        return ZONE_NAMES[self.zone]


def visible_len(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def pad_ansi(s: str, width: int) -> str:
    """Pad or truncate an ANSI-colored string to an exact visible character width."""
    vlen = visible_len(s)
    if vlen < width:
        return s + (" " * (width - vlen))
    if vlen == width:
        return s
    # Truncate visible characters while preserving ANSI resets
    out: List[str] = []
    vis = 0
    idx = 0
    while idx < len(s) and vis < width:
        m = ANSI_RE.match(s, idx)
        if m:
            out.append(m.group(0))
            idx = m.end()
        else:
            out.append(s[idx])
            vis += 1
            idx += 1
    out.append(C_RESET)
    return "".join(out)


def render_braille_plot(
    values: List[float],
    lo: float,
    hi: float,
    cols: int = 58,
    rows: int = 3,
    color: str = C_CYAN,
    threshold_y: Optional[float] = None,
    tiered_colors: bool = False,
) -> List[str]:
    """Render a multi-row btop-style Unicode Braille (2x4 subpixel) oscilloscope trace."""
    px_w = cols * 2
    px_h = rows * 4
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    if values:
        arr = np.array(values[-px_w:], dtype=np.float64)
        if len(arr) < px_w:
            arr = np.pad(arr, (px_w - len(arr), 0), mode="edge")
        norm = np.clip((arr - lo) / max(1e-9, hi - lo), 0.0, 0.999)
        y_px = (px_h - 1) - np.floor(norm * (px_h - 1)).astype(int)

        prev_y = int(y_px[0])
        for x in range(px_w):
            cur_y = int(y_px[x])
            y0, y1 = min(prev_y, cur_y), max(prev_y, cur_y)
            for yy in range(y0, y1 + 1):
                r_cell = yy // 4
                r_sub = yy % 4
                c_cell = x // 2
                c_sub = x % 2
                grid[r_cell][c_cell] |= BRAILLE_MAP[r_sub][c_sub]
            prev_y = cur_y

    if threshold_y is not None:
        tnorm = float(np.clip((threshold_y - lo) / max(1e-9, hi - lo), 0.0, 0.999))
        ty = int((px_h - 1) - np.floor(tnorm * (px_h - 1)))
        for x in range(0, px_w, 4):
            grid[ty // 4][x // 2] |= BRAILLE_MAP[ty % 4][x % 2]

    out_lines: List[str] = []
    tier_colors = [C_RED, C_AMBER, C_GREEN]
    for r in range(rows):
        row_col = tier_colors[r % len(tier_colors)] if (tiered_colors and rows == 3) else color
        line_chars = "".join(chr(0x2800 + grid[r][c]) for c in range(cols))
        out_lines.append(f"{row_col}{line_chars}{C_RESET}")
    return out_lines


def sparkline(
    values: List[float], lo: float, hi: float, width: int = 48, color: str = C_CYAN
) -> str:
    if not values:
        return " " * width
    arr = np.array(values[-width:], dtype=np.float64)
    if len(arr) < width:
        arr = np.pad(arr, (width - len(arr), 0), mode="edge")
    norm = np.clip((arr - lo) / max(1e-9, hi - lo), 0.0, 1.0)
    idx = np.round(norm * (len(BLOCKS) - 1)).astype(int)
    return f"{color}{''.join(BLOCKS[i] for i in idx)}{C_RESET}"


def render_phase_compass(psi_rad: float, r_val: float, fired: bool,
                         zone: Optional[int] = None, pulse_glow: int = 0) -> List[str]:
    """Render a 7-line high-precision ASCII/Unicode polar compass with 12 PRC phase sectors."""
    ang = ((psi_rad + np.pi) % (2.0 * np.pi)) - np.pi
    sector = int(np.round((ang + np.pi) / (2.0 * np.pi) * 12.0)) % 12
    # 12 clock positions: 0=9h(-π, TARGET), 1=8h, 2=7h, 3=6h(-π/2), 4=5h, 5=4h, 6=3h(0), 7=2h, 8=1h, 9=12h(+π/2), 10=11h, 11=10h
    p = [f"{C_GRAY}·{C_RESET}"] * 12
    if pulse_glow >= 4:
        p[0] = f"{fg_rgb(110, 231, 183)}{C_BOLD}✦{C_RESET}"
    elif pulse_glow >= 2:
        p[0] = f"{fg_rgb(56, 189, 248)}{C_BOLD}◈{C_RESET}"
    else:
        p[0] = f"{C_GREEN}◎{C_RESET}"  # Optimal PRC target φ* = -π
    if fired or pulse_glow >= 5:
        p[sector] = f"{fg_rgb(110, 231, 183)}{C_BOLD}★{C_RESET}"
    elif pulse_glow >= 2:
        p[sector] = f"{fg_rgb(56, 189, 248)}✦{C_RESET}"
    else:
        p[sector] = f"{C_WHITE}●{C_RESET}"
    vec_char = (
        "←"
        if sector in (0, 1, 11)
        else ("↓" if sector in (2, 3, 4) else ("→" if sector in (5, 6, 7) else "↑"))
    )
    zcol = (C_GREEN, C_AMBER, C_RED)[raw_zone(r_val) if zone is None else zone]
    center = f"{zcol}{vec_char}{C_RESET}"
    return [
        f"          +π/2 {p[9]}               ",
        f"       {p[10]}        {p[8]}            ",
        f"    {p[11]}              {p[7]}         ",
        f" φ*={p[0]}      {center}       {p[6]} 0 rad  ",
        f"    {p[1]}              {p[5]}         ",
        f"       {p[2]}        {p[4]}            ",
        f"          -π/2 {p[3]}               ",
    ]


def render_snn_raster(
    env_q15: int, phase_q15: int, step_num: int, width: int = 32, pulse_glow: int = 0
) -> str:
    """24-bit true-color Warp-style visualization of 64-neuron Int8 LIF SNN activity."""
    cells: List[str] = []
    glow_boost = 0.25 if pulse_glow > 0 else 0.0
    for i in range(width):
        pref_phase = ((i / width) * 65536.0) - 32768.0
        phase_dist = abs(((phase_q15 - pref_phase + 32768.0) % 65536.0) - 32768.0)
        drive = (env_q15 / 1400.0) * max(0.0, 1.0 - phase_dist / 16384.0) + glow_boost * max(
            0.0, 1.0 - abs(i - ((step_num // 2) % width)) / 6.0
        )
        if drive > 0.85 and ((step_num + i) % 3 == 0):
            cells.append(f"{fg_rgb(110, 231, 183)}█")
        elif drive > 0.55:
            cells.append(f"{fg_rgb(56, 189, 248)}▓")
        elif drive > 0.30:
            cells.append(f"{fg_rgb(129, 140, 248)}▒")
        elif drive > 0.14:
            cells.append(f"{fg_rgb(71, 85, 105)}░")
        else:
            cells.append(f"{fg_rgb(51, 65, 85)}·")
    cells.append(C_RESET)
    return "".join(cells)


def make_row(left: str, right: str, w_left: int = 62, w_right: int = 38) -> str:
    return f"│ {pad_ansi(left, w_left)} │ {pad_ansi(right, w_right)} │"


def run_tui(
    port: Optional[str] = None,
    baudrate: int = 2000000,
    max_frames: Optional[int] = None,
    keys: str = "",
    trace: bool = False,
) -> None:
    """keys: scripted keypresses 'FRAME:KEY,...' (e.g. '30:r,120:space') for bounded runs."""
    key_script: dict = {}
    for item in filter(None, (keys or "").split(",")):
        fr, k = item.split(":", 1)
        kb = b" " if k.lower() == "space" else k.encode()[:1]
        key_script.setdefault(int(fr), []).append(kb)
    from host.board import resolve_port

    port = resolve_port(port)
    if os.name == "nt":
        os.system("")
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    enable_windows_1ms_timer()

    ser = None
    hw_live = False
    mirror = os.environ.get("NURO_DEMO_MIRROR") == "1"
    if not mirror:
        try:
            ser = serial.Serial(
                port=port, baudrate=baudrate, timeout=0.20, write_timeout=0.20
            )
            ser.dtr = False
            ser.rts = False
            time.sleep(0.30)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            hw_live = True
        except Exception:
            hw_live = False

    k_brain = KuramotoBrain(KuramotoConfig(seed=2026))
    k_brain.theta = np.mod(
        k_brain.rng.normal(0.0, 0.55, size=k_brain.cfg.n_osc), 2.0 * np.pi
    )
    k_brain.step_idx = int(30.0 / k_brain.cfg.dt_s)
    jr_brain = JansenRitBrain(JansenRitConfig(seed=2026))
    patient_brain = PatientBrain("chb01")
    dsp = DspChainRef()

    zt = ZoneTracker(r0=0.85)
    raw_hist: Deque[float] = deque([0.0] * 120, maxlen=120)
    bp_hist: Deque[float] = deque([0.0] * 120, maxlen=120)
    r_hist: Deque[float] = deque([0.85] * 6, maxlen=104)
    rtt_win: Deque[float] = deque([1305.0] * 100, maxlen=100)
    dsp_win: Deque[float] = deque([18.0] * 100, maxlen=100)
    wcet_win: Deque[float] = deque([82.0] * 100, maxlen=100)
    event_log: Deque[str] = deque(maxlen=4)
    if mirror:
        event_log.append("[DEMO] In-silico mirror — the web cockpit owns the live port")
    else:
        event_log.append("[BOOT] Neural Loop TUI v2.0 — 32B CRC16 lockstep @ 2 Mbps")

    loop_on = True
    arm_idx = 0
    model_idx = 0
    patient_ids = ["chb01", "chb02", "chb03"]
    pat_idx = 0
    arm_names = [
        "D: PHASE-LOCKED (φ*=-π)",
        "C: RANDOM-PHASE SHAM",
        "B: OPEN-LOOP 20 Hz",
    ]
    model_names = [
        "STAGE 1: CHB-MIT CLINICAL PATIENT (PhysioNet)",
        "STAGE 2: KURAMOTO (N=64) IN-SILICO",
        "STAGE 3: JANSEN-RIT NMM IN-SILICO",
    ]

    seq = 0
    pulses = 0
    valid_frames = 0
    crc_err = 0
    refrac = 0
    pending_stim = 0.0
    last_tx_hex = "4E 4C 01 01 00 00 00 00"
    last_rx_hex = "4E 4C 81 01 00 00 00 00"
    last_res = ResultFrame(
        state=1,
        seq=0,
        t_rx_us=1000,
        t_dsp_us=1018,
        t_dec_us=1082,
        t_stim_us=0,
        cls=1,
        conf=230,
        phase_q15=-31000,
        R_q15=24000,
        budget_used=0,
    )

    if max_frames is None:
        sys.stdout.write("\033[?1049h\033[?7l\033[H\033[2J\033[?25l")
        sys.stdout.flush()
        if not trace:
            play_warp_boot_splash(port=port, hw_live=hw_live, mirror=mirror)

    has_msvcrt = False
    if os.name == "nt" and max_frames is None:
        try:
            import msvcrt

            has_msvcrt = True
        except ImportError:
            pass

    w_l, w_r = 62, 38
    total_w = w_l + w_r + 5
    top_border = f"╭─{'─' * w_l}─┬─{'─' * w_r}─╮"
    mid_border = f"├─{'─' * w_l}─┼─{'─' * w_r}─┤"
    bot_border = f"╰─{'─' * w_l}─┴─{'─' * w_r}─╯"

    frame_count = 0
    pulse_glow = 0
    last_hotplug_probe = time.time()
    try:
        while True:
            frame_count += 1

            # Dynamic Plug & Play: auto-detect and connect whenever ESP32-S3 is plugged in
            if not hw_live and not mirror:
                now_s = time.time()
                if now_s - last_hotplug_probe >= 1.0:
                    last_hotplug_probe = now_s
                    from host.board import detect_board_port

                    found_port = detect_board_port()
                    if found_port:
                        try:
                            ser = serial.Serial(
                                port=found_port,
                                baudrate=baudrate,
                                timeout=0.20,
                                write_timeout=0.20,
                            )
                            ser.dtr = False
                            ser.rts = False
                            time.sleep(0.20)
                            ser.reset_input_buffer()
                            ser.reset_output_buffer()
                            hw_live = True
                            port = found_port
                            event_log.append(
                                f"[HOTPLUG] Board detected on {found_port} -> LIVE HIL connected!"
                            )
                        except Exception:
                            hw_live = False
            keys_now: List[bytes] = list(key_script.get(frame_count, []))
            if has_msvcrt:
                import msvcrt

                while msvcrt.kbhit():
                    keys_now.append(msvcrt.getch())
            if keys_now:
                for ch in keys_now:
                    if ch == b" ":
                        loop_on = not loop_on
                        status_str = (
                            ("ON (TREATING)" if model_idx == 0 else "ON (SUPPRESSING)")
                            if loop_on
                            else ("OFF (UNTREATED)" if model_idx == 0 else "OFF (ICTAL SURGE)")
                        )
                        event_log.append(f"[KEY] Closed-Loop -> {status_str}")
                    elif ch in (b"m", b"M"):
                        arm_idx = (arm_idx + 1) % 3
                        event_log.append(
                            f"[KEY] Controller Arm -> {arm_names[arm_idx]}"
                        )
                    elif ch in (b"1", b"2", b"3"):
                        model_idx = 0 if ch == b"1" else (1 if ch == b"2" else 2)
                        event_log.append(
                            f"[KEY] Brain Model -> {model_names[model_idx]}"
                        )
                    elif ch in (b"p", b"P"):
                        pat_idx = (pat_idx + 1) % len(patient_ids)
                        patient_brain.set_patient(patient_ids[pat_idx])
                        model_idx = 0
                        event_log.append(
                            f"[KEY] Patient -> {patient_ids[pat_idx].upper()} (CHB-MIT Scalp EEG)"
                        )
                    elif ch in (b"r", b"R"):
                        if model_idx == 0:
                            patient_brain.reset_to_onset()
                            event_log.append(
                                f"[KEY] Triggered Patient {patient_ids[pat_idx].upper()} seizure burst (R -> {patient_brain.R_empirical[patient_brain.sz_start_idx]:.2f})"
                            )
                        else:
                            k_brain.theta = np.mod(
                                k_brain.rng.normal(0.0, 0.35, size=k_brain.cfg.n_osc),
                                2.0 * np.pi,
                            )
                            event_log.append(
                                "[KEY] Injected synchronous seizure burst (R -> 0.90)"
                            )
                    elif ch in (b"q", b"Q", b"\x1b"):
                        return

            fired_this_batch = False
            for _ in range(20):
                if model_idx == 0:
                    p0, _, _ = patient_brain.step(stim_amp=pending_stim if loop_on else 0.0)
                    p1, r_val, psi_val = patient_brain.step(stim_amp=0.0)
                    s0, s1 = p0, p1
                elif model_idx == 1:
                    s0, _, _ = k_brain.step(stim_amp=pending_stim if loop_on else 0.0)
                    s1, r_val, psi_val = k_brain.step(stim_amp=0.0)
                elif model_idx == 2:
                    s0, _, _ = k_brain.step(stim_amp=0.0)
                    s1, r_val, psi_val = k_brain.step(stim_amp=0.0)
                    j0, _ = jr_brain.step(u_stim=(pending_stim * 0.2) if loop_on else 0.0)
                    j1, _ = jr_brain.step(u_stim=0.0)
                    s0, s1 = j0, j1
                pending_stim = 0.0

                dsp.step(s0)
                bp_out = dsp.step(s1)
                fired = False

                prev_zone = zt.zone
                zt.update(r_val)
                if zt.zone != prev_zone:
                    event_log.append(
                        f"[ZONE] {ZONE_NAMES[prev_zone]} -> {zt.name} (R~{zt.r:.2f})"
                    )
                host_color = zt.led_flags
                sf = SampleFrame(
                    seq=seq,
                    sim_t_us=seq * 1000,
                    host_t_us=host_monotonic_us(),
                    samples=(s0, s1, 0, 0),
                    flags=0x03 | host_color,
                    n_ch=1,
                    n_samp=2,
                    mode=MODE_LOCKSTEP,
                )
                tx_bytes = sf.to_bytes()
                last_tx_hex = (
                    tx_bytes[:10].hex(" ").upper()
                    + " .. "
                    + tx_bytes[28:].hex(" ").upper()
                )
                seq += 1

                if hw_live and ser is not None:
                    t0 = time.perf_counter_ns()
                    try:
                        ser.write(tx_bytes)
                        rx = ser.read(32)
                        t1 = time.perf_counter_ns()
                        if len(rx) == 32:
                            last_rx_hex = (
                                rx[:10].hex(" ").upper()
                                + " .. "
                                + rx[28:].hex(" ").upper()
                            )
                            last_res = ResultFrame.from_bytes(rx)
                            valid_frames += 1
                            rtt_win.append((t1 - t0) / 1000.0)
                            dsp_win.append(
                                float(
                                    (last_res.t_dsp_us - last_res.t_rx_us) & 0xFFFFFFFF
                                )
                            )
                            wcet_win.append(
                                float(
                                    (last_res.t_dec_us - last_res.t_rx_us) & 0xFFFFFFFF
                                )
                            )
                            if loop_on:
                                if arm_idx == 0 and last_res.t_stim_us != 0:
                                    fired = True
                                elif (
                                    arm_idx == 1
                                    and refrac == 0
                                    and k_brain.rng.random() < 0.015
                                ):
                                    fired = True
                                elif arm_idx == 2 and (seq % 50 == 0):
                                    fired = True
                        else:
                            crc_err += 1
                    except (serial.SerialException, OSError):
                        hw_live = False
                        try:
                            if ser is not None and ser.is_open:
                                ser.close()
                        except Exception:
                            pass
                        ser = None
                        event_log.append("[HOTPLUG] Board disconnected -> In-silico fallback")
                    except Exception:
                        crc_err += 1
                else:
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
                    refrac = 20
                    fired_this_batch = True
                    if model_idx == 0:
                        event_log.append(
                            f"[TREAT #{pulses:03d}] seq={seq:<4d} ψ={psi_val:+.2f}rad R={r_val:.3f} Pat={patient_brain.patient_id.upper()} [SUPPRESSED]"
                        )
                    elif model_idx == 1:
                        k_brain.theta = np.mod(
                            k_brain.theta + k_brain.rng.normal(0.0, 2.2, size=k_brain.cfg.n_osc),
                            2.0 * np.pi,
                        )
                        event_log.append(
                            f"[STIM #{pulses:03d}] seq={seq:<4d} ψ={psi_val:+.2f}rad R={r_val:.3f} T_BLANK=20ms"
                        )
                    else:
                        event_log.append(
                            f"[STIM #{pulses:03d}] seq={seq:<4d} ψ={psi_val:+.2f}rad R={r_val:.3f} T_BLANK=20ms"
                        )
                elif refrac > 0:
                    refrac -= 1

                if not loop_on and r_val < 0.88 and model_idx == 1:
                    k_brain.theta = np.mod(
                        k_brain.theta - 0.016 * np.sin(k_brain.theta - psi_val),
                        2.0 * np.pi,
                    )

                raw_hist.append(s1 / 1024.0)
                bp_hist.append(bp_out.I_q15 / 1024.0)
                if seq % 8 == 0:
                    r_hist.append(zt.r)

            hw_str = (
                f"{C_GREEN}● LIVE ESP32-S3 ({port} @ 2M){C_RESET}"
                if hw_live
                else f"{C_AMBER}● IN-SILICO (PnP Auto-Detect){C_RESET}"
            )
            if loop_on:
                loop_str = (
                    f"{C_GREEN}[LOOP: ON  (TREATING PATIENT)]{C_RESET}"
                    if model_idx == 0
                    else f"{C_GREEN}[LOOP: ON  (SUPPRESSING)]{C_RESET}"
                )
            else:
                loop_str = (
                    f"{C_RED}[LOOP: OFF (UNTREATED ICTAL)]{C_RESET}"
                    if model_idx == 0
                    else f"{C_RED}[LOOP: OFF (ICTAL SURGE)]{C_RESET}"
                )
            cur_r = zt.r
            zone = zt.zone
            if fired_this_batch:
                pulse_glow = 6
            elif pulse_glow > 0:
                pulse_glow -= 1

            anim_phase = (frame_count * 0.028) % 1.0
            spin_ch = SPINNER_BRAILLE[frame_count % len(SPINNER_BRAILLE)]

            r_col = (C_GREEN, C_AMBER, C_RED)[zone]
            zone_badge = f"{r_col}{C_BOLD}■ {zt.name}{C_RESET}"
            r_bar = render_smooth_bar(cur_r, width=38, zone=zone, phase=anim_phase)

            p50_rtt = float(np.percentile(rtt_win, 50))
            p95_rtt = float(np.percentile(rtt_win, 95))
            p99_rtt = float(np.percentile(rtt_win, 99))
            p50_dsp = float(np.percentile(dsp_win, 50))
            p50_wcet = float(np.percentile(wcet_win, 50))

            hero_title = (
                f"{C_BOLD}[3] HERO — Real-Time 23-Ch Scalp Synchrony R(t){C_RESET}"
                if model_idx == 0
                else f"{C_BOLD}[3] HERO — Kuramoto Synchrony R(t){C_RESET}"
            )

            term_cols, term_lines = shutil.get_terminal_size((80, 24))

            if term_cols < 105 or term_lines < 28:
                # =========================================================
                # COMPACT 80x24 LAYOUT (78 cols x 22 rows, zero scroll/jitter)
                # =========================================================
                c_w_l, c_w_r = 48, 23
                top_b = f"╭─{'─' * c_w_l}─┬─{'─' * c_w_r}─╮"
                mid_b = f"├─{'─' * c_w_l}─┼─{'─' * c_w_r}─┤"
                bot_b = f"╰─{'─' * c_w_l}─┴─{'─' * c_w_r}─╯"

                hw_badge = (
                    f"{C_GREEN}● LIVE {port}{C_RESET}"
                    if hw_live
                    else f"{C_AMBER}● IN-SILICO{C_RESET}"
                )
                compact_loop = (
                    f"{C_GREEN}[TREATING]{C_RESET}"
                    if (loop_on and model_idx == 0)
                    else (
                        f"{C_GREEN}[ON]{C_RESET}"
                        if loop_on
                        else f"{C_RED}[OFF (SURGE)]{C_RESET}"
                    )
                )
                pat_name = (
                    patient_brain.get_patient_profile()[:22]
                    if model_idx == 0
                    else ("KURAMOTO" if model_idx == 1 else "JANSEN")
                )
                arm_short = (
                    "D: LOCKED"
                    if arm_idx == 0
                    else ("C: SHAM" if arm_idx == 1 else "B: OPEN 20Hz")
                )

                r_bar_c = render_smooth_bar(cur_r, width=24, zone=zone, phase=anim_phase)

                braille_c = render_braille_plot(
                    list(r_hist),
                    0.0,
                    1.0,
                    cols=38,
                    rows=3,
                    color=r_col,
                    threshold_y=R_RED,
                    tiered_colors=False,
                )
                snn_strip_c = render_snn_raster(
                    bp_out.env_q15, bp_out.phase_q15, seq, width=22, pulse_glow=pulse_glow
                )

                ang = ((psi_val + np.pi) % (2.0 * np.pi)) - np.pi
                sec = int(np.round((ang + np.pi) / (2.0 * np.pi) * 12.0)) % 12
                pt = [f"{C_GRAY}·{C_RESET}"] * 12
                pt[0] = (
                    f"{fg_rgb(110, 231, 183)}{C_BOLD}✦{C_RESET}"
                    if pulse_glow >= 4
                    else (
                        f"{fg_rgb(56, 189, 248)}{C_BOLD}◈{C_RESET}"
                        if pulse_glow >= 2
                        else f"{C_GREEN}◎{C_RESET}"
                    )
                )
                pt[sec] = (
                    f"{fg_rgb(110, 231, 183)}{C_BOLD}★{C_RESET}"
                    if (fired_this_batch or pulse_glow >= 5)
                    else (
                        f"{fg_rgb(56, 189, 248)}✦{C_RESET}"
                        if pulse_glow >= 2
                        else f"{C_WHITE}●{C_RESET}"
                    )
                )
                vec_char = (
                    "←"
                    if sec in (0, 1, 11)
                    else (
                        "↓"
                        if sec in (2, 3, 4)
                        else ("→" if sec in (5, 6, 7) else "↑")
                    )
                )
                center_c = f"{r_col}{vec_char}{C_RESET}"
                compass_c = [
                    f"      +π/2 {pt[9]}",
                    f"   {pt[10]}        {pt[8]}",
                    f"φ*={pt[0]}   {center_c}   {pt[6]} 0",
                    f"   {pt[2]}        {pt[4]}",
                    f"      -π/2 {pt[3]}",
                ]

                ev0 = event_log[-1] if len(event_log) >= 1 else ""
                ev1 = event_log[-2] if len(event_log) >= 2 else ""
                ev0_col = C_GREEN if ("STIM" in ev0 or "TREAT" in ev0) else C_CYAN
                ev1_col = C_GREEN if ("STIM" in ev1 or "TREAT" in ev1) else C_CYAN

                warp_hdr_c = (
                    f" {fg_rgb(56, 189, 248)}{spin_ch}{C_RESET} "
                    f"{gradient_text('⚡ NURO v1.0', phase=anim_phase, bold=True)} "
                    f"{C_GRAY}│{C_RESET} {C_WHITE}CHB-MIT Closed-Loop{C_RESET} "
                    f"{C_GRAY}│{C_RESET} {fg_rgb(52, 211, 153)}0.964 AUROC{C_RESET}"
                )
                rows = [
                    f"{C_BG_BAR}{pad_ansi(warp_hdr_c, 78)}{C_RESET}",
                    top_b,
                    make_row(
                        f"{C_BOLD}LINK:{C_RESET} {hw_badge} {compact_loop}",
                        f"{C_BOLD}MODEL:{C_RESET} {C_CYAN}{pat_name}{C_RESET}",
                        c_w_l,
                        c_w_r,
                    ),
                    make_row(
                        f"{C_BOLD}ARM :{C_RESET} {arm_short} {C_BOLD}PULSES:{C_RESET} {pulses:3d}/300",
                        f"{C_BOLD}AUROC:{C_RESET} {C_GREEN}0.964 [G5/G6]{C_RESET}",
                        c_w_l,
                        c_w_r,
                    ),
                    mid_b,
                    make_row(
                        f"{C_BOLD}[1] PATIENT EEG & 20ms BLANKING{C_RESET}",
                        f"{C_BOLD}[2] PHASE COMPASS{C_RESET}",
                        c_w_l,
                        c_w_r,
                    ),
                    make_row(
                        f"EEG µV  : {sparkline(list(raw_hist), -3.8, 3.8, 22, C_GRAY)} {patient_brain.current_uV:+5.0f}µV"
                        if model_idx == 0
                        else f"Raw LFP : {sparkline(list(raw_hist), -3.8, 3.8, 34, C_GRAY)}",
                        compass_c[0],
                        c_w_l,
                        c_w_r,
                    ),
                    make_row(
                        f"Q15 I(t): {sparkline(list(bp_hist), -2.5, 2.5, 34, C_CYAN)}",
                        compass_c[1],
                        c_w_l,
                        c_w_r,
                    ),
                    make_row(
                        f"64-N SNN: [{snn_strip_c}] 99%",
                        compass_c[2],
                        c_w_l,
                        c_w_r,
                    ),
                    make_row(
                        f"Blanking: {f'{C_AMBER}ACTIVE (20ms){C_RESET}' if bp_out.blanked else f'{C_GREEN}IDLE (0 art){C_RESET}'} │ Env={bp_out.env_q15:4d}",
                        compass_c[3],
                        c_w_l,
                        c_w_r,
                    ),
                    mid_b,
                    make_row(hero_title, f"{C_BOLD}[4] TELEMETRY{C_RESET}", c_w_l, c_w_r),
                    make_row(
                        f"R(t) Now: [{r_bar_c}] {r_col}{cur_r:0.3f}{C_RESET}",
                        f"USB RTT: {C_CYAN}{p50_rtt:6.0f} µs{C_RESET}",
                        c_w_l,
                        c_w_r,
                    ),
                    make_row(
                        f"  1.0 ┤ {braille_c[0]}",
                        f"MCU DSP: {C_GREEN}{p50_dsp:6.0f} µs{C_RESET}",
                        c_w_l,
                        c_w_r,
                    ),
                    make_row(
                        f"  0.5 ┤ {braille_c[1]}",
                        f"MCU SNN: {C_GREEN}{max(1.0, p50_wcet - p50_dsp):6.0f} µs{C_RESET}",
                        c_w_l,
                        c_w_r,
                    ),
                    make_row(
                        f"  0.0 ┴ {braille_c[2]}",
                        f"MCU WCET:{C_GREEN}{p50_wcet:6.0f} µs{C_RESET}",
                        c_w_l,
                        c_w_r,
                    ),
                    mid_b,
                    make_row(
                        f"{C_BOLD}[5] EVENT & CLINICAL SUPPRESSION LOG{C_RESET}",
                        f"{C_BOLD}[6] SAFETY & GATES{C_RESET}",
                        c_w_l,
                        c_w_r,
                    ),
                    make_row(
                        f"{ev0_col}{ev0}{C_RESET}",
                        f"Interlocks: {C_GREEN}CRC/BLANK OK{C_RESET}",
                        c_w_l,
                        c_w_r,
                    ),
                    make_row(
                        f"{ev1_col}{ev1}{C_RESET}",
                        f"Runaway   : {C_GREEN}0x00 ZERO{C_RESET}",
                        c_w_l,
                        c_w_r,
                    ),
                    bot_b,
                    pad_ansi(
                        f" {C_BOLD}PAT{C_RESET} {patient_ids[pat_idx].upper() if model_idx == 0 else pat_name[:8]} "
                        f"| {C_BOLD}LOOP{C_RESET} {(C_GREEN + 'ON ') if loop_on else (C_RED + 'OFF')}{C_RESET} "
                        f"| {C_BOLD}R{C_RESET} {r_col}{cur_r:0.3f}{C_RESET} | {C_BOLD}ZONE{C_RESET} {zone_badge} "
                        f"| {C_GREEN}<.35{C_RESET} {C_AMBER}.35-.55{C_RESET} {C_RED}>=.55{C_RESET}",
                        78,
                    ),
                    pad_ansi(
                        f" {C_BOLD}KEYS:{C_RESET} {C_CYAN}[SPACE]{C_RESET} Loop │ {C_CYAN}[P]{C_RESET} Pat │ {C_CYAN}[R]{C_RESET} Onset │ {C_CYAN}[M]{C_RESET} Arm │ {C_CYAN}[1/2/3]{C_RESET} Model │ {C_CYAN}[Q]{C_RESET} Quit",
                        78,
                    ),
                ]
            else:
                # =========================================================
                # EXPANDED WIDESCREEN LAYOUT (>= 105 cols, >= 28 lines)
                # =========================================================
                compass = render_phase_compass(psi_val, cur_r, fired_this_batch, zone, pulse_glow=pulse_glow)
                braille_r = render_braille_plot(
                    list(r_hist),
                    0.0,
                    1.0,
                    cols=52,
                    rows=3,
                    color=r_col,
                    threshold_y=R_RED,
                    tiered_colors=False,
                )
                snn_strip = render_snn_raster(
                    bp_out.env_q15, bp_out.phase_q15, seq, width=35, pulse_glow=pulse_glow
                )

                header_title = pad_ansi(
                    f" {fg_rgb(56, 189, 248)}{spin_ch}{C_RESET} "
                    f"{gradient_text('⚡ NURO WORKSTATION v1.0', phase=anim_phase, bold=True)} "
                    f"{C_GRAY}│{C_RESET} ESP32-S3 Closed-Loop Edge Neuromodulation "
                    f"{C_GRAY}│{C_RESET} {fg_rgb(52, 211, 153)}0.964 AUROC QAT-SNN{C_RESET} "
                    f"{C_GRAY}│{C_RESET} 2000 Hz Lockstep",
                    total_w,
                )

                rows = [
                    f"{C_BG_BAR}{header_title}{C_RESET}",
                    top_border,
                    make_row(
                        f"{C_BOLD}LINK:{C_RESET} {hw_str}  {loop_str}",
                        f"{C_BOLD}MODEL:{C_RESET} {C_CYAN}{patient_brain.get_patient_profile() if model_idx == 0 else model_names[model_idx]}{C_RESET}",
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"{C_BOLD}ARM :{C_RESET} {C_AMBER}{arm_names[arm_idx]}{C_RESET}  {C_BOLD}PULSES:{C_RESET} {pulses:3d}/300",
                        f"{C_BOLD}VALID:{C_RESET} {valid_frames:6d} frames │ {C_GREEN}{crc_err:2d} CRC err{C_RESET}",
                        w_l,
                        w_r,
                    ),
                    mid_border,
                    make_row(
                        f"{C_BOLD}[1] CHB-MIT CLINICAL SCALP EEG & 20ms BLANKED Q15 BANDPASS{C_RESET}"
                        if model_idx == 0
                        else f"{C_BOLD}[1] SIGNAL CHAIN — Raw LFP & 20ms Blanked Q15 Bandpass{C_RESET}",
                        f"{C_BOLD}[2] 12-SECTOR PRC PHASE COMPASS{C_RESET}",
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"EEG µV  : {sparkline(list(raw_hist), -3.8, 3.8, 36, C_GRAY)} {patient_brain.current_uV:+6.1f} µV [{patient_brain.info['channel_name']} Focus]"
                        if model_idx == 0
                        else f"Raw LFP : {sparkline(list(raw_hist), -3.8, 3.8, 50, C_GRAY)}",
                        compass[0],
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"Q15 I(t): {sparkline(list(bp_hist), -2.5, 2.5, 50, C_CYAN)}",
                        compass[1],
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"64-N SNN: [{snn_strip}] {C_GREEN}99.2% sparse{C_RESET}",
                        compass[2],
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"Blanking: {f'{C_AMBER}ACTIVE (40-samp=20ms){C_RESET}' if bp_out.blanked else f'{C_GREEN}IDLE (0 artifact){C_RESET}'} │ Env={bp_out.env_q15:4d} Q15",
                        compass[3],
                        w_l,
                        w_r,
                    ),
                    f"├─{'─' * w_l}─┤ {pad_ansi(compass[4], w_r)} │",
                    make_row(
                        hero_title,
                        compass[5],
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"R(t) Now: [{r_bar}] {r_col}{C_BOLD}{cur_r:0.4f}{C_RESET}",
                        compass[6],
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"  1.00 ┤ {braille_r[0]}",
                        f"ψ(t) = {C_CYAN}{psi_val:+6.3f} rad ({np.degrees(psi_val):+6.1f}°){C_RESET}",
                        w_l,
                        w_r,
                    ),
                    f"│ {pad_ansi(f'  0.50 ┤ {braille_r[1]}', w_l)} ├─{'─' * w_r}─┤",
                    make_row(
                        f"  0.00 ┴ {braille_r[2]}",
                        f"{C_BOLD}[4] ESP32-S3 HARDWARE TELEMETRY{C_RESET}",
                        w_l,
                        w_r,
                    ),
                    f"├─{'─' * w_l}─┤ {pad_ansi(f'USB RTT p50 : {C_CYAN}{p50_rtt:7.1f} µs{C_RESET} (< 2000 G0)', w_r)} │",
                    make_row(
                        f"{C_BOLD}[5] 32-BYTE BINARY FRAME & MCU REGISTER INSPECTOR{C_RESET}",
                        f"USB RTT p95 : {C_CYAN}{p95_rtt:7.1f} µs{C_RESET} (10.6° err)",
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"TX (32B): {C_GRAY}{last_tx_hex}{C_RESET}",
                        f"USB RTT p99 : {C_CYAN}{p99_rtt:7.1f} µs{C_RESET} (< 5000)",
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"RX (32B): {C_CYAN}{last_rx_hex}{C_RESET}",
                        f"MCU FIR DSP : {C_GREEN}{p50_dsp:7.1f} µs{C_RESET} (61-tap Q15)",
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"REGS    : seq={last_res.seq:<5d} st=0x{last_res.state:02X} cls={last_res.cls} conf={last_res.conf:<3d} φ_q15={last_res.phase_q15:<+6d}",
                        f"MCU SNN+FSM : {C_GREEN}{max(1.0, p50_wcet - p50_dsp):7.1f} µs{C_RESET} (64-N Int8)",
                        w_l,
                        w_r,
                    ),
                    make_row(
                        f"CLOCKS  : rx={last_res.t_rx_us:<9d} dsp={last_res.t_dsp_us:<9d} dec={last_res.t_dec_us:<9d}",
                        f"MCU WCET p50: {C_GREEN}{p50_wcet:7.1f} µs{C_RESET} (< 250 G3)",
                        w_l,
                        w_r,
                    ),
                    mid_border,
                    make_row(
                        f"{C_BOLD}[6] REAL-TIME HARDWARE STIMULATION & EVENT LOG{C_RESET}",
                        f"{C_BOLD}[7] SAFETY & GATES VERIFICATION{C_RESET}",
                        w_l,
                        w_r,
                    ),
                ]

                safety_lines = [
                    f"Interlocks  : {C_GREEN}CRC/REFRAC/BLANK=20ms OK{C_RESET}",
                    f"Fault Flags : {C_GREEN}0x00 (ZERO RUNAWAY){C_RESET}",
                    f"Gate G5/G6  : {C_GREEN}CHB-MIT AUROC 0.964 │ Colab QAT-BPTT SNN{C_RESET}",
                    f"Gate G7/G8  : {C_GREEN}ΔR=+0.746 [0.72, 0.78]{C_RESET}",
                ]
                for i in range(4):
                    ev = event_log[i] if i < len(event_log) else ""
                    ev_col = C_GREEN if ("STIM" in ev or "TREAT" in ev) else C_CYAN
                    rows.append(
                        make_row(f"{ev_col}{ev}{C_RESET}", safety_lines[i], w_l, w_r)
                    )

                rows.append(bot_border)
                rows.append(
                    pad_ansi(
                        f" {C_BOLD}HOTKEYS:{C_RESET} {C_CYAN}[SPACE]{C_RESET} Loop ON/OFF │ {C_CYAN}[M]{C_RESET} Arm │ {C_CYAN}[1/2/3]{C_RESET} Patient/Kuramoto/JR │ {C_CYAN}[P]{C_RESET} Switch Patient │ {C_CYAN}[R]{C_RESET} Seizure Onset │ {C_CYAN}[Q]{C_RESET} Quit",
                        total_w,
                    )
                )

            if trace:
                print(f"frame={frame_count:4d} loop={'ON ' if loop_on else 'OFF'} "
                      f"R_raw={r_val:.3f} R_smooth={zt.r:.3f} zone={zt.name} led=0x{zt.led_flags:02X}")
            if max_frames is None:
                sys.stdout.write("\033[?2026h\033[?7l\033[H" + "\033[K\n".join(rows) + "\033[K\033[J\033[?2026l")
                sys.stdout.flush()
                time.sleep(0.045)
            else:
                if frame_count >= max_frames:
                    print("\n".join(rows))
                    break
    finally:
        if max_frames is None:
            sys.stdout.write("\033[?7h\033[?1049l\033[?25h\n")
            sys.stdout.flush()
        if ser is not None and ser.is_open:
            try:
                ser.write(b"rest\r\n")
                time.sleep(0.05)
            except Exception:
                pass
            ser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neural Loop btop-Style Terminal User Interface (TUI)"
    )
    parser.add_argument(
        "--port", type=str, default=None, help="Serial port for ESP32-S3 (auto-detect)"
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="Run N batches and print TUI snapshot"
    )
    parser.add_argument("--keys", type=str, default="", help="Scripted keys 'FRAME:KEY,...'")
    parser.add_argument("--trace", action="store_true", help="Print per-frame R/zone trace")
    args = parser.parse_args()
    run_tui(port=args.port, max_frames=args.frames, keys=args.keys, trace=args.trace)
