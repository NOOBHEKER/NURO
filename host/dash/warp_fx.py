"""Warp-CLI-inspired 24-bit True-Color Terminal Animation & Splash Engine for NURO.

Provides:
- Smooth 24-bit RGB gradients (Warp cyan -> indigo -> violet -> emerald aurora palette)
- 8-level sub-character fractional progress & gauge bars (▏▎▍▌▋▊▉█)
- Live animated Braille neural waveform banner
- Zero-flicker (\033[?2026h synchronized) Warp-style boot splash for the TUI
- Sleek Warp-style animated launcher screen for the `nuro` CLI
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import time
from typing import List, Optional, Sequence, Tuple

RGB = Tuple[int, int, int]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Warp-inspired Cyber-Aurora True-Color Palette
PALETTE_WARP: Tuple[RGB, ...] = (
    (56, 189, 248),   # Electric Sky Cyan
    (99, 102, 241),   # Warp Indigo
    (192, 132, 252),  # Neon Violet
    (45, 212, 191),   # Bio Teal
    (52, 211, 153),   # Clinical Emerald
    (56, 189, 248),   # Loop back seamlessly
)

PALETTE_GREEN: Tuple[RGB, ...] = ((16, 185, 129), (52, 211, 153), (110, 231, 183))
PALETTE_AMBER: Tuple[RGB, ...] = ((217, 119, 6), (245, 158, 11), (252, 211, 77))
PALETTE_RED: Tuple[RGB, ...] = ((220, 38, 38), (248, 81, 73), (252, 165, 165))

SPINNER_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SUB_BLOCKS = (" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█")

BRAILLE_BITS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)

NURO_LOGO: Tuple[str, ...] = (
    " ███╗   ██╗ ██╗   ██╗ ██████╗   ██████╗ ",
    " ████╗  ██║ ██║   ██║ ██╔══██╗ ██╔═══██╗",
    " ██╔██╗ ██║ ██║   ██║ ██████╔╝ ██║   ██║",
    " ██║╚██╗██║ ██║   ██║ ██╔══██╗ ██║   ██║",
    " ██║ ╚████║ ╚██████╔╝ ██║  ██║ ╚██████╔╝",
    " ╚═╝  ╚═══╝  ╚═════╝  ╚═╝  ╚═╝  ╚═════╝ ",
)


def fg_rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def bg_rgb(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


def sample_palette(t: float, palette: Sequence[RGB] = PALETTE_WARP) -> RGB:
    """Smoothly sample a cyclic or linear RGB palette at t in [0, 1]."""
    n = len(palette)
    if n == 0:
        return (255, 255, 255)
    if n == 1:
        return palette[0]
    pos = (t % 1.0) * (n - 1)
    i0 = int(pos)
    i1 = min(n - 1, i0 + 1)
    frac = pos - i0
    c0, c1 = palette[i0], palette[i1]
    return (
        int(c0[0] + (c1[0] - c0[0]) * frac),
        int(c0[1] + (c1[1] - c0[1]) * frac),
        int(c0[2] + (c1[2] - c0[2]) * frac),
    )


def gradient_text(
    text: str,
    phase: float = 0.0,
    palette: Sequence[RGB] = PALETTE_WARP,
    bold: bool = False,
    spread: float = 0.75,
) -> str:
    """Apply a smooth animated 24-bit RGB gradient wave across visible characters."""
    n = max(1, len(text))
    prefix = BOLD if bold else ""
    out: List[str] = []
    for idx, ch in enumerate(text):
        if ch == " ":
            out.append(" ")
            continue
        r, g, b = sample_palette((idx / n) * spread + phase, palette)
        out.append(f"{prefix}\033[38;2;{r};{g};{b}m{ch}")
    out.append(RESET)
    return "".join(out)


def render_smooth_bar(
    fraction: float,
    width: int = 24,
    zone: Optional[int] = None,
    phase: float = 0.0,
    use_warp_palette: bool = False,
) -> str:
    """Render an 8x sub-character fractional progress/gauge bar with true-color RGB glow."""
    frac = max(0.0, min(1.0, float(fraction)))
    total_eighths = int(round(frac * width * 8))
    full_cells = total_eighths // 8
    rem_eighths = total_eighths % 8

    if use_warp_palette:
        pal = PALETTE_WARP
    elif zone == 0:
        pal = PALETTE_GREEN
    elif zone == 1:
        pal = PALETTE_AMBER
    elif zone == 2:
        pal = PALETTE_RED
    else:
        pal = PALETTE_WARP

    track_col = fg_rgb(56, 62, 76)
    parts: List[str] = []
    for i in range(width):
        t = (i / max(1, width - 1)) * 0.85 + phase
        r, g, b = sample_palette(t if use_warp_palette else (i / max(1, width - 1)), pal)
        if i < full_cells:
            parts.append(f"\033[38;2;{r};{g};{b}m█")
        elif i == full_cells and rem_eighths > 0:
            parts.append(f"\033[38;2;{r};{g};{b}m{SUB_BLOCKS[rem_eighths]}")
        else:
            parts.append(f"{track_col}░")
    parts.append(RESET)
    return "".join(parts)


def render_neural_wave(
    width: int,
    phase: float,
    rows: int = 2,
    amp: float = 0.85,
) -> List[str]:
    """Render a smooth animated 2x4 Braille neural oscillation ribbon with RGB wave colors."""
    px_w = width * 2
    px_h = rows * 4
    grid = [[0 for _ in range(width)] for _ in range(rows)]
    for x in range(px_w):
        u = x / max(1, px_w - 1)
        # Dual-harmonic Kuramoto envelope wave
        env = math.sin(u * math.pi)
        y_val = (
            0.5
            + 0.42
            * amp
            * env
            * (
                0.65 * math.sin(u * 4.5 * math.pi - phase * 6.0)
                + 0.35 * math.sin(u * 11.0 * math.pi + phase * 9.0)
            )
        )
        yy = max(0, min(px_h - 1, int(round(y_val * (px_h - 1)))))
        grid[yy // 4][x // 2] |= BRAILLE_BITS[yy % 4][x % 2]

    lines: List[str] = []
    for r_i in range(rows):
        row_chars: List[str] = []
        for c_i in range(width):
            ch = chr(0x2800 + grid[r_i][c_i])
            r, g, b = sample_palette((c_i / max(1, width)) * 0.8 + phase, PALETTE_WARP)
            row_chars.append(f"\033[38;2;{r};{g};{b}m{ch}")
        row_chars.append(RESET)
        lines.append("".join(row_chars))
    return lines


def _vis_len(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def _truncate_ansi(s: str, width: int) -> str:
    if _vis_len(s) <= width:
        return s
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
    out.append(RESET)
    return "".join(out)


def _pad_center(s: str, width: int) -> str:
    s = _truncate_ansi(s, width)
    vl = _vis_len(s)
    left = (width - vl) // 2
    right = width - vl - left
    return (" " * left) + s + (" " * right)


def _pad_left(s: str, width: int) -> str:
    s = _truncate_ansi(s, width)
    vl = _vis_len(s)
    return s + (" " * (width - vl))


def _key_pressed_nonblock() -> bool:
    if os.name == "nt":
        try:
            import msvcrt

            if msvcrt.kbhit():
                msvcrt.getch()
                return True
        except Exception:
            pass
    return False


def play_warp_boot_splash(
    port: str = "COM6",
    hw_live: bool = False,
    mirror: bool = False,
    duration_s: float = 1.65,
) -> None:
    """Play a silky-smooth Warp-CLI-style animated startup splash inside the TUI alt screen.

    Strictly bounded to <= 76x21 so it never wraps on 80x24 terminals, centered dynamically
    in wider windows, atomic-synced (\033[?2026h), and skippable with any keypress.
    """
    if not sys.stdout.isatty():
        return

    term_cols, term_rows = shutil.get_terminal_size((80, 24))
    inner_w = 68 if term_cols >= 74 else max(54, term_cols - 6)
    box_w = inner_w + 4
    left_pad = " " * max(0, (term_cols - box_w) // 2)
    top_pad = max(0, (term_rows - 21) // 2)

    link_label = (
        f"ESP32-S3 Live Hardware ({port} @ 2.0 Mbps)"
        if hw_live
        else ("Web Cockpit In-Silico Mirror" if mirror else f"ESP32-S3 Plug-and-Play ({port} standby)")
    )
    steps = [
        ("PhysioNet CHB-MIT 23-Ch Clinical EEG", "chb01 / chb02 / chb03"),
        ("64-N Int8 QAT-BPTT LIF SNN Engine", "0.964 AUROC • Bit-Exact"),
        ("Hardware-in-the-Loop Lockstep Link", link_label),
        ("Closed-Loop Safety & Solid RGB Sync", "Zero-Blink Active"),
    ]

    border_col = fg_rgb(71, 85, 105)
    muted = fg_rgb(148, 163, 184)
    mint = fg_rgb(52, 211, 153)
    cyan = fg_rgb(56, 189, 248)
    white = fg_rgb(241, 245, 249)

    fps = 35.0
    total_frames = max(18, int(round(duration_s * fps)))
    t_start = time.perf_counter()

    for f_idx in range(total_frames + 1):
        if _key_pressed_nonblock():
            break

        u = f_idx / float(total_frames)
        # Cubic ease-out for silky Warp-style progress glide
        eased = 1.0 - (1.0 - u) ** 3
        phase = (f_idx * 0.032) % 1.0
        spin = SPINNER_BRAILLE[f_idx % len(SPINNER_BRAILLE)]

        top_b = f"{border_col}╭{'─' * (inner_w + 2)}╮{RESET}"
        mid_b = f"{border_col}├{'─' * (inner_w + 2)}┤{RESET}"
        bot_b = f"{border_col}╰{'─' * (inner_w + 2)}╯{RESET}"

        def box_line(content: str, center: bool = False) -> str:
            body = _pad_center(content, inner_w) if center else _pad_left(content, inner_w)
            return f"{left_pad}{border_col}│{RESET} {body} {border_col}│{RESET}"

        lines: List[str] = ["\n" * top_pad] if top_pad > 0 else []
        lines.append(f"{left_pad}{top_b}")

        # Animated Warp RGB Logo
        for li, logo_row in enumerate(NURO_LOGO):
            grad_row = gradient_text(logo_row, phase=(phase + li * 0.045), bold=True, spread=0.65)
            lines.append(box_line(grad_row, center=True))

        subtitle = (
            f"{BOLD}{cyan}CLOSED-LOOP EDGE NEUROMODULATION{RESET}  "
            f"{muted}•  ESP32-S3  •  v1.0{RESET}"
        )
        lines.append(box_line(subtitle, center=True))

        # Animated Braille Neural Wave Ribbon
        wave = render_neural_wave(width=min(56, inner_w - 4), phase=phase, rows=2, amp=0.95)
        for w_row in wave:
            lines.append(box_line(w_row, center=True))

        lines.append(f"{left_pad}{mid_b}")

        # Staged Warp Checklist
        for s_idx, (title, detail) in enumerate(steps):
            thresh = (s_idx + 1) / (len(steps) + 0.35)
            if eased >= thresh:
                icon = f"{BOLD}{mint}✓{RESET}"
                t_str = f"{white}{title}{RESET}"
                d_str = f"{mint}{detail}{RESET}"
            elif eased >= s_idx / (len(steps) + 0.35):
                icon = f"{BOLD}{cyan}{spin}{RESET}"
                t_str = f"{BOLD}{white}{title}{RESET}"
                d_str = f"{cyan}initializing...{RESET}"
            else:
                icon = f"{muted}○{RESET}"
                t_str = f"{muted}{title}{RESET}"
                d_str = f"{muted}queued{RESET}"
            gap = max(1, inner_w - _vis_len(f" {icon} {t_str} ") - _vis_len(d_str))
            lines.append(box_line(f" {icon} {t_str}{' ' * gap}{d_str}"))

        lines.append(f"{left_pad}{mid_b}")

        # Smooth Sub-Character Progress Bar
        bar_w = min(46, inner_w - 18)
        bar = render_smooth_bar(eased, width=bar_w, phase=phase, use_warp_palette=True)
        pct = int(round(eased * 100))
        status_word = "READY" if pct >= 100 else "WARPING"
        prog_line = f" {BOLD}{cyan}{status_word:<7s}{RESET} [{bar}] {BOLD}{white}{pct:3d}%{RESET}"
        lines.append(box_line(prog_line, center=True))

        hint = f"{muted}Press any key to skip  •  1000 Hz Real-Time Lockstep{RESET}"
        lines.append(box_line(hint, center=True))
        lines.append(f"{left_pad}{bot_b}")

        frame_str = "\033[?2026h\033[H" + "\n".join(line + "\033[K" for line in lines) + "\033[J\033[?2026l"
        sys.stdout.write(frame_str)
        sys.stdout.flush()

        target_t = t_start + (f_idx + 1) / fps
        dt = target_t - time.perf_counter()
        if dt > 0:
            time.sleep(dt)

    # Clean settle pause on 100% READY frame
    time.sleep(0.12)
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()


def print_warp_cli_menu(version: str = "1.0.0", animate: bool = True) -> None:
    """Render a modern Warp-CLI-style starting screen with smooth gradient reveal."""
    if os.name == "nt":
        os.system("")

    is_tty = sys.stdout.isatty()
    inner_w = 68
    border_col = fg_rgb(71, 85, 105)
    muted = fg_rgb(148, 163, 184)
    cyan = fg_rgb(56, 189, 248)
    mint = fg_rgb(52, 211, 153)
    violet = fg_rgb(192, 132, 252)
    amber = fg_rgb(251, 191, 36)
    white = fg_rgb(241, 245, 249)

    def build_frame(phase: float) -> str:
        top_b = f"{border_col}╭{'─' * (inner_w + 2)}╮{RESET}"
        mid_b = f"{border_col}├{'─' * (inner_w + 2)}┤{RESET}"
        bot_b = f"{border_col}╰{'─' * (inner_w + 2)}╯{RESET}"

        def row(content: str, center: bool = False) -> str:
            body = _pad_center(content, inner_w) if center else _pad_left(content, inner_w)
            return f"  {border_col}│{RESET} {body} {border_col}│{RESET}"

        def cmd_row(badge_col: str, cmd: str, desc: str) -> str:
            return row(f"  {BOLD}{badge_col}▸ {cmd:<17s}{RESET} {white}{desc}{RESET}")

        out: List[str] = ["", f"  {top_b}"]
        for li, logo_row in enumerate(NURO_LOGO):
            out.append(row(gradient_text(logo_row, phase=phase + li * 0.05, bold=True, spread=0.65), center=True))

        pill = (
            f"{BOLD}{mint}● v{version}{RESET}  {muted}│{RESET}  "
            f"{cyan}ESP32-S3 Closed-Loop Lab{RESET}  {muted}│{RESET}  "
            f"{violet}0.964 AUROC QAT-SNN{RESET}"
        )
        out.append(row(pill, center=True))
        for w_row in render_neural_wave(width=54, phase=phase, rows=1, amp=0.9):
            out.append(row(w_row, center=True))

        out.append(f"  {mid_b}")
        out.append(row(f" {BOLD}{cyan}CLINICAL NEUROMODULATION{RESET}"))
        out.append(cmd_row(mint, "tui", "Live 60 FPS terminal cockpit (CHB-MIT + ESP32)"))
        out.append(cmd_row(mint, "web  /  demo", "Browser cockpit + full multi-view showcase"))
        out.append(cmd_row(mint, "bench  /  info", "1000-frame live HIL benchmark / board probe"))

        out.append(f"  {mid_b}")
        out.append(row(f" {BOLD}{violet}AI & SCIENTIFIC POCKET LAB{RESET}"))
        out.append(cmd_row(violet, "chat [n]", "Offline PC AI chatbot quiz (default 3 Qs)"))
        out.append(cmd_row(violet, "ask \"question\"", "Ask the local AI anything across all subjects"))
        out.append(cmd_row(violet, "symptoms \"...\"", "Educational symptom triage (not a diagnosis)"))
        out.append(cmd_row(violet, "calc / chem / phy", "Scientific calculator, chemistry & physics"))

        out.append(f"  {mid_b}")
        out.append(row(f" {BOLD}{amber}SYSTEM & VERIFICATION{RESET}"))
        out.append(cmd_row(amber, "doctor / setup", "Hardware readiness check / one-click setup"))
        out.append(cmd_row(amber, "test / figures", "Run full pytest suite / rebuild figures"))
        out.append(cmd_row(amber, "session / obsidian", "Named sessions & Obsidian graph sync"))

        out.append(f"  {mid_b}")
        tip = f"{muted}Type {BOLD}{cyan}tui{RESET}{muted} for cockpit  •  {BOLD}{violet}chat{RESET}{muted} for AI  •  {BOLD}{white}Enter{RESET}{muted} for shell{RESET}"
        out.append(row(tip, center=True))
        out.append(f"  {bot_b}")
        return "\n".join(out)

    if animate and is_tty:
        # Short, snappy Warp-style RGB wave entrance (~0.55s, 14 frames in-place)
        sys.stdout.write("\033[?25l")
        n_frames = 14
        frame_lines = 0
        try:
            for k in range(n_frames):
                rendered = build_frame(phase=(k * 0.065) % 1.0)
                if k > 0 and frame_lines > 0:
                    sys.stdout.write(f"\033[{frame_lines}A\r")
                sys.stdout.write("\033[?2026h" + rendered + "\n\033[?2026l")
                sys.stdout.flush()
                frame_lines = rendered.count("\n") + 1
                time.sleep(0.038)
        finally:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
    else:
        print(build_frame(phase=0.12))
