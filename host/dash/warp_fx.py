"""Minimalist Warp-Agent-CLI 3D Wireframe Animation & Interactive Start Engine for NURO.

Faithfully reverse-engineered from the Warp Agent CLI startup screen (YouTube 0:57-0:59):
- Left column: crisp minimalist session/hardware metadata with generous bottom breathing room
- Right hero canvas: real-time 3D rotating twin-slab isometric sculpture with solid-face Z-occlusion,
  beveled corners, slanted diagonal channel, depth-shaded silhouette edges, and subtle cosmic starfield
- Bottom: sleek bordered interactive prompt box with live block cursor & commands placeholder
- Continuous smooth 30 FPS turntable animation that NEVER stops until the first command is submitted!
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

# ---------------------------------------------------------------------------
# Sampled 4K Warp Agent CLI Palette (Monochrome Silver, Deep Slate, Pastel Sage)
# ---------------------------------------------------------------------------
# 3D Sculpture & Wireframe Edges
COL_EDGE_BRIGHT = "\033[38;2;213;217;229m"  # #d5d9e5 - near silhouette / front creases
COL_EDGE_MID    = "\033[38;2;158;163;176m"  # #9ea3b0 - mid-depth edges
COL_EDGE_DIM    = "\033[38;2;101;105;113m"  # #656971 - far edges / deep shadow
COL_SURF_DOT    = "\033[38;2;115;120;132m"  # #737884 - sparse surface stipple dots

# Starfield Glyphs
COL_STAR_BRIGHT = "\033[38;2;207;210;223m"  # #cfd2df - accent star *
COL_STAR_MID    = "\033[38;2;164;168;176m"  # #a4a8b0 - medium star +
COL_STAR_DIM    = "\033[38;2;101;105;113m"  # #656971 - faint star .

# Typography & Metadata
COL_TITLE       = "\033[38;2;207;210;236m"  # #cfd2ec - bold cool silver-lavender
COL_TEXT_WHITE  = "\033[38;2;231;235;241m"  # #e7ebf1 - bold workspace path & headers
COL_TEXT_MUTED  = "\033[38;2;164;168;176m"  # #a4a8b0 - secondary labels & subtitle
COL_TEXT_DIM    = "\033[38;2;101;105;113m"  # #656971 - deep slate / inactive port
COL_CHECK_SAGE  = "\033[38;2;202;231;164m"  # #cae7a4 - soft pastel sage checkmark & live port
COL_STATUS_TXT  = "\033[38;2;166;170;177m"  # #a6aab1 - checkmark description text

# Box & Prompt
COL_BOX_BORDER  = "\033[38;2;207;210;223m"  # #cfd2df - crisp neutral single-line box
COL_PROMPT_CYAN = "\033[38;2;56;189;248m"   # #38bdf8 - prompt glyph ❯
COL_CURSOR      = "\033[38;2;249;251;255m"  # #f9fbff - bright solid block cursor
COL_FOOTER_MODE = "\033[38;2;221;236;209m"  # #ddecd1 - soft sage-white mode tag
COL_FOOTER_TXT  = "\033[38;2;164;168;176m"  # #a4a8b0 - footer path & key hints

# Cyber-Aurora fallback palette (used by helper widgets in TUI)
PALETTE_WARP: Tuple[RGB, ...] = (
    (56, 189, 248),
    (99, 102, 241),
    (167, 139, 250),
    (125, 211, 252),
    (226, 232, 240),
    (56, 189, 248),
)
PALETTE_GREEN: Tuple[RGB, ...] = ((16, 185, 129), (52, 211, 153), (110, 231, 183))
PALETTE_AMBER: Tuple[RGB, ...] = ((217, 119, 6), (245, 158, 11), (252, 211, 77))
PALETTE_RED: Tuple[RGB, ...]   = ((220, 38, 38), (248, 81, 73), (252, 165, 165))

SPINNER_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SUB_BLOCKS = (" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█")
BRAILLE_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def fg_rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def bg_rgb(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


def sample_palette(t: float, palette: Sequence[RGB] = PALETTE_WARP) -> RGB:
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
    spread: float = 0.55,
) -> str:
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

    track_col = fg_rgb(51, 65, 85)
    parts: List[str] = []
    for i in range(width):
        t = (i / max(1, width - 1)) * 0.65 + phase
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
    px_w = width * 2
    px_h = rows * 4
    grid = [[0 for _ in range(width)] for _ in range(rows)]
    for x in range(px_w):
        u = x / max(1, px_w - 1)
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
            r, g, b = sample_palette((c_i / max(1, width)) * 0.65 + phase, PALETTE_WARP)
            row_chars.append(f"\033[38;2;{r};{g};{b}m{ch}")
        row_chars.append(RESET)
        lines.append("".join(row_chars))
    return lines


def _vis_len(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def _pad_left(s: str, width: int) -> str:
    vl = _vis_len(s)
    if vl >= width:
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
    return s + (" " * (width - vl))


# ---------------------------------------------------------------------------
# Solid 3D Twin-Slab Polyhedral Geometry (Warp Agent CLI Specification)
# ---------------------------------------------------------------------------

class _PolyMesh:
    """A convex 3D polyhedron with vertices, polygonal faces, and outward normals."""

    def __init__(self, vertices: List[Tuple[float, float, float]], faces: List[List[int]]):
        self.vertices = vertices
        self.faces = faces
        self.face_normals: List[Tuple[float, float, float]] = []
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
            e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
            nx = e1[1] * e2[2] - e1[2] * e2[1]
            ny = e1[2] * e2[0] - e1[0] * e2[2]
            nz = e1[0] * e2[1] - e1[1] * e2[0]
            mag = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            self.face_normals.append((nx / mag, ny / mag, nz / mag))

        edge_map = {}
        for fi, face in enumerate(faces):
            n_v = len(face)
            for i in range(n_v):
                i0 = face[i]
                i1 = face[(i + 1) % n_v]
                ekey = (min(i0, i1), max(i0, i1))
                edge_map.setdefault(ekey, []).append(fi)

        self.edges: List[Tuple[int, int, List[int]]] = []
        for (i0, i1), flist in edge_map.items():
            self.edges.append((i0, i1, flist))


def _build_warp_twin_slabs() -> Tuple[_PolyMesh, _PolyMesh]:
    """Construct the two iconic 3D extruded slabs of the Warp logo.

    - Left slab: shifted down-left, beveled corners, slanted right channel wall
    - Right slab: shifted up-right, beveled corners, slanted left channel wall
    - Clean slanted diagonal channel running top-right to bottom-left between them
    """
    d = 0.22  # half-depth (thickness = 0.44)

    # Left slab 2D profile (x, y):
    p_left = [
        (-0.82, -0.56),
        (-0.02, -0.56),
        ( 0.04, -0.46),
        (-0.14,  0.62),
        (-0.20,  0.72),
        (-0.82,  0.72),
        (-0.90,  0.62),
        (-0.90, -0.46),
    ]

    # Right slab 2D profile (x, y):
    p_right = [
        (0.24, -0.72),
        (0.82, -0.72),
        (0.90, -0.62),
        (0.90,  0.46),
        (0.82,  0.56),
        (0.06,  0.56),
        (0.00,  0.46),
        (0.18, -0.62),
    ]

    def make_slab(poly2d: List[Tuple[float, float]], depth: float) -> _PolyMesh:
        n = len(poly2d)
        verts: List[Tuple[float, float, float]] = []
        # Front face vertices: 0 .. n-1 (z = +depth)
        for x, y in poly2d:
            verts.append((x, y, depth))
        # Back face vertices: n .. 2n-1 (z = -depth)
        for x, y in poly2d:
            verts.append((x, y, -depth))

        faces: List[List[int]] = []
        # Front face (CCW)
        faces.append(list(range(n)))
        # Back face (CCW viewed from rear)
        faces.append(list(range(2 * n - 1, n - 1, -1)))
        # Side faces
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([i, next_i, next_i + n, i + n])

        return _PolyMesh(verts, faces)

    return make_slab(p_left, d), make_slab(p_right, d)


_SLAB_LEFT, _SLAB_RIGHT = _build_warp_twin_slabs()

# Deterministic sparse cosmic starfield (15 stars matching Warp Agent CLI 4K crops)
_STARS: Tuple[Tuple[float, float, float, int], ...] = (
    (0.04, 0.12, 0.4, 0),  # .
    (0.12, 0.78, 1.8, 1),  # +
    (0.24, 0.22, 3.2, 0),  # .
    (0.06, 0.58, 4.5, 2),  # *
    (0.32, 0.92, 2.7, 0),  # .
    (0.72, 0.08, 1.1, 2),  # *
    (0.82, 0.82, 3.9, 1),  # +
    (0.88, 0.25, 1.5, 0),  # .
    (0.94, 0.68, 5.1, 0),  # .
    (0.90, 0.92, 2.3, 2),  # *
    (0.96, 0.40, 5.8, 0),  # .
    (0.82, 0.52, 0.7, 1),  # +
    (0.66, 0.95, 3.1, 0),  # .
    (0.52, 0.96, 1.3, 1),  # +
    (0.78, 0.74, 4.1, 0),  # .
)
_STAR_GLYPHS = (".", "+", "*")


# ---------------------------------------------------------------------------
# 3D Canvas Rasterizer with Z-Buffer Occlusion & Clean Slope Glyphs
# ---------------------------------------------------------------------------

def render_warp_3d_canvas(width: int = 48, height: int = 18, t_sec: float = 0.0) -> List[str]:
    """Render one frame of the 3D spinning Warp twin slabs with solid Z-buffer occlusion."""
    chars = [[" " for _ in range(width)] for _ in range(height)]
    z_buf = [[-1e9 for _ in range(width)] for _ in range(height)]
    color_buf = [["" for _ in range(width)] for _ in range(height)]

    # 1. Background subtle twinkling starfield (pure cool monochrome silver)
    for sx, sy, sp, gi in _STARS:
        cx = int(sx * (width - 1))
        cy = int(sy * (height - 1))
        twinkle = 0.5 + 0.5 * math.sin(t_sec * 2.1 + sp)
        if twinkle > 0.32:
            chars[cy][cx] = _STAR_GLYPHS[gi]
            col = COL_STAR_BRIGHT if gi == 2 else (COL_STAR_MID if gi == 1 else COL_STAR_DIM)
            color_buf[cy][cx] = col
            z_buf[cy][cx] = -50.0

    # 2. Pure Y-axis turntable rotation at ~2.4 rad/s (period ~ 2.62s, zero wobbly pitch/roll)
    theta = (t_sec * 2.4) % (2.0 * math.pi)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Aspect ratio compensation: terminal cells are ~2x taller than wide
    scale_x = width * 0.40
    scale_y = height * 0.48
    center_x = width * 0.50
    center_y = height * 0.50

    meshes = (_SLAB_LEFT, _SLAB_RIGHT)

    for mesh in meshes:
        rot_verts: List[Tuple[float, float, float]] = []
        proj_verts: List[Tuple[float, float, float]] = []
        for (x, y, z) in mesh.vertices:
            rx = x * cos_t + z * sin_t
            ry = y
            rz = -x * sin_t + z * cos_t
            rot_verts.append((rx, ry, rz))
            px = center_x + rx * scale_x
            py = center_y + ry * scale_y
            proj_verts.append((px, py, rz))

        rot_normals: List[Tuple[float, float, float]] = []
        for (nx, ny, nz) in mesh.face_normals:
            rnx = nx * cos_t + nz * sin_t
            rny = ny
            rnz = -nx * sin_t + nz * cos_t
            rot_normals.append((rnx, rny, rnz))

        # A. Rasterize solid FRONT faces into the Z-buffer!
        # Front faces act as opaque blockers: back-faces and occluded edges behind them are culled!
        for fi, face in enumerate(mesh.faces):
            rnx, rny, rnz = rot_normals[fi]
            if rnz > 0.05:  # front-facing
                poly_proj = [proj_verts[vi] for vi in face]
                v0 = poly_proj[0]
                for ti in range(1, len(poly_proj) - 1):
                    v1 = poly_proj[ti]
                    v2 = poly_proj[ti + 1]

                    min_x = max(0, int(math.floor(min(v0[0], v1[0], v2[0]))))
                    max_x = min(width - 1, int(math.ceil(max(v0[0], v1[0], v2[0]))))
                    min_y = max(0, int(math.floor(min(v0[1], v1[1], v2[1]))))
                    max_y = min(height - 1, int(math.ceil(max(v0[1], v1[1], v2[1]))))

                    def edge_func(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
                        return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)

                    area = edge_func(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1])
                    if abs(area) < 1e-5:
                        continue

                    inv_area = 1.0 / area
                    for y in range(min_y, max_y + 1):
                        for x in range(min_x, max_x + 1):
                            px = x + 0.5
                            py = y + 0.5
                            w0 = edge_func(v1[0], v1[1], v2[0], v2[1], px, py) * inv_area
                            w1 = edge_func(v2[0], v2[1], v0[0], v0[1], px, py) * inv_area
                            w2 = edge_func(v0[0], v0[1], v1[0], v1[1], px, py) * inv_area
                            if w0 >= -1e-4 and w1 >= -1e-4 and w2 >= -1e-4:
                                pz = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                                if pz > z_buf[y][x]:
                                    z_buf[y][x] = pz
                                    # Very sparse surface stipple dots on front faces (matching Warp 4K crops)
                                    if fi in (0, 1) and (x % 5 == 1) and (y % 2 == 0):
                                        chars[y][x] = "."
                                        color_buf[y][x] = COL_SURF_DOT
                                    else:
                                        chars[y][x] = " "

        # B. Draw visible silhouette & crease edges on top of the solid geometry!
        for i0, i1, flist in mesh.edges:
            # Candidate visible edge: at least one adjacent face faces forward
            is_candidate = any(rot_normals[f_idx][2] > 0.02 for f_idx in flist)
            if not is_candidate:
                continue

            x0, y0, z0 = proj_verts[i0]
            x1, y1, z1 = proj_verts[i1]
            dx = x1 - x0
            dy = y1 - y0
            steps = max(2, int(max(abs(dx) * 1.5, abs(dy) * 2.0)))

            adx = abs(dx)
            ady = abs(dy) * 1.9  # vertical aspect compensation
            if ady < adx * 0.38:
                glyph = "-"
            elif adx < ady * 0.38:
                glyph = "|"
            elif (dx * dy) < 0:
                glyph = "/"
            else:
                glyph = "\\"

            for s in range(steps + 1):
                f = s / float(steps)
                px = x0 + dx * f
                py = y0 + dy * f
                pz = z0 + (z1 - z0) * f
                ix = int(round(px))
                iy = int(round(py))
                if 0 <= ix < width and 0 <= iy < height:
                    # Occlusion test: must be at or in front of the Z-buffer surface
                    if pz >= z_buf[iy][ix] - 0.04:
                        col = COL_EDGE_BRIGHT if pz > 0.05 else (COL_EDGE_MID if pz > -0.15 else COL_EDGE_DIM)
                        chars[iy][ix] = glyph
                        color_buf[iy][ix] = col
                        z_buf[iy][ix] = pz + 0.01

    out_rows: List[str] = []
    for r in range(height):
        parts: List[str] = []
        last_col = ""
        for c in range(width):
            ch = chars[r][c]
            if ch == " ":
                parts.append(" ")
                continue
            col = color_buf[r][c]
            if col != last_col:
                parts.append(col)
                last_col = col
            parts.append(ch)
        if last_col:
            parts.append(RESET)
        out_rows.append("".join(parts))

    return out_rows


# ---------------------------------------------------------------------------
# Warp Agent CLI Full Frame Assembly
# ---------------------------------------------------------------------------

def _build_warp_start_frame(
    t_sec: float,
    version: str,
    port_label: str,
    hw_live: bool,
    cwd_short: str,
    typed: str,
    cursor_pos: int,
    show_cursor: bool = True,
    total_w: int = 76,
) -> str:
    """Assemble the exact Warp-Agent-CLI start screen layout (left metadata + right 3D hero + bottom input box)."""
    left_w = 26
    right_w = total_w - left_w - 2  # 48 cols
    canvas_h = 18

    hw_color = COL_CHECK_SAGE if hw_live else COL_TEXT_DIM
    left_col: List[str] = [
        f"{BOLD}{COL_TITLE}NURO CLI{RESET}",
        f"{COL_TEXT_MUTED}v{version} · ESP32-S3{RESET}",
        f"{COL_TEXT_MUTED}QAT-SNN · 0.964 AUROC{RESET}",
        "",
        f"{BOLD}{COL_TEXT_WHITE}{cwd_short}{RESET}",
        f"{COL_CHECK_SAGE}✓{RESET} {COL_STATUS_TXT}CHB-MIT 23-ch loaded{RESET}",
        f"{COL_CHECK_SAGE}✓{RESET} {COL_STATUS_TXT}64-N Int8 SNN ready{RESET}",
        "",
        f"{BOLD}{COL_TEXT_WHITE}HARDWARE{RESET}",
        f"{hw_color}{port_label}{RESET}",
        f"{COL_STATUS_TXT}tui · web · chat · calc{RESET}",
    ] + [""] * 7  # Generous open breathing room below left column (rows 12-18)

    right_rows = render_warp_3d_canvas(width=right_w, height=canvas_h, t_sec=t_sec)

    lines: List[str] = []
    for i in range(canvas_h):
        l_txt = _pad_left(left_col[i] if i < len(left_col) else "", left_w)
        lines.append(f"  {l_txt}{right_rows[i]}")

    # Bottom minimalist Warp Agent CLI input box
    box_inner = total_w - 2
    max_input_vis = box_inner - 5
    view_txt = typed
    cpos = max(0, min(len(typed), cursor_pos))
    if len(view_txt) > max_input_vis:
        start_i = max(0, cpos - max_input_vis + 2)
        view_txt = view_txt[start_i : start_i + max_input_vis]
        cpos = cpos - start_i

    if show_cursor:
        if cpos < len(view_txt):
            before = view_txt[:cpos]
            under = view_txt[cpos]
            after = view_txt[cpos + 1 :]
            input_render = f"{COL_TEXT_WHITE}{before}\033[7m{under}\033[27m{COL_TEXT_WHITE}{after}{RESET}"
        else:
            blink_on = (int(t_sec * 2.2) % 2) == 0
            cur_glyph = f"{COL_CURSOR}█{RESET}" if blink_on else f"{COL_TEXT_MUTED}▌{RESET}"
            input_render = f"{COL_TEXT_WHITE}{view_txt}{cur_glyph}"
    else:
        input_render = f"{COL_TEXT_WHITE}{view_txt}{RESET}"

    placeholder = ""
    if not typed and show_cursor:
        placeholder = f"{COL_TEXT_MUTED}  Ask NURO anything · tui · web · chat · calc · help{RESET}"

    prompt_line = _pad_left(f" {BOLD}{COL_PROMPT_CYAN}❯{RESET} {input_render}{placeholder}", box_inner)
    lines.append(f"  {COL_BOX_BORDER}┌{'─' * box_inner}┐{RESET}")
    lines.append(f"  {COL_BOX_BORDER}│{RESET}{prompt_line}{COL_BOX_BORDER}│{RESET}")
    lines.append(f"  {COL_BOX_BORDER}└{'─' * box_inner}┘{RESET}")
    lines.append(
        f"  {COL_FOOTER_MODE}Interactive mode{RESET}  {COL_FOOTER_TXT}{cwd_short}  ·  Enter to run  ·  q to exit{RESET}"
    )
    return "\n".join(lines)


def _detect_hw_summary() -> Tuple[str, bool]:
    try:
        from host.board import detect_board_port

        p = detect_board_port()
        if p:
            return f"● {p} @ 2.0 Mbps", True
    except Exception:
        pass
    return "○ Standby · PnP ready", False


def _short_cwd() -> str:
    try:
        p = os.getcwd()
        home = os.path.expanduser("~")
        if p.startswith(home):
            p = "~" + p[len(home) :]
        parts = p.replace("\\", "/").split("/")
        if len(parts) > 2:
            return f"~/{parts[-2]}/{parts[-1]}"[:26]
        return p.replace("\\", "/")[:26]
    except Exception:
        return "~/mit/neural-loop"


def print_warp_cli_menu(version: str = "0.3.0", animate: bool = False) -> None:
    """Render one crisp static frame of the Warp-Agent-CLI start screen (non-interactive fallback)."""
    if os.name == "nt":
        os.system("")
    port_label, hw_live = _detect_hw_summary()
    frame = _build_warp_start_frame(
        t_sec=0.25,
        version=version,
        port_label=port_label,
        hw_live=hw_live,
        cwd_short=_short_cwd(),
        typed="",
        cursor_pos=0,
        show_cursor=False,
    )
    print(frame)


def run_warp_start_screen(version: str = "0.3.0") -> Optional[str]:
    """Run the LIVE continuous 3D Warp-Agent-CLI start screen until the first command is sent.

    The 3D twin-slab sculpture spins smoothly at ~30 FPS with starfield while the user
    types directly into the bottom input box. Pressing Enter locks the command and returns it.
    """
    if os.name == "nt":
        os.system("")

    port_label, hw_live = _detect_hw_summary()
    cwd_short = _short_cwd()

    # Non-TTY / piped stdin fallback (keeps pytest and pipes instant & clean)
    if not sys.stdout.isatty() or not sys.stdin.isatty() or os.name != "nt":
        print_warp_cli_menu(version=version, animate=False)
        try:
            return input("nuro ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

    import msvcrt

    typed = ""
    cursor_pos = 0
    t0 = time.perf_counter()
    fps = 30.0
    frame_period = 1.0 / fps

    sys.stdout.write("\033[?25l\033[H\033[2J")
    sys.stdout.flush()

    submitted: Optional[str] = None
    try:
        while True:
            t_frame = time.perf_counter()
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    ext = msvcrt.getwch() if msvcrt.kbhit() else ""
                    if ext == "K":
                        cursor_pos = max(0, cursor_pos - 1)
                    elif ext == "M":
                        cursor_pos = min(len(typed), cursor_pos + 1)
                    elif ext == "G":
                        cursor_pos = 0
                    elif ext == "O":
                        cursor_pos = len(typed)
                    elif ext == "S":
                        if cursor_pos < len(typed):
                            typed = typed[:cursor_pos] + typed[cursor_pos + 1 :]
                elif ch in ("\r", "\n"):
                    cmd = typed.strip()
                    if cmd:
                        submitted = cmd
                        break
                elif ch == "\x03":  # Ctrl+C
                    submitted = None
                    raise KeyboardInterrupt
                elif ch == "\x1b":  # Esc
                    if typed:
                        typed = ""
                        cursor_pos = 0
                    else:
                        submitted = None
                        raise KeyboardInterrupt
                elif ch == "\x08":  # Backspace
                    if cursor_pos > 0:
                        typed = typed[: cursor_pos - 1] + typed[cursor_pos:]
                        cursor_pos -= 1
                elif ch == "\x15":  # Ctrl+U
                    typed = ""
                    cursor_pos = 0
                elif ch.isprintable():
                    typed = typed[:cursor_pos] + ch + typed[cursor_pos:]
                    cursor_pos += 1

            elapsed = time.perf_counter() - t0
            frame = _build_warp_start_frame(
                t_sec=elapsed,
                version=version,
                port_label=port_label,
                hw_live=hw_live,
                cwd_short=cwd_short,
                typed=typed,
                cursor_pos=cursor_pos,
                show_cursor=(submitted is None),
            )
            payload = (
                "\033[?2026h\033[H"
                + "\n".join(line + "\033[K" for line in frame.splitlines())
                + "\n\033[J\033[?2026l"
            )
            sys.stdout.write(payload)
            sys.stdout.flush()

            if submitted is not None:
                break

            dt = frame_period - (time.perf_counter() - t_frame)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        submitted = None
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    return submitted


def play_warp_boot_splash(
    port: str = "COM6",
    hw_live: bool = False,
    mirror: bool = False,
    duration_s: float = 1.15,
) -> None:
    """Minimalist 3D spinning Warp wireframe splash before entering the TUI (skippable)."""
    if not sys.stdout.isatty():
        return

    term_cols, term_rows = shutil.get_terminal_size((80, 24))
    left_pad = " " * max(0, (term_cols - 76) // 2)
    top_pad = max(0, (term_rows - 23) // 2)
    port_label = f"● {port} @ 2.0 Mbps" if hw_live else ("○ Web Mirror" if mirror else f"○ {port} Standby")
    cwd_short = _short_cwd()

    fps = 30.0
    total_frames = max(14, int(round(duration_s * fps)))
    t0 = time.perf_counter()
    for f_idx in range(total_frames + 1):
        if os.name == "nt":
            try:
                import msvcrt

                if msvcrt.kbhit():
                    msvcrt.getch()
                    break
            except Exception:
                pass
        elapsed = f_idx / fps
        frame = _build_warp_start_frame(
            t_sec=elapsed,
            version="0.3.0",
            port_label=port_label,
            hw_live=hw_live,
            cwd_short=cwd_short,
            typed="tui --live",
            cursor_pos=10,
            show_cursor=False,
        )
        lines = ([""] * top_pad) + [left_pad + ln for ln in frame.splitlines()]
        sys.stdout.write("\033[?2026h\033[H" + "\n".join(l + "\033[K" for l in lines) + "\033[J\033[?2026l")
        sys.stdout.flush()
        dt = (t0 + (f_idx + 1) / fps) - time.perf_counter()
        if dt > 0:
            time.sleep(dt)

    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()
