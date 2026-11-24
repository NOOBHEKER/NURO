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
    d = 0.46  # deep sculptural front-to-back thickness (total depth = 0.92!)

    # Left slab 2D profile (x, y) — sleeker, narrower width with bold vertical presence:
    p_left = [
        (-0.82, -0.58),
        (-0.04, -0.58),
        ( 0.03, -0.47),
        (-0.15,  0.64),
        (-0.22,  0.75),
        (-0.82,  0.75),
        (-0.88,  0.64),
        (-0.88, -0.47),
    ]

    # Right slab 2D profile (x, y) — matching sleeker twin:
    p_right = [
        (0.22, -0.75),
        (0.82, -0.75),
        (0.88, -0.64),
        (0.88,  0.47),
        (0.82,  0.58),
        (0.04,  0.58),
        (-0.03, 0.47),
        (0.15, -0.64),
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

# Deterministic cosmic starfield & sparkle layer (28 multi-tier stars + diamond glints)
# Tuple: (x_norm, y_norm, phase, glyph_idx, twinkle_speed)
_STARS: Tuple[Tuple[float, float, float, int, float], ...] = (
    (0.03, 0.10, 0.4, 0, 2.1),   # ·
    (0.08, 0.34, 2.9, 3, 3.4),   # ✧
    (0.12, 0.78, 1.8, 1, 2.4),   # +
    (0.18, 0.14, 5.2, 2, 3.1),   # *
    (0.24, 0.22, 3.2, 0, 1.9),   # ·
    (0.05, 0.58, 4.5, 4, 3.8),   # ✦
    (0.15, 0.90, 0.9, 3, 2.7),   # ✧
    (0.30, 0.06, 2.2, 1, 2.9),   # +
    (0.32, 0.92, 2.7, 0, 2.0),   # ·
    (0.42, 0.05, 4.1, 3, 3.6),   # ✧
    (0.56, 0.04, 1.4, 2, 2.8),   # *
    (0.72, 0.08, 1.1, 4, 3.5),   # ✦
    (0.84, 0.12, 3.7, 3, 2.9),   # ✧
    (0.93, 0.18, 0.6, 2, 3.3),   # *
    (0.88, 0.25, 1.5, 0, 2.2),   # ·
    (0.96, 0.40, 5.8, 3, 3.7),   # ✧
    (0.86, 0.48, 2.4, 1, 2.6),   # +
    (0.94, 0.68, 5.1, 4, 3.9),   # ✦
    (0.82, 0.56, 0.7, 1, 2.3),   # +
    (0.82, 0.82, 3.9, 2, 3.2),   # *
    (0.91, 0.90, 2.3, 4, 3.6),   # ✦
    (0.76, 0.94, 4.8, 3, 2.8),   # ✧
    (0.66, 0.95, 3.1, 0, 2.1),   # ·
    (0.52, 0.96, 1.3, 1, 3.0),   # +
    (0.78, 0.74, 4.1, 0, 2.5),   # ·
    (0.20, 0.48, 5.5, 0, 2.7),   # ·
    (0.10, 0.68, 1.9, 2, 3.1),   # *
    (0.64, 0.12, 4.4, 0, 2.4),   # ·
)
_STAR_GLYPHS = ("·", "+", "*", "✧", "✦")
_COL_SPARKLE_ICE = f"{BOLD}{fg_rgb(220, 244, 255)}"
_COL_SPARKLE_CYAN = fg_rgb(138, 214, 245)


# ---------------------------------------------------------------------------
# 3D Canvas Rasterizer with Z-Buffer Occlusion & Clean Slope Glyphs
# ---------------------------------------------------------------------------

def render_warp_3d_canvas(width: int = 48, height: int = 18, t_sec: float = 0.0) -> List[str]:
    """Render one frame of the 3D spinning Warp twin slabs with solid Z-buffer occlusion."""
    chars = [[" " for _ in range(width)] for _ in range(height)]
    z_buf = [[-1e9 for _ in range(width)] for _ in range(height)]
    color_buf = [["" for _ in range(width)] for _ in range(height)]

    # 1. Background twinkling cosmic starfield + animated diamond sparkles
    for sx, sy, sp, gi, spd in _STARS:
        cx = int(sx * (width - 1))
        cy = int(sy * (height - 1))
        twinkle = 0.5 + 0.5 * math.sin(t_sec * spd + sp)
        if twinkle > 0.24:
            # Peak twinkle briefly upgrades '*' / '✧' to a bright crisp '✦' flash
            if twinkle > 0.90 and gi >= 2:
                chars[cy][cx] = "✦"
                col = _COL_SPARKLE_ICE
            elif twinkle > 0.72 and gi >= 3:
                chars[cy][cx] = _STAR_GLYPHS[gi]
                col = _COL_SPARKLE_CYAN
            else:
                chars[cy][cx] = _STAR_GLYPHS[gi]
                col = COL_STAR_BRIGHT if gi >= 2 else (COL_STAR_MID if gi == 1 else COL_STAR_DIM)
            color_buf[cy][cx] = col
            z_buf[cy][cx] = -50.0

    # 2. Smooth Y-axis turntable rotation + subtle 9° forward elevation pitch so front-face depth pops!
    theta = (t_sec * 2.2) % (2.0 * math.pi)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    pitch = 0.16  # ~9.2° subtle top-down perspective reveals thick top/channel walls right on the front!
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)

    # Strong 3D perspective camera (dramatic front-to-back depth!) + narrower, sleeker width
    cam_dist = 2.65
    scale_x = width * 0.29
    scale_y = height * 0.39
    center_x = width * 0.50
    center_y = height * 0.50

    # Directional key light (upper-front-left) for rich 3D volume shading on side walls & front bevel
    lx, ly, lz = -0.38, -0.42, 0.82

    col_edge_hero = f"{BOLD}{fg_rgb(248, 250, 255)}"
    col_edge_front = fg_rgb(198, 214, 236)
    col_edge_mid = fg_rgb(112, 128, 154)
    col_edge_far = fg_rgb(62, 72, 90)

    meshes = (_SLAB_LEFT, _SLAB_RIGHT)

    for mesh in meshes:
        rot_verts: List[Tuple[float, float, float]] = []
        proj_verts: List[Tuple[float, float, float]] = []
        for (x, y, z) in mesh.vertices:
            # Rotate Y (turntable)
            rx = x * cos_t + z * sin_t
            rz0 = -x * sin_t + z * cos_t
            # Subtle X elevation tilt so front-facing top/bottom/channel depth walls are clearly visible
            ry = y * cos_p - rz0 * sin_p
            rz = y * sin_p + rz0 * cos_p
            rot_verts.append((rx, ry, rz))
            persp = cam_dist / (cam_dist - rz)
            px = center_x + rx * persp * scale_x
            py = center_y + ry * persp * scale_y
            proj_verts.append((px, py, rz))

        rot_normals: List[Tuple[float, float, float]] = []
        for (nx, ny, nz) in mesh.face_normals:
            rnx = nx * cos_t + nz * sin_t
            rnz0 = -nx * sin_t + nz * cos_t
            rny = ny * cos_p - rnz0 * sin_p
            rnz = ny * sin_p + rnz0 * cos_p
            rot_normals.append((rnx, rny, rnz))

        # A. Rasterize solid FRONT faces into the Z-buffer with directional 3D lighting!
        for fi, face in enumerate(mesh.faces):
            rnx, rny, rnz = rot_normals[fi]
            if rnz > 0.04:  # front-facing
                ndotl = max(0.0, rnx * lx + rny * ly + rnz * lz)
                is_side_wall = fi >= 2
                # Compute sleek architectural side-wall shade (makes 3D thickness pop!)
                if is_side_wall:
                    if ndotl > 0.68:
                        wall_ch = "▒"
                        wall_col = fg_rgb(132, 152, 184)
                    elif ndotl > 0.42:
                        wall_ch = "░"
                        wall_col = fg_rgb(96, 114, 144)
                    elif ndotl > 0.20:
                        wall_ch = ":"
                        wall_col = fg_rgb(76, 90, 114)
                    else:
                        wall_ch = "·"
                        wall_col = fg_rgb(58, 68, 86)
                else:
                    # Front/back faces stay clean dark negative space with light-reactive stipple
                    dot_lum = int(68 + ndotl * 64)
                    face_dot_col = fg_rgb(dot_lum, dot_lum + 8, min(255, dot_lum + 22))

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
                                    if is_side_wall:
                                        chars[y][x] = wall_ch
                                        color_buf[y][x] = wall_col
                                    elif (x % 4 == 1) and (y % 2 == 0):
                                        # Depth-graded front-face micro-relief (brighter closer to viewer)
                                        zn = max(0.0, min(1.0, (pz + 0.45) / 0.95))
                                        dl = int(58 + (0.45 * ndotl + 0.55 * zn) * 115)
                                        chars[y][x] = "·" if zn > 0.60 else "."
                                        color_buf[y][x] = fg_rgb(dl, dl + 8, min(255, dl + 24))
                                    else:
                                        chars[y][x] = " "

        # B. Draw bold, thick foreground silhouette & crease edges (heavy strokes in front, lighter in back)
        for i0, i1, flist in mesh.edges:
            is_candidate = any(rot_normals[f_idx][2] > 0.02 for f_idx in flist)
            if not is_candidate:
                continue

            x0, y0, z0 = proj_verts[i0]
            x1, y1, z1 = proj_verts[i1]
            dx = x1 - x0
            dy = y1 - y0
            steps = max(2, int(max(abs(dx) * 1.8, abs(dy) * 2.4)))

            adx = abs(dx)
            ady = abs(dy) * 1.9
            is_horiz = ady < adx * 0.38
            is_vert = adx < ady * 0.38

            for s in range(steps + 1):
                f = s / float(steps)
                px = x0 + dx * f
                py = y0 + dy * f
                pz = z0 + (z1 - z0) * f
                ix = int(round(px))
                iy = int(round(py))

                if pz > 0.24:
                    col = col_edge_hero
                    if is_horiz:
                        glyph = "━"
                    elif is_vert:
                        glyph = "┃"
                    else:
                        glyph = "╱" if (dx * dy) < 0 else "╲"
                elif pz > 0.0:
                    col = col_edge_front
                    if is_horiz:
                        glyph = "━"
                    elif is_vert:
                        glyph = "┃"
                    else:
                        glyph = "╱" if (dx * dy) < 0 else "╲"
                elif pz > -0.24:
                    col = col_edge_mid
                    if is_horiz:
                        glyph = "─"
                    elif is_vert:
                        glyph = "│"
                    else:
                        glyph = "/" if (dx * dy) < 0 else "\\"
                else:
                    col = col_edge_far
                    if is_horiz:
                        glyph = "-"
                    elif is_vert:
                        glyph = "|"
                    else:
                        glyph = "/" if (dx * dy) < 0 else "\\"

                # Thicken foreground edges by 1 adjacent cell so front strokes look bold & weighty
                offsets = ((0, 0), (1, 0)) if (pz > 0.14 and not is_horiz) else ((0, 0),)
                for ox, oy in offsets:
                    nx_i, ny_i = ix + ox, iy + oy
                    if 0 <= nx_i < width and 0 <= ny_i < height:
                        if pz >= z_buf[ny_i][nx_i] - 0.05:
                            chars[ny_i][nx_i] = glyph
                            color_buf[ny_i][nx_i] = col
                            z_buf[ny_i][nx_i] = pz + 0.01

        # C. Specular vertex glints — crisp brief '✦' sparkle on leading bright corners
        for vi, (px, py, pz) in enumerate(proj_verts):
            if pz > 0.28:
                glint = 0.5 + 0.5 * math.sin(t_sec * 5.2 + vi * 1.7)
                if glint > 0.88:
                    gx, gy = int(round(px)), int(round(py))
                    if 0 <= gx < width and 0 <= gy < height and pz >= z_buf[gy][gx] - 0.08:
                        chars[gy][gx] = "✦" if glint > 0.94 else "✧"
                        color_buf[gy][gx] = _COL_SPARKLE_ICE
                        z_buf[gy][gx] = pz + 0.02

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

def _wrap_chat_lines(
    chat_log: List[Tuple[str, str]],
    frame_w: int,
    ai_busy: bool,
    t_sec: float,
) -> List[str]:
    """Word-wrap chat Q&A history in a clean ChatGPT/Claude-style feed scrolling upwards."""
    import textwrap

    wrap_w = max(24, frame_w - 8)
    out: List[str] = []
    spin_chars = ("✦", "✧", "·", "✧")
    spin = spin_chars[int(t_sec * 6.0) % len(spin_chars)]
    blink_cur = f"{COL_PROMPT_CYAN}▌{RESET}" if (int(t_sec * 4.0) % 2) == 0 else " "
    for idx, (role, text) in enumerate(chat_log):
        clean = " ".join((text or "").replace("\r", " ").split())
        is_last = idx == len(chat_log) - 1
        if role == "user":
            if out:
                out.append(" " * frame_w)  # breathable spacer between Q&A turns
            wrapped = textwrap.wrap(clean, width=wrap_w) or [""]
            for j, seg in enumerate(wrapped):
                bar = f"  {BOLD}{COL_PROMPT_CYAN}▌{RESET} " if j == 0 else f"  {COL_TEXT_DIM}│{RESET} "
                out.append(_pad_left(f"{bar}{BOLD}{COL_TEXT_WHITE}{seg}{RESET}", frame_w))
        else:
            if not clean and ai_busy and is_last:
                out.append(
                    _pad_left(
                        f"  {COL_CHECK_SAGE}{spin}{RESET} {COL_TEXT_MUTED}NURO is thinking…{RESET}",
                        frame_w,
                    )
                )
            else:
                wrapped = textwrap.wrap(clean, width=wrap_w) or [""]
                for j, seg in enumerate(wrapped):
                    prefix = f"  {BOLD}{COL_CHECK_SAGE}✦{RESET} " if j == 0 else "    "
                    tail = f" {blink_cur}" if (ai_busy and is_last and j == len(wrapped) - 1) else ""
                    out.append(_pad_left(f"{prefix}{COL_STATUS_TXT}{seg}{RESET}{tail}", frame_w))
    return out


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
    max_rows: int = 24,
    chat_log: Optional[List[Tuple[str, str]]] = None,
    ai_busy: bool = False,
) -> str:
    """Assemble the Warp-Agent-CLI start screen with live spinning 3D hero + upward-scrolling Q&A."""
    frame_w = max(50, int(total_w))
    left_w = 24 if frame_w < 70 else 26
    right_w = max(20, frame_w - left_w - 2)

    q = typed.strip()
    show_palette = q.startswith("/") or q.startswith("?")
    palette_lines: List[str] = []

    if show_palette:
        is_slash = q.startswith("/")
        flt = q[1:].lower()
        items = _SLASH_FUNCS if is_slash else _ALL_COMMANDS
        hits = [it for it in items if flt in it[0].lower()] or items
        title = "FUNCTIONS" if is_slash else "ALL COMMANDS"
        palette_lines.append(
            _pad_left(
                f"  {BOLD}{COL_TEXT_WHITE}{title}{RESET}  {COL_TEXT_MUTED}type to filter · Enter to run{RESET}",
                frame_w,
            )
        )
        n_cols = 3 if (frame_w >= 86 and len(hits) > 8) else 2
        n_rows = (len(hits) + n_cols - 1) // n_cols
        # Reserve at least 8 rows for hero + 3 for box + 1 header
        max_pal_rows = max(3, max_rows - 13)
        n_rows = min(n_rows, max_pal_rows)
        col_w = (frame_w - 2) // n_cols
        cmd_w = 10 if n_cols == 3 else 11
        for r_i in range(n_rows):
            cells: List[str] = []
            for c_i in range(n_cols):
                idx = c_i * n_rows + r_i
                if idx < len(hits):
                    nm, desc = hits[idx]
                    cells.append(
                        _pad_left(
                            f"{BOLD}{COL_PROMPT_CYAN}{nm:<{cmd_w}s}{RESET}{COL_STATUS_TXT}{desc}{RESET}",
                            col_w,
                        )
                    )
                else:
                    cells.append(" " * col_w)
            palette_lines.append(_pad_left("  " + "".join(cells), frame_w))
    else:
        hint = "clear · " if chat_log else ""
        palette_lines.append(
            _pad_left(
                f"  {COL_FOOTER_MODE}Interactive mode{RESET}  {COL_FOOTER_TXT}{cwd_short}  ·  / functions  ·  ? commands  ·  {hint}q exit{RESET}",
                frame_w,
            )
        )

    # Strict vertical budget: hero + chat_pane + 3 (input box) + palette <= max_rows - 1 (NEVER scrolls!)
    avail_above_box = max(8, max_rows - 1 - 3 - len(palette_lines))
    chat_visible_lines: List[str] = []
    if chat_log and not show_palette and avail_above_box >= 13:
        # Keep the 3D spinning hero alive at the top (9-15 rows) and scroll Q&A upwards below it!
        canvas_h = max(9, min(15, avail_above_box - 5))
        chat_budget = avail_above_box - canvas_h - 1  # 1 subtle divider line
        all_chat = _wrap_chat_lines(chat_log, frame_w, ai_busy=ai_busy, t_sec=t_sec)
        chat_visible_lines = all_chat[-chat_budget:] if chat_budget > 0 else []
    else:
        canvas_h = max(8, min(24, avail_above_box))

    hw_color = COL_CHECK_SAGE if hw_live else COL_TEXT_DIM
    left_col: List[str] = [
        f"{BOLD}{COL_PROMPT_CYAN}▌{RESET} {BOLD}{COL_TITLE}NURO CLI{RESET}",
        f"  {COL_TEXT_MUTED}v{version} · ESP32-S3{RESET}",
        f"  {COL_TEXT_MUTED}QAT-SNN · 0.964 AUROC{RESET}",
        "",
        f"{BOLD}{COL_TEXT_WHITE}{cwd_short}{RESET}",
        f"{COL_CHECK_SAGE}✓{RESET} {COL_STATUS_TXT}CHB-MIT 23-ch loaded{RESET}",
        f"{COL_CHECK_SAGE}✓{RESET} {COL_STATUS_TXT}64-N Int8 SNN ready{RESET}",
        "",
        f"{BOLD}{COL_TEXT_WHITE}HARDWARE{RESET}",
        f"{hw_color}{port_label}{RESET}",
        f"{COL_STATUS_TXT}tui · web · chat · calc{RESET}",
    ]

    right_rows = render_warp_3d_canvas(width=right_w, height=canvas_h, t_sec=t_sec)

    # Vertically center the left info column beside the 3D hero on tall/fullscreen windows
    left_top_pad = max(0, (canvas_h - len(left_col)) // 2)
    lines: List[str] = []
    for i in range(canvas_h):
        li = i - left_top_pad
        l_txt = _pad_left(left_col[li] if 0 <= li < len(left_col) else "", left_w)
        lines.append(_pad_left(f"  {l_txt}{right_rows[i]}", frame_w))

    if chat_visible_lines:
        lines.append(f"  {COL_BOX_BORDER}{'─' * (frame_w - 2)}{RESET}")
        lines.extend(chat_visible_lines)

    # Bottom minimalist Warp Agent CLI input box (exact width == frame_w)
    box_inner = frame_w - 4  # "  ┌" (3) + box_inner + "┐" (1) == frame_w
    max_input_vis = max(8, box_inner - 5)
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
        placeholder = f"{COL_TEXT_MUTED}  Ask NURO anything · / functions · ? commands{RESET}"

    prompt_line = _pad_left(f" {BOLD}{COL_PROMPT_CYAN}❯{RESET} {input_render}{placeholder}", box_inner)
    lines.append(f"  {COL_BOX_BORDER}┌{'─' * box_inner}┐{RESET}")
    lines.append(f"  {COL_BOX_BORDER}│{RESET}{prompt_line}{COL_BOX_BORDER}│{RESET}")
    lines.append(f"  {COL_BOX_BORDER}└{'─' * box_inner}┘{RESET}")
    lines.extend(palette_lines)
    return "\n".join(lines)


_SLASH_FUNCS = [
    ("/tui", "Live closed-loop cockpit"), ("/web", "Browser cockpit"),
    ("/demo", "Web + TUI + GUI"), ("/gui", "Matplotlib cockpit"),
    ("/bench", "1000-frame HIL bench"), ("/info", "Board probe"),
    ("/doctor", "Env & board check"), ("/test", "Run pytest suite"),
    ("/chat", "Offline AI quiz"), ("/ask", "Ask local AI"),
    ("/calc", "Math shell"), ("/chem", "Chemistry engine"),
    ("/phy", "Physics engine"), ("/symptoms", "Symptom triage"),
    ("/figures", "Rebuild figures"), ("/session", "Named sessions"),
]
_ALL_COMMANDS = [(n.lstrip("/"), d) for n, d in _SLASH_FUNCS] + [
    ("setup", "Bootstrap this PC"), ("eval", "Synthetic gates"),
    ("obsidian", "Vault sync"), ("term", "Serial terminal"),
    ("watch", "USB auto-launch"), ("llm", "Local model status"),
    ("solve", "One-shot math"), ("q", "Exit"),
]


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


def _enable_win_vt_utf8() -> None:
    """Enable UTF-8 code page (65001) and Win32 VT processing without spawning a cmd subshell."""
    if os.name != "nt":
        return
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.SetConsoleOutputCP(65001)
        k32.SetConsoleCP(65001)
        h_out = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h_out, ctypes.byref(mode)):
            k32.SetConsoleMode(h_out, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        os.system("")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def print_warp_cli_menu(version: str = "0.3.0", animate: bool = False) -> None:
    """Render one crisp static frame of the Warp-Agent-CLI start screen (non-interactive fallback)."""
    _enable_win_vt_utf8()
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


def _apply_key(
    typed: str,
    cursor_pos: int,
    ch: str,
    ext: str = "",
) -> Tuple[str, int, Optional[str], bool]:
    """Pure, testable keystroke state transition -> (typed, cursor_pos, submitted, aborted)."""
    if ch in ("\x00", "\xe0"):
        if ext == "K":  # Left
            cursor_pos = max(0, cursor_pos - 1)
        elif ext == "M":  # Right
            cursor_pos = min(len(typed), cursor_pos + 1)
        elif ext == "G":  # Home
            cursor_pos = 0
        elif ext == "O":  # End
            cursor_pos = len(typed)
        elif ext == "S":  # Delete
            if cursor_pos < len(typed):
                typed = typed[:cursor_pos] + typed[cursor_pos + 1 :]
        return typed, cursor_pos, None, False
    if ch in ("\r", "\n"):
        cmd = typed.strip().lstrip("/")
        if cmd in ("?", ""):
            return typed, cursor_pos, None, False
        return typed, cursor_pos, cmd, False
    if ch == "\x03":  # Ctrl+C
        return typed, cursor_pos, None, True
    if ch == "\x1b":  # Esc
        if typed:
            return "", 0, None, False
        return typed, cursor_pos, None, True
    if ch == "\x08":  # Backspace
        if cursor_pos > 0:
            typed = typed[: cursor_pos - 1] + typed[cursor_pos:]
            cursor_pos -= 1
        return typed, cursor_pos, None, False
    if ch == "\x7f":  # Ctrl+Backspace (delete previous word)
        if cursor_pos > 0:
            j = cursor_pos - 1
            while j > 0 and typed[j - 1] == " ":
                j -= 1
            while j > 0 and typed[j - 1] != " ":
                j -= 1
            typed = typed[:j] + typed[cursor_pos:]
            cursor_pos = j
        return typed, cursor_pos, None, False
    if ch == "\x15":  # Ctrl+U
        return "", 0, None, False
    if ch.isprintable():
        typed = typed[:cursor_pos] + ch + typed[cursor_pos:]
        cursor_pos += 1
    return typed, cursor_pos, None, False


def _diff_frame_payload(prev_rows: List[str], new_rows: List[str]) -> str:
    """Build a minimal, zero-scroll, explicit-row ANSI diff payload (never clears screen mid-frame)."""
    chunks: List[str] = ["\033[?2026h"]
    n_new = len(new_rows)
    n_prev = len(prev_rows)
    for r in range(n_new):
        if r >= n_prev or prev_rows[r] != new_rows[r]:
            chunks.append(f"\033[{r + 1};1H{new_rows[r]}\033[K")
    for r in range(n_new, n_prev):
        chunks.append(f"\033[{r + 1};1H\033[K")
    chunks.append("\033[?2026l")
    return "".join(chunks)


def _atomic_write_stdout(text: str) -> None:
    """Write a complete frame via Python's WriteConsoleW-backed stdout (immune to CP437 mojibake)."""
    sys.stdout.write(text)
    sys.stdout.flush()


_KNOWN_INLINE_CMDS = {
    "setup", "tui", "neural", "mneuro", "live", "treat", "patient", "clinic",
    "demo", "web", "gui", "bench", "eval", "test", "figures", "chat", "chatbot",
    "quiz", "ask", "ai", "symptoms", "health", "check", "llm", "calc", "chem",
    "phy", "session", "obsidian", "graph", "term", "serial", "solve", "info",
    "doctor", "watch", "pnp", "plug-and-play", "autolaunch", "help", "commands",
    "q", "quit", "exit",
}


def run_warp_start_screen(
    version: str = "0.3.0",
    chat_log: Optional[List[Tuple[str, str]]] = None,
) -> Optional[str]:
    """Run the LIVE continuous 3D Warp-Agent-CLI start screen with inline upward-scrolling AI chat.

    - Natural messages (e.g. 'hey', 'what is gravity?') stream LIVE inside the screen while the
      3D sparkling logo keeps spinning smoothly at 24 FPS and Q&A scrolls upwards!
    - Typing ANY command (/tui, tui, /web, /calc, /info, etc.) at any time immediately dispatches it.
    """
    _enable_win_vt_utf8()

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
    import threading

    if chat_log is None:
        chat_log = []
    chat_lock = threading.Lock()
    ai_state = {"busy": False, "gen": 0}

    def _dispatch_inline_chat(prompt_text: str) -> None:
        from host.cli import _instant_fast_reply, _start_llm_background_warmup

        fast = _instant_fast_reply(prompt_text)
        if fast is not None:
            _start_llm_background_warmup()
            with chat_lock:
                chat_log.append(("user", prompt_text))
                chat_log.append(("ai", fast))
            return

        with chat_lock:
            ai_state["gen"] += 1
            my_gen = ai_state["gen"]
            ai_state["busy"] = True
            hist_snapshot = [
                {"role": ("user" if r == "user" else "assistant"), "content": txt}
                for r, txt in chat_log[-6:]
                if txt
            ]
            chat_log.append(("user", prompt_text))
            chat_log.append(("ai", ""))
            ans_idx = len(chat_log) - 1

        def _worker() -> None:
            try:
                from host import llm_backend

                if not llm_backend.ensure_server(wait_seconds=18.0):
                    from host.chatbot import answer_question

                    ans = answer_question(prompt_text)
                    with chat_lock:
                        if my_gen == ai_state["gen"] and ans_idx < len(chat_log):
                            chat_log[ans_idx] = ("ai", ans)
                    return

                acc: List[str] = []
                for piece in llm_backend.ask_stream(prompt_text, history=hist_snapshot, max_tokens=200):
                    acc.append(piece)
                    with chat_lock:
                        if my_gen != ai_state["gen"]:
                            return
                        if ans_idx < len(chat_log):
                            chat_log[ans_idx] = ("ai", "".join(acc))
                if not acc:
                    with chat_lock:
                        if my_gen == ai_state["gen"] and ans_idx < len(chat_log):
                            chat_log[ans_idx] = ("ai", "No response from local model.")
            except Exception as exc:
                with chat_lock:
                    if my_gen == ai_state["gen"] and ans_idx < len(chat_log):
                        chat_log[ans_idx] = ("ai", f"(error: {exc})")
            finally:
                with chat_lock:
                    if my_gen == ai_state["gen"]:
                        ai_state["busy"] = False

        threading.Thread(target=_worker, daemon=True).start()

    typed = ""
    cursor_pos = 0
    t0 = time.perf_counter()
    fps = 24.0
    frame_period = 1.0 / fps
    prev_rows: List[str] = []
    prev_size = (0, 0)

    # Disable auto-wrap and hide hardware cursor; clear viewport ONCE on start
    _atomic_write_stdout("\033[?7l\033[?25l\033[H\033[2J")

    submitted: Optional[str] = None
    last_rows_count = 0
    try:
        while True:
            t_frame = time.perf_counter()
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                ext = msvcrt.getwch() if ch in ("\x00", "\xe0") else ""
                typed, cursor_pos, sub, aborted = _apply_key(typed, cursor_pos, ch, ext)
                if aborted:
                    submitted = None
                    raise KeyboardInterrupt
                if sub is not None:
                    first_tok = sub.split()[0].lower() if sub.split() else ""
                    if first_tok == "clear":
                        with chat_lock:
                            ai_state["gen"] += 1
                            ai_state["busy"] = False
                            chat_log.clear()
                        typed = ""
                        cursor_pos = 0
                        continue
                    from host.cli import _looks_like_math_or_sci

                    if first_tok in _KNOWN_INLINE_CMDS or _looks_like_math_or_sci(sub):
                        submitted = sub
                        break
                    # Stay right inside the live spinning 3D screen and stream Q&A upwards!
                    _dispatch_inline_chat(sub)
                    typed = ""
                    cursor_pos = 0

            elapsed = time.perf_counter() - t0
            term_cols, term_rows = shutil.get_terminal_size((80, 24))
            if (term_cols, term_rows) != prev_size:
                if prev_size != (0, 0):
                    _atomic_write_stdout("\033[H\033[2J")
                    prev_rows = []
                prev_size = (term_cols, term_rows)

            # Responsive sizing + dead-center layout (like agy CLI / OpenCode) on any window or fullscreen!
            dyn_w = min(124, max(50, term_cols - 4))
            margin_x = " " * max(0, (term_cols - dyn_w) // 2)
            with chat_lock:
                log_snap = list(chat_log)
                busy_snap = ai_state["busy"]
            frame = _build_warp_start_frame(
                t_sec=elapsed,
                version=version,
                port_label=port_label,
                hw_live=hw_live,
                cwd_short=cwd_short,
                typed=typed,
                cursor_pos=cursor_pos,
                show_cursor=(submitted is None),
                total_w=dyn_w,
                max_rows=max(16, term_rows),
                chat_log=log_snap,
                ai_busy=busy_snap,
            )
            raw_rows = frame.splitlines()
            max_vis_rows = max(1, term_rows - 1)
            slack_y = max(0, max_vis_rows - len(raw_rows))
            # Center vertically on start screen; gently anchor with top breathing room in chat mode
            margin_y = (slack_y // 2) if not log_snap else min(2, slack_y // 2)
            new_rows = ([""] * margin_y + [margin_x + r for r in raw_rows])[:max_vis_rows]
            last_rows_count = len(new_rows)
            payload = _diff_frame_payload(prev_rows, new_rows)
            if len(payload) > 16:  # skip no-op frames
                _atomic_write_stdout(payload)
            prev_rows = new_rows

            if submitted is not None:
                break

            # Sub-slice sleep (5 ms slices) so keypresses respond in < 5 ms!
            deadline = t_frame + frame_period
            while time.perf_counter() < deadline:
                if msvcrt.kbhit():
                    break
                time.sleep(0.005)
    except KeyboardInterrupt:
        submitted = None
    finally:
        # Park cursor cleanly below the rendered screen and restore wrap + cursor
        _atomic_write_stdout(f"\033[{last_rows_count + 1};1H\033[?7h\033[?25h\n")

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
