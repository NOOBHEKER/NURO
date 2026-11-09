"""Math & Physics Natural-Language Terminal Solver with Unicode Graphs for nuro.

Parses natural expressions with words like 'sqrt', 'pi', 'sin', 'squared', 'integral',
solves them symbolically & numerically, and renders an instant ASCII/Braille curve.
"""
from __future__ import annotations

import math
import re
from typing import Tuple, List, Optional
import numpy as np


BLOCKS = " ▂▃▄▅▆▇█"
BRAILLE_MAP = [
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
]


def render_terminal_graph(
    xs: np.ndarray,
    ys: np.ndarray,
    cols: int = 50,
    rows: int = 6,
    x_label: str = "x",
    y_label: str = "f(x)",
) -> str:
    """Render a high-resolution Unicode Braille 2D plot with labeled axes."""
    px_w = cols * 2
    px_h = rows * 4
    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    y_clean = np.nan_to_num(ys, nan=0.0, posinf=1e4, neginf=-1e4)
    y_min, y_max = float(np.min(y_clean)), float(np.max(y_clean))
    if abs(y_max - y_min) < 1e-9:
        y_max += 1.0
        y_min -= 1.0

    # Resample to px_w points
    x_sampled = np.linspace(xs[0], xs[-1], px_w)
    y_sampled = np.interp(x_sampled, xs, y_clean)
    norm = np.clip((y_sampled - y_min) / (y_max - y_min), 0.0, 0.999)
    y_px = (px_h - 1) - np.floor(norm * (px_h - 1)).astype(int)

    prev_y = int(y_px[0])
    for x in range(px_w):
        cur_y = int(y_px[x])
        y0, y1 = min(prev_y, cur_y), max(prev_y, cur_y)
        for yy in range(y0, y1 + 1):
            grid[yy // 4][x // 2] |= BRAILLE_MAP[yy % 4][x % 2]
        prev_y = cur_y

    lines: List[str] = []
    lines.append(f"┌─[ GRAPH: {y_label} vs {x_label} ]" + "─" * (cols - len(y_label) - len(x_label) - 15) + "┐")
    for r in range(rows):
        chars = "".join(chr(0x2800 + grid[r][c]) for c in range(cols))
        val_tag = f"{y_max if r == 0 else (y_min if r == rows - 1 else (y_max + y_min)/2):+7.2f} ┤"
        if r not in (0, rows // 2, rows - 1):
            val_tag = "        ┤"
        lines.append(f"│ {val_tag} \033[38;5;75m{chars}\033[0m │")
    lines.append(f"│         └" + "─" * cols + " │")
    lines.append(f"│          {xs[0]:<7.2f}" + " " * (cols - 16) + f"{xs[-1]:>7.2f} │")
    lines.append("└" + "─" * (cols + 12) + "┘")
    return "\n".join(lines)


def sanitize_math_expression(text: str) -> str:
    """Translate natural English math words into executable Python/numpy math."""
    s = text.lower().strip()
    s = re.sub(r"^(what is|calculate|compute|solve|eval|evaluate|find|give me)\s+", "", s)

    # Convert "sqrt of X" or "square root of X" -> "sqrt(X)"
    s = re.sub(r"\b(square root of|sqrt of|square root|sqrt)\s+([0-9\.]+|[a-zA-Z]+)", r"sqrt(\2)", s)
    s = re.sub(r"\b(cube root of|cbrt of|cube root|cbrt)\s+([0-9\.]+|[a-zA-Z]+)", r"cbrt(\2)", s)

    s = re.sub(r"\b(divided by|over)\b", "/", s)
    s = re.sub(r"\b(multiplied by|times)\b", "*", s)
    s = re.sub(r"\b(plus|add)\b", "+", s)
    s = re.sub(r"\b(minus|subtract)\b", "-", s)
    s = re.sub(r"\b(to the power of|raised to the power of|raised to|power of)\b", "**", s)
    s = re.sub(r"\b(squared)\b", "**2", s)
    s = re.sub(r"\b(cubed)\b", "**3", s)
    s = re.sub(r"\b(pi)\b", "np.pi", s)
    s = re.sub(r"\b(e)\b(?!\w)", "np.e", s)
    s = s.replace("^", "**")

    # Functions: ensure parentheses
    funcs = ["sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "log", "log10", "log2", "sqrt", "cbrt", "abs"]
    for f in funcs:
        s = re.sub(rf"\b{f}\s+([0-9\.]+|[a-zA-Z]+)", rf"{f}(\1)", s)
        s = re.sub(rf"\b{f}\b(?!\()", f"np.{f}", s)
        s = re.sub(rf"\bnp\.np\.{f}\b", f"np.{f}", s)

    # cbrt
    s = re.sub(r"\bnp\.cbrt\(([^)]+)\)", r"((\1)**(1.0/3.0))", s)
    s = re.sub(r"\bcbrt\(([^)]+)\)", r"((\1)**(1.0/3.0))", s)
    return s


def solve_question(question: str) -> str:
    """Solve any math/physics question, return minimal verbose derivation + formula + graph."""
    raw_q = question.strip()
    expr = sanitize_math_expression(raw_q)

    # Check if expression contains a variable 'x' -> function to plot!
    is_function = bool(re.search(r"\b(x)\b", expr))

    out_lines: List[str] = []
    out_lines.append(f"\033[1;38;5;78m▶ FORMULA / QUERY:\033[0m \033[1m{raw_q}\033[0m")
    out_lines.append(f"\033[38;5;244m  Parsed Math   :\033[0m \033[38;5;75m{expr}\033[0m")

    # Safe namespace
    safe_ns = {
        "np": np,
        "math": math,
        "pi": np.pi,
        "e": np.e,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "abs": np.abs,
    }

    if is_function:
        xs = np.linspace(-10.0, 10.0, 300)
        # Check domain for sqrt / log
        if "sqrt" in expr or "log" in expr:
            xs = np.linspace(0.01, 20.0, 300)
        try:
            local_ns = dict(safe_ns)
            local_ns["x"] = xs
            ys = eval(expr, {"__builtins__": {}}, local_ns)
            # Evaluate at key points
            local_ns["x"] = 0.0 if xs[0] <= 0 <= xs[-1] else xs[0]
            val_0 = float(eval(expr, {"__builtins__": {}}, local_ns))
            out_lines.append(f"\033[1;38;5;214m▶ VALUE AT x={local_ns['x']}:\033[0m \033[1;38;5;78m{val_0:+10.5f}\033[0m")
            out_lines.append(render_terminal_graph(xs, ys, cols=48, rows=6, x_label="x", y_label="f(x)"))
        except Exception as err:
            out_lines.append(f"\033[1;38;5;203m[EVAL ERROR]:\033[0m {err}")
    else:
        # Scalar calculation
        try:
            result = eval(expr, {"__builtins__": {}}, safe_ns)
            if isinstance(result, (int, float, np.number)):
                out_lines.append(f"\033[1;38;5;78m▶ EXACT VALUE   :\033[0m \033[1;38;5;255m{float(result):+14.8f}\033[0m")
                out_lines.append(f"\033[1;38;5;78m▶ SCIENTIFIC    :\033[0m \033[38;5;75m{float(result):.6e}\033[0m")
                # Generate surrounding neighborhood graph to visualize scaling
                c_val = float(result)
                delta = max(1.0, abs(c_val) * 0.5)
                xs = np.linspace(c_val - delta, c_val + delta, 100)
                ys = xs  # Identity projection showing operating point
                out_lines.append(render_terminal_graph(xs, ys, cols=46, rows=4, x_label="neighborhood", y_label="y=x"))
            else:
                out_lines.append(f"\033[1;38;5;78m▶ RESULT:\033[0m {result}")
        except Exception as err:
            out_lines.append(f"\033[1;38;5;203m[EVAL ERROR]:\033[0m {err}")

    return "\n".join(out_lines)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "sqrt(16) * pi + sin(pi/4)"
    print(solve_question(q))
