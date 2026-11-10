"""Whole Scientific Graphing Calculator & Calculus Engine for nuro.

Features:
- Symbolic & numerical math (SymPy + NumPy + SciPy)
- Natural language words ('sqrt', 'cube root', 'pi', 'times', 'integral', 'derivative', 'solve')
- Automatic Calculus: Exact derivatives f'(x), indefinite/definite integrals, roots, extrema
- High-res Unicode Braille 2D graph with zero-axis reticle
- Multi-function overlay plotting (e.g., 'sin(x), cos(x)')
- Interactive REPL shell (type 'nuro calc') or single-command CLI (e.g. 'nuro calc "x^3 - 3*x"')
"""
from __future__ import annotations

import math
import os
import re
import sys
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import sympy as sp


BLOCKS = " ▂▃▄▅▆▇█"
BRAILLE_MAP = [
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
]

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[38;5;75m"
C_GREEN = "\033[38;5;78m"
C_AMBER = "\033[38;5;214m"
C_RED = "\033[38;5;203m"
C_GRAY = "\033[38;5;244m"
C_WHITE = "\033[38;5;255m"
C_PURPLE = "\033[38;5;141m"


def render_braille_curve(
    xs: np.ndarray,
    curves: List[Tuple[np.ndarray, str, str]],  # (y_values, color_code, label)
    cols: int = 56,
    rows: int = 8,
) -> str:
    """Render a multi-curve high-resolution Unicode Braille 2D plot with labeled axes and 0-axis."""
    px_w = cols * 2
    px_h = rows * 4

    all_ys: List[float] = []
    for y_arr, _, _ in curves:
        y_clean = np.nan_to_num(y_arr, nan=0.0, posinf=1e5, neginf=-1e5)
        all_ys.extend(y_clean.tolist())

    if not all_ys:
        return ""

    y_min, y_max = float(np.percentile(all_ys, 1)), float(np.percentile(all_ys, 99))
    if abs(y_max - y_min) < 1e-9:
        y_max += 1.0
        y_min -= 1.0

    # Expand 8% margin
    margin = (y_max - y_min) * 0.08
    y_min -= margin
    y_max += margin

    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    # Draw zero axis in braille if in range
    if y_min <= 0.0 <= y_max:
        z_norm = float((0.0 - y_min) / (y_max - y_min))
        z_px = int((px_h - 1) - np.floor(z_norm * (px_h - 1)))
        for x in range(0, px_w, 2):
            grid[z_px // 4][x // 2] |= BRAILLE_MAP[z_px % 4][x % 2]

    # Plot each curve
    x_sub = np.linspace(xs[0], xs[-1], px_w)
    for y_arr, _, _ in curves:
        y_clean = np.nan_to_num(y_arr, nan=0.0, posinf=y_max, neginf=y_min)
        y_interp = np.interp(x_sub, xs, y_clean)
        norm = np.clip((y_interp - y_min) / (y_max - y_min), 0.0, 0.999)
        y_px = (px_h - 1) - np.floor(norm * (px_h - 1)).astype(int)

        prev_y = int(y_px[0])
        for x in range(px_w):
            cur_y = int(y_px[x])
            y0, y1 = min(prev_y, cur_y), max(prev_y, cur_y)
            for yy in range(y0, y1 + 1):
                grid[yy // 4][x // 2] |= BRAILLE_MAP[yy % 4][x % 2]
            prev_y = cur_y

    out_lines: List[str] = []
    # Header legend
    legend_items = "  ".join(f"{c}■ {lbl}{C_RESET}" for _, c, lbl in curves)
    out_lines.append(f"╭─[ GRAPH: {legend_items} ]" + "─" * max(2, cols - 18) + "╮")

    for r in range(rows):
        chars = "".join(chr(0x2800 + grid[r][c]) for c in range(cols))
        cur_val = y_max - (r / (rows - 1)) * (y_max - y_min)
        tag = f"{cur_val:+8.2f} ┤" if (r in (0, rows // 2, rows - 1)) else "         ┤"
        out_lines.append(f"│ {C_GRAY}{tag}{C_RESET} \033[38;5;75m{chars}\033[0m │")

    out_lines.append(f"│          └" + "─" * cols + " │")
    out_lines.append(f"│           {C_GRAY}{xs[0]:<8.2f}" + " " * max(1, cols - 19) + f"{xs[-1]:>8.2f}{C_RESET} │")
    out_lines.append("╰" + "─" * (cols + 14) + "╯")
    return "\n".join(out_lines)


def natural_to_sympy_str(text: str) -> str:
    """Convert natural words ('sqrt of x', 'divided by', 'squared', etc.) to SymPy syntax."""
    s = text.strip()
    # Strip question starters
    s = re.sub(r"^(what is|calculate|compute|solve|eval|evaluate|graph|plot|find)\s+", "", s, flags=re.I)

    # Single-key operator shorthands: m -> *, a -> +, s -> -, d -> /
    s = re.sub(r"([0-9\)])\s*m\s*([0-9\(])", r"\1 * \2", s, flags=re.I)
    s = re.sub(r"([0-9\)])\s*a\s*([0-9\(])", r"\1 + \2", s, flags=re.I)
    s = re.sub(r"([0-9\)])\s*s\s*([0-9\(])", r"\1 - \2", s, flags=re.I)
    s = re.sub(r"([0-9\)])\s*d\s*([0-9\(])", r"\1 / \2", s, flags=re.I)
    s = re.sub(r"\b([0-9\.]+)\s+m\s+([0-9\.]+)\b", r"\1 * \2", s, flags=re.I)
    s = re.sub(r"\b([0-9\.]+)\s+a\s+([0-9\.]+)\b", r"\1 + \2", s, flags=re.I)
    s = re.sub(r"\b([0-9\.]+)\s+s\s+([0-9\.]+)\b", r"\1 - \2", s, flags=re.I)
    s = re.sub(r"\b([0-9\.]+)\s+d\s+([0-9\.]+)\b", r"\1 / \2", s, flags=re.I)

    # Clean words
    replacements = [
        (r"\b(square root of|sqrt of|square root)\b", "sqrt"),
        (r"\b(cube root of|cbrt of|cube root)\b", "cbrt"),
        (r"\b(divided by|over)\b", "/"),
        (r"\b(multiplied by|times)\b", "*"),
        (r"\b(plus|add)\b", "+"),
        (r"\b(minus|subtract)\b", "-"),
        (r"\b(to the power of|raised to the power of|raised to|power of)\b", "**"),
        (r"\b(squared)\b", "**2"),
        (r"\b(cubed)\b", "**3"),
        (r"\b(pi)\b", "pi"),
        (r"\b(infinity|inf)\b", "oo"),
        (r"\^", "**"),
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s, flags=re.I)

    # Replace function words without parentheses: e.g. "sqrt 25" -> "sqrt(25)"
    funcs = ["sqrt", "cbrt", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log", "ln", "abs"]
    for f in funcs:
        s = re.sub(rf"\b{f}\s+([0-9\.]+|[a-zA-Z]+)", rf"{f}(\1)", s, flags=re.I)
        if f == "ln":
            s = re.sub(r"\bln\(", "log(", s, flags=re.I)
        if f == "cbrt":
            s = re.sub(r"\bcbrt\(([^)]+)\)", r"((\1)**(1/3))", s, flags=re.I)

    # Implicit multiplication expansion:
    # 1. Number adjacent to variable: 5x -> 5*x, 3x^2 -> 3*x^2
    s = re.sub(r"(\d+)\s*([a-df-zA-DF-Z])", r"\1*\2", s)
    s = re.sub(r"(\d+)\s*(e)(?![+-]?\d)", r"\1*\2", s)

    # 2. Parens adjacent to parens, numbers, or variables: (x-1)(x+2) -> (x-1)*(x+2), 2(x+1) -> 2*(x+1)
    s = re.sub(r"\)\s*\(", ")*(", s)
    s = re.sub(r"(\d+)\s*\(", r"\1*(", s)
    s = re.sub(r"\)\s*([a-zA-Z0-9])", r")*\1", s)
    s = re.sub(r"\b([xXyYzZ])\s*\(", r"\1*(", s)

    # 3. Closing paren adjacent to letter or digit: (x+1)x -> (x+1)*x, (x+1)5 -> (x+1)*5
    s = re.sub(r"(\))\s*([a-zA-Z0-9])", r"\1*\2", s)

    return s


def calculate_and_graph(query: str, domain: Optional[Tuple[float, float]] = None) -> str:
    """Primary Scientific Graphing Calculator engine."""
    q_raw = query.strip()
    if not q_raw:
        return ""

    from host.obsidian_sync import extract_obsidian_directive, record_session_calculation, launch_obsidian_vault
    q_clean, target_vault = extract_obsidian_directive(q_raw)
    q_clean_low = q_clean.lower()

    # Session switching inside calc
    if q_clean_low.startswith("session"):
        from host.cli import handle_session_command
        return handle_session_command(q_clean.split()[1:])

    if q_clean_low in ("obsidian", "graph", "vault", "sync"):
        return launch_obsidian_vault(target_vault)

    # 0. Direct AI / LLM Query trigger: e.g. "ai explain quadratic formula" or "ask ..."
    if q_clean_low.startswith(("ai ", "ask ", "llm ", "explain ")):
        prompt_text = re.sub(r"^(ai|ask|llm|explain)\s+", "", q_clean, flags=re.I).strip()
        from host.ai import query_gemini
        return query_gemini(prompt_text)

    # Auto-normalize "solve quadratic ...", "quadratic ..." to an equation
    if "quadratic" in q_clean_low and "=" not in q_clean:
        q_clean = re.sub(r"^(solve quadratic equation|solve quadratic|quadratic equation|quadratic)\s+", "", q_clean, flags=re.I).strip()
        if "=" not in q_clean:
            q_clean = f"{q_clean} = 0"

    out: List[str] = []
    x = sp.Symbol("x")

    # 1. Check for Equation Solving: "x^2 - 5x + 6 = 0" or "solve 2x + 4 = 10"
    if "=" in q_clean:
        try:
            left_str, right_str = q_clean.split("=", 1)
            sym_left = sp.sympify(natural_to_sympy_str(left_str))
            sym_right = sp.sympify(natural_to_sympy_str(right_str))
            eq = sp.Eq(sym_left, sym_right)
            syms = list(eq.free_symbols)
            var_x = syms[0] if syms else x
            f_diff = sym_left - sym_right
            f_lamb = sp.lambdify(var_x, f_diff, modules=["numpy"])

            # Check if this is a Quadratic Equation (degree 2 polynomial)
            try:
                poly = sp.Poly(f_diff, var_x)
                if poly.degree() == 2:
                    coeffs = poly.all_coeffs()
                    a_val, b_val, c_val = coeffs[0], coeffs[1], coeffs[2]
                    disc = b_val**2 - 4 * a_val * c_val
                    out.append(f"{C_BOLD}{C_GREEN}⚡ QUADRATIC EQUATION SOLVER:{C_RESET} {C_WHITE}{a_val}·{var_x}² + ({b_val})·{var_x} + ({c_val}) = 0{C_RESET}")
                    out.append(f"  {C_GRAY}Coefficients          :{C_RESET} a = {a_val}, b = {b_val}, c = {c_val}")
                    out.append(f"  {C_AMBER}Discriminant Δ=b²-4ac  :{C_RESET} {C_BOLD}{C_CYAN}{disc}{C_RESET} ({float(disc.evalf()):.4f})")
                    if disc > 0:
                        nature = "Two Distinct Real Roots"
                    elif disc == 0:
                        nature = "One Repeated Real Root (Double Root)"
                    else:
                        nature = "Two Complex Conjugate Roots"
                    out.append(f"  {C_GRAY}Nature of Roots       :{C_RESET} {nature}")
                    out.append(f"  {C_GRAY}Quadratic Formula     :{C_RESET} {var_x} = (-b ± √Δ) / 2a")

                    roots = sp.solve(eq, var_x)
                    r_strs = [f"{r} ({float(r.evalf()):.4f})" if getattr(r, 'is_real', False) else str(r) for r in roots]
                    out.append(f"  {C_BOLD}{C_GREEN}▶ SOLUTIONS FOR {var_x}     :{C_RESET} {C_BOLD}{C_WHITE}{', '.join(r_strs)}{C_RESET}")

                    h_v = -b_val / (2 * a_val)
                    k_v = c_val - (b_val**2) / (4 * a_val)
                    out.append(f"  {C_PURPLE}Parabola Vertex (h,k) :{C_RESET} ({float(h_v.evalf()):.3f}, {float(k_v.evalf()):.3f})  │  Axis: {var_x} = {float(h_v.evalf()):.3f}")

                    h_f = float(h_v.evalf())
                    xs = np.linspace(h_f - 6.0, h_f + 6.0, 300)
                    ys = f_lamb(xs)
                    out.append(render_braille_curve(xs, [(ys, C_CYAN, f"y = {f_diff}")]))
                    res = "\n".join(out)
                    from host.session_manager import record_calc_in_session
                    record_calc_in_session("Math", q_clean, res)
                    if target_vault:
                        note_p = record_session_calculation("Math", f"Quadratic - {q_clean[:30]}", q_clean, res, ["math", "quadratic"], ["Quadratic Equation", "Algebra"], vault_name=target_vault)
                        res += f"\n\n{C_BOLD}{C_GREEN}💾 SAVED TO OBSIDIAN VAULT [{target_vault}]:{C_RESET}\n  {C_GRAY}Session Note :{C_RESET} {C_CYAN}{note_p.name}{C_RESET}"
                    return res
            except Exception:
                pass

            sol = sp.solve(eq, var_x)
            out.append(f"{C_BOLD}{C_GREEN}⚡ EQUATION SOLVER:{C_RESET} {C_WHITE}{eq}{C_RESET}")
            out.append(f"{C_AMBER}▶ SOLUTIONS FOR {var_x}:{C_RESET} {C_BOLD}{C_CYAN}{sol}{C_RESET}")

            xs = np.linspace(-10.0, 10.0, 300)
            ys = f_lamb(xs)
            out.append(render_braille_curve(xs, [(ys, C_CYAN, f"{f_diff} = 0")]))
            res = "\n".join(out)
            from host.session_manager import record_calc_in_session
            record_calc_in_session("Math", q_clean, res)
            if target_vault:
                note_p = record_session_calculation("Math", f"Equation - {q_clean[:30]}", q_clean, res, ["math", "equations"], ["Algebra"], vault_name=target_vault)
                res += f"\n\n{C_BOLD}{C_GREEN}💾 SAVED TO OBSIDIAN VAULT [{target_vault}]:{C_RESET}\n  {C_GRAY}Session Note :{C_RESET} {C_CYAN}{note_p.name}{C_RESET}"
            return res
        except Exception:
            # Fallback to LLM on parsing failure
            from host.ai import query_gemini
            return query_gemini(f"Solve equation: {q_clean}")

    # 2. Check for Derivative query: "derivative of ..." / "diff ..."
    if q_clean.lower().startswith(("diff", "derivative")):
        inner = re.sub(r"^(diff|derivative of|derivative)\s+", "", q_clean, flags=re.I)
        sym_expr = sp.sympify(natural_to_sympy_str(inner))
        d_expr = sp.diff(sym_expr, x)

        out.append(f"{C_BOLD}{C_GREEN}⚡ CALCULUS // DERIVATIVE:{C_RESET}")
        out.append(f"  {C_GRAY}f(x)  ={C_RESET} {C_WHITE}{sym_expr}{C_RESET}")
        out.append(f"  {C_AMBER}f'(x) ={C_RESET} {C_BOLD}{C_CYAN}{d_expr}{C_RESET}")

        xs = np.linspace(-6.0, 6.0, 300)
        f_l = sp.lambdify(x, sym_expr, modules=["numpy"])
        d_l = sp.lambdify(x, d_expr, modules=["numpy"])
        ys_f = np.array(f_l(xs), dtype=np.float64)
        ys_d = np.array(d_l(xs), dtype=np.float64)
        out.append(render_braille_curve(xs, [(ys_f, C_CYAN, "f(x)"), (ys_d, C_AMBER, "f'(x)")]))
        return "\n".join(out)

    # 3. Check for Integral query: "integral of ... from a to b" / "integrate ..."
    if q_clean.lower().startswith(("integral", "integrate")):
        m_bounds = re.search(r"from\s+([0-9\.\-]+)\s+to\s+([0-9\.\-]+)", q_clean, flags=re.I)
        inner = re.sub(r"from\s+.*", "", q_clean, flags=re.I)
        inner = re.sub(r"^(integral of|integrate|integral)\s+", "", inner, flags=re.I)
        sym_expr = sp.sympify(natural_to_sympy_str(inner))

        if m_bounds:
            a_val, b_val = float(m_bounds.group(1)), float(m_bounds.group(2))
            exact_val = sp.integrate(sym_expr, (x, a_val, b_val))
            out.append(f"{C_BOLD}{C_GREEN}⚡ DEFINITE INTEGRAL [a={a_val}, b={b_val}]:{C_RESET}")
            out.append(f"  {C_GRAY}∫ f(x) dx ={C_RESET} {C_BOLD}{C_CYAN}{exact_val}{C_RESET}  ({float(exact_val):.6f})")
            xs = np.linspace(min(a_val, -5.0), max(b_val, 5.0), 300)
        else:
            indef_val = sp.integrate(sym_expr, x)
            out.append(f"{C_BOLD}{C_GREEN}⚡ INDEFINITE INTEGRAL:{C_RESET}")
            out.append(f"  {C_GRAY}∫ f(x) dx ={C_RESET} {C_BOLD}{C_CYAN}{indef_val} + C{C_RESET}")
            xs = np.linspace(-6.0, 6.0, 300)

        f_l = sp.lambdify(x, sym_expr, modules=["numpy"])
        ys = np.array(f_l(xs), dtype=np.float64)
        out.append(render_braille_curve(xs, [(ys, C_CYAN, f"f(x) = {sym_expr}")]))
        return "\n".join(out)

    # 4. Multi-Function Plotting: "sin(x), cos(x)"
    if "," in q_clean and "x" in q_clean:
        parts = [p.strip() for p in q_clean.split(",")]
        curves: List[Tuple[np.ndarray, str, str]] = []
        colors = [C_CYAN, C_AMBER, C_GREEN, C_PURPLE]
        xs = np.linspace(-8.0, 8.0, 300)

        out.append(f"{C_BOLD}{C_GREEN}⚡ MULTI-FUNCTION PLOT:{C_RESET}")
        for idx, part in enumerate(parts):
            s_expr = sp.sympify(natural_to_sympy_str(part))
            fn = sp.lambdify(x, s_expr, modules=["numpy"])
            y_vals = np.array(fn(xs), dtype=np.float64)
            curves.append((y_vals, colors[idx % len(colors)], str(s_expr)))
            out.append(f"  {colors[idx % len(colors)]}Curve {idx+1}:{C_RESET} {s_expr}")

        out.append(render_braille_curve(xs, curves))
        return "\n".join(out)

    # 5. Standard Expression (Function f(x) or Scalar Calculation)
    parsed_s = natural_to_sympy_str(q_clean)
    sym_expr = sp.sympify(parsed_s)

    sym_names = [s.name for s in sym_expr.free_symbols]
    if "x" in sym_names:
        # It's a function of x!
        x = sp.Symbol("x")
        d_expr = sp.diff(sym_expr, x)
        integ_expr = sp.integrate(sym_expr, x)

        out.append(f"{C_BOLD}{C_GREEN}⚡ FUNCTION CALCULATOR:{C_RESET} {C_WHITE}f(x) = {sym_expr}{C_RESET}")
        out.append(f"  {C_GRAY}Derivative f'(x)     :{C_RESET} {C_CYAN}{d_expr}{C_RESET}")
        out.append(f"  {C_GRAY}Integral   ∫ f(x) dx :{C_RESET} {C_PURPLE}{integ_expr} + C{C_RESET}")

        # Find roots f(x) = 0
        try:
            roots = sp.solve(sym_expr, x)
            real_roots = [r for r in roots if sp.re(r) == r or r.is_real]
            if real_roots:
                r_str = ", ".join(f"{float(r):.4f}" for r in real_roots[:4])
                out.append(f"  {C_GRAY}Roots (f=0)      :{C_RESET} {C_AMBER}{r_str}{C_RESET}")
        except Exception:
            pass

        # Domain selection
        xs = np.linspace(domain[0] if domain else -4.0, domain[1] if domain else 4.0, 320)
        if "sqrt" in str(sym_expr) or "log" in str(sym_expr):
            xs = np.linspace(0.01, 12.0, 320)

        f_l = sp.lambdify(x, sym_expr, modules=["numpy"])
        d_l = sp.lambdify(x, d_expr, modules=["numpy"])
        ys_f = np.array(f_l(xs), dtype=np.float64)
        ys_d = np.array(d_l(xs), dtype=np.float64)

        out.append(render_braille_curve(xs, [(ys_f, C_CYAN, f"f(x) = {sym_expr}"), (ys_d, C_AMBER, "f'(x)")], cols=56, rows=8))
    else:
        # Pure Numerical / Symbolic Scalar calculation!
        val_eval = sym_expr.evalf()
        out.append(f"{C_BOLD}{C_GREEN}⚡ SCIENTIFIC CALCULATION:{C_RESET} {C_WHITE}{q_clean}{C_RESET}")
        out.append(f"  {C_GRAY}Symbolic Form :{C_RESET} {C_CYAN}{sym_expr}{C_RESET}")
        out.append(f"  {C_BOLD}{C_GREEN}Exact Decimal :{C_RESET} {C_WHITE}{float(val_eval):+16.10f}{C_RESET}")
        out.append(f"  {C_GRAY}Scientific    :{C_RESET} {C_AMBER}{float(val_eval):.8e}{C_RESET}")

        # Quick neighborhood visualization
        c_val = float(val_eval)
        d_val = max(1.0, abs(c_val) * 0.5)
        xs = np.linspace(c_val - d_val, c_val + d_val, 100)
        out.append(render_braille_curve(xs, [(xs, C_CYAN, f"value={c_val:.4f}")], cols=48, rows=4))

    result_str = "\n".join(out)

    # Record in active session history
    from host.session_manager import record_calc_in_session, get_active_session
    record_calc_in_session("Math", q_clean, result_str)
    active_sess = get_active_session()

    # ONLY write to Obsidian if explicitly requested with vault name!
    if target_vault:
        note_p = record_session_calculation(
            domain="Math",
            title=f"Math - {q_clean[:30]}",
            query=q_clean,
            result_text=result_str,
            tags=["math", "calculus", "scientific_calculator"],
            related_links=["Calculus", "Functions", "Equations"],
            vault_name=target_vault,
        )
        conf = f"\n\n{C_BOLD}{C_GREEN}💾 SAVED TO OBSIDIAN VAULT [{target_vault}]:{C_RESET}\n  {C_GRAY}Session Note :{C_RESET} {C_CYAN}{note_p.name}{C_RESET}\n  {C_GRAY}Session Link :{C_RESET} [[Session - {active_sess}]]"
        return result_str + conf

    return result_str


def calc_get_line(prompt: str, shortcuts_active: bool = True) -> Tuple[str, bool]:
    """Read a line with live single-key operator mapping:
    - Pressing 'm' -> prints '*' and buffers '*'
    - Pressing 'a' -> prints '+' and buffers '+'
    - Pressing 's' -> prints '-' and buffers '-'
    - Pressing 'd' -> prints '/' and buffers '/'
    - Tab or Ctrl+T toggles between [Shortcuts (m=* a=+ s=- d=/)] and [Alpha Mode]
    """
    if os.name != "nt":
        try:
            return input(prompt), shortcuts_active
        except Exception:
            return "", shortcuts_active

    import msvcrt
    sys.stdout.write(prompt)
    sys.stdout.flush()
    buf: List[str] = []
    mode = shortcuts_active

    while True:
        try:
            ch = msvcrt.getwch()
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt

        if ch == "\r":  # Enter
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "".join(buf), mode
        elif ch == "\x03":  # Ctrl+C
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise KeyboardInterrupt
        elif ch in ("\t", "\x14"):  # Tab or Ctrl+T -> Toggle Mode
            mode = not mode
            # Brief visual feedback on right, then reset
            tag = f" {C_AMBER}[m=* a=+ s=- d=/]{C_RESET}" if mode else f" {C_GRAY}[ALPHA]{C_RESET}"
            sys.stdout.write("\r\033[K" + prompt + "".join(buf) + tag)
            sys.stdout.flush()
            sys.stdout.write("\r\033[K" + prompt + "".join(buf))
            sys.stdout.flush()
        elif ch == "\x08":  # Backspace
            if buf:
                buf.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
        elif ch in ("\x00", "\xe0"):
            msvcrt.getwch()  # skip arrow/function extension code
        else:
            if mode:
                if ch in ("m", "M"):
                    ch = "*"
                elif ch in ("a", "A"):
                    ch = "+"
                elif ch in ("s", "S"):
                    ch = "-"
                elif ch in ("d", "D"):
                    ch = "/"
            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()


def run_calc_repl() -> None:
    """Interactive Scientific Graphing Calculator Shell."""
    if os.name == "nt":
        os.system("")
    print(f"\n{C_BOLD}{C_CYAN}╔═══════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}║     ⚡ nuro-calc — Interactive Scientific Graphing Shell      ║{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}╚═══════════════════════════════════════════════════════════════╝{C_RESET}")
    print(f"{C_GRAY}Shortcuts Active:{C_RESET} {C_GREEN}m = *{C_RESET} │ {C_CYAN}a = +{C_RESET} │ {C_AMBER}s = -{C_RESET} │ {C_PURPLE}d = /{C_RESET}  ({C_WHITE}Tab / Ctrl+T{C_RESET} toggles Alpha mode)")
    print(f"{C_GRAY}Commands:{C_RESET} Type formula, 'obsidian' (open graph view), or 'q' to quit")
    print(f"{C_GRAY}Examples:{C_RESET} {C_AMBER}5m3a10{C_RESET} (5*3+10) │ {C_CYAN}sin(x)a2{C_RESET} │ {C_PURPLE}x^2 s 4 = 0{C_RESET} │ {C_GREEN}obsidian{C_RESET}\n")

    shortcuts_mode = True
    while True:
        try:
            raw, shortcuts_mode = calc_get_line(f"{C_BOLD}{C_GREEN}nuro-calc>{C_RESET} ", shortcuts_active=shortcuts_mode)
            raw = raw.strip()
            if not raw:
                continue
            if raw.lower() in ("exit", "quit", "q"):
                print(f"{C_GRAY}Exiting nuro-calc.{C_RESET}")
                break
            print()
            print(calculate_and_graph(raw))
            print()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C_GRAY}Exiting nuro-calc.{C_RESET}")
            break
        except Exception as e:
            print(f"{C_RED}[CALC ERROR]:{C_RESET} {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_input = " ".join(sys.argv[1:])
        print(calculate_and_graph(query_input))
    else:
        run_calc_repl()
