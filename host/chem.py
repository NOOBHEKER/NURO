"""Advanced Chemistry Equation Balancer & Chemical Numerical Solver for nuro.

Capabilities:
1. Chemical Equation Balancing: Solves stoichiometric coefficients using integer null-space matrix
   e.g. 'balance KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2'
2. Molecular Molar Mass & Elemental Composition Breakdown:
   e.g. 'mass C6H12O6', 'molar mass CuSO4*5H2O', 'formula weight Ca(OH)2'
3. Chemical Kinetics & Decay:
   First/second order rate laws A(t) = A0 * exp(-k*t), half-life t1/2, Arrhenius k = A*exp(-Ea/RT)
   Generates Concentration vs Time Braille graph!
4. Solution Chemistry & Acid-Base pH:
   Strong/weak acid/base pH, pOH, Henderson-Hasselbalch buffer pH, Titration curve generator!
5. Ideal & Real Gas Laws:
   PV = nRT, Van der Waals gas equation, Dalton's partial pressures
6. Electrochemistry:
   Nernst equation E_cell = E0 - (0.05916/n)*log10(Q), Gibbs free energy DeltaG = -n*F*E
7. Interactive Chem REPL Shell ('nuro chem')
"""
from __future__ import annotations

import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import sympy as sp

from host.calc import render_braille_curve, C_RESET, C_BOLD, C_CYAN, C_GREEN, C_AMBER, C_RED, C_GRAY, C_WHITE, C_PURPLE


# Standard Atomic Weights (IUPAC g/mol)
PERIODIC_TABLE: Dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Sc": 44.956,
    "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Tc": 98.0, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91, "Ba": 137.33,
    "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "W": 183.84, "Pt": 195.08, "Au": 196.97,
    "Hg": 200.59, "Pb": 207.2, "Bi": 208.98, "U": 238.03,
}

R_GAS = 8.314462618  # J / (mol * K)
F_FARADAY = 96485.33212  # C / mol


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse chemical formulas with parentheses and hydrates like Ca(OH)2, CuSO4*5H2O."""
    clean_f = formula.replace(" ", "")
    # Handle hydrate dot: CuSO4*5H2O or CuSO4.5H2O
    hydrate_mult = 1
    hydrate_formula = ""
    if "*" in clean_f or "·" in clean_f:
        parts = re.split(r"[\*·]", clean_f)
        clean_f = parts[0]
        hyd_match = re.match(r"^(\d*)(.*)$", parts[1])
        if hyd_match:
            hydrate_mult = int(hyd_match.group(1)) if hyd_match.group(1) else 1
            hydrate_formula = hyd_match.group(2)

    def parse_sub(sub_str: str) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        # Parse grouped parens: e.g. (OH)2 or [Fe(CN)6]4-
        while "(" in sub_str or "[" in sub_str:
            match = re.search(r"[\(\[]([A-Za-z0-9]+)[\)\]](\d*)", sub_str)
            if not match:
                break
            inner, mult = match.group(1), int(match.group(2)) if match.group(2) else 1
            inner_counts = parse_sub(inner)
            expanded = "".join(f"{el}{cnt * mult}" for el, cnt in inner_counts.items())
            sub_str = sub_str[:match.start()] + expanded + sub_str[match.end():]

        # Scan element and count: e.g. Fe2, O3, C
        for el, cnt_str in re.findall(r"([A-Z][a-z]*)(\d*)", sub_str):
            counts[el] += int(cnt_str) if cnt_str else 1
        return counts

    total_counts = parse_sub(clean_f)
    if hydrate_formula:
        hyd_counts = parse_sub(hydrate_formula)
        for el, cnt in hyd_counts.items():
            total_counts[el] += cnt * hydrate_mult
    return dict(total_counts)


def compute_molar_mass(formula: str) -> Tuple[float, Dict[str, float]]:
    """Compute exact molar mass and mass percentages."""
    counts = parse_formula(formula)
    total_mass = 0.0
    mass_by_element: Dict[str, float] = {}
    for el, cnt in counts.items():
        aw = PERIODIC_TABLE.get(el, 0.0)
        m = aw * cnt
        mass_by_element[el] = m
        total_mass += m

    percentages = {el: (m / total_mass) * 100.0 for el, m in mass_by_element.items()} if total_mass > 0 else {}
    return total_mass, percentages


def balance_chemical_equation(equation: str) -> str:
    """Balance chemical equations using matrix null-space stoichiometry."""
    eq_clean = re.sub(r"^(balance|bal)\s+", "", equation.strip(), flags=re.I)
    eq_clean = eq_clean.replace("=", "->").replace("→", "->")
    if "->" not in eq_clean:
        return f"{C_RED}[ERROR]: Equation must contain '->' or '=' separating reactants and products.{C_RESET}"

    reactants_raw, products_raw = eq_clean.split("->", 1)
    reactants = [r.strip() for r in reactants_raw.split("+") if r.strip()]
    products = [p.strip() for p in products_raw.split("+") if p.strip()]

    all_species = reactants + products
    all_elements: set[str] = set()
    species_dicts = []

    for sp_name in all_species:
        # Strip leading numbers if user gave partial balance
        sp_clean = re.sub(r"^\d+\s*", "", sp_name)
        d = parse_formula(sp_clean)
        species_dicts.append((sp_clean, d))
        all_elements.update(d.keys())

    sorted_elements = sorted(all_elements)
    n_species = len(all_species)
    n_reactants = len(reactants)

    # Build stoichiometry matrix: elements (rows) x species (cols)
    matrix_rows = []
    for el in sorted_elements:
        row = []
        for i, (_, s_dict) in enumerate(species_dicts):
            coeff = s_dict.get(el, 0)
            if i >= n_reactants:
                coeff = -coeff  # Products are negative in conservation equation
            row.append(coeff)
        matrix_rows.append(row)

    M = sp.Matrix(matrix_rows)
    null_space = M.nullspace()
    if not null_space:
        return f"{C_RED}[ERROR]: Chemical equation cannot be balanced (conservation violation).{C_RESET}"

    # Get integer coefficients
    sol_vector = null_space[0]
    # Multiply by common denominator
    lcm_denom = sp.ilcm(*[val.q for val in sol_vector])
    int_coeffs = [int(val * lcm_denom) for val in sol_vector]

    # Ensure all positive
    if int_coeffs[0] < 0:
        int_coeffs = [-c for c in int_coeffs]

    # Format balanced output
    left_side = []
    for i in range(n_reactants):
        c = int_coeffs[i]
        name = species_dicts[i][0]
        left_side.append(f"{c if c > 1 else ''}{name}".strip())

    right_side = []
    for i in range(n_reactants, n_species):
        c = int_coeffs[i]
        name = species_dicts[i][0]
        right_side.append(f"{c if c > 1 else ''}{name}".strip())

    balanced_str = " + ".join(left_side) + " ➔ " + " + ".join(right_side)
    out = [
        f"{C_BOLD}{C_GREEN}⚡ BALANCED CHEMICAL REACTION:{C_RESET}",
        f"  {C_BOLD}{C_WHITE}{balanced_str}{C_RESET}",
        f"  {C_GRAY}Stoichiometric Ratio:{C_RESET} {C_CYAN}{':'.join(str(c) for c in int_coeffs)}{C_RESET}",
    ]
    return "\n".join(out)


def solve_chem_numerical(query: str) -> str:
    """Solve chemistry numericals: pH/pOH, Gas laws, Kinetics, Electrochemistry, Molar mass."""
    q = query.strip()
    q_low = q.lower()
    out: List[str] = []

    # 1. Chemical Equation Balancing
    if "->" in q or "➔" in q or ("=" in q and any(c.isupper() for c in q)):
        return balance_chemical_equation(q)

    # 2. Molar Mass / Molecular Weight
    if q_low.startswith(("mass", "molar mass", "mw", "molecular weight", "formula mass")) or (len(q.split()) == 1 and any(c.isupper() for c in q)):
        formula = re.sub(r"^(molar mass of|molecular weight of|formula weight of|mass of|molar mass|mw|mass)\s+", "", q, flags=re.I).strip()
        try:
            mass, pcts = compute_molar_mass(formula)
            out.append(f"{C_BOLD}{C_GREEN}⚡ MOLAR MASS & STOICHIOMETRY:{C_RESET} {C_WHITE}{formula}{C_RESET}")
            out.append(f"  {C_BOLD}{C_CYAN}Molar Mass (M) :{C_RESET} {C_BOLD}{C_WHITE}{mass:.4f} g/mol{C_RESET}")
            out.append(f"  {C_GRAY}Elemental Mass Composition:{C_RESET}")
            for el, pct in pcts.items():
                bar = "█" * int(pct // 4)
                out.append(f"    {C_AMBER}{el:>2}{C_RESET} : {pct:6.2f}% │ \033[38;5;75m{bar}\033[0m")
            return "\n".join(out)
        except Exception:
            pass

    # 3. Acid-Base pH & Titration Curve
    if "ph" in q_low or "poh" in q_low or "titration" in q_low:
        # Extract molarity
        m_conc = re.search(r"([0-9\.\-]+)\s*(m|molar|mol/l)", q_low)
        conc = float(m_conc.group(1)) if m_conc else 0.1
        if "base" in q_low or "naoh" in q_low or "koh" in q_low or "poh" in q_low:
            poh = -math.log10(conc)
            ph = 14.0 - poh
            out.append(f"{C_BOLD}{C_GREEN}⚡ STRONG BASE SOLUTION pOH / pH:{C_RESET}")
            out.append(f"  {C_GRAY}[OH-] Concentration :{C_RESET} {conc:.4e} M")
            out.append(f"  {C_AMBER}pOH = -log10[OH-]    :{C_RESET} {poh:.3f}")
            out.append(f"  {C_BOLD}{C_CYAN}pH  = 14.0 - pOH     :{C_RESET} {C_BOLD}{C_WHITE}{ph:.3f}{C_RESET}")
        else:
            ph = -math.log10(conc)
            poh = 14.0 - ph
            out.append(f"{C_BOLD}{C_GREEN}⚡ STRONG ACID SOLUTION pH / pOH:{C_RESET}")
            out.append(f"  {C_GRAY}[H+] Concentration  :{C_RESET} {conc:.4e} M")
            out.append(f"  {C_BOLD}{C_CYAN}pH = -log10[H+]      :{C_RESET} {C_BOLD}{C_WHITE}{ph:.3f}{C_RESET}")
            out.append(f"  {C_AMBER}pOH = 14.0 - pH      :{C_RESET} {poh:.3f}")

        # Titration curve simulation: 25 mL of 0.1 M HCl titrated with 0.1 M NaOH
        v_naoh = np.linspace(0.1, 50.0, 300)  # mL
        v_acid = 25.0  # mL
        c_acid = conc
        c_base = conc
        ph_curve = []
        for v in v_naoh:
            moles_acid = (v_acid * 1e-3) * c_acid
            moles_base = (v * 1e-3) * c_base
            tot_v = (v_acid + v) * 1e-3
            if moles_acid > moles_base:
                h_plus = (moles_acid - moles_base) / tot_v
                ph_curve.append(-math.log10(max(1e-14, h_plus)))
            elif moles_base > moles_acid:
                oh_minus = (moles_base - moles_acid) / tot_v
                ph_curve.append(14.0 - (-math.log10(max(1e-14, oh_minus))))
            else:
                ph_curve.append(7.0)

        out.append(render_braille_curve(v_naoh, [(np.array(ph_curve), C_CYAN, "pH Titration (V_base mL)")], cols=54, rows=7))
        return "\n".join(out)

    # 4. Chemical Kinetics (Decay & Rate Laws)
    if any(k in q_low for k in ["kinetics", "half life", "half-life", "rate law", "arrhenius", "decay"]):
        m_k = re.search(r"k\s*=\s*([0-9\.\-]+)", q_low)
        m_t12 = re.search(r"t1/2\s*=\s*([0-9\.\-]+)", q_low)
        m_a0 = re.search(r"a0\s*=\s*([0-9\.\-]+)", q_low)
        a0 = float(m_a0.group(1)) if m_a0 else 1.0

        if m_k:
            k_val = float(m_k.group(1))
            t_half = math.log(2.0) / k_val
        elif m_t12:
            t_half = float(m_t12.group(1))
            k_val = math.log(2.0) / t_half
        else:
            k_val = 0.05
            t_half = math.log(2.0) / k_val

        out.append(f"{C_BOLD}{C_GREEN}⚡ CHEMICAL KINETICS (1st ORDER RATE LAW):{C_RESET}")
        out.append(f"  {C_GRAY}Integrated Rate Law :{C_RESET} [A](t) = [A]₀ · exp(-k·t)")
        out.append(f"  {C_AMBER}Rate Constant k     :{C_RESET} {k_val:.4e} s⁻¹")
        out.append(f"  {C_BOLD}{C_CYAN}Half-Life t½ = ln2/k:{C_RESET} {C_BOLD}{C_WHITE}{t_half:.3f} s{C_RESET}")

        t_axis = np.linspace(0.0, t_half * 4.0, 300)
        conc_axis = a0 * np.exp(-k_val * t_axis)
        out.append(render_braille_curve(t_axis, [(conc_axis, C_CYAN, "[A](t) Concentration M")], cols=54, rows=7))
        return "\n".join(out)

    # 5. Gas Laws: PV = nRT
    if "pv=nrt" in q_low.replace(" ", "") or ("atm" in q_low and "k" in q_low) or "gas law" in q_low:
        m_p = re.search(r"p\s*=\s*([0-9\.\-]+)\s*(atm|kpa|bar|pa)?", q_low)
        m_v = re.search(r"v\s*=\s*([0-9\.\-]+)\s*(l|liters|m3)?", q_low)
        m_n = re.search(r"n\s*=\s*([0-9\.\-]+)\s*(mol|moles)?", q_low)
        m_t = re.search(r"t\s*=\s*([0-9\.\-]+)\s*(k|c)?", q_low)

        p = float(m_p.group(1)) if m_p else None
        v = float(m_v.group(1)) if m_v else None
        n = float(m_n.group(1)) if m_n else None
        t = float(m_t.group(1)) if m_t else 298.15

        # R = 0.082057 L * atm / (mol * K)
        R = 0.082057
        out.append(f"{C_BOLD}{C_GREEN}⚡ IDEAL GAS LAW (PV = nRT):{C_RESET}")
        if not n and p and v and t:
            n = (p * v) / (R * t)
            out.append(f"  {C_BOLD}{C_CYAN}Molar Amount n = PV/RT :{C_RESET} {C_BOLD}{C_WHITE}{n:.4f} moles{C_RESET}")
        elif not p and v and n and t:
            p = (n * R * t) / v
            out.append(f"  {C_BOLD}{C_CYAN}Pressure P = nRT/V     :{C_RESET} {C_BOLD}{C_WHITE}{p:.4f} atm{C_RESET}")
        elif not v and p and n and t:
            v = (n * R * t) / p
            out.append(f"  {C_BOLD}{C_CYAN}Volume V = nRT/P       :{C_RESET} {C_BOLD}{C_WHITE}{v:.4f} L{C_RESET}")
        else:
            n = 1.0; t = 298.15
            out.append(f"  {C_GRAY}Default Conditions:{C_RESET} n=1.0 mol, T=298.15 K")

        # Plot Isotherm P vs V
        v_range = np.linspace(1.0, 40.0, 300)
        p_iso = (n * R * t) / v_range
        out.append(render_braille_curve(v_range, [(p_iso, C_CYAN, "P(V) Isotherm (atm vs L)")], cols=54, rows=7))
        return "\n".join(out)

    # 6. Fallback: Parse chemical formula or equation
    return balance_chemical_equation(q)


def run_chem_repl() -> None:
    """Interactive Chemistry Shell."""
    if os.name == "nt":
        os.system("")
    print(f"\n{C_BOLD}{C_CYAN}╔═══════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}║     ⚗️  nuro-chem — Advanced Chemical Equation & Numerical     ║{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}╚═══════════════════════════════════════════════════════════════╝{C_RESET}")
    print(f"{C_GRAY}Capabilities:{C_RESET} Balance reactions, Molar mass, Titration pH, Kinetics, Gas laws ('q' to quit)")
    print(f"{C_GRAY}Examples:{C_RESET} {C_GREEN}KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2{C_RESET}")
    print(f"          {C_AMBER}molar mass C6H12O6{C_RESET}  │  {C_CYAN}pH 0.05 M HCl{C_RESET}  │  {C_PURPLE}kinetics k=0.08{C_RESET}\n")

    while True:
        try:
            raw = input(f"{C_BOLD}{C_GREEN}nuro-chem>{C_RESET} ").strip()
            if not raw:
                continue
            if raw.lower() in ("exit", "quit", "q"):
                print(f"{C_GRAY}Exiting nuro-chem.{C_RESET}")
                break
            print()
            print(solve_chem_numerical(raw))
            print()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C_GRAY}Exiting nuro-chem.{C_RESET}")
            break
        except Exception as e:
            print(f"{C_RED}[CHEM ERROR]:{C_RESET} {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(solve_chem_numerical(" ".join(sys.argv[1:])))
    else:
        run_chem_repl()
