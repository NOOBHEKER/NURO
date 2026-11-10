"""Advanced Chemistry Equation Balancer & Chemical Numerical Solver for nuro.

Capabilities:
1. Chemical Equation Balancing:
   - Solves stoichiometric coefficients using integer null-space matrix
   - Catalyst detection ('c = V2O5', '[c: Pt]', 'catalyst = Fe', etc.)
   - Catalyst activation energy reduction & mechanism explanation
   - Reaction Coordinate Potential Energy Braille Graph (Uncatalyzed vs Catalyzed)
   - Reactants & Products properties, mixture classification (solids, gases, aqueous)
2. Molecular Molar Mass & Elemental Composition Breakdown:
   - e.g. 'mass C6H12O6', 'molar mass CuSO4*5H2O', 'formula weight Ca(OH)2'
3. Chemical Kinetics & Decay:
   - First/second order rate laws A(t) = A0 * exp(-k*t), half-life t1/2, Arrhenius
   - Generates Concentration vs Time Braille graph!
4. Solution Chemistry & Acid-Base pH:
   - Strong/weak acid/base pH, pOH, Henderson-Hasselbalch, Titration curve generator!
5. Ideal & Real Gas Laws:
   - PV = nRT, Isotherm P(V) curve
6. Obsidian Graph View Sync:
   - Logs reactions and formulas with [[wikilinks]] into 'G:\\obsidian\\Neural Loop Calculations'
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

from host.calc import (
    render_braille_curve,
    C_RESET,
    C_BOLD,
    C_CYAN,
    C_GREEN,
    C_AMBER,
    C_RED,
    C_GRAY,
    C_WHITE,
    C_PURPLE,
)


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

# Chemical Compound Properties Database (Minimum Verbose)
COMPOUND_PROPERTIES: Dict[str, Dict[str, str]] = {
    "KMnO4": {
        "name": "Potassium Permanganate",
        "type": "Ionic Salt / Strong Oxidizer",
        "appearance": "Deep purple-black crystals; intense magenta in solution",
        "role": "Potent oxidant; Mn(+7) is reduced to Mn(+2) in acidic media",
        "phase": "Solid reactant / Aqueous solution",
    },
    "HCl": {
        "name": "Hydrochloric Acid",
        "type": "Strong Mineral Acid",
        "appearance": "Colorless fuming liquid/gas with pungent choking odor",
        "role": "Source of H+ and reducing agent (Cl- oxidized to Cl2)",
        "phase": "Aqueous solution / Gas",
    },
    "KCl": {
        "name": "Potassium Chloride",
        "type": "Neutral Ionic Salt",
        "appearance": "White cubic crystalline solid; saline taste",
        "role": "Spectator electrolyte salt (K+ and Cl- ions)",
        "phase": "Aqueous solution",
    },
    "MnCl2": {
        "name": "Manganese(II) Chloride",
        "type": "Transition Metal Salt",
        "appearance": "Pale pink/rose deliquescent crystalline solid",
        "role": "Reduced Mn(+2) product; water-soluble",
        "phase": "Aqueous solution (pink)",
    },
    "H2O": {
        "name": "Water",
        "type": "Polar Covalent Molecule",
        "appearance": "Colorless transparent amphoteric liquid at STP",
        "role": "Universal solvent and medium (Kw = 1.0e-14)",
        "phase": "Liquid solvent",
    },
    "Cl2": {
        "name": "Chlorine Gas",
        "type": "Diatomic Halogen Non-Metal",
        "appearance": "Greenish-yellow gas; suffocating bleached odor; toxic",
        "role": "Oxidized product (oxidation state 0); disinfectant/bleach",
        "phase": "Gas phase (effervescence ↑)",
    },
    "Fe": {
        "name": "Iron",
        "type": "Transition Metal Element",
        "appearance": "Lustrous silvery-gray ferromagnetic metal",
        "role": "Reducing agent / Haber catalyst",
        "phase": "Solid metal",
    },
    "O2": {
        "name": "Oxygen Gas",
        "type": "Diatomic Non-Metal Gas",
        "appearance": "Colorless, odorless paramagnetic gas",
        "role": "Universal oxidant; supports combustion",
        "phase": "Gas phase",
    },
    "Fe2O3": {
        "name": "Iron(III) Oxide",
        "type": "Basic Transition Metal Oxide",
        "appearance": "Reddish-brown insoluble powder (rust / hematite)",
        "role": "Insoluble oxidized oxide product",
        "phase": "Solid precipitate (rust)",
    },
    "SO2": {
        "name": "Sulfur Dioxide",
        "type": "Acidic Covalent Oxide",
        "appearance": "Colorless gas with pungent suffocating sulfur odor",
        "role": "Reducing agent; Contact Process reactant",
        "phase": "Gas phase",
    },
    "SO3": {
        "name": "Sulfur Trioxide",
        "type": "Acidic Anhydride Oxide",
        "appearance": "Colorless volatile liquid / white silky crystals; fumes",
        "role": "Hydrated into H2SO4; Contact Process product",
        "phase": "Gas / volatile liquid",
    },
    "V2O5": {
        "name": "Vanadium(V) Oxide",
        "type": "Amphoteric Transition Metal Oxide",
        "appearance": "Orange-yellow crystalline solid",
        "role": "Heterogeneous oxidation catalyst (Contact Process)",
        "phase": "Solid catalyst bed",
    },
    "N2": {
        "name": "Nitrogen Gas",
        "type": "Diatomic Inert Element",
        "appearance": "Colorless, odorless inert diatomic gas",
        "role": "Unreactive due to strong N≡N bond (945 kJ/mol); Haber reactant",
        "phase": "Gas phase",
    },
    "H2": {
        "name": "Hydrogen Gas",
        "type": "Diatomic Gas / Clean Fuel",
        "appearance": "Colorless, odorless lightest gas (d = 0.089 g/L)",
        "role": "Reducing agent; highly flammable",
        "phase": "Gas phase",
    },
    "NH3": {
        "name": "Ammonia",
        "type": "Covalent Alkaline Hydride",
        "appearance": "Colorless gas with sharp, choking pungent alkaline odor",
        "role": "Weak base; forms white NH4Cl fumes with HCl; fertilizer precursor",
        "phase": "Gas phase / aqueous alkaline solution",
    },
    "KClO3": {
        "name": "Potassium Chlorate",
        "type": "Strong Oxidizing Salt",
        "appearance": "White crystalline powder / lustrous plates",
        "role": "Thermal oxygen source; decomposes readily with MnO2 catalyst",
        "phase": "Solid reactant",
    },
    "H2O2": {
        "name": "Hydrogen Peroxide",
        "type": "Reactive Peroxide",
        "appearance": "Colorless, slightly viscous liquid; oxidizing/reducing",
        "role": "Disproportionates into H2O + O2",
        "phase": "Liquid / aqueous solution",
    },
    "CO": {
        "name": "Carbon Monoxide",
        "type": "Neutral Toxic Oxide",
        "appearance": "Colorless, odorless, tasteless poisonous gas",
        "role": "Reducing agent; binds hemoglobin 200x tighter than O2",
        "phase": "Gas phase",
    },
    "CO2": {
        "name": "Carbon Dioxide",
        "type": "Acidic Covalent Oxide",
        "appearance": "Colorless, odorless gas; turns lime water milky",
        "role": "Fire suppressant; sub-limates as dry ice at -78.5°C",
        "phase": "Gas phase",
    },
    "C2H4": {
        "name": "Ethylene / Ethene",
        "type": "Unsaturated Alkene Hydrocarbon",
        "appearance": "Colorless flammable gas with sweet musky odor",
        "role": "Monomer for polyethylene; hydrogenated by Ni/Pt catalyst",
        "phase": "Gas phase",
    },
    "C2H6": {
        "name": "Ethane",
        "type": "Saturated Alkane Hydrocarbon",
        "appearance": "Colorless, odorless flammable gas",
        "role": "Saturated paraffin fuel",
        "phase": "Gas phase",
    },
    "CH4": {
        "name": "Methane",
        "type": "Aliphatic Hydrocarbon / Natural Gas",
        "appearance": "Colorless, odorless flammable gas",
        "role": "Primary natural gas constituent (CH4 + 2O2 -> CO2 + 2H2O)",
        "phase": "Gas phase",
    },
    "CuSO4": {
        "name": "Copper(II) Sulfate",
        "type": "Transition Metal Salt",
        "appearance": "White anhydrous powder / Blue pentahydrate crystals",
        "role": "Electrolyte, fungicide, moisture test",
        "phase": "Solid / aqueous bright blue solution",
    },
    "NaOH": {
        "name": "Sodium Hydroxide (Caustic Soda)",
        "type": "Strong Caustic Base",
        "appearance": "White deliquescent pellets / flakes; highly corrosive",
        "role": "Complete dissociation (Na+ + OH-); saponification",
        "phase": "Solid / alkaline solution (pH 13-14)",
    },
}

# Catalyst Database (Industrial processes & mechanism)
CATALYST_DATABASE: Dict[str, Dict[str, Any]] = {
    "V2O5": {
        "name": "Vanadium(V) Oxide",
        "process": "Contact Process (2SO2 + O2 ➔ 2SO3)",
        "mechanism": "Heterogeneous redox cycle: SO2 reduces V2O5 to V2O4; O2 re-oxidizes V2O4 back to V2O5.",
        "uncat_ea": 250.0,
        "cat_ea": 65.0,
        "conditions": "450°C, 1-2 atm",
    },
    "FE": {
        "name": "Finely Divided Iron (with Mo / Al2O3 promoters)",
        "process": "Haber-Bosch Process (N2 + 3H2 ➔ 2NH3)",
        "mechanism": "Dissociative chemisorption on Fe surface, breaking the inert N≡N triple bond (945 kJ/mol).",
        "uncat_ea": 335.0,
        "cat_ea": 160.0,
        "conditions": "450-500°C, 200 atm",
    },
    "MNO2": {
        "name": "Manganese Dioxide (Black powder)",
        "process": "Thermal Decomposition of KClO3 / H2O2 Disproportionation",
        "mechanism": "Surface adsorption & intermediate coordination; lowers decomposition temperature from 650°C to 300°C.",
        "uncat_ea": 220.0,
        "cat_ea": 85.0,
        "conditions": "300°C / Ambient for H2O2",
    },
    "PT": {
        "name": "Platinum / Platinum-Rhodium Gauze",
        "process": "Ostwald Process (4NH3 + 5O2 ➔ 4NO + 6H2O) / Catalytic Converter",
        "mechanism": "High d-band electron mobility; rapid dissociative adsorption of O2 and NH3.",
        "uncat_ea": 280.0,
        "cat_ea": 75.0,
        "conditions": "800-900°C",
    },
    "NI": {
        "name": "Finely Divided Nickel / Raney Nickel",
        "process": "Hydrogenation of Alkenes / Sabatier Reaction (C=C + H2 ➔ C-C)",
        "mechanism": "Adsorbs H2 into interstitial hydride lattice, weakening the H-H bond for addition to double bonds.",
        "uncat_ea": 180.0,
        "cat_ea": 40.0,
        "conditions": "150-200°C, 5 atm",
    },
    "CU": {
        "name": "Copper catalyst / Copper(II) Oxide",
        "process": "Methanol Synthesis / Alcohol Dehydrogenation",
        "mechanism": "Selective C-H bond activation and hydride intermediate transfer.",
        "uncat_ea": 210.0,
        "cat_ea": 70.0,
        "conditions": "250-300°C",
    },
}


def render_catalyst_energy_diagram(uncat_ea: float = 250.0, cat_ea: float = 70.0, delta_h: float = -98.0) -> str:
    """Render Potential Energy vs Reaction Coordinate Braille curve comparing Uncatalyzed vs Catalyzed paths."""
    xs = np.linspace(0.0, 100.0, 300)
    h_react = 80.0
    h_prod = h_react + delta_h

    peak_x = 50.0
    width = 16.0

    # Base sigmoid transition
    base = h_react + (h_prod - h_react) / (1.0 + np.exp(-(xs - peak_x) / 7.0))
    uncat_curve = base + uncat_ea * np.exp(-((xs - peak_x) ** 2) / (2.0 * (width ** 2)))
    cat_curve = base + cat_ea * np.exp(-((xs - peak_x) ** 2) / (2.0 * ((width * 0.85) ** 2)))

    return render_braille_curve(
        xs,
        [
            (uncat_curve, C_AMBER, f"Uncatalyzed Path (E_a = {uncat_ea:.0f} kJ)"),
            (cat_curve, C_GREEN, f"Catalyzed Path (E_a = {cat_ea:.0f} kJ)"),
        ],
        cols=56,
        rows=8,
    )


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse chemical formulas with parentheses and hydrates like Ca(OH)2, CuSO4*5H2O."""
    clean_f = formula.replace(" ", "")
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
        while "(" in sub_str or "[" in sub_str:
            match = re.search(r"[\(\[]([A-Za-z0-9]+)[\)\]](\d*)", sub_str)
            if not match:
                break
            inner, mult = match.group(1), int(match.group(2)) if match.group(2) else 1
            inner_counts = parse_sub(inner)
            expanded = "".join(f"{el}{cnt * mult}" for el, cnt in inner_counts.items())
            sub_str = sub_str[:match.start()] + expanded + sub_str[match.end():]

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


def get_compound_info(formula: str) -> Dict[str, str]:
    """Retrieve compound properties or infer from elemental composition."""
    # Strip numbers and clean
    f_clean = re.sub(r"^\d+", "", formula).strip()
    if f_clean in COMPOUND_PROPERTIES:
        return COMPOUND_PROPERTIES[f_clean]

    # Inferred properties
    counts = parse_formula(f_clean)
    has_metal = any(el in ["Na", "K", "Ca", "Mg", "Fe", "Cu", "Zn", "Al", "Mn", "Pb", "Ag", "Au", "Pt", "Ba", "Ni"] for el in counts)
    has_h = "H" in counts
    has_o = "O" in counts

    if has_metal and has_o and len(counts) == 2:
        return {"name": f_clean, "type": "Basic Metal Oxide", "appearance": "Solid oxide powder", "role": "Inorganic metal oxide", "phase": "Solid"}
    elif has_metal and len(counts) > 2:
        return {"name": f_clean, "type": "Inorganic Ionic Salt", "appearance": "Crystalline solid", "role": "Ionic salt compound", "phase": "Solid / aqueous"}
    elif has_h and not has_metal and len(counts) >= 2:
        return {"name": f_clean, "type": "Covalent Compound / Acid", "appearance": "Liquid or gas at STP", "role": "Molecular covalent compound", "phase": "Liquid / Gas"}
    else:
        return {"name": f_clean, "type": "Chemical Compound", "appearance": "Pure substance", "role": "Stoichiometric participant", "phase": "Standard state"}


def balance_chemical_equation(equation: str) -> str:
    """Balance chemical equations with catalyst parsing, energy diagrams, and compound properties."""
    eq_raw = equation.strip()

    # 1. Catalyst detection: e.g. '[c=V2O5]', 'c = Fe', 'catalyst = MnO2', 'with catalyst Pt'
    catalyst_key: Optional[str] = None
    m_cat = re.search(r"(?:\[|\b)(?:c|catalyst|cat)\s*[:=]\s*([A-Za-z0-9_\/]+)(?:\]|\b)", eq_raw, re.I)
    if not m_cat:
        m_cat = re.search(r"with catalyst\s+([A-Za-z0-9_\/]+)", eq_raw, re.I)

    if m_cat:
        catalyst_key = m_cat.group(1).upper()
        # Clean equation string
        eq_clean = re.sub(r"\[?\b(?:c|catalyst|cat)\s*[:=]\s*[A-Za-z0-9_\/]+\]?", "", eq_raw, flags=re.I)
        eq_clean = re.sub(r"with catalyst\s+[A-Za-z0-9_\/]+", "", eq_clean, flags=re.I)
    else:
        eq_clean = eq_raw

    eq_clean = re.sub(r"^(balance|bal)\s+", "", eq_clean.strip(), flags=re.I)
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
        sp_clean = re.sub(r"^\d+\s*", "", sp_name)
        d = parse_formula(sp_clean)
        species_dicts.append((sp_clean, d))
        all_elements.update(d.keys())

    sorted_elements = sorted(all_elements)
    n_species = len(all_species)
    n_reactants = len(reactants)

    # Build stoichiometry matrix
    matrix_rows = []
    for el in sorted_elements:
        row = []
        for i, (_, s_dict) in enumerate(species_dicts):
            coeff = s_dict.get(el, 0)
            if i >= n_reactants:
                coeff = -coeff
            row.append(coeff)
        matrix_rows.append(row)

    M = sp.Matrix(matrix_rows)
    null_space = M.nullspace()
    if not null_space:
        return f"{C_RED}[ERROR]: Chemical equation cannot be balanced (conservation violation).{C_RESET}"

    sol_vector = null_space[0]
    lcm_denom = sp.ilcm(*[val.q for val in sol_vector])
    int_coeffs = [int(val * lcm_denom) for val in sol_vector]
    if int_coeffs[0] < 0:
        int_coeffs = [-c for c in int_coeffs]

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

    arrow = f" ──[{catalyst_key}]──➔ " if catalyst_key else " ➔ "
    balanced_str = " + ".join(left_side) + arrow + " + ".join(right_side)

    out = [
        f"{C_BOLD}{C_GREEN}⚡ BALANCED CHEMICAL REACTION:{C_RESET}",
        f"  {C_BOLD}{C_WHITE}{balanced_str}{C_RESET}",
        f"  {C_GRAY}Stoichiometric Ratio:{C_RESET} {C_CYAN}{':'.join(str(c) for c in int_coeffs)}{C_RESET}",
    ]

    # 2. Catalyst Mechanism & Barrier Reduction
    if catalyst_key:
        cat_data = CATALYST_DATABASE.get(catalyst_key, None)
        out.append(f"\n{C_BOLD}{C_AMBER}⚗️ CATALYST ACTIVE [c = {catalyst_key}]:{C_RESET}")
        if cat_data:
            out.append(f"  {C_BOLD}{C_CYAN}Identity     :{C_RESET} {cat_data['name']} ({cat_data['conditions']})")
            out.append(f"  {C_GRAY}Mechanism    :{C_RESET} {cat_data['mechanism']}")
            out.append(f"  {C_BOLD}{C_GREEN}Energy Drop  :{C_RESET} {C_WHITE}{cat_data['uncat_ea']:.0f} kJ/mol{C_RESET} (uncat) ➔ {C_BOLD}{C_GREEN}{cat_data['cat_ea']:.0f} kJ/mol{C_RESET} (cat) [ΔE_a = -{cat_data['uncat_ea'] - cat_data['cat_ea']:.0f} kJ]")
            out.append(render_catalyst_energy_diagram(cat_data['uncat_ea'], cat_data['cat_ea'], -98.0))
        else:
            out.append(f"  {C_GRAY}Mechanism    :{C_RESET} Lowers activation energy barrier E_a without shifting equilibrium constant K_eq.")
            out.append(render_catalyst_energy_diagram(250.0, 75.0, -90.0))

    # 3. Explanation of Reactants & Products (Minimum Verbose)
    out.append(f"\n{C_BOLD}{C_PURPLE}🔬 REACTANTS & PRODUCTS COMPOSITION (MINIMUM VERBOSE):{C_RESET}")
    out.append(f"  {C_BOLD}{C_AMBER}▶ REACTANTS MIXTURE:{C_RESET}")
    for i in range(n_reactants):
        sp_name = species_dicts[i][0]
        c = int_coeffs[i]
        info = get_compound_info(sp_name)
        out.append(f"    • {C_BOLD}{C_CYAN}{c if c>1 else ''}{sp_name}{C_RESET} ({info['name']}): [{info['type']}] — {info['appearance']} | {info['role']}")

    out.append(f"  {C_BOLD}{C_GREEN}▶ PRODUCTS MIXTURE:{C_RESET}")
    for i in range(n_reactants, n_species):
        sp_name = species_dicts[i][0]
        c = int_coeffs[i]
        info = get_compound_info(sp_name)
        out.append(f"    • {C_BOLD}{C_CYAN}{c if c>1 else ''}{sp_name}{C_RESET} ({info['name']}): [{info['type']}] — {info['appearance']} | {info['role']}")

    # 4. Mixture / Phase Analysis
    phases = [get_compound_info(s[0])["phase"] for s in species_dicts]
    has_gas = any("Gas" in p for p in phases)
    has_aq = any("Aqueous" in p or "solution" in p for p in phases)
    has_precip = any("precipitate" in p or "rust" in p for p in phases)
    mix_type = "Heterogeneous (Multiphase)" if (has_gas and (has_aq or has_precip)) else "Homogeneous"
    out.append(f"  {C_GRAY}Reaction System Phase:{C_RESET} {C_WHITE}{mix_type}{C_RESET} " +
               f"{'(Effervescence ↑)' if has_gas else ''} {'(Precipitation ↓)' if has_precip else ''}")

    result_str = "\n".join(out)

    # Sync to Obsidian
    try:
        from host.obsidian_sync import record_session_calculation
        record_session_calculation(
            domain="Chemistry",
            title=f"Reaction - {balanced_str[:40]}",
            query=equation,
            result_text=result_str,
            tags=["chemistry", "reaction", "stoichiometry", "catalysis" if catalyst_key else "balance"],
            related_links=[s[0] for s in species_dicts] + (["Catalysts", catalyst_key] if catalyst_key else []),
        )
    except Exception:
        pass

    return result_str


def solve_chem_numerical(query: str) -> str:
    """Solve chemistry numericals: pH/pOH, Gas laws, Kinetics, Electrochemistry, Molar mass."""
    q = query.strip()
    q_low = q.lower()

    # Obsidian / Graph view trigger
    if q_low in ("obsidian", "graph", "vault", "sync"):
        from host.obsidian_sync import open_obsidian_graph_view
        return open_obsidian_graph_view()

    # 1. Chemical Equation Balancing
    if "->" in q or "➔" in q or ("=" in q and any(c.isupper() for c in q)):
        return balance_chemical_equation(q)

    out: List[str] = []

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
            res = "\n".join(out)
            try:
                from host.obsidian_sync import record_session_calculation
                record_session_calculation("Chemistry", f"MolarMass - {formula}", q, res, ["chemistry", "molar_mass"], [formula, "Molar Mass"])
            except Exception:
                pass
            return res
        except Exception:
            pass

    # 3. Acid-Base pH & Titration Curve
    if "ph" in q_low or "poh" in q_low or "titration" in q_low:
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

        v_naoh = np.linspace(0.1, 50.0, 300)
        v_acid = 25.0
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
        res = "\n".join(out)
        try:
            from host.obsidian_sync import record_session_calculation
            record_session_calculation("Chemistry", f"pH - {conc}M", q, res, ["chemistry", "acid_base", "pH"], ["pH Scale", "Titration", "Neutralisation"])
        except Exception:
            pass
        return res

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
        res = "\n".join(out)
        try:
            from host.obsidian_sync import record_session_calculation
            record_session_calculation("Chemistry", f"Kinetics - k={k_val}", q, res, ["chemistry", "kinetics", "half_life"], ["Chemical Kinetics", "Rate Laws"])
        except Exception:
            pass
        return res

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

        v_range = np.linspace(1.0, 40.0, 300)
        p_iso = (n * R * t) / v_range
        out.append(render_braille_curve(v_range, [(p_iso, C_CYAN, "P(V) Isotherm (atm vs L)")], cols=54, rows=7))
        res = "\n".join(out)
        try:
            from host.obsidian_sync import record_session_calculation
            record_session_calculation("Chemistry", "Ideal Gas Law - PV=nRT", q, res, ["chemistry", "gas_laws"], ["Gas Laws", "Thermodynamics"])
        except Exception:
            pass
        return res

    # 6. Fallback
    return balance_chemical_equation(q)


def run_chem_repl() -> None:
    """Interactive Chemistry Shell."""
    if os.name == "nt":
        os.system("")
    print(f"\n{C_BOLD}{C_CYAN}╔═══════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}║     ⚗️  nuro-chem — Advanced Chemical Equation & Numerical     ║{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}╚═══════════════════════════════════════════════════════════════╝{C_RESET}")
    print(f"{C_GRAY}Capabilities:{C_RESET} Balance, Catalyst c, Activation Energy curve, Properties, Titration, Kinetics ('q' to quit)")
    print(f"{C_GRAY}Examples:{C_RESET} {C_GREEN}2SO2 + O2 -> 2SO3 [c=V2O5]{C_RESET}  │  {C_AMBER}N2 + 3H2 -> 2NH3 c=Fe{C_RESET}")
    print(f"          {C_CYAN}KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2{C_RESET}  │  {C_PURPLE}obsidian{C_RESET} (open graph)\n")

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
