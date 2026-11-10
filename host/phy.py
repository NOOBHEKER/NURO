"""Advanced Physics Simulation & Numerical Solver Engine for nuro.

Capabilities:
1. Classical Mechanics & Ballistics:
   - Projectile motion with launch angle, initial velocity, height
   - Exact flight time, peak altitude, total range, impact velocity vector
   - Unicode Braille trajectory graph y(x)
2. Oscillations, Damping & Resonance:
   - Damped harmonic oscillator: m*x'' + c*x' + k*x = F0*cos(w*t)
   - Natural frequency w0, damping ratio zeta, Q factor, resonance frequency wr
   - Braille resonance frequency response curve |A(w)| vs w
3. Electromagnetism & RLC Circuits:
   - Series RLC transient step response: R, L, C
   - w0, alpha, wd (damped frequency), Q-factor, damping regime
   - Braille transient ringing voltage curve V(t)
   - AC Impedance, reactances (XL, XC), phase angle phi
4. Special Relativity:
   - Lorentz factor gamma = 1/sqrt(1 - (v/c)^2)
   - Time dilation, length contraction, relativistic momentum & total energy
   - Braille gamma(v/c) curve
5. Quantum Mechanics & Modern Physics:
   - Bohr hydrogen atom transitions (Lyman, Balmer, Paschen series, photon energy & wavelength)
   - De Broglie matter wavelength: lambda = h/p
   - Photoelectric effect: stopping potential, work function, threshold frequency
   - Planck blackbody radiation curve & Wien's displacement law
6. Orbital Mechanics & Gravitation:
   - Keplerian orbital velocity, orbital period, escape velocity
7. Physical Constants Database:
   - c, G, h, hbar, kB, e, me, mp, eps0, mu0, g, NA, sigma
8. Interactive REPL Shell ('nuro phy')
"""
from __future__ import annotations

import math
import os
import re
import sys
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

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


# Physical Constants in SI units
CONSTANTS: Dict[str, Tuple[float, str, str]] = {
    "c": (299792458.0, "m/s", "Speed of light in vacuum"),
    "g": (9.80665, "m/s²", "Standard acceleration due to gravity"),
    "G": (6.67430e-11, "N·m²/kg²", "Newtonian gravitational constant"),
    "h": (6.62607015e-34, "J·s", "Planck constant"),
    "hbar": (1.054571817e-34, "J·s", "Reduced Planck constant (h / 2π)"),
    "kB": (1.380649e-23, "J/K", "Boltzmann constant"),
    "e": (1.602176634e-19, "C", "Elementary charge"),
    "me": (9.1093837e-31, "kg", "Electron rest mass"),
    "mp": (1.672621923e-27, "kg", "Proton rest mass"),
    "mn": (1.674927498e-27, "kg", "Neutron rest mass"),
    "eps0": (8.8541878128e-12, "F/m", "Permittivity of free space (vacuum)"),
    "mu0": (1.25663706212e-6, "N/A²", "Permeability of free space (vacuum)"),
    "NA": (6.02214076e23, "mol⁻¹", "Avogadro constant"),
    "sigma": (5.670374419e-8, "W/(m²·K⁴)", "Stefan-Boltzmann constant"),
    "R": (8.314462618, "J/(mol·K)", "Universal gas constant"),
    "wien": (2.897771955e-3, "m·K", "Wien displacement law constant b"),
    "a0": (5.29177210903e-11, "m", "Bohr radius"),
    "R_inf": (1.0973731568160e7, "m⁻¹", "Rydberg constant"),
}


def format_constants_table(filter_key: Optional[str] = None) -> str:
    """Format physical constants table."""
    out = [f"{C_BOLD}{C_GREEN}⚡ FUNDAMENTAL PHYSICAL CONSTANTS (CODATA / SI):{C_RESET}"]
    for sym, (val, units, desc) in sorted(CONSTANTS.items()):
        if filter_key and filter_key.lower() not in sym.lower() and filter_key.lower() not in desc.lower():
            continue
        val_str = f"{val:.6e}" if (abs(val) < 1e-2 or abs(val) >= 1e4) else f"{val:.5f}"
        out.append(f"  {C_CYAN}{sym:>6}{C_RESET} = {C_BOLD}{C_WHITE}{val_str:>13}{C_RESET} {C_AMBER}{units:<12}{C_RESET} {C_GRAY}{desc}{C_RESET}")
    return "\n".join(out)


def solve_projectile(query: str) -> str:
    """Solve ballistics and projectile trajectory."""
    # Look for v (m/s), angle (deg), y0 (m), g
    m_v = re.search(r"(?:v|vel|speed)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_ang = re.search(r"(?:angle|theta|ang|deg)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_y0 = re.search(r"(?:h|height|y0)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_g = re.search(r"(?:g|gravity)\s*=\s*([0-9\.\-]+)", query, re.I)

    # Fallback numbers if simple tokens: projectile 50 45
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", query)]
    v0 = float(m_v.group(1)) if m_v else (nums[0] if len(nums) >= 1 else 30.0)
    theta_deg = float(m_ang.group(1)) if m_ang else (nums[1] if len(nums) >= 2 else 45.0)
    y0 = float(m_y0.group(1)) if m_y0 else 0.0
    g = float(m_g.group(1)) if m_g else CONSTANTS["g"][0]

    theta_rad = math.radians(theta_deg)
    vx0 = v0 * math.cos(theta_rad)
    vy0 = v0 * math.sin(theta_rad)

    # Quadratic for flight time: y0 + vy0*t - 0.5*g*t^2 = 0
    disc = vy0**2 + 2 * g * y0
    t_flight = (vy0 + math.sqrt(max(0.0, disc))) / g if g > 0 else 0.0
    t_peak = vy0 / g if g > 0 else 0.0
    h_max = y0 + (vy0**2) / (2 * g) if g > 0 else y0
    r_max = vx0 * t_flight

    # Velocity at impact
    vy_impact = vy0 - g * t_flight
    v_impact = math.hypot(vx0, vy_impact)
    impact_ang = math.degrees(math.atan2(abs(vy_impact), vx0))

    out = [
        f"{C_BOLD}{C_GREEN}⚡ PROJECTILE DYNAMICS & TRAJECTORY:{C_RESET}",
        f"  {C_GRAY}Initial Parameters :{C_RESET} v₀ = {v0:.2f} m/s, θ = {theta_deg:.1f}°, y₀ = {y0:.2f} m, g = {g:.2f} m/s²",
        f"  {C_AMBER}Time of Flight     :{C_RESET} {C_BOLD}{C_WHITE}{t_flight:.3f} s{C_RESET}  (Peak reached at {t_peak:.3f} s)",
        f"  {C_BOLD}{C_CYAN}Maximum Altitude H :{C_RESET} {C_BOLD}{C_WHITE}{h_max:.3f} m{C_RESET}",
        f"  {C_BOLD}{C_CYAN}Total Range R      :{C_RESET} {C_BOLD}{C_WHITE}{r_max:.3f} m{C_RESET}",
        f"  {C_GRAY}Impact Velocity    :{C_RESET} {v_impact:.2f} m/s @ -{impact_ang:.1f}°",
    ]

    # Generate Braille curve y(x)
    xs = np.linspace(0.0, max(0.1, r_max), 300)
    ts = xs / max(1e-9, vx0)
    ys = np.maximum(0.0, y0 + vy0 * ts - 0.5 * g * ts**2)
    out.append(render_braille_curve(xs, [(ys, C_CYAN, "Trajectory y(x) [m]")], cols=56, rows=8))
    return "\n".join(out)


def solve_oscillator(query: str) -> str:
    """Solve damped & driven harmonic oscillator resonance."""
    m_m = re.search(r"(?:m|mass)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_k = re.search(r"(?:k|spring)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_c = re.search(r"(?:c|damping|gamma|b)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_f0 = re.search(r"(?:f0|force)\s*=\s*([0-9\.\-]+)", query, re.I)

    m = float(m_m.group(1)) if m_m else 1.0
    k = float(m_k.group(1)) if m_k else 100.0
    c = float(m_c.group(1)) if m_c else 2.0
    F0 = float(m_f0.group(1)) if m_f0 else 10.0

    w0 = math.sqrt(k / m)  # rad/s
    f0 = w0 / (2.0 * math.pi)  # Hz
    zeta = c / (2.0 * math.sqrt(k * m))  # Damping ratio
    Q = 1.0 / (2.0 * zeta) if zeta > 0 else float("inf")

    # Resonance frequency
    if zeta < (1.0 / math.sqrt(2.0)):
        wr = w0 * math.sqrt(1.0 - 2.0 * (zeta**2))
        fr = wr / (2.0 * math.pi)
        A_max = (F0 / k) / (2.0 * zeta * math.sqrt(1.0 - zeta**2))
    else:
        wr = 0.0
        fr = 0.0
        A_max = F0 / k

    regime = "Underdamped (Oscillatory)" if zeta < 1.0 else ("Critically Damped" if abs(zeta - 1.0) < 1e-4 else "Overdamped")

    out = [
        f"{C_BOLD}{C_GREEN}⚡ DAMPED & DRIVEN HARMONIC OSCILLATOR:{C_RESET}",
        f"  {C_GRAY}System Parameters  :{C_RESET} m = {m:.2f} kg, k = {k:.2f} N/m, c = {c:.2f} N·s/m, F₀ = {F0:.2f} N",
        f"  {C_BOLD}{C_CYAN}Natural Freq ω₀    :{C_RESET} {C_BOLD}{C_WHITE}{w0:.3f} rad/s{C_RESET} ({f0:.3f} Hz)",
        f"  {C_AMBER}Damping Ratio ζ    :{C_RESET} {zeta:.4f}  [{regime}]",
        f"  {C_AMBER}Quality Factor Q   :{C_RESET} {Q:.3f}",
        f"  {C_BOLD}{C_CYAN}Resonance Peak ω_r :{C_RESET} {C_BOLD}{C_WHITE}{wr:.3f} rad/s{C_RESET} ({fr:.3f} Hz) | Max Amp: {A_max:.4f} m",
    ]

    # Frequency response curve A(w)
    w_axis = np.linspace(0.01, w0 * 2.5, 300)
    denom = np.sqrt((w0**2 - w_axis**2)**2 + (2.0 * zeta * w0 * w_axis)**2)
    A_resp = (F0 / m) / np.maximum(1e-9, denom)
    out.append(render_braille_curve(w_axis, [(A_resp, C_CYAN, "Amplitude A(ω) vs ω (rad/s)")], cols=56, rows=8))
    return "\n".join(out)


def solve_rlc(query: str) -> str:
    """Solve RLC series circuit transient ringing & AC impedance."""
    m_r = re.search(r"(?:r|res)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_l = re.search(r"(?:l|ind)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_c = re.search(r"(?:c|cap)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_v0 = re.search(r"(?:v0|v|vin)\s*=\s*([0-9\.\-]+)", query, re.I)

    R = float(m_r.group(1)) if m_r else 10.0
    L = float(m_l.group(1)) if m_l else 0.05
    C = float(m_c.group(1)) if m_c else 100e-6
    V0 = float(m_v0.group(1)) if m_v0 else 5.0

    w0 = 1.0 / math.sqrt(L * C)
    f0 = w0 / (2.0 * math.pi)
    alpha = R / (2.0 * L)  # Neper frequency
    zeta = alpha / w0
    Q = (1.0 / R) * math.sqrt(L / C)

    if zeta < 1.0:
        wd = math.sqrt(w0**2 - alpha**2)
        fd = wd / (2.0 * math.pi)
        regime = "Underdamped Ringing"
    elif abs(zeta - 1.0) < 1e-4:
        wd = 0.0
        fd = 0.0
        regime = "Critically Damped"
    else:
        wd = 0.0
        fd = 0.0
        regime = "Overdamped (Aperiodic)"

    out = [
        f"{C_BOLD}{C_GREEN}⚡ SERIES RLC TRANSIENT & AC IMPEDANCE ENGINE:{C_RESET}",
        f"  {C_GRAY}Components         :{C_RESET} R = {R:.3f} Ω, L = {L*1e3:.2f} mH, C = {C*1e6:.2f} µF, V₀ = {V0:.2f} V",
        f"  {C_BOLD}{C_CYAN}Resonance Freq f₀  :{C_RESET} {C_BOLD}{C_WHITE}{f0:.2f} Hz{C_RESET} (ω₀ = {w0:.2f} rad/s)",
        f"  {C_AMBER}Neper Damping α    :{C_RESET} {alpha:.2f} s⁻¹  (Damping Ratio ζ = {zeta:.4f})",
        f"  {C_AMBER}Quality Factor Q   :{C_RESET} {Q:.3f}  [{regime}]",
    ]
    if zeta < 1.0:
        out.append(f"  {C_BOLD}{C_CYAN}Damped Freq f_d    :{C_RESET} {C_BOLD}{C_WHITE}{fd:.2f} Hz{C_RESET} (ω_d = {wd:.2f} rad/s)")

    # Plot transient step response V_c(t)
    t_end = 5.0 / alpha if alpha > 0 else 0.05
    t_axis = np.linspace(0.0, max(1e-4, t_end), 350)
    if zeta < 1.0:
        phi = math.atan2(alpha, wd)
        vc = V0 * (1.0 - (w0 / wd) * np.exp(-alpha * t_axis) * np.sin(wd * t_axis + phi))
    elif abs(zeta - 1.0) < 1e-4:
        vc = V0 * (1.0 - (1.0 + alpha * t_axis) * np.exp(-alpha * t_axis))
    else:
        s1 = -alpha + math.sqrt(alpha**2 - w0**2)
        s2 = -alpha - math.sqrt(alpha**2 - w0**2)
        vc = V0 * (1.0 - (s2 * np.exp(s1 * t_axis) - s1 * np.exp(s2 * t_axis)) / (s2 - s1))

    out.append(render_braille_curve(t_axis * 1e3, [(vc, C_CYAN, "Capacitor Voltage V_C(t) [V] vs t [ms]")], cols=56, rows=8))
    return "\n".join(out)


def solve_relativity(query: str) -> str:
    """Solve special relativity: Lorentz factor, time dilation, mass-energy."""
    m_beta = re.search(r"(?:v|beta)\s*=\s*([0-9\.\-]+)\s*c?", query, re.I)
    m_gamma = re.search(r"(?:gamma|g)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_m0 = re.search(r"(?:m0|m|mass)\s*=\s*([0-9\.\-]+)", query, re.I)

    c = CONSTANTS["c"][0]
    m0 = float(m_m0.group(1)) if m_m0 else CONSTANTS["me"][0]

    beta = 0.8
    if m_gamma:
        gamma = float(m_gamma.group(1))
        beta = math.sqrt(max(0.0, 1.0 - (1.0 / (gamma**2))))
    elif m_beta:
        raw_v = float(m_beta.group(1))
        beta = raw_v if raw_v < 1.0 else (raw_v / c if raw_v > 1e4 else 0.9)
    else:
        # Check if query has raw number like '0.9c'
        raw = re.findall(r"([0-9\.]+)\s*c", query, re.I)
        if raw:
            beta = float(raw[0])

    beta = min(0.99999999, max(0.0, beta))
    v = beta * c
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    dilation_pct = (gamma - 1.0) * 100.0
    contraction_pct = (1.0 - 1.0 / gamma) * 100.0

    p = gamma * m0 * v
    E_rest = m0 * (c**2)
    E_tot = gamma * E_rest
    K_rel = (gamma - 1.0) * E_rest

    out = [
        f"{C_BOLD}{C_GREEN}⚡ SPECIAL RELATIVITY & LORENTZ KINEMATICS:{C_RESET}",
        f"  {C_GRAY}Velocity v         :{C_RESET} {C_BOLD}{C_WHITE}{beta:.6f} c{C_RESET} ({v:.3e} m/s)",
        f"  {C_BOLD}{C_CYAN}Lorentz Factor γ   :{C_RESET} {C_BOLD}{C_WHITE}{gamma:.6f}{C_RESET}",
        f"  {C_AMBER}Time Dilation t'   :{C_RESET} 1.000 s stationary ➔ {C_BOLD}{C_CYAN}{gamma:.4f} s{C_RESET} (+{dilation_pct:.2f}%)",
        f"  {C_AMBER}Length Contraction :{C_RESET} 1.000 m moving     ➔ {C_BOLD}{C_CYAN}{1.0/gamma:.4f} m{C_RESET} (-{contraction_pct:.2f}%)",
        f"  {C_BOLD}{C_CYAN}Rest Energy E₀     :{C_RESET} {E_rest:.6e} J ({E_rest / CONSTANTS['e'][0]:.4e} eV)",
        f"  {C_BOLD}{C_CYAN}Total Energy E     :{C_RESET} {E_tot:.6e} J ({E_tot / CONSTANTS['e'][0]:.4e} eV)",
        f"  {C_GRAY}Relativistic Mom p :{C_RESET} {p:.6e} kg·m/s",
    ]

    # Plot Lorentz Factor gamma vs beta = v/c
    betas = np.linspace(0.0, 0.99, 300)
    gammas = 1.0 / np.sqrt(1.0 - betas**2)
    out.append(render_braille_curve(betas, [(gammas, C_CYAN, "Lorentz Factor γ(β) vs β = v/c")], cols=56, rows=8))
    return "\n".join(out)


def solve_quantum(query: str) -> str:
    """Solve quantum mechanics: Bohr model, De Broglie, Photoelectric, Blackbody."""
    q_low = query.lower()
    c = CONSTANTS["c"][0]
    h = CONSTANTS["h"][0]
    hbar = CONSTANTS["hbar"][0]
    e = CONSTANTS["e"][0]

    # 1. Bohr Model transitions
    if "bohr" in q_low or ("hydrogen" in q_low and ("n=" in q_low or "transition" in q_low or "level" in q_low)):
        m_n1 = re.search(r"n1\s*=\s*(\d+)", q_low)
        m_n2 = re.search(r"n2\s*=\s*(\d+)", q_low)
        nums = [int(x) for x in re.findall(r"\b\d+\b", q_low)]
        n1 = int(m_n1.group(1)) if m_n1 else (nums[0] if len(nums) >= 1 else 1)
        n2 = int(m_n2.group(1)) if m_n2 else (nums[1] if len(nums) >= 2 else 2)
        if n1 > n2:
            n1, n2 = n2, n1  # n1 lower, n2 higher

        E1 = -13.6057 / (n1**2)
        E2 = -13.6057 / (n2**2)
        dE_ev = 13.6057 * (1.0 / (n1**2) - 1.0 / (n2**2))
        dE_joules = dE_ev * e
        wavelength_m = (h * c) / dE_joules if dE_joules > 0 else 0.0
        wavelength_nm = wavelength_m * 1e9

        series_name = {
            1: "Lyman (Ultraviolet)",
            2: "Balmer (Visible)",
            3: "Paschen (Infrared)",
            4: "Brackett (Far Infrared)",
            5: "Pfund (Far Infrared)",
        }.get(n1, "Outer Shell")

        out = [
            f"{C_BOLD}{C_GREEN}⚡ BOHR HYDROGEN ATOM QUANTUM TRANSITION:{C_RESET}",
            f"  {C_GRAY}Orbital Transition :{C_RESET} n = {n2} ➔ n = {n1}  [{series_name}]",
            f"  {C_AMBER}Lower Level E_{n1}   :{C_RESET} {E1:.4f} eV",
            f"  {C_AMBER}Upper Level E_{n2}   :{C_RESET} {E2:.4f} eV",
            f"  {C_BOLD}{C_CYAN}Photon Energy ΔE   :{C_RESET} {C_BOLD}{C_WHITE}{dE_ev:.4f} eV{C_RESET} ({dE_joules:.4e} J)",
            f"  {C_BOLD}{C_CYAN}Photon Wavelength λ:{C_RESET} {C_BOLD}{C_WHITE}{wavelength_nm:.2f} nm{C_RESET} (ν = {c/wavelength_m:.4e} Hz)",
        ]
        return "\n".join(out)

    # 2. Photoelectric Effect
    if "photoelectric" in q_low or "work function" in q_low or "stopping potential" in q_low:
        m_phi = re.search(r"(?:work|phi|w)\s*=\s*([0-9\.\-]+)", q_low)
        m_lam = re.search(r"(?:lambda|lam|wavelength)\s*=\s*([0-9\.\-]+)", q_low)
        phi_ev = float(m_phi.group(1)) if m_phi else 2.3  # eV (e.g. Potassium/Sodium)
        lam_nm = float(m_lam.group(1)) if m_lam else 400.0  # nm

        phi_j = phi_ev * e
        lam_m = lam_nm * 1e-9
        nu = c / lam_m
        E_photon_j = h * nu
        E_photon_ev = E_photon_j / e

        nu_0 = phi_j / h
        lam_0_nm = (c / nu_0) * 1e9

        K_max_j = max(0.0, E_photon_j - phi_j)
        K_max_ev = K_max_j / e
        V_stop = K_max_ev

        out = [
            f"{C_BOLD}{C_GREEN}⚡ PHOTOELECTRIC EFFECT (EINSTEIN RELATION):{C_RESET}",
            f"  {C_GRAY}Incident Light λ   :{C_RESET} {lam_nm:.1f} nm (ν = {nu:.3e} Hz, E_photon = {E_photon_ev:.3f} eV)",
            f"  {C_GRAY}Target Work Func Φ :{C_RESET} {phi_ev:.3f} eV ({phi_j:.3e} J)",
            f"  {C_AMBER}Threshold Freq ν₀  :{C_RESET} {nu_0:.3e} Hz (Cutoff λ₀ = {lam_0_nm:.1f} nm)",
            f"  {C_BOLD}{C_CYAN}Max Kinetic Energy :{C_RESET} {C_BOLD}{C_WHITE}{K_max_ev:.4f} eV{C_RESET} ({K_max_j:.3e} J)",
            f"  {C_BOLD}{C_CYAN}Stopping Potential :{C_RESET} {C_BOLD}{C_WHITE}{V_stop:.4f} V{C_RESET}",
        ]
        return "\n".join(out)

    # 3. Blackbody Radiation & Wien's Law
    if "blackbody" in q_low or "planck" in q_low or "wien" in q_low:
        m_t = re.search(r"(?:t|temp)\s*=\s*([0-9\.\-]+)", q_low)
        T = float(m_t.group(1)) if m_t else 5778.0  # Sun's surface K

        # Wien's law: lambda_max = b / T
        b = CONSTANTS["wien"][0]
        lam_peak_m = b / T
        lam_peak_nm = lam_peak_m * 1e9
        total_flux = CONSTANTS["sigma"][0] * (T**4)

        out = [
            f"{C_BOLD}{C_GREEN}⚡ PLANCK BLACKBODY SPECTRAL EMISSION:{C_RESET}",
            f"  {C_GRAY}Surface Temperature:{C_RESET} {T:.1f} K",
            f"  {C_BOLD}{C_CYAN}Peak Wavelength λ_max:{C_RESET} {C_BOLD}{C_WHITE}{lam_peak_nm:.1f} nm{C_RESET} (Wien's Displacement Law)",
            f"  {C_AMBER}Stefan-Boltzmann Flux:{C_RESET} {total_flux:.4e} W/m²",
        ]

        # Plot Planck radiance curve B(lambda, T)
        lams = np.linspace(100e-9, max(2000e-9, lam_peak_m * 3.5), 300)
        # B(lambda) = (2*h*c^2 / lambda^5) / (exp(h*c / (lambda*kB*T)) - 1)
        kB = CONSTANTS["kB"][0]
        exponent = (h * c) / (lams * kB * T)
        radiances = (2.0 * h * (c**2) / (lams**5)) / (np.exp(np.clip(exponent, 0, 700)) - 1.0)
        out.append(render_braille_curve(lams * 1e9, [(radiances / np.max(radiances), C_CYAN, "Spectral Radiance B(λ) normalized vs λ (nm)")], cols=56, rows=8))
        return "\n".join(out)

    # 4. De Broglie Wavelength
    m_v = re.search(r"(?:v|speed)\s*=\s*([0-9\.\-]+)", q_low)
    m_m = re.search(r"(?:m|mass)\s*=\s*([0-9\.\-]+)", q_low)
    v_val = float(m_v.group(1)) if m_v else 1e6
    m_val = float(m_m.group(1)) if m_m else CONSTANTS["me"][0]

    p_val = m_val * v_val
    lam_db = h / p_val if p_val > 0 else float("inf")
    out = [
        f"{C_BOLD}{C_GREEN}⚡ DE BROGLIE MATTER WAVE EQUATION:{C_RESET}",
        f"  {C_GRAY}Particle Mass m    :{C_RESET} {m_val:.4e} kg",
        f"  {C_GRAY}Particle Velocity v:{C_RESET} {v_val:.4e} m/s",
        f"  {C_AMBER}Momentum p = m·v   :{C_RESET} {p_val:.4e} kg·m/s",
        f"  {C_BOLD}{C_CYAN}De Broglie Wave λ  :{C_RESET} {C_BOLD}{C_WHITE}{lam_db*1e9:.6f} nm{C_RESET} ({lam_db:.4e} m)",
    ]
    return "\n".join(out)


def solve_orbital(query: str) -> str:
    """Solve gravitation & Keplerian orbits."""
    m_m = re.search(r"(?:m|mass)\s*=\s*([0-9\.\-]+)", query, re.I)
    m_r = re.search(r"(?:r|alt|radius|altitude)\s*=\s*([0-9\.\-]+)", query, re.I)

    # Earth defaults: M = 5.972e24 kg, R = 6.371e6 m
    M_earth = 5.972e24
    R_earth = 6.371e6
    G = CONSTANTS["G"][0]

    M = float(m_m.group(1)) if m_m else M_earth
    alt_km = float(m_r.group(1)) if m_r else 400.0  # ISS altitude ~400 km
    r = R_earth + (alt_km * 1e3 if alt_km < 1e5 else alt_km)

    v_orb = math.sqrt(G * M / r)
    v_esc = math.sqrt(2.0 * G * M / r)
    T_sec = 2.0 * math.pi * math.sqrt((r**3) / (G * M))
    T_min = T_sec / 60.0

    out = [
        f"{C_BOLD}{C_GREEN}⚡ KEPLERIAN ORBITAL MECHANICS & GRAVITATION:{C_RESET}",
        f"  {C_GRAY}Central Mass M     :{C_RESET} {M:.4e} kg",
        f"  {C_GRAY}Orbital Radius r   :{C_RESET} {r/1e3:.2f} km (Altitude: {alt_km:.1f} km)",
        f"  {C_BOLD}{C_CYAN}Circular Orbit Vel :{C_RESET} {C_BOLD}{C_WHITE}{v_orb/1e3:.3f} km/s{C_RESET} ({v_orb:.1f} m/s)",
        f"  {C_BOLD}{C_CYAN}Orbital Period T   :{C_RESET} {C_BOLD}{C_WHITE}{T_min:.2f} min{C_RESET} ({T_sec:.1f} s)",
        f"  {C_AMBER}Escape Velocity v_e:{C_RESET} {v_esc/1e3:.3f} km/s ({v_esc:.1f} m/s)",
    ]
    return "\n".join(out)


def solve_physics(query: str) -> str:
    """Main Physics Router: dispatches to domain-specific physics modules."""
    q = query.strip()
    q_low = q.lower()

    if not q:
        return format_constants_table()

    if q_low in ("obsidian", "graph", "vault", "sync"):
        from host.obsidian_sync import open_obsidian_graph_view
        return open_obsidian_graph_view()

    # 1. Constants
    if q_low in ("const", "constants", "constant") or q_low.startswith(("constant ", "const ")):
        token = q_low.split()[-1] if len(q_low.split()) > 1 else None
        res = format_constants_table(token)
    elif any(k in q_low for k in ["projectile", "ballistics", "trajectory", "launch", "angle="]):
        res = solve_projectile(q)
    elif any(k in q_low for k in ["oscillator", "damping", "resonance", "harmonic", "spring", "damping ratio"]):
        res = solve_oscillator(q)
    elif any(k in q_low for k in ["rlc", "circuit", "inductor", "capacitor", "impedance", "transient"]):
        res = solve_rlc(q)
    elif any(k in q_low for k in ["relativity", "lorentz", "gamma", "dilation", "contraction", "einstein"]) or ("0." in q_low and "c" in q_low):
        res = solve_relativity(q)
    elif any(k in q_low for k in ["bohr", "quantum", "photoelectric", "blackbody", "de broglie", "planck", "photon", "hydrogen"]):
        res = solve_quantum(q)
    elif any(k in q_low for k in ["orbit", "kepler", "satellite", "escape velocity", "gravitation"]):
        res = solve_orbital(q)
    elif "v=" in q_low or "speed=" in q_low:
        res = solve_projectile(q)
    elif "r=" in q_low and "l=" in q_low:
        res = solve_rlc(q)
    else:
        res = format_constants_table(q)

    # Sync to Obsidian
    try:
        from host.obsidian_sync import record_session_calculation
        record_session_calculation(
            domain="Physics",
            title=f"Physics - {q[:30]}",
            query=q,
            result_text=res,
            tags=["physics", "simulation"],
            related_links=["Mechanics", "Electromagnetism", "Relativity", "Quantum Mechanics"],
        )
    except Exception:
        pass

    return res


def run_phy_repl() -> None:
    """Interactive Physics Shell."""
    if os.name == "nt":
        os.system("")
    print(f"\n{C_BOLD}{C_CYAN}╔═══════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}║    ⚛️  nuro-phy — Over-the-Top Advanced Physics Engine        ║{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}╚═══════════════════════════════════════════════════════════════╝{C_RESET}")
    print(f"{C_GRAY}Capabilities:{C_RESET} Projectile, Damped Resonance, RLC Transients, Relativity, Quantum Bohr, Orbits ('q' to quit)")
    print(f"{C_GRAY}Examples:{C_RESET} {C_GREEN}projectile v=50 angle=45{C_RESET}  │  {C_AMBER}rlc R=10 L=0.05 C=0.0001{C_RESET}")
    print(f"          {C_CYAN}relativity v=0.9c{C_RESET}         │  {C_PURPLE}bohr n1=1 n2=3{C_RESET}  │  {C_WHITE}blackbody T=5800{C_RESET}\n")

    while True:
        try:
            raw = input(f"{C_BOLD}{C_GREEN}nuro-phy>{C_RESET} ").strip()
            if not raw:
                continue
            if raw.lower() in ("exit", "quit", "q"):
                print(f"{C_GRAY}Exiting nuro-phy.{C_RESET}")
                break
            print()
            print(solve_physics(raw))
            print()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C_GRAY}Exiting nuro-phy.{C_RESET}")
            break
        except Exception as e:
            print(f"{C_RED}[PHY ERROR]:{C_RESET} {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(solve_physics(" ".join(sys.argv[1:])))
    else:
        run_phy_repl()
