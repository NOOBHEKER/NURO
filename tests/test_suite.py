"""Automated verification suite for Gates G1 (Protocol & CRC16), G4 (Determinism), and G7 (Sham vs Phase-Locked)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from host.hil.protocol import (
    SampleFrame,
    ResultFrame,
    crc16_ccitt_false,
    enable_windows_1ms_timer,
    MODE_LOCKSTEP,
    STATE_ARMED,
)
from host.hil.dsp_ref import DspChainRef, atan2_q15, isqrt32
from host.analysis.prc_and_bakeoff import identify_empirical_prc, run_five_arm_bakeoff
from host.analysis.figures import generate_all_figures
from host.dash.app import render_dashboard_snapshot
from host.dash.tui import visible_len, pad_ansi, render_braille_plot
from host.dash.web_cockpit import TuiCockpitEngine
from host.replay.predictor_eval import (
    evaluate_gate_g5_detector,
    evaluate_gate_g6_predictor,
)
from host.virtualbrain.kuramoto import KuramotoBrain, KuramotoConfig
from host.virtualbrain.jansen_rit import JansenRitBrain, JansenRitConfig


GOLDEN_DIR = Path(__file__).parent / "golden"


def test_crc16_ccitt_false_standard_vector() -> None:
    """Gate G1: Standard CCITT-FALSE check value for ASCII '123456789' is 0x29B1."""
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_sample_and_result_frame_binary_contract() -> None:
    """Gate G1: 32-byte exact packing, little-endian layout, and CRC rejection."""
    frame = SampleFrame(
        seq=42,
        sim_t_us=21_000,
        host_t_us=21_050,
        samples=(512, -512, 1024, -1024),
        flags=0x01,
        n_ch=2,
        n_samp=2,
        mode=MODE_LOCKSTEP,
    )
    raw = frame.to_bytes()
    assert len(raw) == 32

    decoded = SampleFrame.from_bytes(raw)
    assert decoded == frame

    # Compare against the checked-in golden vector. Tests must never rewrite it.
    golden_path = GOLDEN_DIR / "crc16_sample.json"
    assert golden_path.exists(), f"Missing golden vector: {golden_path}"
    golden_data = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden_data["frame_hex"] == raw.hex()
    assert golden_data["crc16_hex"] == f"0x{crc16_ccitt_false(raw[:28]):04X}"
    assert golden_data["seq"] == frame.seq
    assert golden_data["samples"] == list(frame.samples)

    # Flip 1 bit in payload -> must raise ValueError on CRC mismatch
    corrupted = bytearray(raw)
    corrupted[20] ^= 0x01
    with pytest.raises(ValueError, match="CRC mismatch"):
        SampleFrame.from_bytes(bytes(corrupted))

    res = ResultFrame(
        state=STATE_ARMED,
        seq=42,
        t_rx_us=1000,
        t_dsp_us=1045,
        t_dec_us=1140,
        t_stim_us=1152,
        cls=1,
        conf=230,
        phase_q15=-8192,
        R_q15=24576,
        budget_used=7,
    )
    res_raw = res.to_bytes()
    assert len(res_raw) == 32
    assert ResultFrame.from_bytes(res_raw) == res


def test_gate_g4_twin_run_determinism() -> None:
    """Gate G4: Twin runs with identical seeds must produce bit-identical state hashes."""
    # Stage 1: Kuramoto
    b1 = KuramotoBrain(KuramotoConfig(seed=2026))
    b2 = KuramotoBrain(KuramotoConfig(seed=2026))
    for _ in range(500):
        s1, r1, _ = b1.step(stim_amp=0.0)
        s2, r2, _ = b2.step(stim_amp=0.0)
        assert s1 == s2
        assert r1 == r2
    assert b1.trajectory_hash() == b2.trajectory_hash()

    # Stage 2: Jansen-Rit Neural Mass Model
    j1 = JansenRitBrain(JansenRitConfig(seed=2026))
    j2 = JansenRitBrain(JansenRitConfig(seed=2026))
    for _ in range(500):
        y1, lfp1 = j1.step(u_stim=0.0)
        y2, lfp2 = j2.step(u_stim=0.0)
        assert y1 == y2
        assert lfp1 == lfp2
    assert j1.trajectory_hash() == j2.trajectory_hash()


def test_gate_g7_phase_locked_beats_sham_and_no_stim() -> None:
    """Gate G7: Phase-locked stimulation derived from PRC must beat random-phase sham on paired seeds.

    By Ott-Antonsen reduction for Z(theta) = -sin(theta):
        dR/dt|_stim = (u(t) / 2) * (1 - R^2) * cos(psi)
    Therefore, positive pulses u(t) > 0 maximally desynchronize (dR/dt < 0) when cos(psi) = -1 (psi = +/- pi),
    whereas random-phase pulses average E[cos(psi)] = 0 and fail to collapse synchrony.
    """
    r_no_stim = []
    r_sham = []
    r_locked = []

    pulse_u = 600.0  # 600 rad/s * 0.5 ms = 0.30 rad (17.2 deg) per pulse
    n_steps = 2000  # 1.0 s window at 2 kHz

    def make_ictal_brain(seed: int) -> KuramotoBrain:
        b = KuramotoBrain(KuramotoConfig(seed=seed))
        # Initialize in synchronized ictal cluster (R ~= 0.82) at t = 30.0 s (K = 6.0)
        b.theta = np.mod(b.rng.normal(0.0, 0.60, size=b.cfg.n_osc), 2.0 * np.pi)
        b.step_idx = int(30.0 / b.cfg.dt_s)
        return b

    for seed in range(100, 105):
        # Arm A: No stimulation
        brain_a = make_ictal_brain(seed)
        r_trace_a = []
        for _ in range(n_steps):
            _, r_val, _ = brain_a.step(stim_amp=0.0)
            r_trace_a.append(r_val)
        r_no_stim.append(float(np.mean(r_trace_a[-400:])))

        # Arm D: Phase-locked stimulation at optimal desynchronizing phase psi ~= +/- pi (cos(psi) < -0.85)
        brain_d = make_ictal_brain(seed)
        r_trace_d = []
        pulses_fired = 0
        refractory = 0
        for _ in range(n_steps):
            r_curr, psi_curr = brain_d.order_parameter()
            stim = 0.0
            if refractory == 0 and r_curr > 0.20 and np.cos(psi_curr) < -0.85:
                stim = pulse_u
                pulses_fired += 1
                refractory = 8  # 4 ms refractory
            elif refractory > 0:
                refractory -= 1
            _, r_val, _ = brain_d.step(stim_amp=stim)
            r_trace_d.append(r_val)
        r_locked.append(float(np.mean(r_trace_d[-400:])))

        # Arm C: Random-phase sham with the EXACT same number of pulses as Arm D
        brain_c = make_ictal_brain(seed)
        sham_rng = np.random.default_rng(seed + 999)
        sham_steps = set(
            sham_rng.choice(range(n_steps), size=pulses_fired, replace=False)
        )
        r_trace_c = []
        for step_i in range(n_steps):
            stim = pulse_u if step_i in sham_steps else 0.0
            _, r_val, _ = brain_c.step(stim_amp=stim)
            r_trace_c.append(r_val)
        r_sham.append(float(np.mean(r_trace_c[-400:])))

    mean_a = float(np.mean(r_no_stim))
    mean_c = float(np.mean(r_sham))
    mean_d = float(np.mean(r_locked))

    # Pre-registered Gate G7 pass conditions:
    # 1. Phase-locked reduces R by > 0.20 vs No-Stim
    # 2. Phase-locked beats Random-Phase Sham by > 0.10
    assert (mean_a - mean_d) > 0.20, (
        f"Expected delta_R(A-D) > 0.20, got {mean_a - mean_d:.3f} (A={mean_a:.3f}, D={mean_d:.3f})"
    )
    assert (mean_c - mean_d) > 0.10, (
        f"Expected delta_R(C-D) > 0.10, got {mean_c - mean_d:.3f} (C={mean_c:.3f}, D={mean_d:.3f})"
    )


def test_windows_timer_helper() -> None:
    assert enable_windows_1ms_timer() is True


def test_gate_g3_dsp_quadrature_phase_and_artifact_blanking() -> None:
    """Gate G3: Verify 61-tap Q15 quadrature bandpass, phase accuracy < 3 deg, and 20 ms artifact blanking."""
    dsp = DspChainRef()
    fs = 2000.0
    f_test = 20.0
    amp = 2000.0

    # 1. Feed 500 samples (250 ms) of clean 20 Hz cosine and verify envelope & phase linearity
    phases = []
    envs = []
    outputs_golden = []
    for n in range(1000):
        t = n / fs
        raw = int(np.round(amp * np.cos(2.0 * np.pi * f_test * t)))
        # At sample 600, inject a massive +30000 LSB stimulation artifact spike right after notify_stim()
        if n == 600:
            dsp.notify_stim()
        if 600 <= n < 615:
            raw = 30000  # 15x full-scale artifact transient

        out = dsp.step(raw)
        if 200 <= n < 500:
            envs.append(out.env_q15)
            phases.append(out.phase_q15)
        if n < 200:
            outputs_golden.append(
                {
                    "n": n,
                    "raw": raw,
                    "I_q15": out.I_q15,
                    "Q_q15": out.Q_q15,
                    "env_q15": out.env_q15,
                    "phase_q15": out.phase_q15,
                }
            )
        if 600 <= n < 640:
            assert out.blanked is True, (
                f"Sample {n} must be blanked during 40-sample (20 ms) window"
            )
            assert out.env_q15 < 3500, (
                f"Artifact leaked into envelope at sample {n}: {out.env_q15}"
            )
        elif n >= 640:
            assert out.blanked is False

    mean_env = float(np.mean(envs))
    # Gain at 20 Hz should be within 5% of unity (2000 LSB)
    assert 1850.0 <= mean_env <= 2150.0, (
        f"Expected passband envelope ~2000, got {mean_env:.1f}"
    )

    # Verify phase progression per sample at 20 Hz: d_phase = 20 / 2000 * 65536 = 655.36 Q15 units/sample
    d_phase = np.diff(np.unwrap(np.array(phases, dtype=np.float64) * np.pi / 32768.0))
    measured_freq_hz = float(np.mean(d_phase)) * fs / (2.0 * np.pi)
    assert abs(measured_freq_hz - 20.0) < 0.15, (
        f"Expected 20.0 Hz phase slope, got {measured_freq_hz:.3f} Hz"
    )

    # Compare against the checked-in DSP fixture (read-only; tests never rewrite goldens).
    golden_path = GOLDEN_DIR / "dsp_fixture.json"
    assert golden_path.exists(), f"Missing golden fixture: {golden_path}"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert outputs_golden[:50] == golden


def test_gate_g8_prc_identification_and_five_arm_bakeoff() -> None:
    """Gate G8: Empirical PRC identification (12 phases) + 5-arm bakeoff with 95% bootstrap CIs."""
    res = run_five_arm_bakeoff(n_seeds=8, n_steps=1600, pulse_u=600.0)

    # 1. Empirical PRC must identify optimal phase near +/- pi (where cos(phi*) < -0.8)
    phi_star = res["prc"]["phi_star_rad"]
    assert np.cos(phi_star) < -0.80, (
        f"Expected optimal PRC phase near pi, got {phi_star:.3f} rad"
    )
    assert res["prc"]["phi_star_delta_R"] < -0.05

    # 2. 95% Bootstrap CIs for (A - D) and (C - D) must strictly exclude 0
    ad = res["contrasts"]["delta_R_A_minus_D"]
    cd = res["contrasts"]["delta_R_C_minus_D"]
    assert ad["ci95_low"] > 0.20, f"CI for A-D must exclude 0 (>0.20), got {ad}"
    assert cd["ci95_low"] > 0.10, f"CI for C-D must exclude 0 (>0.10), got {cd}"

    # 3. Armed phase-locked (Arm E) must use fewer pulses than unarmed (Arm D) while still collapsing R
    pulses_d = res["arms"]["D_phase_locked"]["pulses_mean"]
    pulses_e = res["arms"]["E_phase_locked_armed"]["pulses_mean"]
    assert pulses_e < pulses_d, (
        f"Armed controller (E={pulses_e}) must reduce pulse burden vs D={pulses_d}"
    )

    # Compare the reproducible summary against the checked-in golden file (read-only).
    golden_path = GOLDEN_DIR / "five_arm_bakeoff.json"
    assert golden_path.exists(), f"Missing golden bakeoff: {golden_path}"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert res["prc"]["phi_star_rad"] == golden["prc"]["phi_star_rad"]
    assert res["prc"]["phi_star_delta_R"] == golden["prc"]["phi_star_delta_R"]
    assert res["contrasts"] == golden["contrasts"]


def test_gate_g5_and_g6_detector_and_predictor() -> None:
    """Gates G5 & G6 (SYNTHETIC SURROGATE): Int8 LIF detector + int8 readout pre-ictal predictor.

    These gates exercise the pipeline on signals generated inside the repo. They
    are not CHB-MIT results and must not be reported as clinical evidence.
    """
    g5 = evaluate_gate_g5_detector()
    assert g5["auroc_int8_lif_snn"] >= 0.95
    assert g5["auroc_int8_lif_snn"] > g5["auroc_amplitude_baseline"]
    assert g5["auroc_int8_lif_snn"] > g5["auroc_envelope_z_baseline"]
    assert g5["resting_active_firing_pct"] < 15.0
    assert g5["resting_spike_sparsity_pct"] > 85.0

    g6 = evaluate_gate_g6_predictor()
    assert g6["split_verified_disjoint"] is True
    for pid, pinfo in g6["per_patient"].items():
        assert pinfo["auroc_int8_cnn"] >= 0.95, (
            f"Patient {pid} AUROC too low: {pinfo['auroc_int8_cnn']}"
        )
        assert pinfo["auroc_int8_cnn"] > pinfo["auroc_persistence_baseline"]
        assert pinfo["median_lead_time_s"] >= 30.0
    assert g6["summary"]["mean_fa_per_hour"] <= 0.50


def test_gate_g10_reproducible_figures_and_dashboard(tmp_path: Path) -> None:
    """Gate G10: One command regenerates all 9 publication figures (PNG + PDF) and the dashboard.

    Everything is written under pytest's ``tmp_path`` so a test run never dirties
    the tracked ``docs/figures`` tree.
    """
    fig_files = generate_all_figures(out_dir=tmp_path / "figures")
    assert len(fig_files) == 18
    for f in fig_files:
        assert f.exists() and f.stat().st_size > 2048, f"Missing or empty figure: {f}"

    dash_file = render_dashboard_snapshot(
        show_gui=False, out_path=tmp_path / "dashboard_hero.png"
    )
    assert dash_file.exists() and dash_file.stat().st_size > 10_000


def test_tui_and_web_cockpit_engines() -> None:
    """Verify ANSI-aware TUI alignment, Unicode Braille oscilloscope, and TuiCockpitEngine CLI shell."""
    colored = "\033[38;5;78mLIVE ESP32-S3\033[0m"
    assert visible_len(colored) == 13
    padded = pad_ansi(colored, 20)
    assert visible_len(padded) == 20

    braille = render_braille_plot(
        [0.85, 0.70, 0.50, 0.30, 0.20], 0.0, 1.0, cols=20, rows=3, threshold_y=0.25
    )
    assert len(braille) == 3
    for b_row in braille:
        assert visible_len(b_row) == 20

    eng = TuiCockpitEngine(port="COM_NONEXISTENT")
    eng.step_batch(10)
    eng.execute_command("cli:loop off")
    assert eng.loop_on is False
    eng.execute_command("cli:arm sham")
    assert eng.arm_idx == 1
    eng.execute_command("cli:model jansen")
    assert eng.model_idx == 2
    eng.execute_command("cli:burst")
    snap = eng.snapshot()
    assert "hex_inspector" in snap and len(snap["raw"]) == 360


def test_scientific_calculator_and_graph() -> None:
    """Verify natural-language scientific calculator, derivatives, integrals, and Braille plot."""
    from host.calc import calculate_and_graph

    res1 = calculate_and_graph("sqrt of 16 times pi")
    assert "12.56637" in res1

    res2 = calculate_and_graph("x^3 - 3*x")
    assert "f'(x)" in res2
    assert "∫ f(x) dx" in res2
    assert "GRAPH:" in res2

    res3 = calculate_and_graph("x^2 - 4*x - 5 = 0")
    assert "SOLUTIONS FOR x" in res3

    # Normal arithmetic signs: *, +, -, /
    res4 = calculate_and_graph("5 * 3 + 10")
    assert "25" in res4
    res5 = calculate_and_graph("100 / 4 - 5")
    assert "20" in res5

    # Direct quadratic shortcut inside terminal
    res_quad = calculate_and_graph("quad 1 -5 6")
    assert "3.0000" in res_quad and "2.0000" in res_quad


def test_chemistry_balancer_and_numericals() -> None:
    """Verify chemical equation balancing, catalyst c support, molar mass, and pH/kinetics solvers."""
    from host.chem import (
        balance_chemical_equation,
        compute_molar_mass,
        solve_chem_numerical,
    )

    balanced = balance_chemical_equation("KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2")
    assert "2KMnO4 + 16HCl ➔ 2KCl + 2MnCl2 + 8H2O + 5Cl2" in balanced
    assert "Potassium Permanganate" in balanced
    assert "Hydrochloric Acid" in balanced

    # Catalyst c support + Activation energy curve
    cat_res = balance_chemical_equation("2SO2 + O2 -> 2SO3 [c=V2O5]")
    assert "2SO2 + O2 ──[V2O5]──➔ 2SO3" in cat_res
    assert "CATALYST ACTIVE" in cat_res
    assert "Vanadium(V) Oxide" in cat_res
    assert "Uncatalyzed Path" in cat_res

    mass, pcts = compute_molar_mass("CuSO4*5H2O")
    assert abs(mass - 249.677) < 0.1
    assert "Cu" in pcts and "O" in pcts

    ph_res = solve_chem_numerical("ph 0.01 M HCl")
    assert "pH = -log10[H+]" in ph_res
    assert "Titration" in ph_res

    kin_res = solve_chem_numerical("kinetics k=0.08")
    assert "Half-Life" in kin_res


def test_obsidian_graph_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify directive extraction, vault resolution, and sessions — fully sandboxed.

    The real Obsidian app is never launched and the user's session state and
    ``G:\\obsidian`` vault are never touched.
    """
    import host.obsidian_sync as obs
    import host.session_manager as sess

    # Test explicit vault directive extraction
    q_clean, vault = obs.extract_obsidian_directive(
        "2SO2 + O2 -> 2SO3 to obsidian Acids, Bases and Salts"
    )
    assert q_clean == "2SO2 + O2 -> 2SO3"
    assert vault == "Acids, Bases and Salts"

    monkeypatch.setattr(obs, "OBSIDIAN_ROOT", tmp_path / "obsidian")
    monkeypatch.setattr(obs, "OBSIDIAN_EXE", tmp_path / "does-not-exist.exe")
    monkeypatch.setattr(obs.webbrowser, "open", lambda *_a, **_k: True)

    v_path, v_name = obs.resolve_vault_path("ICSE Class 10")
    assert v_name == "ICSE Class 10"
    assert v_path == tmp_path / "obsidian" / "ICSE Class 10"
    assert v_path.exists()

    out = obs.launch_obsidian_vault()
    assert "OBSIDIAN VAULT SYNCED" in out

    # Session switching against a temporary state file.
    monkeypatch.setattr(sess, "STATE_FILE", tmp_path / "session_state.json")
    sess.set_active_session("Physics-Exam-Prep")
    assert sess.get_active_session() == "Physics-Exam-Prep"
    assert sess.load_state()["active_session"] == "Physics-Exam-Prep"


def test_physics_simulation_and_constants() -> None:
    """Verify projectile motion, RLC ringing, relativity, Bohr quantum model, Pascal's law, and constants."""
    from host.phy import solve_physics, CONSTANTS

    assert "c" in CONSTANTS and abs(CONSTANTS["c"][0] - 299792458.0) < 1.0

    # Pascal's Law conceptual theory & numerical solver
    pascal = solve_physics("pascals law F1=50 A1=0.01 A2=0.5")
    assert "PASCAL'S LAW" in pascal
    assert "Mechanical Advantage" in pascal
    assert "Output Lift Force F₂" in pascal
    assert "Diameter Ratio" in pascal

    proj = solve_physics("projectile v=50 angle=45")
    assert "Time of Flight" in proj
    assert "Maximum Altitude H" in proj
    assert "Trajectory y(x)" in proj

    rlc = solve_physics("rlc R=5 L=0.05 C=0.0001")
    assert "Underdamped Ringing" in rlc
    assert "Capacitor Voltage" in rlc

    rel = solve_physics("relativity v=0.9c")
    assert "Lorentz Factor γ" in rel
    assert "2.294" in rel

    bohr = solve_physics("bohr n1=2 n2=3")
    assert "Balmer" in bohr
    assert "656" in bohr
