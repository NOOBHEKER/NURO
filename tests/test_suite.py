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

    # Save golden vector
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / "crc16_sample.json"
    golden_data = {
        "frame_hex": raw.hex(),
        "crc16_hex": f"0x{crc16_ccitt_false(raw[:28]):04X}",
        "seq": frame.seq,
        "samples": list(frame.samples),
    }
    golden_path.write_text(json.dumps(golden_data, indent=2), encoding="utf-8")

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
    n_steps = 2000   # 1.0 s window at 2 kHz

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
        sham_steps = set(sham_rng.choice(range(n_steps), size=pulses_fired, replace=False))
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
    assert (mean_a - mean_d) > 0.20, f"Expected delta_R(A-D) > 0.20, got {mean_a - mean_d:.3f} (A={mean_a:.3f}, D={mean_d:.3f})"
    assert (mean_c - mean_d) > 0.10, f"Expected delta_R(C-D) > 0.10, got {mean_c - mean_d:.3f} (C={mean_c:.3f}, D={mean_d:.3f})"


def test_windows_timer_helper() -> None:
    assert enable_windows_1ms_timer() is True
