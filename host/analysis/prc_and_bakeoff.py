"""Science Plan Engine: Empirical PRC Identification (12 phases x 20 seeds) & 5-Arm Controller Bake-Off."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

from host.virtualbrain.kuramoto import KuramotoBrain, KuramotoConfig


def identify_empirical_prc(
    n_phases: int = 12,
    n_seeds: int = 20,
    pulse_u: float = 600.0,
) -> Dict[str, Any]:
    """Measure the Phase-Response Curve (PRC) across 12 target phases x 20 seeds.

    At each candidate phase phi_k in [-pi, pi), delivers a single test pulse to a synchronized
    Kuramoto cluster (R ~= 0.80) and measures the immediate change delta_R = R_post - R_pre.
    Returns the empirically optimal desynchronizing phase phi_star = argmin_phi E[delta_R(phi)].
    """
    candidate_phases = np.linspace(-np.pi, np.pi, n_phases, endpoint=False)
    mean_delta_r: List[float] = []
    sd_delta_r: List[float] = []

    for phi_target in candidate_phases:
        deltas: List[float] = []
        for s in range(n_seeds):
            cfg = KuramotoConfig(seed=5000 + s, noise_sigma=0.05)
            brain = KuramotoBrain(cfg)
            # Initialize synchronized cluster centered at phi_target
            brain.theta = np.mod(
                brain.rng.normal(phi_target, 0.55, size=cfg.n_osc),
                2.0 * np.pi,
            )
            brain.step_idx = int(30.0 / cfg.dt_s)

            r_pre, _ = brain.order_parameter()
            _, r_post, _ = brain.step(stim_amp=pulse_u, inject_artifact=False)
            deltas.append(float(r_post - r_pre))

        mean_delta_r.append(float(np.mean(deltas)))
        sd_delta_r.append(float(np.std(deltas, ddof=1)))

    best_idx = int(np.argmin(mean_delta_r))
    worst_idx = int(np.argmax(mean_delta_r))

    return {
        "n_phases": n_phases,
        "n_seeds": n_seeds,
        "phases_rad": [float(p) for p in candidate_phases],
        "mean_delta_R": mean_delta_r,
        "sd_delta_R": sd_delta_r,
        "phi_star_rad": float(candidate_phases[best_idx]),
        "phi_star_delta_R": float(mean_delta_r[best_idx]),
        "phi_anti_rad": float(candidate_phases[worst_idx]),
        "phi_anti_delta_R": float(mean_delta_r[worst_idx]),
    }


def bootstrap_paired_ci(
    diff: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute mean difference and 95% percentile bootstrap CI over paired seeds."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = np.mean(diff[idx], axis=1)
    return (
        float(np.mean(diff)),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def run_five_arm_bakeoff(
    n_seeds: int = 20,
    n_steps: int = 2000,
    pulse_u: float = 600.0,
) -> Dict[str, Any]:
    """Run the 5 controller arms on identical paired seeds after empirical PRC identification.

    Arms:
    - A_none: No stimulation
    - B_open_loop: Fixed 20 Hz periodic pulses (every 100 steps = 50 ms)
    - C_sham: Pulse-matched random-phase stimulation
    - D_phase_locked: Phase-locked to empirical phi_star from PRC
    - E_phase_locked_armed: Phase-locked + pre-ictal envelope gating (reduces pulse burden)
    """
    prc = identify_empirical_prc(n_phases=12, n_seeds=n_seeds, pulse_u=pulse_u)
    phi_star = prc["phi_star_rad"]

    arms = ["A_none", "B_open_loop", "C_sham", "D_phase_locked", "E_phase_locked_armed"]
    r_final: Dict[str, List[float]] = {a: [] for a in arms}
    pulses_used: Dict[str, List[int]] = {a: [] for a in arms}

    def init_paired_brain(seed: int) -> KuramotoBrain:
        b = KuramotoBrain(KuramotoConfig(seed=seed))
        b.theta = np.mod(b.rng.normal(0.0, 0.55, size=b.cfg.n_osc), 2.0 * np.pi)
        b.step_idx = int(30.0 / b.cfg.dt_s)
        return b

    for s in range(n_seeds):
        seed = 1000 + s

        # Arm A: No stimulation
        ba = init_paired_brain(seed)
        ra = [ba.step(0.0)[1] for _ in range(n_steps)]
        r_final["A_none"].append(float(np.mean(ra[-400:])))
        pulses_used["A_none"].append(0)

        # Arm B: Open-loop 20 Hz (1 pulse every 100 steps)
        bb = init_paired_brain(seed)
        rb = []
        pb = 0
        for step_i in range(n_steps):
            u = pulse_u if (step_i % 100 == 0) else 0.0
            if u > 0:
                pb += 1
            rb.append(bb.step(u)[1])
        r_final["B_open_loop"].append(float(np.mean(rb[-400:])))
        pulses_used["B_open_loop"].append(pb)

        # Arm D: Phase-locked to empirically identified phi_star
        bd = init_paired_brain(seed)
        rd = []
        pd = 0
        refrac = 0
        for _ in range(n_steps):
            r_curr, psi_curr = bd.order_parameter()
            # Shortest angular distance between psi_curr and phi_star
            ang_diff = abs(np.angle(np.exp(1j * (psi_curr - phi_star))))
            u = 0.0
            if refrac == 0 and r_curr > 0.22 and ang_diff < 0.45:
                u = pulse_u
                pd += 1
                refrac = 8
            elif refrac > 0:
                refrac -= 1
            rd.append(bd.step(u)[1])
        r_final["D_phase_locked"].append(float(np.mean(rd[-400:])))
        pulses_used["D_phase_locked"].append(pd)

        # Arm C: Random-phase sham matched to Arm D's exact pulse count
        bc = init_paired_brain(seed)
        sham_rng = np.random.default_rng(seed + 7777)
        sham_steps = set(sham_rng.choice(range(n_steps), size=pd, replace=False))
        rc = []
        for step_i in range(n_steps):
            u = pulse_u if step_i in sham_steps else 0.0
            rc.append(bc.step(u)[1])
        r_final["C_sham"].append(float(np.mean(rc[-400:])))
        pulses_used["C_sham"].append(pd)

        # Arm E: Phase-locked + armed (only stimulates when R > 0.42 burst threshold, slashing pulse count)
        be = init_paired_brain(seed)
        re = []
        pe = 0
        refrac = 0
        for _ in range(n_steps):
            r_curr, psi_curr = be.order_parameter()
            ang_diff = abs(np.angle(np.exp(1j * (psi_curr - phi_star))))
            u = 0.0
            if refrac == 0 and r_curr > 0.42 and ang_diff < 0.40:
                u = pulse_u
                pe += 1
                refrac = 12
            elif refrac > 0:
                refrac -= 1
            re.append(be.step(u)[1])
        r_final["E_phase_locked_armed"].append(float(np.mean(re[-400:])))
        pulses_used["E_phase_locked_armed"].append(pe)

    arr_a = np.array(r_final["A_none"])
    arr_c = np.array(r_final["C_sham"])
    arr_d = np.array(r_final["D_phase_locked"])

    diff_a_d, ci_lo_ad, ci_hi_ad = bootstrap_paired_ci(arr_a - arr_d)
    diff_c_d, ci_lo_cd, ci_hi_cd = bootstrap_paired_ci(arr_c - arr_d)

    summary = {
        "prc": prc,
        "arms": {
            arm: {
                "R_mean": float(np.mean(r_final[arm])),
                "R_sd": float(np.std(r_final[arm], ddof=1)),
                "pulses_mean": float(np.mean(pulses_used[arm])),
            }
            for arm in arms
        },
        "contrasts": {
            "delta_R_A_minus_D": {
                "mean": diff_a_d,
                "ci95_low": ci_lo_ad,
                "ci95_high": ci_hi_ad,
            },
            "delta_R_C_minus_D": {
                "mean": diff_c_d,
                "ci95_low": ci_lo_cd,
                "ci95_high": ci_hi_cd,
            },
        },
    }
    return summary
