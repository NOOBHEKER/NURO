"""Stage 1 Analytical Virtual Brain: Kuramoto coupled phase oscillators + artifact injection."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class KuramotoConfig:
    n_osc: int = 200
    f0_hz: float = 20.0
    sigma_f_hz: float = 0.35  # Critical coupling Kc = sqrt(8/pi)*2*pi*0.35 ~= 3.5 rad/s (so K=0.5 is healthy, K=6.0 is ictal)
    dt_s: float = 0.0005      # 2 kHz step (500 us)
    noise_sigma: float = 0.2
    scale_int16: float = 1024.0
    artifact_amp: float = 8.0       # 8x full-scale LFP amplitude
    artifact_tau_s: float = 0.005   # 5 ms capacitive decay
    seed: int = 42


class KuramotoBrain:
    """Deterministic Kuramoto network with scheduled ictal transitions and PRC coupling."""

    def __init__(self, cfg: Optional[KuramotoConfig] = None) -> None:
        self.cfg = cfg or KuramotoConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.omega = self.rng.normal(
            2.0 * np.pi * self.cfg.f0_hz,
            2.0 * np.pi * self.cfg.sigma_f_hz,
            size=self.cfg.n_osc,
        )
        self.theta = self.rng.uniform(0.0, 2.0 * np.pi, size=self.cfg.n_osc)
        self.step_idx = 0
        self.last_stim_t_s: Optional[float] = None

    @staticmethod
    def scheduled_coupling_k(t_s: float) -> float:
        """Deterministic 50 s seizure schedule: Baseline -> Ramp -> Ictal -> Recovery."""
        if t_s < 10.0:
            return 0.5
        if t_s < 30.0:
            return 0.5 + (6.0 - 0.5) * ((t_s - 10.0) / 20.0)
        if t_s < 40.0:
            return 6.0
        if t_s < 50.0:
            return 6.0 - (6.0 - 0.5) * ((t_s - 40.0) / 10.0)
        return 0.5

    def order_parameter(self) -> Tuple[float, float]:
        """Compute Kuramoto order parameter R in [0, 1] and mean phase psi in [-pi, pi]."""
        z = np.mean(np.exp(1j * self.theta))
        return float(np.abs(z)), float(np.angle(z))

    def step(self, stim_amp: float = 0.0, inject_artifact: bool = True) -> Tuple[int, float, float]:
        """Integrate one 500 us step using O(N) mean-field form.

        Returns:
            (raw_sample_i16, ground_truth_R, ground_truth_psi)
        """
        dt = self.cfg.dt_s
        t_s = self.step_idx * dt
        R, psi = self.order_parameter()
        K = self.scheduled_coupling_k(t_s)

        # PRC: Z(theta) = -sin(theta)
        if stim_amp != 0.0:
            self.last_stim_t_s = t_s

        prc = -np.sin(self.theta)
        noise = self.rng.normal(0.0, 1.0, size=self.cfg.n_osc) * np.sqrt(dt) * self.cfg.noise_sigma
        dtheta = (self.omega + K * R * np.sin(psi - self.theta) + prc * stim_amp) * dt + noise
        self.theta = np.mod(self.theta + dtheta, 2.0 * np.pi)
        self.step_idx += 1

        # Observed LFP proxy (never reveals R directly; R is ground truth only)
        lfp = float(np.mean(np.cos(self.theta)))
        if inject_artifact and self.last_stim_t_s is not None:
            dt_stim = t_s - self.last_stim_t_s
            if 0.0 <= dt_stim <= 0.025:
                lfp += self.cfg.artifact_amp * float(np.exp(-dt_stim / self.cfg.artifact_tau_s))

        raw_i16 = int(np.clip(np.round(lfp * self.cfg.scale_int16), -32768, 32767))
        return raw_i16, R, psi

    def trajectory_hash(self) -> str:
        """Return SHA-256 hash of current oscillator phases for Gate G4 determinism check."""
        return hashlib.sha256(self.theta.tobytes()).hexdigest()
