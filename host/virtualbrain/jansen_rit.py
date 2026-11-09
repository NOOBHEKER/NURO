"""Stage 2 Biophysical Virtual Brain: Jansen-Rit Cortical Neural Mass Model (NMM)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class JansenRitConfig:
    A: float = 3.25        # Excitatory synaptic gain (mV)
    B: float = 22.0        # Inhibitory synaptic gain (mV)
    a: float = 100.0       # Excitatory rate constant (s^-1)
    b: float = 50.0        # Inhibitory rate constant (s^-1)
    C: float = 135.0       # Base connectivity constant
    v0: float = 6.0        # PSP threshold (mV)
    nu_max: float = 2.5    # Half maximum firing rate (s^-1)
    r: float = 0.56        # Sigmoid steepness (mV^-1)
    dt_s: float = 0.0005   # 2 kHz step (500 us)
    scale_int16: float = 128.0
    artifact_amp_mv: float = 80.0
    artifact_tau_s: float = 0.005
    seed: int = 42


class JansenRitBrain:
    """Three-population cortical column ODE generator (pyramidal, excitatory, inhibitory)."""

    def __init__(self, cfg: Optional[JansenRitConfig] = None) -> None:
        self.cfg = cfg or JansenRitConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        # State vector: [y0, y1, y2, dy0, dy1, dy2]
        self.state = np.zeros(6, dtype=np.float64)
        self.step_idx = 0
        self.last_stim_t_s: Optional[float] = None

    def sigmoid(self, v: float) -> float:
        c = self.cfg
        return float((2.0 * c.nu_max) / (1.0 + np.exp(c.r * (c.v0 - v))))

    def derivatives(self, y: np.ndarray, p_ext: float, u_stim: float) -> np.ndarray:
        c = self.cfg
        c1 = c.C
        c2 = 0.8 * c.C
        c3 = 0.25 * c.C
        c4 = 0.25 * c.C

        y0, y1, y2, dy0, dy1, dy2 = y
        ddy0 = c.A * c.a * self.sigmoid(y1 - y2) - 2.0 * c.a * dy0 - (c.a ** 2) * y0
        ddy1 = (
            c.A * c.a * (p_ext + c2 * self.sigmoid(c1 * y0) + u_stim)
            - 2.0 * c.a * dy1
            - (c.a ** 2) * y1
        )
        ddy2 = (
            c.B * c.b * (c4 * self.sigmoid(c3 * y0))
            - 2.0 * c.b * dy2
            - (c.b ** 2) * y2
        )
        return np.array([dy0, dy1, dy2, ddy0, ddy1, ddy2], dtype=np.float64)

    def step(self, u_stim: float = 0.0, hyperexcitable: bool = True, inject_artifact: bool = True) -> Tuple[int, float]:
        """Integrate one 500 us step via 4th-order Runge-Kutta (RK4).

        Returns:
            (raw_sample_i16, clean_lfp_mv)
        """
        c = self.cfg
        dt = c.dt_s
        t_s = self.step_idx * dt

        if u_stim != 0.0:
            self.last_stim_t_s = t_s

        # Extrinsic cortical input p(t)
        p_mean = 220.0 if hyperexcitable else 150.0
        p_ext = float(self.rng.normal(p_mean, 22.0))

        # RK4 integration
        k1 = self.derivatives(self.state, p_ext, u_stim)
        k2 = self.derivatives(self.state + 0.5 * dt * k1, p_ext, u_stim)
        k3 = self.derivatives(self.state + 0.5 * dt * k2, p_ext, u_stim)
        k4 = self.derivatives(self.state + dt * k3, p_ext, u_stim)
        self.state = self.state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.step_idx += 1

        clean_lfp_mv = float(self.state[1] - self.state[2])
        observed_mv = clean_lfp_mv

        if inject_artifact and self.last_stim_t_s is not None:
            dt_stim = t_s - self.last_stim_t_s
            if 0.0 <= dt_stim <= 0.025:
                observed_mv += c.artifact_amp_mv * float(np.exp(-dt_stim / c.artifact_tau_s))

        raw_i16 = int(np.clip(np.round(observed_mv * c.scale_int16), -32768, 32767))
        return raw_i16, clean_lfp_mv

    def trajectory_hash(self) -> str:
        """Return SHA-256 hash of state vector for Gate G4 determinism check."""
        return hashlib.sha256(self.state.tobytes()).hexdigest()
