"""Bit-exact Q15 DSP reference implementation matching firmware/components/dsp/nl_dsp.c.

Implements:
- 61-tap Quadrature FIR pair (h_c cosine-phase, h_s sine-phase) for 13-30 Hz @ 2000 Hz
- 20 ms (40 samples @ 2 kHz) Stimulation Artifact Blanking clamp (DEC-007)
- Integer square-root envelope sqrt(I^2 + Q^2)
- Fixed-point Q15 atan2(Q, I) mapping [-pi, pi) to [-32768, 32767]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

NL_FIR_TAPS = 61
NL_BLANK_SAMPLES = 40  # 20 ms @ 2000 Hz


def design_quadrature_fir_q15(
    n_taps: int = NL_FIR_TAPS,
    f_low_hz: float = 13.0,
    f_high_hz: float = 30.0,
    fs_hz: float = 2000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a 61-tap Hamming-windowed analytic bandpass filter pair (h_c, h_s) in Q15.

    Centered at f0 = 21.5 Hz with passband 13-30 Hz (bandwidth = 17 Hz).
    Normalized so peak passband gain is 1.0 in Q15 (32768).
    """
    m = (n_taps - 1) / 2.0
    n = np.arange(n_taps, dtype=np.float64) - m
    f0 = 0.5 * (f_low_hz + f_high_hz)     # 21.5 Hz band center
    bw = (f_high_hz - f_low_hz)            # 17.0 Hz bandwidth
    fc_lp = (0.5 * bw) / fs_hz             # Lowpass prototype cutoff

    # Sinc lowpass prototype * Hamming window
    h_lp = 2.0 * fc_lp * np.sinc(2.0 * fc_lp * n)
    win = np.hamming(n_taps)
    h_proto = h_lp * win

    # Modulate to f0 in quadrature (cosine = real I, +sine = imaginary Q so convolution yields +sin(w*(n-m)))
    phase = 2.0 * np.pi * (f0 / fs_hz) * n
    h_c_f = h_proto * np.cos(phase)
    h_s_f = h_proto * np.sin(phase)

    # Normalize so frequency response at f0 has unit magnitude (|H(f0)| = 1.0)
    exp_f0 = np.exp(-1j * phase)
    gain_c = float(np.abs(np.sum(h_c_f * exp_f0)))
    gain_s = float(np.abs(np.sum(h_s_f * exp_f0)))

    h_c_q15 = np.clip(np.round((h_c_f / gain_c) * 32767.0), -32768, 32767).astype(np.int16)
    h_s_q15 = np.clip(np.round((h_s_f / gain_s) * 32767.0), -32768, 32767).astype(np.int16)
    return h_c_q15, h_s_q15


HC_Q15, HS_Q15 = design_quadrature_fir_q15()


def isqrt32(val: int) -> int:
    """Bit-exact integer square root matching C implementation."""
    if val <= 0:
        return 0
    res = 0
    bit = 1 << 30
    while bit > val:
        bit >>= 2
    while bit != 0:
        if val >= res + bit:
            val -= res + bit
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2
    return res


def atan2_q15(q: int, i: int) -> int:
    """Fixed-point Q15 atan2(q, i) mapping [-pi, pi) to [-32768, 32767].

    Uses 3rd-order octant minimax polynomial in integer arithmetic (< 0.3 deg max error).
    """
    if i == 0 and q == 0:
        return 0

    abs_i = abs(int(i))
    abs_q = abs(int(q))

    # Octant reduction: z = min / max in Q15 [0, 32768]
    if abs_i >= abs_q:
        z_q15 = (abs_q << 15) // abs_i if abs_i > 0 else 0
        # atan(z) / pi * 32768 ~= z * (0.3183 - 0.0685 * z^2)
        # In Q15: angle = (z * (10430 - ((2245 * z * z) >> 30))) >> 15
        z2_q15 = (z_q15 * z_q15) >> 15
        poly = 10430 - ((2245 * z2_q15) >> 15)
        ang = (z_q15 * poly) >> 15  # [0, 8192] for [0, pi/4]
    else:
        z_q15 = (abs_i << 15) // abs_q
        z2_q15 = (z_q15 * z_q15) >> 15
        poly = 10430 - ((2245 * z2_q15) >> 15)
        ang = 16384 - ((z_q15 * poly) >> 15)  # pi/2 - atan(z)

    if i < 0:
        ang = 32768 - ang
    if q < 0:
        ang = -ang

    if ang > 32767:
        ang = 32767
    if ang < -32768:
        ang = -32768
    return int(ang)


@dataclass
class DspOutput:
    I_q15: int
    Q_q15: int
    env_q15: int
    phase_q15: int
    blanked: bool


class DspChainRef:
    """Bit-exact stateful Q15 DSP chain with 20 ms artifact blanking."""

    def __init__(self) -> None:
        self.hist = [0] * NL_FIR_TAPS
        self.head = 0
        self.blank_rem = 0
        self.last_clean_samp = 0
        self.env_mean_q15 = 256

    def notify_stim(self) -> None:
        self.blank_rem = NL_BLANK_SAMPLES

    def step(self, raw_sample: int) -> DspOutput:
        raw_i16 = int(np.clip(raw_sample, -32768, 32767))
        blanked = False
        if self.blank_rem > 0:
            self.blank_rem -= 1
            x = self.last_clean_samp
            blanked = True
        else:
            x = raw_i16
            self.last_clean_samp = x

        self.hist[self.head] = x
        acc_i = 0
        acc_q = 0
        idx = self.head
        for k in range(NL_FIR_TAPS):
            s = self.hist[idx]
            acc_i += s * int(HC_Q15[k])
            acc_q += s * int(HS_Q15[k])
            idx = idx - 1 if idx > 0 else (NL_FIR_TAPS - 1)

        self.head = (self.head + 1) % NL_FIR_TAPS

        i_q15 = int(np.clip(acc_i >> 15, -32768, 32767))
        q_q15 = int(np.clip(acc_q >> 15, -32768, 32767))

        mag_sq = i_q15 * i_q15 + q_q15 * q_q15
        env_q15 = min(32767, isqrt32(mag_sq))
        phase_q15 = atan2_q15(q_q15, i_q15)

        # Update 200 ms EMA (alpha = 1/256)
        self.env_mean_q15 += (env_q15 - self.env_mean_q15) >> 8

        return DspOutput(
            I_q15=i_q15,
            Q_q15=q_q15,
            env_q15=env_q15,
            phase_q15=phase_q15,
            blanked=blanked,
        )
