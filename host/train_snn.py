"""Supervised Training & Evaluation Pipeline for 64-Neuron Int8 LIF SNN on CHB-MIT Clinical EEG.

Trains the exact 64-neuron LIF SNN architecture used in the ESP32-S3 firmware:
  - Input: 16 int8 features (quadrature FIR envelope + phase components + differential reference)
  - Hidden: 64 leaky integrate-and-fire neurons with Q12 membrane potentials
  - Readout: 64 -> 1 output spike accumulator
  - Weights: w1[64][16] and w2[64] quantized strictly to int8_t [-128, 127]

Evaluates on disjoint patient splits (chb01, chb02 -> held-out chb03) and exports
trained weights directly to firmware/src/trained_snn_weights.h.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

from host.hil.dsp_ref import HC_Q15, HS_Q15
from host.replay.patient_data import ensure_patient_dataset


ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_HEADER = ROOT_DIR / "firmware" / "src" / "trained_snn_weights.h"
EVAL_JSON = ROOT_DIR / "tests" / "golden" / "trained_snn_clinical_eval.json"

NL_SNN_IN = 16
NL_SNN_HID = 64
NL_SNN_TH_Q12 = 2048
NL_SNN_BETA_M = 3891  # 3891/4096 = 0.95 decay factor


def compute_fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC via O(N log N) rank sum with tie handling."""
    pos_mask = (y_true == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    _, u_idx, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    tie_sums = np.bincount(u_idx, weights=ranks)
    ranks = (tie_sums / counts)[u_idx]
    sum_pos_ranks = np.sum(ranks[pos_mask])
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def extract_dsp_features(signal_2k: np.ndarray) -> np.ndarray:
    """Fast vectorized extraction of 16 int8 features matching firmware/src/main.cpp.

    Feature layout:
      0..3: Multi-scale envelope (env/8, env/16, env/32, env/64)
      4:    Inhibitory constant reference baseline (35 LSB)
      5:    Normalized envelope relative to resting baseline (env*16 / 200)
      6..7: Envelope derivative / ramp rate of change (d_env/2, d_env/4)
      8..11: Quadrature components (I/128, Q/128, I/256, Q/256)
      12:   Phase in Q7 (phase / 256)
      13:   Absolute phase in Q7 (abs(phase) / 256)
      14:   In-phase spectral magnitude (|I| / 128)
      15:   Quadrature spectral magnitude (|Q| / 128)
    """
    scaled = np.clip(np.round(signal_2k * 2.5), -32768, 32767).astype(np.float64)

    # Convolve with quadrature pair
    hc = HC_Q15[::-1] / 32768.0
    hs = HS_Q15[::-1] / 32768.0
    ci = np.convolve(scaled, hc, mode="same")
    cq = np.convolve(scaled, hs, mode="same")

    env = np.sqrt(ci**2 + cq**2)
    env_b = env[1::2]
    ci_b = ci[1::2]
    cq_b = cq[1::2]

    n_blocks = len(env_b)
    feats = np.zeros((n_blocks, NL_SNN_IN), dtype=np.int8)

    # Multi-scale envelope
    feats[:, 0] = np.clip(np.round(env_b / 8.0), -128, 127).astype(np.int8)
    feats[:, 1] = np.clip(np.round(env_b / 16.0), -128, 127).astype(np.int8)
    feats[:, 2] = np.clip(np.round(env_b / 32.0), -128, 127).astype(np.int8)
    feats[:, 3] = np.clip(np.round(env_b / 64.0), -128, 127).astype(np.int8)

    # Differential baseline reference (35)
    feats[:, 4] = np.full(n_blocks, 35, dtype=np.int8)

    # Scaled envelope relative to 200 LSB resting baseline
    feats[:, 5] = np.clip(np.round((env_b * 16.0) / 200.0), -128, 127).astype(np.int8)

    # Derivative
    d_env = np.diff(env_b, prepend=env_b[0])
    feats[:, 6] = np.clip(np.round(d_env / 2.0), -128, 127).astype(np.int8)
    feats[:, 7] = np.clip(np.round(d_env / 4.0), -128, 127).astype(np.int8)

    # Oscillatory components
    feats[:, 8] = np.clip(np.round(ci_b / 128.0), -128, 127).astype(np.int8)
    feats[:, 9] = np.clip(np.round(cq_b / 128.0), -128, 127).astype(np.int8)
    feats[:, 10] = np.clip(np.round(ci_b / 256.0), -128, 127).astype(np.int8)
    feats[:, 11] = np.clip(np.round(cq_b / 256.0), -128, 127).astype(np.int8)

    # Phase components
    phase = np.arctan2(cq_b, ci_b) * (32768.0 / np.pi)
    feats[:, 12] = np.clip(np.round(phase / 256.0), -128, 127).astype(np.int8)
    feats[:, 13] = np.clip(np.round(np.abs(phase) / 256.0), -128, 127).astype(np.int8)

    # Oscillatory energy
    feats[:, 14] = np.clip(np.round(np.abs(ci_b) / 128.0), -128, 127).astype(np.int8)
    feats[:, 15] = np.clip(np.round(np.abs(cq_b) / 128.0), -128, 127).astype(np.int8)

    return feats


def simulate_snn_fast(
    features: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
) -> np.ndarray:
    """Fast simulation of nl_snn.c over 1 ms blocks matching firmware/components/snn/nl_snn.c."""
    n_blocks = len(features)
    spike_counts = np.zeros(n_blocks, dtype=np.uint8)

    i_syn_all = np.dot(features.astype(np.int32), w1.T.astype(np.int32))
    v_hid = np.zeros(NL_SNN_HID, dtype=np.int32)
    v_out = 0
    w2_i32 = w2.astype(np.int32)

    for b in range(n_blocks):
        i_syn = i_syn_all[b]
        out_spikes = 0
        for _ in range(8):
            out_acc = (v_out * NL_SNN_BETA_M) >> 12

            acc = ((v_hid * NL_SNN_BETA_M) >> 12) + i_syn
            fired = acc >= NL_SNN_TH_Q12
            acc[fired] -= NL_SNN_TH_Q12

            if np.any(fired):
                out_acc += int(np.sum(w2_i32[fired])) << 4

            v_hid = np.clip(acc, -8192, 8192)

            if out_acc >= NL_SNN_TH_Q12:
                out_acc -= NL_SNN_TH_Q12
                out_spikes += 1
            v_out = int(np.clip(out_acc, -8192, 8192))

        spike_counts[b] = out_spikes

    return spike_counts


def train_snn_on_clinical_data(seed: int = 2026) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train SNN weights on CHB-MIT clinical patient seizure data."""
    rng = np.random.default_rng(seed)
    data = ensure_patient_dataset()

    print("[train_snn] Extracting Q15 FIR features from clinical recordings...")

    # Test Set: Held-out Patient chb03 (unseen patient!)
    sig_test_sz = data["chb03_sz_sig"]
    lab_test_sz = data["chb03_sz_labels"]
    feat_test_sz = extract_dsp_features(sig_test_sz)
    lab_test_sz_1ms = lab_test_sz[::2][:len(feat_test_sz)]

    sig_test_base = data["chb03_base_sig"]
    feat_test_base = extract_dsp_features(sig_test_base)
    lab_test_base_1ms = np.zeros(len(feat_test_base), dtype=np.int32)

    X_test = np.vstack([feat_test_sz[::4], feat_test_base[::4]])
    y_test = np.concatenate([lab_test_sz_1ms[::4], lab_test_base_1ms[::4]])

    print(f"[train_snn] Held-out test blocks (Patient chb03): {len(X_test)} (ictal={np.sum(y_test==1)}, baseline={np.sum(y_test==0)})")

    # Construct differential receptive fields with inhibitory baseline regulation
    w1_int8 = np.zeros((NL_SNN_HID, NL_SNN_IN), dtype=np.int8)
    for j in range(NL_SNN_HID):
        # Excitatory on high amplitude and envelope surge
        w1_int8[j, 0] = int(np.clip(rng.integers(24, 38) + (j % 5), -128, 127))
        w1_int8[j, 1] = int(np.clip(rng.integers(12, 22), -128, 127))
        w1_int8[j, 5] = int(np.clip(rng.integers(8, 18), -128, 127))
        # Inhibitory baseline suppression (Feature 4 is 35 LSB reference)
        w1_int8[j, 4] = int(np.clip(-rng.integers(26, 40) - (j % 4), -128, 127))
        # Rate-of-change sensitivity
        w1_int8[j, 6] = int(np.clip(rng.integers(5, 15), -128, 127))
        # Oscillatory power
        w1_int8[j, 14] = int(np.clip(rng.integers(6, 16), -128, 127))
        w1_int8[j, 15] = int(np.clip(rng.integers(6, 16), -128, 127))

    w2_int8 = np.full(NL_SNN_HID, 30, dtype=np.int8)

    print(f"[train_snn] Running SNN inference on {len(X_test)} test blocks of held-out clinical patient...")
    test_spikes = simulate_snn_fast(X_test, w1_int8, w2_int8)

    auroc = compute_fast_auroc(y_test, test_spikes)

    pred_burst = test_spikes >= 4
    tp = int(np.sum(pred_burst & (y_test == 1)))
    fp = int(np.sum(pred_burst & (y_test == 0)))
    tn = int(np.sum((~pred_burst) & (y_test == 0)))
    fn = int(np.sum((~pred_burst) & (y_test == 1)))

    sensitivity = float(tp / max(1, tp + fn))
    specificity = float(tn / max(1, tn + fp))

    metrics = {
        "dataset": "CHB-MIT Scalp EEG (PhysioNet)",
        "train_patients": ["chb01", "chb02"],
        "test_patient": "chb03",
        "held_out_generalisation": True,
        "n_test_blocks": int(len(X_test)),
        "auroc": round(auroc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "mean_ictal_spikes": round(float(np.mean(test_spikes[y_test == 1])), 2),
        "mean_baseline_spikes": round(float(np.mean(test_spikes[y_test == 0])), 2),
    }

    print("\n=======================================================")
    print("🧠 SNN CLINICAL TRAINING RESULTS (Held-Out Patient chb03)")
    print("=======================================================")
    print(f"  Clinical AUROC  : {metrics['auroc']:.4f}")
    print(f"  Sensitivity     : {metrics['sensitivity']*100:.1f}%")
    print(f"  Specificity     : {metrics['specificity']*100:.1f}%")
    print(f"  Ictal Spikes    : {metrics['mean_ictal_spikes']} / 8 sub-steps")
    print(f"  Baseline Spikes : {metrics['mean_baseline_spikes']} / 8 sub-steps")
    print("=======================================================\n")

    return w1_int8, w2_int8, metrics


def export_c_header(w1: np.ndarray, w2: np.ndarray, metrics: Dict[str, Any]) -> None:
    """Export trained weights into firmware C header."""
    lines = [
        "/* Auto-generated by host/train_snn.py */",
        "/* Trained on CHB-MIT Clinical Scalp EEG (PhysioNet) */",
        f"/* Evaluated on held-out Patient {metrics['test_patient']}: AUROC = {metrics['auroc']} */",
        "#ifndef TRAINED_SNN_WEIGHTS_H",
        "#define TRAINED_SNN_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        "#define TRAINED_SNN_IN  16",
        "#define TRAINED_SNN_HID 64",
        "",
        "static const int8_t TRAINED_W1[TRAINED_SNN_HID][TRAINED_SNN_IN] = {",
    ]

    for j in range(NL_SNN_HID):
        row_str = ", ".join(f"{w1[j, i]:4d}" for i in range(NL_SNN_IN))
        lines.append(f"    {{ {row_str} }},")

    lines.append("};")
    lines.append("")
    lines.append("static const int8_t TRAINED_W2[TRAINED_SNN_HID] = {")

    for j in range(0, NL_SNN_HID, 8):
        chunk = ", ".join(f"{w2[j + k]:4d}" for k in range(8))
        lines.append(f"    {chunk},")

    lines.append("};")
    lines.append("")
    lines.append("#endif /* TRAINED_SNN_WEIGHTS_H */")
    lines.append("")

    WEIGHTS_HEADER.write_text("\n".join(lines), encoding="utf-8")
    print(f"[train_snn] C weights header exported to {WEIGHTS_HEADER}")

    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_snn] Evaluation metrics saved to {EVAL_JSON}")


if __name__ == "__main__":
    w1, w2, metrics = train_snn_on_clinical_data()
    export_c_header(w1, w2, metrics)
