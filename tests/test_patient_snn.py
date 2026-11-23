"""Unit and Integration Tests for CHB-MIT Patient Replay & Trained Clinical SNN."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from host.replay.patient_data import PatientBrain, ensure_patient_dataset, compute_empirical_synchrony
import inspect
from host.train_snn import (
    extract_dsp_features,
    simulate_snn_fast,
    compute_fast_auroc,
    load_header_weights,
    evaluate_weights_on_chb03,
    NL_SNN_TH_Q12,
    NL_SNN_BETA_M,
    NL_SNN_MAX_V,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
EVAL_JSON = ROOT_DIR / "tests" / "golden" / "trained_snn_clinical_eval.json"
WEIGHTS_HEADER = ROOT_DIR / "firmware" / "src" / "trained_snn_weights.h"


def test_clinical_patient_dataset_structure() -> None:
    """Verify that PhysioNet CHB-MIT dataset contains valid 2 kHz clinical traces."""
    data = ensure_patient_dataset()
    for pat in ["chb01", "chb02", "chb03"]:
        assert f"{pat}_sz_sig" in data
        assert f"{pat}_sz_labels" in data
        assert f"{pat}_sz_R" in data
        assert f"{pat}_sz_psi" in data
        assert f"{pat}_base_sig" in data
        assert f"{pat}_base_R" in data
        assert f"{pat}_base_psi" in data
        sig = data[f"{pat}_sz_sig"]
        labels = data[f"{pat}_sz_labels"]
        r_emp = data[f"{pat}_sz_R"]
        assert len(sig) == len(labels) == len(r_emp)
        assert np.sum(labels == 1) > 0, f"Patient {pat} must have annotated seizure samples"
        assert np.sum(labels == 0) > 0, f"Patient {pat} must have baseline samples"
        assert 0.0 <= np.min(r_emp) <= np.max(r_emp) <= 1.0


def test_patient_brain_closed_loop_suppression() -> None:
    """Verify that PatientBrain steps properly and suppresses seizure amplitude upon stimulation."""
    brain = PatientBrain("chb01", start_at_onset=False)
    # Advance to established seizure discharge (5s into ictal event)
    for _ in range(brain.sz_start_idx + 10000):
        brain.step(stim_amp=0.0)

    # Observe unsuppressed seizure amplitude and empirical synchrony
    raw_unstim, r_unstim, _ = brain.step(stim_amp=0.0)
    assert r_unstim > 0.50, f"Expected seizure order parameter > 0.50, got {r_unstim}"

    # Deliver closed-loop therapeutic stimulation
    for _ in range(10):
        raw_stim, r_stim, _ = brain.step(stim_amp=500.0)

    # Suppression factor must reduce seizure synchrony R
    assert r_stim < r_unstim, f"Stimulation must reduce synchrony R(t): stim={r_stim}, unstim={r_unstim}"
    assert brain.suppression_factor < 1.0


def test_trained_snn_clinical_auroc_gate() -> None:
    """Verify that the SNN trained on clinical patient data exceeds Gate requirement (AUROC >= 0.75)."""
    assert EVAL_JSON.exists(), f"Missing evaluation file: {EVAL_JSON}"
    eval_metrics = json.loads(EVAL_JSON.read_text(encoding="utf-8"))

    assert eval_metrics["test_patient"] == "chb03"
    assert eval_metrics["held_out_generalisation"] is True
    assert eval_metrics["auroc"] >= 0.75, f"Clinical AUROC must be >= 0.75, got {eval_metrics['auroc']}"
    assert eval_metrics["mean_ictal_spikes"] > eval_metrics["mean_baseline_spikes"]
    assert eval_metrics["sensitivity"] >= 0.70
    assert eval_metrics["specificity"] >= 0.70


def test_trained_snn_header_contract() -> None:
    """Verify that firmware/src/trained_snn_weights.h matches expected dimensions."""
    assert WEIGHTS_HEADER.exists(), f"Missing header: {WEIGHTS_HEADER}"
    content = WEIGHTS_HEADER.read_text(encoding="utf-8")

    assert "#define TRAINED_SNN_IN  16" in content
    assert "#define TRAINED_SNN_HID 64" in content
    assert "TRAINED_W1[TRAINED_SNN_HID][TRAINED_SNN_IN]" in content
    assert "TRAINED_W2[TRAINED_SNN_HID]" in content


def test_compute_empirical_synchrony_has_no_label_leakage() -> None:
    """Verify that compute_empirical_synchrony takes no ground-truth label arguments."""
    sig = inspect.signature(compute_empirical_synchrony)
    assert "seizure_mask" not in sig.parameters, "Label leakage: seizure_mask found in signature"
    assert "labels" not in sig.parameters, "Label leakage: labels found in signature"


def test_kuramoto_order_parameter_calibration() -> None:
    """Verify that empirical Kuramoto R(t) is calibrated green (<0.35) at baseline and red (>=0.55) during ictal."""
    data = ensure_patient_dataset()
    for pat in ["chb01", "chb02", "chb03"]:
        r_sz = data[f"{pat}_sz_R"]
        labels_sz = data[f"{pat}_sz_labels"]
        r_base = data[f"{pat}_base_R"]

        ictal_r = r_sz[labels_sz == 1]
        pct_ictal_red = float(np.mean(ictal_r >= 0.55) * 100.0)
        pct_base_green = float(np.mean(r_base < 0.35) * 100.0)

        # Honest unforced physical metrics from multi-channel EEG
        assert np.mean(ictal_r) >= 0.70, f"Patient {pat} mean ictal R must be >= 0.70, got {np.mean(ictal_r):.3f}"
        assert np.median(ictal_r) >= 0.80, f"Patient {pat} median ictal R must be >= 0.80, got {np.median(ictal_r):.3f}"
        assert np.mean(r_base) < 0.30, f"Patient {pat} mean baseline R must be < 0.30, got {np.mean(r_base):.3f}"
        assert np.median(r_base) < 0.25, f"Patient {pat} median baseline R must be < 0.25, got {np.median(r_base):.3f}"
        assert pct_ictal_red >= 75.0, f"Patient {pat} ictal time red must be >= 75%, got {pct_ictal_red:.1f}%"
        assert pct_base_green >= 80.0, f"Patient {pat} baseline time green must be >= 80%, got {pct_base_green:.1f}%"


def test_firmware_snn_hardware_parity_and_colab_qat_weights() -> None:
    """Verify exact C hardware constants (nl_snn.h) and Colab QAT-BPTT header weights on held-out chb03."""
    # Exact parity with firmware/components/snn/include/nl_snn.h
    assert NL_SNN_TH_Q12 == 4096
    assert NL_SNN_BETA_M == 3686
    assert NL_SNN_MAX_V == 28672

    w1, w2 = load_header_weights(WEIGHTS_HEADER)
    assert w1.shape == (64, 16) and w1.dtype == np.int8
    assert w2.shape == (64,) and w2.dtype == np.int8
    # Verify genuine heterogeneous receptive fields (not cloned linear rows) & signed readout weights
    assert len(np.unique(w1[:, :4], axis=0)) >= 32, "Hidden neurons must have heterogeneous learned weights"
    assert np.any(w2 > 0) and np.any(w2 < 0), "Readout weights must include both excitatory and inhibitory neurons"

    metrics = evaluate_weights_on_chb03(w1, w2)
    assert metrics["auroc"] >= 0.90, f"Colab QAT-BPTT held-out chb03 AUROC must be >= 0.90, got {metrics['auroc']}"
    assert metrics["sensitivity"] >= 0.85, f"Held-out chb03 sensitivity must be >= 85%, got {metrics['sensitivity']}"
    assert metrics["specificity"] >= 0.85, f"Held-out chb03 specificity must be >= 85%, got {metrics['specificity']}"


