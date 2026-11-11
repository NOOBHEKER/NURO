"""Unit and Integration Tests for CHB-MIT Patient Replay & Trained Clinical SNN."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from host.replay.patient_data import PatientBrain, ensure_patient_dataset
from host.train_snn import extract_dsp_features, simulate_snn_fast, compute_fast_auroc


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
    # Advance to established seizure discharge (0.5s into ictal event)
    for _ in range(brain.sz_start_idx + 1000):
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
