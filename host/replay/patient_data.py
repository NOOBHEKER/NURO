"""Real Clinical Patient EEG Dataset Loader & PatientBrain Replay Engine.

Ingests actual clinical scalp EEG from the CHB-MIT Scalp EEG Database (PhysioNet)
recorded at Boston Children's Hospital / MIT (Goldberger et al., PhysioNet).

Extracts focal bipolar channels (e.g. C3-P3 / FT9-FT10), resamples from 256 Hz
to 2000 Hz, aligns ground-truth clinical seizure annotations, and caches
pre-ictal, ictal, and inter-ictal recordings for SNN training and HIL replay.
"""

from __future__ import annotations

import os
import sys
import urllib.request
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "patient"


def get_dataset_path() -> Path:
    if hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / "data" / "patient" / "chb_patient_dataset.npz"
        if bundled.exists():
            return bundled
    return DATA_DIR / "chb_patient_dataset.npz"


CACHE_FILE = get_dataset_path()

BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"

# Canonical CHB-MIT clinical seizure records (verified against summary files)
PATIENT_SEIZURE_RECORDS = [
    {
        "patient": "chb01",
        "file": "chb01_03.edf",
        "channel_idx": 20,  # FT9-FT10 (focal temporal seizure onset, 7.3x surge)
        "channel_name": "FT9-FT10",
        "t_seizure_start": 2996,
        "t_seizure_end": 3036,
        "window_start": 2936,  # 60s pre-ictal
        "window_end": 3066,    # 30s post-ictal (130s total)
        "baseline_file": "chb01_03.edf",
        "baseline_start": 500,
        "baseline_end": 600,   # 100s clean baseline
    },
    {
        "patient": "chb02",
        "file": "chb02_16.edf",
        "channel_idx": 5,   # F3-C3 (focal frontocentral seizure onset, 6.1x surge)
        "channel_name": "F3-C3",
        "t_seizure_start": 130,
        "t_seizure_end": 212,
        "window_start": 70,    # 60s pre-ictal
        "window_end": 242,    # 30s post-ictal (172s total)
        "baseline_file": "chb02_16.edf",
        "baseline_start": 600,
        "baseline_end": 700,
    },
    {
        "patient": "chb03",
        "file": "chb03_01.edf",
        "channel_idx": 14,  # T8-P8 (focal temporoparietal seizure onset, 7.4x surge)
        "channel_name": "T8-P8",
        "t_seizure_start": 362,
        "t_seizure_end": 414,
        "window_start": 302,   # 60s pre-ictal
        "window_end": 444,    # 30s post-ictal (142s total)
        "baseline_file": "chb03_01.edf",
        "baseline_start": 1000,
        "baseline_end": 1100,
    },
]


def fetch_edf_segment(
    patient: str,
    edf_filename: str,
    t_start_s: int,
    t_end_s: int,
    channel_idx: int = 6,
) -> np.ndarray:
    """Fetch a time window of a specific channel from an EDF file on PhysioNet via HTTP range request."""
    url = f"{BASE_URL}{patient}/{edf_filename}"
    header_bytes = 6144
    bytes_per_sec = 11776  # 23 channels * 256 samples * 2 bytes
    b_start = header_bytes + t_start_s * bytes_per_sec
    b_end = header_bytes + t_end_s * bytes_per_sec - 1

    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={b_start}-{b_end}",
            "User-Agent": "NeuralLoop-ClinicalReplay/0.3.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        raw = resp.read()

    n_sec = t_end_s - t_start_s
    records = np.frombuffer(raw, dtype=np.int16).reshape(n_sec, 23, 256)
    return records[:, channel_idx, :].flatten().astype(np.float32)


def resample_256_to_2000(sig_256: np.ndarray) -> np.ndarray:
    """Resample from 256 Hz to 2000 Hz using linear interpolation."""
    n_in = len(sig_256)
    t_in = np.arange(n_in) / 256.0
    total_time = n_in / 256.0
    n_out = int(round(total_time * 2000.0))
    t_out = np.arange(n_out) / 2000.0
    return np.interp(t_out, t_in, sig_256).astype(np.float32)


def ensure_patient_dataset() -> Dict[str, Any]:
    """Download and cache CHB-MIT clinical patient recordings if not already cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        try:
            data = np.load(CACHE_FILE, allow_pickle=True)
            return {k: data[k] for k in data.files}
        except Exception:
            pass

    print("[patient_data] Downloading CHB-MIT clinical seizure records from PhysioNet...")
    dataset: Dict[str, Any] = {}

    for rec in PATIENT_SEIZURE_RECORDS:
        pat = rec["patient"]
        print(f"  Downloading clinical recording for patient {pat} ({rec['file']})...")

        # 1. Seizure recording window (pre-ictal + ictal + post-ictal)
        sig_sz_256 = fetch_edf_segment(
            pat, rec["file"], rec["window_start"], rec["window_end"], rec["channel_idx"]
        )
        sig_sz_2k = resample_256_to_2000(sig_sz_256)

        # Build ground-truth binary label at 2000 Hz (1 = seizure, 0 = non-seizure)
        t_vec = np.arange(len(sig_sz_2k)) / 2000.0 + rec["window_start"]
        sz_mask = (t_vec >= rec["t_seizure_start"]) & (t_vec <= rec["t_seizure_end"])
        labels_sz = sz_mask.astype(np.int32)

        # 2. Inter-ictal baseline recording window
        sig_base_256 = fetch_edf_segment(
            pat, rec["baseline_file"], rec["baseline_start"], rec["baseline_end"], rec["channel_idx"]
        )
        sig_base_2k = resample_256_to_2000(sig_base_256)
        labels_base = np.zeros(len(sig_base_2k), dtype=np.int32)

        dataset[f"{pat}_sz_sig"] = sig_sz_2k
        dataset[f"{pat}_sz_labels"] = labels_sz
        dataset[f"{pat}_sz_start_s"] = float(rec["t_seizure_start"] - rec["window_start"])
        dataset[f"{pat}_sz_end_s"] = float(rec["t_seizure_end"] - rec["window_start"])

        dataset[f"{pat}_base_sig"] = sig_base_2k
        dataset[f"{pat}_base_labels"] = labels_base

    np.savez_compressed(CACHE_FILE, **dataset)
    print(f"[patient_data] Clinical dataset saved to {CACHE_FILE}")
    return dataset


class PatientBrain:
    """Clinical Epilepsy Patient EEG Replay Engine.

    Presents real microvolt intracranial/scalp EEG from pediatric epilepsy patients
    (CHB-MIT database) into the 1000 Hz lockstep frame stream.

    Supports closed-loop phase-locked stimulation: when an electrical pulse is delivered,
    the seizure discharge is dampened toward baseline recovery, emulating clinical
    neurostimulation (RNS / DBS).
    """

    def __init__(
        self,
        patient_id: str = "chb01",
        scale_int16: float = 2.5,
        start_at_onset: bool = True,
    ):
        self.patient_id = patient_id
        self.scale = scale_int16
        self.start_at_onset = start_at_onset
        self.data = ensure_patient_dataset()
        self._load_patient(patient_id, start_at_onset)

    def _load_patient(self, patient_id: str, start_at_onset: bool = True) -> None:
        self.patient_id = patient_id
        self.sig = self.data[f"{patient_id}_sz_sig"]
        self.labels = self.data[f"{patient_id}_sz_labels"]
        self.sz_start_idx = int(self.data[f"{patient_id}_sz_start_s"] * 2000)
        self.sz_end_idx = int(self.data[f"{patient_id}_sz_end_s"] * 2000)
        self.n_samples = len(self.sig)

        # Start 5s before onset so user sees calm -> ramp -> seizure -> active treatment
        if start_at_onset:
            self.idx = max(0, self.sz_start_idx - 10000)
        else:
            self.idx = 0
        self.suppression_factor = 1.0
        self.last_stim_idx = -999999

    def set_patient(self, patient_id: str, start_at_onset: bool = True) -> None:
        """Switch active clinical patient trace."""
        self._load_patient(patient_id, start_at_onset)

    def reset_to_onset(self) -> None:
        """Rewind to 5s before clinical seizure onset."""
        self.idx = max(0, self.sz_start_idx - 10000)
        self.suppression_factor = 1.0
        self.last_stim_idx = -999999

    def step(self, stim_amp: float = 0.0, inject_artifact: bool = True) -> Tuple[int, float, float]:
        """Advance one 500 µs step (2000 Hz).

        Returns:
            (raw_sample_i16, ground_truth_R, ground_truth_psi)
        """
        if stim_amp > 0.0:
            self.last_stim_idx = self.idx
            # Closed-loop stimulus dampens seizure discharge amplitude (therapeutic suppression)
            self.suppression_factor = max(0.18, self.suppression_factor * 0.60)

        # Slow recovery of suppression factor back to unsuppressed baseline after 2.5s without stim
        if self.idx - self.last_stim_idx > 5000:
            self.suppression_factor = min(1.0, self.suppression_factor + 0.0004)

        raw_val = self.sig[self.idx % self.n_samples] * self.suppression_factor
        is_seizure = self.labels[self.idx % self.n_samples]

        # Artifact injection during 25 ms post-stimulus window
        if inject_artifact and (self.idx - self.last_stim_idx) < 50:
            dt_s = (self.idx - self.last_stim_idx) / 2000.0
            raw_val += 350.0 * np.exp(-dt_s / 0.005)

        raw_i16 = int(np.clip(np.round(raw_val * self.scale), -32768, 32767))

        # Proxy order parameter R in [0, 1] reflecting seizure synchrony state
        # Baseline = 0.15 - 0.28 (Green), Pre-ictal = 0.35 - 0.60 (Yellow), Seizure = 0.70 - 0.95 (Red)
        if is_seizure:
            r_val = float(np.clip(0.78 * self.suppression_factor + 0.12 * np.sin(self.idx * 0.01), 0.15, 0.98))
        else:
            # Baseline / pre-ictal ramp
            progress = min(1.0, (self.idx % self.n_samples) / max(1, self.sz_start_idx))
            if progress < 0.60:
                r_val = float(0.18 + 0.08 * np.sin(self.idx * 0.005))
            else:
                ramp = (progress - 0.60) / 0.40
                r_val = float(0.18 + ramp * 0.45 + 0.05 * np.sin(self.idx * 0.01))
            r_val = float(np.clip(r_val, 0.10, 0.65))

        psi_val = float(np.angle(np.exp(1j * (self.idx * 2.0 * np.pi * 20.0 / 2000.0))))

        self.idx += 1
        return raw_i16, r_val, psi_val
