import urllib.request
import numpy as np

def bandpass_hilbert(x, fs=256.0, f_low=3.0, f_high=28.0):
    N = len(x)
    freqs = np.fft.fftfreq(N, 1.0 / fs)
    Xf = np.fft.fft(x)
    bp_mask = (abs(freqs) >= f_low) & (abs(freqs) <= f_high)
    Xf_filt = Xf * bp_mask
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return np.fft.ifft(Xf_filt * h)

def resample_to_2000(sig, fs_orig=256.0):
    n_in = len(sig)
    t_in = np.arange(n_in) / fs_orig
    total_time = n_in / fs_orig
    n_out = int(round(total_time * 2000.0))
    t_out = np.arange(n_out) / 2000.0
    return np.interp(t_out, t_in, sig).astype(np.float32)

records_meta = [
    {
        "pat": "chb01",
        "file": "chb01_03.edf",
        "t_start": 2936,
        "t_end": 3066,
        "sz_start": 60,
        "sz_end": 100,
        "focal_idx": 20,
        "focal_name": "FT9-FT10",
        "dx": "11yo Female, Refractory Temporal Lobe Epilepsy",
    },
    {
        "pat": "chb02",
        "file": "chb02_16.edf",
        "t_start": 70,
        "t_end": 242,
        "sz_start": 60,
        "sz_end": 142,
        "focal_idx": 5,
        "focal_name": "F3-C3",
        "dx": "11yo Male, Frontocentral Motor Epilepsy",
    },
    {
        "pat": "chb03",
        "file": "chb03_01.edf",
        "t_start": 302,
        "t_end": 444,
        "sz_start": 60,
        "sz_end": 112,
        "focal_idx": 14,
        "focal_name": "T8-P8",
        "dx": "14yo Female, Intractable Temporoparietal Epilepsy",
    },
]

for m in records_meta:
    pat = m["pat"]
    fname = m["file"]
    url = f"https://physionet.org/files/chbmit/1.0.0/{pat}/{fname}"
    header_bytes = 6144
    bytes_per_sec = 11776
    b_start = header_bytes + m["t_start"] * bytes_per_sec
    b_end = header_bytes + m["t_end"] * bytes_per_sec - 1
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes={b_start}-{b_end}", "User-Agent": "NeuralLoop/0.3.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    n_sec = m["t_end"] - m["t_start"]
    recs = np.frombuffer(raw, dtype=np.int16).reshape(n_sec, 23, 256)
    channels = recs.transpose(1, 0, 2).reshape(23, -1).astype(np.float32)

    # Compute instantaneous phase for all 23 channels
    phases = np.zeros_like(channels)
    for k in range(23):
        z = bandpass_hilbert(channels[k], fs=256.0, f_low=3.0, f_high=28.0)
        phases[k] = np.angle(z)

    # Kuramoto order parameter across all 23 patient electrodes
    phasors = np.exp(1j * phases)
    mean_phasor = np.mean(phasors, axis=0)
    R_raw = np.abs(mean_phasor)
    psi_raw = np.angle(mean_phasor)

    # Smooth with 0.25s moving average (64 samples at 256 Hz)
    kernel = np.ones(64) / 64.0
    R_smooth = np.convolve(R_raw, kernel, mode="same")

    s_idx = int(m["sz_start"] * 256)
    e_idx = int(m["sz_end"] * 256)
    r_base = np.mean(R_smooth[:s_idx])
    r_seiz = np.mean(R_smooth[s_idx:e_idx])
    r_peak = np.max(R_smooth[s_idx:e_idx])
    r_min_base = np.min(R_smooth[:s_idx])

    focal_sig = channels[m["focal_idx"]]
    uV_base = np.std(focal_sig[:s_idx])
    uV_seiz = np.std(focal_sig[s_idx:e_idx])

    print(f"=== PATIENT: {pat.upper()} ({m['dx']}) ===")
    print(f"  Focal Channel     : {m['focal_name']}")
    print(f"  Baseline R (mean) : {r_base:.3f} (min: {r_min_base:.3f}) [Calm]")
    print(f"  Seizure R (mean)  : {r_seiz:.3f} (peak: {r_peak:.3f}) [Ictal Surge]")
    print(f"  Focal RMS Voltage : {uV_base:.1f} µV (baseline) -> {uV_seiz:.1f} µV (ictal, {uV_seiz/max(1e-3, uV_base):.1f}x surge)")
    print()
