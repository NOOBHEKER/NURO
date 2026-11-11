"""Hardware-in-the-Loop (HIL) transport, clock calibration, and benchmark runner."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import serial

from host.board import resolve_port
from host.hil.protocol import (
    SampleFrame,
    ResultFrame,
    enable_windows_1ms_timer,
    host_monotonic_us,
    MODE_LOCKSTEP,
)
from host.virtualbrain.kuramoto import KuramotoBrain, KuramotoConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def send_repl_command(
    ser: serial.Serial,
    command: str,
    expect: bytes,
    timeout_s: float = 2.0,
) -> bool:
    """Send a firmware REPL command on an open port and wait for its ack.

    The board's REPL shares the UART with the binary HIL frames, so this is only
    safe while no binary frames are in flight. Returns ``True`` when ``expect``
    appears in the response before the timeout.
    """
    ser.reset_input_buffer()
    ser.write(command.encode("ascii") + b"\r\n")
    deadline = time.time() + timeout_s
    buf = b""
    while time.time() < deadline:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk
            if expect in buf:
                return True
    return False


def run_hil_benchmark(
    port: Optional[str] = None,
    baudrate: int = 2000000,
    n_frames: int = 1000,
    seed: int = 2026,
    arm: bool = False,
    runs_root: Path = REPO_ROOT / "runs",
) -> Dict[str, Any]:
    """Run a hardware-in-the-loop closed-loop session against the ESP32-S3.

    Records:
    - Host-to-host round-trip latency (us): p50, p95, p99, max
    - Device DSP compute time (t_dsp_us - t_rx_us): p50, p95, p99, max
    - Device total compute time (t_dec_us - t_rx_us): p50, p95, p99, max
    - Dropped frames, sequence gaps, and stimulation pulses delivered
    """
    port = resolve_port(port)
    enable_windows_1ms_timer()
    brain = KuramotoBrain(KuramotoConfig(seed=seed))
    # Start in ictal window (t = 30.0 s, K = 6.0)
    brain.theta = np.mod(brain.rng.normal(0.0, 0.55, size=brain.cfg.n_osc), 2.0 * np.pi)
    brain.step_idx = int(30.0 / brain.cfg.dt_s)

    ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.25, write_timeout=0.25)
    # Ensure RTS/DTR do not hold ESP32-S3 in reset
    ser.dtr = False
    ser.rts = False
    time.sleep(0.35)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    if arm:
        # Opening the port resets the board, so the gate must be enabled on this
        # connection, after the reset — never before it.
        if not send_repl_command(ser, "arm", b"[ARM]"):
            ser.close()
            raise RuntimeError(
                "the board did not acknowledge `arm`; refusing to run a "
                "benchmark that claims to be armed"
            )
        print("[bench] stimulation gate: ARMED for this run")

    rtt_us_list: List[float] = []
    dsp_us_list: List[int] = []
    compute_us_list: List[int] = []
    r_trace: List[float] = []
    seq_gaps = 0
    dropped = 0
    stim_pulses = 0
    pending_stim = 0.0

    for seq in range(n_frames):
        # Advance virtual brain by 2 samples (1.0 ms @ 2 kHz)
        s0, r0, _ = brain.step(stim_amp=pending_stim)
        s1, r1, _ = brain.step(stim_amp=0.0)
        pending_stim = 0.0
        r_trace.append(r1)

        sim_t_us = int(seq * 1000)
        t0_ns = time.perf_counter_ns()
        h_us = (t0_ns // 1_000) & 0xFFFFFFFF

        frame = SampleFrame(
            seq=seq,
            sim_t_us=sim_t_us,
            host_t_us=h_us,
            samples=(s0, s1, 0, 0),
            flags=0x03,  # bit0=lockstep, bit1=arm-hint
            n_ch=1,
            n_samp=2,
            mode=MODE_LOCKSTEP,
        )
        raw_tx = frame.to_bytes()
        ser.write(raw_tx)

        raw_rx = ser.read(32)
        t1_ns = time.perf_counter_ns()

        if len(raw_rx) != 32:
            dropped += 1
            continue

        try:
            res = ResultFrame.from_bytes(raw_rx)
        except ValueError:
            dropped += 1
            continue

        if res.seq != seq:
            seq_gaps += 1

        rtt_us = (t1_ns - t0_ns) / 1_000.0
        dsp_us = (res.t_dsp_us - res.t_rx_us) & 0xFFFFFFFF
        comp_us = (res.t_dec_us - res.t_rx_us) & 0xFFFFFFFF

        rtt_us_list.append(rtt_us)
        dsp_us_list.append(int(dsp_us))
        compute_us_list.append(int(comp_us))

        if res.t_stim_us != 0:
            stim_pulses += 1
            pending_stim = 500.0

    if arm:
        if not send_repl_command(ser, "disarm", b"[DISARM]", timeout_s=1.5):
            print("[bench] WARNING: disarm was not acknowledged; reset the board.")
    ser.close()

    rtt_arr = (
        np.array(rtt_us_list, dtype=np.float64) if rtt_us_list else np.array([0.0])
    )
    dsp_arr = (
        np.array(dsp_us_list, dtype=np.float64) if dsp_us_list else np.array([0.0])
    )
    comp_arr = (
        np.array(compute_us_list, dtype=np.float64)
        if compute_us_list
        else np.array([0.0])
    )

    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "port": port,
        "baudrate": baudrate,
        "armed": arm,
        "n_frames_sent": n_frames,
        "n_frames_valid": len(rtt_us_list),
        "dropped_frames": dropped,
        "seq_gaps": seq_gaps,
        "rtt_p50_us": float(np.percentile(rtt_arr, 50)),
        "rtt_p95_us": float(np.percentile(rtt_arr, 95)),
        "rtt_p99_us": float(np.percentile(rtt_arr, 99)),
        "rtt_max_us": float(np.max(rtt_arr)),
        "device_dsp_p50_us": float(np.percentile(dsp_arr, 50)),
        "device_dsp_p99_us": float(np.percentile(dsp_arr, 99)),
        "device_compute_p50_us": float(np.percentile(comp_arr, 50)),
        "device_compute_p95_us": float(np.percentile(comp_arr, 95)),
        "device_compute_p99_us": float(np.percentile(comp_arr, 99)),
        "device_compute_max_us": float(np.max(comp_arr)),
        "stim_pulses_fired": stim_pulses,
        "r_initial": float(r_trace[0]) if r_trace else 0.0,
        "r_final_mean": float(np.mean(r_trace[-200:])) if len(r_trace) >= 200 else 0.0,
    }

    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{ts_slug}__lockstep__kuramoto__phase_locked"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    metrics["run_dir"] = str(run_dir)
    return metrics


if __name__ == "__main__":
    out = run_hil_benchmark()
    print(json.dumps(out, indent=2))
