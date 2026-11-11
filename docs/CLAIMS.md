# Claims Ledger — what is real, what is simulated

> **Purpose.** Every public claim about this project must be traceable to one of
> the statuses below. If a sentence is not supported here, it does not go in the
> README, a figure caption, a slide, or an MIT application.
>
> Last audited: **2026-11-10** against commit `ce6472d` + the working tree that
> adds the N16R8 board manifest and the P0 truth labels.

## Status vocabulary

| Status | Meaning |
|---|---|
| **VERIFIED** | Measured on the physical board or produced by a deterministic test that is in the repo. Reproducible with a listed command. |
| **IMPLEMENTED** | Code exists and is covered by a test or a build, but the specific end-to-end claim has not been re-measured on hardware. |
| **SYNTHETIC** | Produced from signals generated inside the repo. Useful as a pipeline check, never as clinical evidence. |
| **PLANNED** | Designed or scaffolded, but not implemented. Do not describe it in the present tense. |

## Hardware and firmware

| Claim | Status | Evidence | Caveat |
|---|---|---|---|
| ESP32-S3-WROOM-1-**N16R8** module | **VERIFIED** | `python -m host.cli info` → `Chip: ESP32-S3 rev v0.2`, `Flash: 16MB`, `PSRAM: 8MB`, MAC `84:c7:bb:58:23:ec` | None. |
| 16 MB flash, 8 MB octal PSRAM build | **IMPLEMENTED** | `firmware/boards/esp32s3_n16r8.json` (`qio_opi`, `default_16MB.csv`); build reports 16 MB flash and `BOARD_HAS_PSRAM` | The new firmware has **not been flashed** yet. The running image is the pre-rebuild one. |
| 32-byte sample/result frames with CRC16-CCITT-FALSE | **VERIFIED** | `tests/test_suite.py::test_sample_and_result_frame_binary_contract`; golden `tests/golden/crc16_sample.json` | None. |
| 61-tap Q15 quadrature FIR, phase, 20 ms blanking | **VERIFIED** (in-silico) / **IMPLEMENTED** (on-device) | `test_gate_g3_dsp_quadrature_phase_and_artifact_blanking`; firmware `components/dsp` | The on-device DSP is compiled but its latency was measured on the previous firmware image. |
| 64-neuron int8 LIF SNN | **IMPLEMENTED** | `firmware/components/snn`; build succeeds | **Weights are formula-initialised, not trained.** It is a fixed random-ish projection, not a learned classifier. |
| Hardware task watchdog | **VERIFIED** (boots clean) | `init_watchdog()` in `firmware/src/main.cpp`; flashed 2026-11-11; boot log shows no `TWDT already initialized` error; 1 kHz loop runs | Fault injection (deliberate hang) is **not** tested. The watchdog is armed, not proven to fire. |
| Explicit `arm` gate before GPIO4 | **VERIFIED** (state machine) | Boot self-test prints `Arm gate: DISARMED`; `arm` → `[ARM] GPIO4 stimulation marker ENABLED` + status `ARMED`; `disarm` → `[DISARM]` + status `DISARMED`. Live 5/5 check on COM6, 2026-11-11, board left disarmed | Proves the gate changes state; it does not prove GPIO4 was electrically low. |
| 10 µs stimulation marker pulse on GPIO4 | **IMPLEMENTED** and flashed | `NL_STIM_PULSE_US = 10` + `delayMicroseconds()`; boot self-test prints `Stim pulse: 10 us on GPIO4` | Firmware parameter only — **not measured on a scope**. Do not claim a verified pulse width. |
| Stimulation requires an explicit `arm` | **VERIFIED** (state machine) | `if (g_armed && …)` guards every GPIO4 write in the HIL loop; `nuro bench` reports `stim_pulses_fired: 0` on a disarmed board | The host tools never arm automatically. A demo that shows pulses must arm the board first (firmware REPL). |
| FreeRTOS task decomposition / custom USB transport | **PLANNED** | — | The current firmware is a single Arduino `loop()`. Do not claim FreeRTOS tasks or a custom USB stack. |

## Host-side performance

| Claim | Status | Evidence | Caveat |
|---|---|---|---|
| Host round-trip p50 1.28 ms, p99 1.42 ms (500 frames, **current firmware**) | **VERIFIED** | `runs/20261111_135815__lockstep__kuramoto__phase_locked/metrics.json` | Post-flash N16R8 build, COM6 @ 2 Mbps, 500/500 valid, 0 drops, 0 seq gaps. Host↔board USB latency, not a clinical loop latency. |
| Host round-trip p50 1.26 ms, p99 1.64 ms (500 frames, pre-rebuild firmware) | **VERIFIED** (historical) | `runs/20261110_111749__lockstep__kuramoto__phase_locked/metrics.json` | Kept for comparison: the safety firmware changed p50 by +21 µs and p99 by −215 µs at n = 500. |
| Host round-trip p50 1.30 ms, p99 1.58 ms (1000 frames, pre-rebuild firmware) | **VERIFIED** (historical) | `runs/20261109_173009__lockstep__kuramoto__phase_locked/metrics.json` | Same machine, same cable, earlier firmware. |
| Device DSP p50 18 µs / p99 22 µs | **VERIFIED** (reported by firmware) | `runs/20261111_135815…/metrics.json`; pre-rebuild reported 20/27 µs | Values are self-reported by the firmware frame; they were not independently scoped. |
| "Sub-5 ms closed loop" | **VERIFIED** for USB RTT | p99 1.42 ms < 5 ms | Do not extend this to "sub-5 ms therapeutic loop". |
| 25/25 automated tests pass | **VERIFIED** | `.venv\Scripts\python -m pytest tests -q` → `25 passed` | Tests are hermetic: no golden writes, no Obsidian launch, figures to `tmp_path`. |

## Gates G5–G8 (model-level)

| Claim | Status | Evidence | Caveat |
|---|---|---|---|
| G5 Int8 LIF detector AUROC 0.9998 vs 0.884 amplitude | **SYNTHETIC** | `nuro eval`; `host/replay/predictor_eval.py` | Signals are generated in `simulate_patient_recording()` / synthetic background code. **Not CHB-MIT.** |
| G6 pre-ictal AUROC 0.990, 53 s median lead, 0.125 FA/h | **SYNTHETIC** | `nuro eval`; Figure 9 | `chb20`–`chb23` are labels on synthetic pseudo-patients. The model is an **int8 linear readout + causal exponential kernel**, not a 1D CNN. |
| G7 phase-locked beats random-phase sham | **VERIFIED** in simulation | `test_gate_g7_phase_locked_beats_sham_and_no_stim`; 5-arm bakeoff | Kuramoto model only. No tissue, no electrodes, no clinical claim. |
| G8 empirical PRC + bootstrap CIs | **VERIFIED** in simulation | `test_gate_g8_prc_identification_and_five_arm_bakeoff`; `tests/golden/five_arm_bakeoff.json` | Same. |
| CHB-MIT EDF ingestion and patient-wise split | **PLANNED** | No `data/` directory exists | Phase 2: download the real 23-patient set (~1.5 GB), parse EDF, then re-run G5/G6. Until then, no CHB-MIT number may be quoted. |
| 1D-CNN pre-ictal classifier | **PLANNED** | — | Current surrogate is a linear readout + exponential kernel. Do not call it a CNN. |

## Demo and tooling

| Claim | Status | Evidence | Caveat |
|---|---|---|---|
| `nuro demo` shows web + TUI + matplotlib | **IMPLEMENTED** | `host/demo.py`; smoke-tested with `--frames 2` | Only the web cockpit owns the live serial port; TUI/GUI are explicitly labelled in-silico mirrors. |
| `nuro setup` bootstraps a fresh computer | **IMPLEMENTED** | `host/setup_env.py`; `setup.cmd` / `setup.ps1` / `setup.sh`; `pip install -e .` verified locally | Not yet exercised on a second physical machine. |
| `nuro` console script from any directory | **VERIFIED** | `pyproject.toml` → `.venv\Scripts\nuro.exe --version` → `nuro 0.3.0` | Windows verified; POSIX entry point untested here. |
| Port auto-ranking (CH343 → native USB → CP210x → CH340) | **VERIFIED** (unit) | `tests/test_board_ports.py` (10 hermetic tests) | Ranking is tested against synthetic port lists; a second PC with a different bridge is still untested. |
| `nuro info` probes the real board | **IMPLEMENTED** | `python -m host.cli info` → real chip/MAC/flash/PSRAM | Read-only; it resets into the ROM bootloader via esptool. |
| `nuro doctor` checks the environment | **IMPLEMENTED** | `python -m host.cli doctor` → 19 pass / 1 warn / 0 fail locally | The git warning is expected while the tree is dirty. |
| Port auto-detection with COM6 fallback | **IMPLEMENTED** | `host/board.py`; `NURO_PORT` env var | Explicit `--port` always wins. |
| Verified full-flash recovery image | **VERIFIED** | `backups/20261110_203223/` (git-ignored), SHA-256 `9f4f013c0cf6ba85…`, read-back verified | Restore is guarded by `--confirm RESTORE`, SHA-256 and MAC. |

## Sentences that must not appear

- "CHB-MIT AUROC 0.99" / "trained on CHB-MIT" / "validated on 23 patients".
- "1D-CNN" for the current predictor.
- "trained SNN" / "learned spiking network".
- "FreeRTOS-based firmware" / "custom USB transport".
- "closed-loop seizure suppression" without the words **in-silico** and **no tissue**.
- "sub-5 ms therapeutic" — the measured number is USB round-trip latency.

## How to reproduce the numbers

```powershell
cd "G:\MIT Planner\mit\neural-loop"
.\.venv\Scripts\python.exe -m pytest tests -q          # 14/14
.\.venv\Scripts\python.exe -m host.cli info            # real board probe
.\.venv\Scripts\python.exe -m host.cli doctor          # environment
.\.venv\Scripts\python.exe -m host.cli eval            # SYNTHETIC gates
.\.venv\Scripts\python.exe -m host.cli bench --frames 500   # live HIL, writes runs/
```

## 🔗 Links

- [[Numbers to Defend]]
- [[Gates & Acceptance Criteria]]
- [[Decision Log]]
- `docs/RECOVERY.md`
