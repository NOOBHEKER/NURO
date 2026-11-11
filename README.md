> **ETHICS & SCOPE — READ FIRST.** This is a **simulation testbed**. No human or animal tissue is recorded, stimulated, or involved at any stage. No electrodes are used. The "brain" is a numerical model on a laptop; the "stimulation" is a 3.3 V logic pulse on a single GPIO sent back to that laptop over USB. Public, de-identified datasets are used offline for model development only. Nothing here is a medical device, a therapy, or a clinical claim.

# Neural Loop — a closed-loop edge neuromodulation testbed on ESP32-S3

**One-line pitch:** a single ESP32-S3-WROOM-1-N16R8 runs a two-timescale neuromodulation testbed — an offline pre-ictal predictor proposes when to arm, and a 1 kHz on-device loop (61-tap Q15 quadrature FIR + 64-neuron int8 LIF SNN + 20 ms artifact blanking) phase-locks a 10 µs marker pulse to an empirically identified phase-response curve.

The project is also a self-contained scientific calculator (math, chemistry, physics) with an Obsidian graph sync. The calculator is real and tested; the neural-loop claims are labelled below.

---

## What is real, what is simulated

This table is the short version. The full evidence ledger is [`docs/CLAIMS.md`](docs/CLAIMS.md).

| Claim | Status |
|---|---|
| ESP32-S3-WROOM-1-N16R8, 16 MB flash, 8 MB octal PSRAM, MAC `84:c7:bb:58:23:ec` | ✅ **Verified** by a read-only esptool probe |
| 32-byte frames + CRC16-CCITT-FALSE, 2 Mbps UART | ✅ **Verified** by tests and live HIL runs |
| 61-tap Q15 quadrature FIR, phase, 20 ms blanking | ✅ **Verified** in-silico; **implemented** on-device |
| 64-neuron int8 LIF SNN | ⚠️ **Implemented**, but **weights are formula-initialised, not trained** |
| Host↔board RTT p50 ≈ 1.28 ms, p99 ≈ 1.42 ms (500 frames) | ✅ **Verified** on the current, flashed firmware |
| G7 phase-locked vs random-phase sham (Kuramoto) | ✅ **Verified** in simulation |
| G8 empirical PRC + bootstrap CIs | ✅ **Verified** in simulation |
| G5 detector AUROC 0.9998 / G6 predictor AUROC 0.990, 53 s lead | 🧪 **Synthetic surrogate — NOT CHB-MIT** |
| "1D-CNN" pre-ictal predictor | 🚧 **Planned**; current model is an int8 linear readout + exponential kernel |
| CHB-MIT EDF ingestion / 23-patient split | 🚧 **Planned** (Phase 2) |
| FreeRTOS task decomposition / custom USB transport | 🚧 **Planned**; current firmware is one Arduino `loop()` |
| Task watchdog, explicit `arm` gate, 10 µs pulse | 🛠️ **Flashed and boot-verified** (2026-11-11); pulse width not scope-measured |

> **Do not cite the G5/G6 numbers as clinical or CHB-MIT results.** They come from signals generated inside this repository. The real-data pipeline is the next milestone.

---

## Hardware (probed, not assumed)

| Property | Value |
|---|---|
| Module | `ESP32-S3-WROOM-1-N16R8` |
| MCU | ESP32-S3 (QFN56) rev v0.2, dual-core LX7 @ 240 MHz |
| Flash | 16 MB quad SPI (`manuf 0x68`, `device 0x4018`) |
| PSRAM | 8 MB octal, 3.3 V |
| DevKit | ESP32-S3-DevKitC-1 v1.3 |
| Host link | WCH CH343 USB-UART on `COM6`, 2 Mbps |
| Native USB | GPIO19/20 (second USB-C port) |
| MAC | `84:c7:bb:58:23:ec` |

The stock PlatformIO `esp32-s3-devkitc-1` board is the **N8 / no-PSRAM** variant. This repo ships `firmware/boards/esp32s3_n16r8.json` so the build log names the real module and enables `qio_opi` + the 16 MB partition table.

---

## Quick start

```powershell
# Copy or clone the repository anywhere, then bootstrap once:
.\setup.cmd                # Windows (creates .venv, installs nuro, runs doctor)
# ./setup.sh               # Linux / macOS

# Optional, once per computer: put `nuro` on PATH
.\install.ps1 -Apply

# Showcase: web cockpit + TUI + matplotlib GUI
nuro demo
```

That is the whole procedure on a fresh machine, including a different COM port
or a board that has never been plugged into this PC. Details, drivers and
per-OS notes: [`docs/PLUG_AND_PLAY.md`](docs/PLUG_AND_PLAY.md).

`nuro demo` starts three views:

1. **Web cockpit** — live localhost workstation, owns the ESP32-S3 serial port.
2. **Terminal TUI** — animated 8-pane cockpit in the current terminal.
3. **Matplotlib GUI** — interactive 4-panel window.

Windows serial ports are exclusive, so the TUI and GUI deliberately run as **labelled in-silico mirrors** while the web cockpit owns the board. Press `q` in the TUI to stop all three cleanly.

### Useful commands

| Command | What it does |
|---|---|
| `nuro setup` | Create `.venv`, install the project, run the doctor |
| `nuro demo` | All three views in one command |
| `nuro tui` | Terminal cockpit (live board if free) |
| `nuro web` | Localhost cockpit + browser |
| `nuro gui` | Matplotlib cockpit |
| `nuro info` | **Real** read-only board probe (chip, MAC, flash, PSRAM) |
| `nuro doctor` | Environment, dependencies, PATH, drivers, board and repo readiness |
| `nuro bench --frames 500` | Live HIL benchmark; writes `runs/<id>/metrics.json`. Add `--arm` to enable GPIO4 for the run |
| `nuro eval` | Gates G5/G6 on **synthetic surrogates** |
| `nuro test` | Full pytest suite |
| `nuro figures` | Regenerate publication figures (PNG + PDF) |
| `nuro calc "x^3 - 3*x"` | Scientific calculator / calculus / graph |
| `nuro chem "KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2"` | Chemistry balancer |
| `nuro phy "projectile v=50 angle=45"` | Physics engine |
| `nuro session switch Physics` | Named calculation sessions |
| `nuro obsidian` | Sync and open the Obsidian graph |

Port selection is centralised in `host/board.py`:

```
--port  >  NURO_PORT  >  auto-detected CH343/ESP32-S3  >  COM6 (Windows only)
```

### Stimulation is opt-in

The flashed firmware boots **disarmed**: GPIO4 stays low and `nuro bench` reports
`stim_pulses_fired: 0` until the gate is enabled. Two ways to enable it for a run:

```powershell
nuro bench --frames 500 --arm   # armed on this connection, disarmed afterwards
nuro demo --arm                 # armed for the demo, disarmed on exit
```

Opening the serial port resets the board, so the gate can never survive a
reconnect — it always comes back disarmed. A 5 s task watchdog, a boot self-test
and a 20 ms artifact-blanking curtain are in the same build; see
[`docs/RECOVERY.md`](docs/RECOVERY.md) before flashing anything.

---

## Architecture

```text
neural-loop/
├── setup.cmd / setup.ps1 / setup.sh   one-command bootstrap (any computer)
├── pyproject.toml                     package metadata + `nuro` entry point
├── firmware/                     ESP32-S3 Arduino/PlatformIO project
│   ├── boards/                   N16R8 board manifest (16 MB, qio_opi)
│   ├── components/
│   │   ├── link/                 32-byte wire protocol + CRC16-CCITT-FALSE
│   │   ├── dsp/                  61-tap Q15 quadrature FIR, phase, blanking
│   │   └── snn/                  64-neuron int8 LIF kernel
│   └── src/main.cpp              1 kHz loop, arm gate, watchdog, boot self-test
├── host/
│   ├── board.py                  port policy + read-only identity probe
│   ├── cli.py                    `nuro` command surface
│   ├── setup_env.py              venv bootstrap used by `nuro setup`
│   ├── demo.py                   three-view orchestrator
│   ├── doctor.py                 environment/board readiness
│   ├── hil/                      protocol codec, DSP reference, HIL transport
│   ├── virtualbrain/             Kuramoto + Jansen-Rit neural mass models
│   ├── replay/                   synthetic G5/G6 surrogates (CHB-MIT planned)
│   ├── analysis/                 PRC identification, 5-arm bakeoff, figures
│   ├── dash/                     TUI, web cockpit, matplotlib GUI
│   ├── calc.py / chem.py / phy.py scientific calculator engines
│   └── obsidian_sync.py          vault + graph sync
├── tests/
│   ├── golden/                   committed byte-exact fixtures (read-only)
│   ├── test_suite.py             hermetic verification suite
│   └── test_board_ports.py       hermetic port-ranking tests
├── tools/                        verified full-flash backup / restore
├── docs/                         CLAIMS.md, PLUG_AND_PLAY.md, RECOVERY.md, figures
├── backups/                      git-ignored full-flash images + manifest
└── runs/                         HIL metrics (metrics.json tracked)
```

### The two loops

- **Fast loop (on-device, 1 kHz):** sample frame → Q15 quadrature FIR → envelope/phase → int8 LIF SNN → phase-gated pulse on GPIO4 → 20 ms blanking.
- **Slow loop (host, offline):** Kuramoto/Jansen-Rit simulation → empirical PRC → 5-arm bakeoff → synthetic G5/G6 surrogate evaluation → arming hint.

The current predictor is **not** a 1D CNN and is **not** trained on CHB-MIT. See [`docs/CLAIMS.md`](docs/CLAIMS.md).

---

## Gates

| Gate | Meaning | Status |
|---|---|---|
| G0 | 32-byte frame + CRC16 contract | ✅ tests |
| G3 | DSP quadrature phase + 20 ms blanking | ✅ tests |
| G4 | Twin-run determinism (Kuramoto + Jansen-Rit) | ✅ tests |
| G5 | Int8 LIF detector vs baselines | 🧪 synthetic surrogate |
| G6 | Pre-ictal predictor, leakage guards | 🧪 synthetic surrogate |
| G7 | Phase-locked beats sham and no-stim | ✅ simulation |
| G8 | Empirical PRC + bootstrap CIs | ✅ simulation |
| G10 | Reproducible figures + dashboard | ✅ tests (to `tmp_path`) |

---

## Safety contract

The board is a single point of failure, so the safety rules are not negotiable:

- Only **GPIO 4** is driven as a stimulation marker; 3.3 V logic, 10 µs pulse, into the laptop's USB input or a jumper. **Never to a body.**
- Never drive strapping pins **GPIO 0/3/45/46**, flash/PSRAM pins **GPIO 26–37**, native USB **19/20**, or UART0 **43/44**.
- Boot is **disarmed**. GPIO4 can only pulse after an explicit `arm` command.
- A task watchdog resets a hung loop instead of leaving GPIO4 in an unknown state.
- **Never** use `erase-flash` "to fix it". Restore the verified image instead.

### Recovery

A full 16 MB flash image was read, SHA-256 hashed, and verified byte-for-byte before any firmware change:

```text
backups/20261110_203223/flash.bin   (16,777,216 bytes)
SHA-256 9f4f013c0cf6ba851f15bab9ff956a9a4be047484fc8d4c527304f73eedc3bed
```

Restore:

```powershell
python tools\board_restore.py --backup "backups\20261110_203223" --confirm RESTORE
```

The restore tool checks the confirm token, SHA-256 and board MAC, backs up the current flash first, and verifies the write. Full procedure: [`docs/RECOVERY.md`](docs/RECOVERY.md).

---

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The suite is **hermetic**:

- goldens are read and compared, never rewritten;
- figures and the dashboard are written to pytest's `tmp_path`;
- Obsidian is never launched and the user's vault/session state is never touched;
- PDF metadata is deterministic.

Current result: **25 passed** (14 in `test_suite.py`, 11 port-ranking tests in
`test_board_ports.py`).

---

## Roadmap

1. **P0 (done):** verified backup/restore, N16R8 build, truth labels, hermetic tests, real `nuro info`, central port policy.
2. **P1 (next):** flash the safety firmware with user approval, re-measure HIL latency, scope the 10 µs pulse, add CI.
3. **P2:** download the real CHB-MIT 23-patient set, parse EDF, implement a genuine patient-wise split and a 1D CNN; re-run G5/G6 and replace every synthetic label with measured values.
4. **P3:** portfolio polish — README figures, demo video, MIT license, vault rewrite.

---

## License

MIT. See [`LICENSE`](LICENSE).

## Links

- [`docs/CLAIMS.md`](docs/CLAIMS.md) — evidence ledger and forbidden sentences
- [`docs/PLUG_AND_PLAY.md`](docs/PLUG_AND_PLAY.md) — fresh-computer setup, drivers, ports
- [`docs/RECOVERY.md`](docs/RECOVERY.md) — board safety and unbrick procedure
- [`AGENTS.md`](AGENTS.md) — guardrails for future contributors/agents
