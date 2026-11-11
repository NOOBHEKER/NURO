# Changelog

All notable changes to this repository. The project is research code; the
[`docs/CLAIMS.md`](docs/CLAIMS.md) ledger defines what each result may be
called.

## 0.3.0 — 2026-11-11

### Added

- **Plug and play on a fresh computer.** `setup.cmd` / `setup.ps1` / `setup.sh`
  create `.venv`, install the project and run the doctor; `install.ps1 -Apply`
  puts the `nuro` console script on the user PATH.
- **`nuro setup`** — the same bootstrap from inside the CLI.
- **`pyproject.toml`** — package metadata, dependency list and the `nuro`
  entry point, so `pip install -e .` works on any machine.
- **`host/board.py`** — one port policy for the whole host side:
  `--port` > `NURO_PORT`/`NL_PORT` > auto-detected bridge > platform fallback.
  Ranking understands CH343, native ESP32-S3 USB, CP210x and CH340, on Windows,
  Linux and macOS, and explains a missing driver.
- **`nuro info`** — real read-only esptool probe (chip, revision, MAC, flash,
  PSRAM, features) instead of hardcoded strings.
- **`nuro doctor`** — Python, packages, venv, PATH, PlatformIO, repo files,
  firmware build, recovery backup, serial ports, board identity and git state.
- **`nuro demo`** — web cockpit + terminal TUI + matplotlib GUI in one command.
- **`host/demo.py`**, **`host/doctor.py`**, **`host/setup_env.py`**.
- **`docs/PLUG_AND_PLAY.md`**, **`docs/CLAIMS.md`**, **`AGENTS.md`**,
  **`LICENSE`** (MIT), **`.github/workflows/ci.yml`**.
- Hermetic port-ranking tests in `tests/test_board_ports.py`.
- **`--arm` on `nuro bench` and `nuro demo`** — enables the GPIO4 gate on the
  same serial connection, refuses to run if the board does not acknowledge
  `[ARM]`, and disarms on exit. Covered by hermetic REPL-handshake tests.

### Verified on hardware (2026-11-11)

- The N16R8 safety firmware was **flashed** to the board and boot-verified:
  boot self-test PASS (16 MB flash, 8 MB octal PSRAM, MAC
  `84:c7:bb:58:23:ec`), `arm` → ARMED → `disarm` checked live (5/5), watchdog
  boots without a TWDT error. The 10 µs pulse remains a firmware value, not a
  scope measurement.
- Post-flash benchmark, 500 frames: p50 **1.28 ms**, p99 **1.42 ms**,
  500/500 valid, 0 drops, 0 seq gaps, device DSP p50 18 µs.
- `stim_pulses_fired: 0` is now the expected result of an un-armed run: the
  board boots disarmed and the host never arms it automatically.
- A second verified full-flash backup (`backups/20261111_140215`, SHA-256
  `ba492bb2…`) records the flashed state; the pre-rebuild image remains at
  `backups/20261110_203223`.
- Figure 4 was regenerated from the post-flash run and now states the real
  frame count and port.

### Changed

- Firmware targets the real **ESP32-S3-WROOM-1-N16R8** (`firmware/boards/esp32s3_n16r8.json`,
  16 MB flash, `qio_opi` octal PSRAM) instead of the stock N8/no-PSRAM board.
- Firmware boots **disarmed**; GPIO4 requires an explicit `arm`. Added a 5 s
  task watchdog, a boot self-test and a defined **10 µs** stimulation marker.
- `platformio.ini` no longer pins `upload_port` / `monitor_port`, so flashing
  auto-detects the board on any computer.
- `nuro eval` and Figure 9 are labelled **SYNTHETIC — NOT CHB-MIT**; the
  predictor is described as an int8 linear readout + exponential kernel, not a
  1D CNN.
- The test suite is hermetic: no golden rewrites, no tracked figure writes, no
  Obsidian launch, no session mutation.
- The Obsidian root is configurable (`NURO_OBSIDIAN_ROOT`) and defaults to
  `~/obsidian` when `G:\obsidian` is absent.
- README rewritten around an ethics/scope block and a claim-status table.

### Fixed

- `nuro` failed with `UnicodeEncodeError` when stdout was redirected on
  Windows; both streams are now reconfigured to UTF-8.
- `esptool_available()` used `--version`, which esptool v5 rejects; it now uses
  the `version` subcommand.

## 0.2.0 — earlier

- On-chip int8 LIF SNN, 61-tap Q15 quadrature FIR, 20 ms blanking, 32-byte
  CRC16 frames, Kuramoto and Jansen-Rit virtual brains, PRC identification,
  five-arm bakeoff, HIL benchmark, dashboards, scientific calculator shell.

## 0.1.0 — initial

- ESP32-S3 testbed skeleton and the `nuro` calculator prototype.
