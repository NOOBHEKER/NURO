# AGENTS.md — guardrails for this repository

These rules exist because the project has one physical ESP32-S3, a credibility
problem it is actively fixing, and tests that used to mutate tracked files.
Read this before changing firmware, tests, figures, or public claims.

---

## 1. Board safety (non-negotiable)

- **Never flash without the user's explicit approval.** The last verified
  full-flash backup is `backups/20261110_203223` (git-ignored). If it is
  missing, take a new backup with `python tools/board_backup.py` first.
- **Never run `erase-flash`.** Restore the verified image instead:
  `python tools/board_restore.py --backup <dir> --confirm RESTORE`.
- Only **GPIO 4** may be driven as the stimulation marker. Never drive
  strapping pins (0/3/45/46), flash/PSRAM pins (26–37), native USB (19/20), or
  UART0 (43/44).
- The pulse is **10 µs, 3.3 V**, into a jumper or the laptop's USB input.
  **Never to a body.** No electrodes, no current source.
- Boot must stay **disarmed**; only the `arm` command may enable GPIO4.
- Any new peripheral or pin change requires a user check-in before flashing.

## 2. Truth and provenance

- Every public claim must have a row in [`docs/CLAIMS.md`](docs/CLAIMS.md).
- G5/G6 currently run on **synthetic surrogates**. Label every output, figure,
  slide, and vault note as `SYNTHETIC — NOT CHB-MIT` until the real EDF
  pipeline lands.
- Do not call the current predictor a **1D-CNN**. It is an int8 linear readout
  + causal exponential kernel.
- Do not call the SNN **trained**. Its weights are formula-initialised.
- Do not claim **FreeRTOS tasks** or a **custom USB transport**; the firmware
  is one Arduino `loop()`.
- The measured latency is **USB host↔board RTT**, not a therapeutic loop.

## 3. Tests must stay hermetic

`pytest` must never:

- rewrite `tests/golden/*.json` (read and compare only);
- launch Obsidian or touch `G:\obsidian`;
- mutate `host/session_state.json` or the user's active session;
- write figures/dashboards into `docs/figures` (use `tmp_path`);
- produce non-deterministic PDF metadata.

Run the suite before every commit:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 4. One port policy

All host entry points resolve serial ports through `host/board.py`:

```
--port  >  NURO_PORT  >  auto-detected CH343/ESP32-S3  >  COM6 fallback
```

Do not add new hardcoded `"COM6"` defaults. Add a `--port` argument with
`default=None` and call `resolve_port()`.

The Windows fallback is the only place `COM6` may appear in code. On Linux and
macOS the fallback is an empty port so the caller can show the in-silico view
instead of guessing `/dev/tty*`.

## 4b. Plug and play

The repository must work on a fresh computer with no edits:

- `setup.cmd` / `setup.ps1` / `setup.sh` create `.venv` and install the project;
- `pyproject.toml` owns the `nuro` console entry point and the dependency list;
- nothing outside `host/obsidian_sync.py` may assume a drive letter such as
  `G:\`; that module uses `NURO_OBSIDIAN_ROOT` with a home-directory fallback;
- `firmware/platformio.ini` must not pin `upload_port` or `monitor_port`;
- `nuro doctor` must explain a missing driver, a missing PATH entry and a
  missing board in one line each.

## 5. Firmware build

- The real module is **N16R8** (16 MB flash, 8 MB octal PSRAM). Use
  `firmware/boards/esp32s3_n16r8.json`; do not revert to the stock N8 board.
- Build only (no upload) with:

```powershell
python -m platformio run -d firmware
```

- The build log must say `16MB Flash` and the board name must mention N16R8.
- Recovery and board facts: [`docs/RECOVERY.md`](docs/RECOVERY.md).

## 6. Style and commits

- Keep the existing structure and naming. Prefer small, focused commits.
- Do not delete unfamiliar files or user work without investigating first.
- Keep `runs/*/metrics.json` tracked; `backups/*` and `*.bin` stay ignored.
- Update `README.md` and `docs/CLAIMS.md` when a status changes.
- The vault at `G:\obsidian` follows the user's ICSE notes convention; do not
  restructure it casually.

## 7. Useful commands

```powershell
.\.venv\Scripts\python.exe -m host.cli doctor
.\.venv\Scripts\python.exe -m host.cli info
.\.venv\Scripts\python.exe -m host.cli demo --frames 2 --no-browser
.\.venv\Scripts\python.exe -m host.cli eval
.\.venv\Scripts\python.exe -m host.cli bench --frames 500
python tools\board_info.py
python tools\board_backup.py
```
