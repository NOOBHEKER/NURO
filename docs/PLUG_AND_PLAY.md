# Plug and play — run this on any computer

The board is not tied to one machine or one COM port. Detection happens on the
host, the port is never hardcoded, and flashing does not pin a port either.

This page is the whole procedure for a **fresh computer**.

---

## 1. Copy the repository

Clone it, or copy the `neural-loop` folder. Any path works — no drive letters
are assumed.

```text
neural-loop\
├── setup.cmd      ← Windows: double-click or run
├── setup.ps1      ← Windows: PowerShell version
├── setup.sh       ← Linux / macOS
└── host\
```

## 2. Bootstrap once

**Windows (double-click or terminal):**

```powershell
cd "C:\path\to\neural-loop"
.\setup.cmd
# or: powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

**Linux / macOS:**

```bash
cd /path/to/neural-loop
chmod +x setup.sh
./setup.sh
```

`setup` checks for Python 3.10+, creates `.venv`, installs the project in
editable mode, and runs `nuro doctor`. It never opens the serial port and never
touches the board. The first run downloads the dependencies (numpy, matplotlib,
pyserial, sympy, esptool, pytest, PlatformIO), so it needs internet once.

## 3. Put `nuro` on PATH (once per computer)

**Windows:**

```powershell
.\install.ps1 -Apply
# open a new terminal afterwards
```

**Linux / macOS** — add one line to `~/.bashrc` or `~/.zshrc`:

```bash
export PATH="/path/to/neural-loop/.venv/bin:$PATH"
```

Then, from **any** directory:

```powershell
nuro demo
```

---

## What "plug and play" actually means here

| Piece | How it is portable |
|---|---|
| **Port** | `host/board.py` ranks every serial port by USB VID:PID and name: CH343 → native ESP32-S3 USB → CP210x → CH340. No `COM6` in any code path except the last-resort Windows fallback. |
| **Port override** | `--port COM7`, or set `NURO_PORT` once and every command uses it. |
| **Launcher** | `setup` installs a real console script (`.venv\Scripts\nuro.exe` or `.venv/bin/nuro`), so `nuro` works from any directory. |
| **Paths** | Nothing in the host code assumes `G:\`. The Obsidian root is `NURO_OBSIDIAN_ROOT`, otherwise `G:\obsidian` if it exists, otherwise `~/obsidian`. |
| **Firmware** | `platformio.ini` no longer pins `upload_port` / `monitor_port`; PlatformIO auto-detects. Override with `--upload-port COM7`. |
| **No board** | `nuro tui`, `nuro web`, `nuro gui` and `nuro demo` fall back to an explicitly labelled in-silico mirror instead of crashing. |

Resolution order, in one line:

```text
--port  >  NURO_PORT / NL_PORT  >  auto-detected ESP32-S3 bridge  >  COM6 (Windows only)
```

---

## Which USB port on the board?

The ESP32-S3-DevKitC-1 has two USB-C ports.

| Port | Silkscreen | Host sees | Use it for |
|---|---|---|---|
| **UART** | `UART` | **WCH CH343** (`1A86:55D3`) as `COMx` | **The testbed.** 2 Mbps HIL, `nuro info`, `nuro bench`, `nuro term`. Needs the WCH driver on Windows. |
| **Native USB** | `USB` | Espressif USB-Serial-JTAG (`303A:1001`) | Flashing and debugging without the bridge. **Not** the HIL link with the current firmware: `Serial` is UART0, so `nuro tui`/`bench` need the UART port. |

Both ports are auto-detected. The UART bridge wins because it carries the
firmware's `Serial` stream at 2 Mbps. If only the native USB port is plugged
in, `nuro doctor` says so explicitly and tells you to use the UART port.

### Drivers

| OS | What to install |
|---|---|
| Windows 10/11 | **WCH CH343 driver** — <https://www.wch-ic.com/downloads/CH343SER_ZIP.html>. Without it, no `COMx` appears and `nuro doctor` says "no serial ports found". |
| Linux | Usually built in (`ch341`, `cdc_acm`). Add your user to `dialout`: `sudo usermod -aG dialout $USER`, then log out and back in. |
| macOS | Usually built in. No driver needed for CH343 on modern macOS; the port shows as `/dev/cu.usbserial-*` or `/dev/cu.wchusbserial*`. |

Use a **data-capable** USB-C cable. Charge-only cables enumerate nothing.

---

## Verify

```powershell
nuro doctor     # environment, PATH, drivers, board, backup, git
nuro info       # read-only chip/MAC/flash/PSRAM probe
nuro demo       # web cockpit + TUI + matplotlib GUI
```

Expected board facts for this unit:

```text
Chip   : ESP32-S3 rev v0.2
Flash  : 16MB
PSRAM  : 8 MB octal
MAC    : 84:c7:bb:58:23:ec
```

Any other N16R8 board works too — only the MAC differs.

---

## Flashing from another computer

The firmware build is self-contained (PlatformIO downloads the toolchain on the
first build), and the port is auto-detected:

```powershell
python -m platformio run -d firmware -t upload
# explicit port if you have several boards:
python -m platformio run -d firmware -t upload --upload-port COM7
```

> [!warning] Board safety
> Read [`RECOVERY.md`](RECOVERY.md) first. This repository's verified full-flash
> backup lives on the original machine, under `backups/` (git-ignored). On a new
> computer, take a fresh backup **before** flashing anything:
> `python tools/board_backup.py`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nuro: command not found` | PATH not updated | Open a **new** terminal, or run `.\install.ps1 -Apply` again; or call `.venv\Scripts\nuro.exe` directly. |
| `No serial ports found` on Windows | CH343 driver missing, or charge-only cable | Install the WCH driver, replug, check Device Manager → Ports. |
| Port appears then disappears | Another program owns it (Arduino IDE, PuTTY, `nuro term`) | Close the other program. Windows serial ports are exclusive. |
| `nuro tui` shows "IN-SILICO" | Board busy/unplugged, or you are inside `nuro demo` (the cockpit owns the port) | Expected in `demo`; otherwise run `nuro doctor`. |
| Wrong port chosen (several boards) | Ranking picked the first CH343 | `--port COM7` or `set NURO_PORT=COM7`. |
| `Permission denied` on Linux | User not in `dialout` | `sudo usermod -aG dialout $USER`, then re-login. |
| `esptool is not installed` | venv missing the dependency | `nuro setup` (or `pip install -e .`). |
| `nuro info` says "NOT RESPONDING" | Board in a sketch that blocks the ROM loader, or held in reset | Hold **BOOT**, tap **RESET**, release **BOOT**, retry. |

---

## Uninstall

```powershell
# Windows
.\install.ps1   # shows the PATH entry; remove it in Settings → Environment Variables
Remove-Item -Recurse -Force .venv
```

Nothing outside the repository and the user PATH entry is modified.
