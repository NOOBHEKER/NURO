# ESP32-S3 Recovery & Board Safety

> **Read this before the first `upload`.** The board is a single point of failure
> for the whole project. A verified full-flash backup turns "bricked" into
> "restore in five minutes".

---

## 1. The exact board

| Field | Value |
|---|---|
| Module | **ESP32-S3-WROOM-1-N16R8** |
| MCU | ESP32-S3 (QFN56) rev v0.2, dual-core LX7 @ 240 MHz |
| Flash | 16 MB quad SPI |
| PSRAM | 8 MB **octal** PSRAM |
| DevKit | ESP32-S3-DevKitC-1 v1.3 |
| UART bridge | WCH CH343, `COM6` on this machine |
| Native USB | GPIO19 (D−) / GPIO20 (D+), second USB-C port |
| MAC | `84:c7:bb:58:23:ec` |

> The PlatformIO board id `esp32-s3-devkitc-1` defaults to the **N8 / no-PSRAM**
> variant. This project overrides flash size, PSRAM type and partition table in
> `firmware/platformio.ini`. If a build log says "8MB Flash / 320KB RAM", the
> override did not apply — stop and fix it before flashing.

---

## 2. The safety contract (never break these)

| Rule | Why |
|---|---|
| Only **GPIO 4** is driven as a stimulation marker | It is a free, general-purpose pin on this board revision |
| Never drive **GPIO 0, 3, 45, 46** | Strapping pins — can prevent boot or change boot mode |
| Never drive **GPIO 26–37** | Connected to the octal PSRAM / flash |
| Never drive GPIO 19/20 or 43/44 | Native USB and UART0; the drivers own them |
| GPIO4 goes **only into the laptop's USB input / a jumper** | 3.3 V logic, 10 µs pulse; no electrode, no current source, no body |
| Never feed 5 V into a 3.3 V pin | The LDO and the SoC are not 5 V tolerant |
| Never use `erase-flash` "to fix it" | It destroys the bootloader, partition table and NVS; restore the backup instead |
| Keep the USB cable on the **UART** port for the data link | `Serial` is UART0; the CH343 bridge carries the 2 Mbps HIL stream. The native USB port is for flashing/debug and does not carry `Serial` with this firmware |

---

## 3. What "recovery" means here

A full-flash image at offset `0x0` contains **everything**:

```
0x000000  bootloader (2nd stage)
0x008000  partition table
0x00F000  NVS / PHY / other data
0x010000  application (firmware.bin)
...       OTA / SPIFFS / coredump, up to 16 MB
```

Restoring it returns the board to the exact byte-level state it had when the
backup was taken — firmware, partition layout, calibration data, NVS and all.

The backup is **not** committed to git (it is 16 MB and may contain NVS data).
`backups/` is git-ignored; only this document and `backups/README.md` are tracked.

---

## 4. One-time backup (already done — re-run after any known-good firmware)

```powershell
cd "G:\MIT Planner\mit\neural-loop"
python tools/board_info.py
python tools/board_backup.py
```

Expected output ends with:

```
=== BACKUP COMPLETE ===
  File     : flash.bin (16.0 MB)
  SHA-256  : <64 hex chars>
  Verified : True
```

The script:

1. reads chip / flash identity with `esptool flash-id` (read-only),
2. reads all 16 MB with `read-flash` (read-only),
3. checks the file is exactly 16,777,216 bytes,
4. computes SHA-256,
5. reads the flash again with `verify-flash` and byte-compares,
6. writes `backups/<timestamp>/manifest.json`.

**If the verify step fails, do not flash anything.** Re-run the backup.

---

## 5. Restore (the "unbrick" command)

```powershell
cd "G:\MIT Planner\mit\neural-loop"
python tools/board_restore.py --backup "backups\20261110_120000" --confirm RESTORE
```

Safety checks before a single byte is written:

- `--confirm RESTORE` must be exact.
- The backup file's SHA-256 must match `manifest.json`.
- The connected board's MAC must match the backup's MAC.
- The current flash is backed up to `backups/pre-restore/...` first.
- After writing, the tool runs `verify-flash` and compares.

If the board is not responding, force the ROM bootloader:

1. Hold **BOOT** (GPIO0).
2. Press and release **RESET**.
3. Release **BOOT**.
4. Re-run `python tools/board_info.py` — it should now answer.

If `board_info.py` still cannot connect, try the other USB-C port (native USB
uses a different bridge) and check Device Manager for a CH343 COM port.

---

## 6. Recovery decision tree

```
Board not responding?
├─ Does COM6 exist in Device Manager?
│  ├─ No  -> cable/port problem. Try the other USB-C port and a known cable.
│  └─ Yes -> Hold BOOT, tap RESET, release BOOT. Re-run board_info.py.
├─ Does board_info.py answer?
│  ├─ Yes -> run board_restore.py with the last verified backup.
│  └─ No  -> try the other USB-C port; if still no, the board may need
│            a manual BOOT/RESET sequence or a different PC.
└─ Board boots but firmware is wrong?
   └─ Flash the known-good build, or restore the last full backup.
```

---

## 7. Backup ledger

| Date | Git tag / commit | Backup folder | SHA-256 (first 16) | Verified |
|---|---|---|---|---|
| 2026-11-10 | `pre-rebuild-2026-11-10` (`456b5b4`) | `backups/20261110_203223` | `9f4f013c0cf6ba85` | ✅ byte-for-byte |
| 2026-11-11 | `1c5cd35` + built firmware | `backups/20261111_140215` | `ba492bb215feafaf` | ✅ byte-for-byte |

> The 2026-11-11 backup is the **current** state: N16R8 safety firmware
> (boot self-test, disarmed `arm` gate, 5 s watchdog, 10 µs pulse) as flashed and
> boot-verified that day. The 2026-11-10 image remains the pre-rebuild recovery
> point.
>
> Update this table after every known-good firmware change. The full hash lives
> in `backups/<timestamp>/manifest.json`.

---

## 8. Related files

- `tools/board_common.py` — port detection, esptool wrapper, hashing
- `tools/board_info.py` — read-only identity probe
- `tools/board_backup.py` — full read + verify + manifest
- `tools/board_restore.py` — verified restore (only writer)
- `firmware/platformio.ini` — N16R8 / PSRAM / 16 MB partition config
- `backups/README.md` — why the backup is not in git

---

## 🔗 Links

- [[Board Facts - ESP32-S3 DevKitC-1 v1.3]]
- [[Safety & Watchdog]]
- [[Week 0 Checklist]]
- [[Decision Log]]
