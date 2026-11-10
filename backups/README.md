# Local flash backups

This folder is **git-ignored on purpose**.

A full ESP32-S3 flash image is 16 MB, can contain NVS/calibration data, and is
only useful on the exact board it came from. Committing it would bloat the repo
and leak data that has nothing to do with the source code.

Each backup lives in `backups/<timestamp>/`:

```
flash.bin        16,777,216 bytes, full chip image from 0x0
manifest.json    chip, MAC, flash size, SHA-256, git state, verification flag
```

Create one:

```powershell
python tools/board_backup.py
```

Restore one (exact command, exact MAC, SHA-256 checked):

```powershell
python tools/board_restore.py --backup "backups\<timestamp>" --confirm RESTORE
```

See `docs/RECOVERY.md` for the full procedure and the board-safety contract.
