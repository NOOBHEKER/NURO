"""Restore a verified full-flash backup to the matching ESP32-S3.

Usage:
    python tools/board_restore.py --backup backups/20261110_120000 --confirm RESTORE

Safety:
- Refuses to run without the exact `--confirm RESTORE` token.
- Refuses a file whose SHA-256 does not match manifest.json.
- Refuses a board whose MAC does not match the manifest (unless overridden).
- Backs up the current flash before overwriting it (unless --skip-current-backup).
- Verifies the write, then hard-resets the board.

This is the only script in tools/ that writes flash. It never drives a GPIO.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from board_common import (  # noqa: E402
    BACKUP_ROOT,
    DEFAULT_BAUD,
    find_latest_backup,
    force_utf8_stdout,
    human_bytes,
    normalize_mac,
    parse_flash_id,
    read_manifest,
    resolve_port,
    run_esptool,
    sha256_file,
    timestamp_slug,
)


def main() -> int:
    force_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Restore a verified full-flash ESP32-S3 backup"
    )
    parser.add_argument(
        "--backup", type=Path, default=None, help="Backup folder (default: latest)"
    )
    parser.add_argument(
        "--port", default=None, help="Serial port (auto-detect by default)"
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--confirm", default="", help="Must be exactly RESTORE")
    parser.add_argument(
        "--skip-current-backup",
        action="store_true",
        help="Do not back up the current flash first (not recommended)",
    )
    parser.add_argument(
        "--allow-different-board",
        action="store_true",
        help="Allow restoring to a board whose MAC differs from the manifest",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate everything, write nothing"
    )
    args = parser.parse_args()

    if args.confirm != "RESTORE":
        print(
            "REFUSED: pass --confirm RESTORE to acknowledge that this overwrites the whole flash.",
            file=sys.stderr,
        )
        return 2

    backup_dir = args.backup or find_latest_backup()
    if not backup_dir:
        print(
            "REFUSED: no backup folder found. Run tools/board_backup.py first.",
            file=sys.stderr,
        )
        return 2
    backup_dir = backup_dir.resolve()
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"REFUSED: {manifest_path} is missing.", file=sys.stderr)
        return 2

    manifest = read_manifest(manifest_path)
    flash_path = backup_dir / manifest.get("file", "flash.bin")
    if not flash_path.exists():
        print(f"REFUSED: {flash_path} is missing.", file=sys.stderr)
        return 2

    expected_bytes = int(manifest.get("flash_bytes", 0))
    if expected_bytes <= 0:
        print("REFUSED: manifest has no flash_bytes.", file=sys.stderr)
        return 2
    actual_bytes = flash_path.stat().st_size
    if actual_bytes != expected_bytes:
        print(
            f"REFUSED: size mismatch: manifest {expected_bytes}, file {actual_bytes}.",
            file=sys.stderr,
        )
        return 2

    print("  Verifying backup SHA-256 before touching the board...")
    digest = sha256_file(flash_path)
    if digest.lower() != str(manifest.get("sha256", "")).lower():
        print(
            "REFUSED: SHA-256 mismatch. The backup file is corrupted or was modified.",
            file=sys.stderr,
        )
        return 2
    print(f"  SHA-256 OK: {digest}")

    port = resolve_port(args.port or manifest.get("port"))
    print("\n=== ESP32-S3 FULL-FLASH RESTORE ===")
    print(f"  Backup    : {backup_dir}")
    print(f"  Image     : {flash_path.name} ({human_bytes(actual_bytes)})")
    print(f"  Port      : {port}")
    print(f"  Board MAC : {manifest.get('mac')} (from backup)")
    print()

    if args.dry_run:
        print("  DRY RUN: everything validated; no bytes written.")
        return 0

    # Identify the board currently connected and confirm it is the intended one.
    try:
        id_proc = run_esptool(
            ["--chip", "esp32s3", "--port", port, "--baud", str(args.baud), "flash-id"],
            quiet=True,
        )
        id_text = (id_proc.stdout or "") + (id_proc.stderr or "")
        facts = parse_flash_id(id_text)

        mac_proc = run_esptool(
            ["--chip", "esp32s3", "--port", port, "--baud", str(args.baud), "read-mac"],
            quiet=True,
        )
        mac_text = (mac_proc.stdout or "") + (mac_proc.stderr or "")
        current_mac = None
        for line in mac_text.splitlines():
            if "MAC:" in line:
                current_mac = normalize_mac(line.split("MAC:", 1)[1])
                break
    except Exception as exc:
        print(f"REFUSED: cannot identify the connected board: {exc}", file=sys.stderr)
        return 2

    expected_mac = normalize_mac(manifest.get("mac"))
    if expected_mac and current_mac != expected_mac:
        if not args.allow_different_board:
            print(
                f"REFUSED: MAC mismatch. Backup is for {expected_mac}, connected board is {current_mac}.\n"
                "         Use --allow-different-board only if you are certain.",
                file=sys.stderr,
            )
            return 2
        print(
            f"  WARNING: restoring {expected_mac} image onto {current_mac} (explicitly allowed)."
        )

    if not args.skip_current_backup:
        print("\n  Saving the current flash first (so this operation is reversible)...")
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "board_backup.py"),
            "--port",
            port,
            "--baud",
            str(args.baud),
            "--out",
            str(BACKUP_ROOT / "pre-restore"),
        ]
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            print(
                "REFUSED: could not back up the current flash. Nothing was written.",
                file=sys.stderr,
            )
            return 2

    print("\n  Writing the verified image to 0x00000000 ...")
    try:
        run_esptool(
            [
                "--chip",
                "esp32s3",
                "--port",
                port,
                "--baud",
                str(args.baud),
                "write-flash",
                "--flash-size",
                "16MB",
                "0x0",
                str(flash_path),
            ],
            capture=False,
        )
        print("\n  Verifying the write...")
        run_esptool(
            [
                "--chip",
                "esp32s3",
                "--port",
                port,
                "--baud",
                str(args.baud),
                "verify-flash",
                "--flash-size",
                "16MB",
                "0x0",
                str(flash_path),
            ],
            capture=False,
        )
    except Exception as exc:
        print(f"\n[RESTORE FAILED] {exc}", file=sys.stderr)
        print(
            "  Do NOT unplug. Re-run the restore command; the bootloader is still reachable.",
            file=sys.stderr,
        )
        return 1

    print("\n=== RESTORE COMPLETE ===")
    print(f"  Restored  : {flash_path.name}")
    print(f"  SHA-256   : {digest}")
    print(f"  Board MAC : {current_mac}")
    print("  The board should now boot the exact firmware that was backed up.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
