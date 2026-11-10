"""Create a verified, full-flash recovery image of the ESP32-S3.

Usage:
    python tools/board_backup.py
    python tools/board_backup.py --port COM6 --baud 460800
    python tools/board_backup.py --no-verify

What it does:
1. Reads the chip / flash identity (read-only).
2. Reads the entire flash to backups/<timestamp>/flash.bin.
3. Checks the file size and computes SHA-256.
4. Reads the flash again with `verify-flash` and compares (unless --no-verify).
5. Writes manifest.json with everything needed to restore the exact board.

The board resets into its ROM bootloader a few times. Nothing is erased or
written. No GPIO is driven.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from board_common import (  # noqa: E402
    BACKUP_ROOT,
    DEFAULT_BAUD,
    flash_size_bytes,
    force_utf8_stdout,
    git_state,
    human_bytes,
    normalize_mac,
    parse_flash_id,
    resolve_port,
    run_esptool,
    sha256_file,
    timestamp_slug,
    write_manifest,
)


def main() -> int:
    force_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Verified full-flash backup (non-destructive)"
    )
    parser.add_argument(
        "--port", default=None, help="Serial port (auto-detect by default)"
    )
    parser.add_argument(
        "--baud", type=int, default=DEFAULT_BAUD, help=f"Baud (default {DEFAULT_BAUD})"
    )
    parser.add_argument(
        "--out", type=Path, default=BACKUP_ROOT, help="Backup root folder"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip the read-back verification pass"
    )
    parser.add_argument("--flash-size", default=None, help="Override size, e.g. 16MB")
    args = parser.parse_args()

    port = resolve_port(args.port)
    out_dir = args.out / timestamp_slug()
    suffix = 1
    while out_dir.exists():
        out_dir = args.out / f"{timestamp_slug()}_{suffix}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    flash_path = out_dir / "flash.bin"
    part_path = out_dir / "flash.bin.part"

    print("=== ESP32-S3 FULL-FLASH BACKUP (READ-ONLY) ===")
    print(f"  Port     : {port}")
    print(f"  Baud     : {args.baud}")
    print(f"  Output   : {out_dir}")
    print()

    try:
        id_proc = run_esptool(
            ["--chip", "esp32s3", "--port", port, "--baud", str(args.baud), "flash-id"]
        )
        id_text = (id_proc.stdout or "") + (id_proc.stderr or "")
        facts = parse_flash_id(id_text)

        mac_proc = run_esptool(
            ["--chip", "esp32s3", "--port", port, "--baud", str(args.baud), "read-mac"]
        )
        mac_text = (mac_proc.stdout or "") + (mac_proc.stderr or "")
        mac = None
        for line in mac_text.splitlines():
            if "MAC:" in line:
                mac = normalize_mac(line.split("MAC:", 1)[1])
                break

        label = args.flash_size or facts.get("flash_size")
        total = flash_size_bytes(label)
        if total > 16 * 1024 * 1024:
            raise RuntimeError(
                f"Refusing suspicious flash size {label} ({total} bytes)"
            )
        size_label = f"{total // (1024 * 1024)}MB"
        print(
            f"\n  Chip     : {facts.get('chip', 'ESP32-S3')} rev {facts.get('chip_revision', '?')}"
        )
        print(f"  MAC      : {mac or '?'}")
        print(
            f"  Flash    : {facts.get('flash_size', label or '?')} "
            f"(manuf {facts.get('manufacturer', '?')}, device {facts.get('device', '?')})"
        )
        print(f"  Reading  : 0x00000000 .. 0x{total:08X} ({human_bytes(total)})")
        print("  This is non-destructive. Do not unplug the board.\n")

        run_esptool(
            [
                "--chip",
                "esp32s3",
                "--port",
                port,
                "--baud",
                str(args.baud),
                "read-flash",
                "--flash-size",
                size_label,
                "0x0",
                hex(total),
                str(part_path),
            ],
            capture=False,
        )
        if not part_path.exists():
            raise RuntimeError("read-flash reported success but produced no file")
        actual = part_path.stat().st_size
        if actual != total:
            raise RuntimeError(f"Backup size mismatch: expected {total}, got {actual}")

        digest = sha256_file(part_path)
        part_path.replace(flash_path)

        verified = False
        if not args.no_verify:
            print("\n  Verifying by reading the flash back and comparing...")
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
                    size_label,
                    "0x0",
                    str(flash_path),
                ],
                capture=False,
            )
            verified = True
            print("  Verification: PASS (flash matches backup byte-for-byte)\n")

        manifest = {
            "schema": 1,
            "created": out_dir.name,
            "port": port,
            "baud": args.baud,
            "chip": facts.get("chip", "ESP32-S3"),
            "chip_revision": facts.get("chip_revision"),
            "mac": mac,
            "flash_size": label,
            "flash_bytes": total,
            "file": flash_path.name,
            "sha256": digest,
            "verified": verified,
            "git": git_state(),
            "note": "Full 16 MB image. Restore only to the matching MAC/board.",
        }
        write_manifest(out_dir / "manifest.json", manifest)

        print("=== BACKUP COMPLETE ===")
        print(f"  Folder   : {out_dir}")
        print(f"  File     : {flash_path.name} ({human_bytes(total)})")
        print(f"  SHA-256  : {digest}")
        print(f"  Verified : {verified}")
        print("\n  Restore with:")
        print(
            f'    python tools/board_restore.py --backup "{out_dir}" --confirm RESTORE'
        )
        print("\n  Keep this folder safe. It is intentionally git-ignored.\n")
        return 0
    except Exception as exc:
        print(f"\n[BACKUP FAILED] {exc}", file=sys.stderr)
        if part_path.exists():
            print(f"  Partial file kept for diagnosis: {part_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
