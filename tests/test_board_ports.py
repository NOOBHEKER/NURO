"""Hermetic tests for the single serial-port policy.

These do not open a serial port and do not need the board: they feed the
ranking functions a list of synthetic ports and assert the choice.
"""

from __future__ import annotations

import os

import pytest

import host.board as board
from host.board import SerialPort, detect_board_port, parse_flash_id, resolve_port


def _port(
    device: str,
    vid: int | None = None,
    pid: int | None = None,
    description: str = "",
    hwid: str = "",
) -> SerialPort:
    return SerialPort(
        device=device,
        description=description,
        hwid=hwid,
        vid=vid,
        pid=pid,
    )


CH343 = _port(
    "COM6", 0x1A86, 0x55D3, "USB-Enhanced-SERIAL CH343", "USB VID:PID=1A86:55D3"
)
CP210X = _port("COM3", 0x10C4, 0xEA60, "Silicon Labs CP210x", "USB VID:PID=10C4:EA60")
CH340 = _port("COM4", 0x1A86, 0x7523, "USB-SERIAL CH340", "USB VID:PID=1A86:7523")
NATIVE = _port(
    "COM7", 0x303A, 0x1001, "USB JTAG/serial debug unit", "USB VID:PID=303A:1001"
)


def _use_ports(monkeypatch: pytest.MonkeyPatch, ports: list[SerialPort]) -> None:
    monkeypatch.setattr(board, "list_serial_ports", lambda: ports)


def test_auto_detect_prefers_the_on_board_ch343(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_ports(monkeypatch, [CP210X, CH340, CH343])
    assert detect_board_port() == "COM6"


def test_auto_detect_falls_back_to_native_usb(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_ports(monkeypatch, [CP210X, NATIVE])
    assert detect_board_port() == "COM7"


def test_native_usb_is_flagged_but_a_bridge_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_ports(monkeypatch, [NATIVE])
    assert board.is_native_usb("COM7")
    assert not board.is_native_usb("COM99")
    _use_ports(monkeypatch, [NATIVE, CH343])
    assert detect_board_port() == "COM6"
    assert not board.is_native_usb("COM6")


def test_unknown_ports_are_never_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_ports(monkeypatch, [_port("COM1", description="Communications Port")])
    assert detect_board_port() is None


def test_explicit_port_beats_environment_and_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_ports(monkeypatch, [CH343])
    monkeypatch.setenv("NURO_PORT", "COM9")
    assert resolve_port("COM1") == "COM1"


def test_environment_port_beats_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_ports(monkeypatch, [CH343])
    monkeypatch.setenv("NURO_PORT", "COM9")
    assert resolve_port() == "COM9"


def test_legacy_environment_alias_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_ports(monkeypatch, [CH343])
    monkeypatch.delenv("NURO_PORT", raising=False)
    monkeypatch.setenv("NL_PORT", "COM11")
    assert resolve_port() == "COM11"


def test_fallback_is_platform_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_ports(monkeypatch, [])
    monkeypatch.delenv("NURO_PORT", raising=False)
    monkeypatch.delenv("NL_PORT", raising=False)
    expected = "COM6" if os.name == "nt" else ""
    assert resolve_port() == expected


def test_posix_device_paths_are_recognised_without_vid_pid() -> None:
    assert board.score_port(_port("/dev/ttyUSB0")) >= 60
    assert board.score_port(_port("/dev/ttyACM0")) >= 70
    assert board.score_port(_port("/dev/cu.usbmodem14201")) >= 70


def test_driver_hint_is_actionable_when_nothing_is_plugged_in() -> None:
    hint = board.driver_hint([])
    assert hint
    assert "port" in hint.lower() or "cable" in hint.lower()


def test_parse_flash_id_reads_esptool_v5_output() -> None:
    sample = """
    Chip type:          ESP32-S3 (QFN56) (revision v0.2)
    MAC:                84:c7:bb:58:23:ec
    Detected flash size: 16MB
    Flash type set in eFuse: quad (4 data lines)
    Features:           Wi-Fi, BT 5 (LE), Dual Core + LP Core, 240MHz, Embedded PSRAM 8MB (AP_3v3)
    Manufacturer:       0x68
    Device:             0x4018
    """
    facts = parse_flash_id(sample)
    assert facts["chip"] == "ESP32-S3"
    assert facts["chip_revision"] == "v0.2"
    assert facts["mac"] == "84:c7:bb:58:23:ec"
    assert facts["flash_size"] == "16MB"
    assert "PSRAM 8MB" in facts["features"]
    assert facts["manufacturer"] == "68"
    assert facts["device"] == "4018"
