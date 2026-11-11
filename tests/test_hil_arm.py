"""Hermetic tests for the arm-gate REPL handshake.

These never open a serial port: a fake serial object is injected, so the suite
stays hardware-free and safe.
"""

from __future__ import annotations

from host.hil.transport import send_repl_command


class FakeSerial:
    """Minimal stand-in for ``serial.Serial`` with a scripted response."""

    def __init__(self, response: bytes) -> None:
        self.response = response
        self.written = b""
        self.reset_calls = 0

    def reset_input_buffer(self) -> None:
        self.reset_calls += 1

    def write(self, data: bytes) -> int:
        self.written += data
        return len(data)

    def read(self, _size: int) -> bytes:
        return self.response


def test_arm_command_is_written_with_crlf() -> None:
    ser = FakeSerial(b"\r\n  [ARM] GPIO4 stimulation marker ENABLED (green LED).\r\n")
    assert send_repl_command(ser, "arm", b"[ARM]") is True
    assert ser.written == b"arm\r\n"
    assert ser.reset_calls == 1


def test_missing_acknowledgement_returns_false() -> None:
    ser = FakeSerial(b"\r\nesp32-s3> ")
    assert send_repl_command(ser, "arm", b"[ARM]", timeout_s=0.05) is False


def test_disarm_acknowledgement_is_matched_exactly() -> None:
    ser = FakeSerial(
        b"  [DISARM] GPIO4 stimulation marker DISABLED (blue LED). Board is safe.\r\n"
    )
    assert send_repl_command(ser, "disarm", b"[DISARM]", timeout_s=0.05) is True
    assert ser.written == b"disarm\r\n"
