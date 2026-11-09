"""Wire protocol codec (32-byte SAMPLE & RESULT frames) and Windows 11 timer setup."""
from __future__ import annotations

import ctypes
import struct
import sys
import time
from dataclasses import dataclass
from typing import Tuple

MAGIC_SAMPLE = 0xA5C3
MAGIC_RESULT = 0x5C3A
PROTOCOL_VER = 1

MODE_LOCKSTEP = 0
MODE_REALTIME = 1
MODE_STANDALONE = 2

STATE_IDLE = 0
STATE_ARMED = 1
STATE_SUPPRESS = 2
STATE_REFRACTORY = 3
STATE_FAULT = 4

# Little-endian 32-byte struct formats matching #pragma pack(1) in nl_protocol.h
_SAMPLE_FMT_NO_CRC = "<HBBIIIBBBBhhhh"  # 28 bytes (0..27)
_SAMPLE_FMT_FULL = "<HBBIIIBBBBhhhhHH"  # 32 bytes (0..31)
_RESULT_FMT_FULL = "<HBBIIIIIBBhhH"     # 32 bytes (0..31)

assert struct.calcsize(_SAMPLE_FMT_NO_CRC) == 28
assert struct.calcsize(_SAMPLE_FMT_FULL) == 32
assert struct.calcsize(_RESULT_FMT_FULL) == 32


def enable_windows_1ms_timer() -> bool:
    """DEC-009: Enable 1 ms multimedia timer resolution on Windows 11."""
    if sys.platform == "win32":
        try:
            winmm = ctypes.WinDLL("winmm")
            res = winmm.timeBeginPeriod(1)
            return res == 0
        except Exception:
            return False
    return True


def host_monotonic_us() -> int:
    """Return monotonic host timestamp in microseconds (uint32 wrapped)."""
    return (time.perf_counter_ns() // 1_000) & 0xFFFFFFFF


def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly=0x1021, init=0xFFFF, refin=false, refout=false, xorout=0x0000)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class SampleFrame:
    seq: int
    sim_t_us: int
    host_t_us: int
    samples: Tuple[int, int, int, int]
    flags: int = 0
    n_ch: int = 2
    n_samp: int = 2
    mode: int = MODE_LOCKSTEP
    ver: int = PROTOCOL_VER

    def to_bytes(self) -> bytes:
        s0, s1, s2, s3 = (int(x) for x in self.samples)
        header_and_payload = struct.pack(
            _SAMPLE_FMT_NO_CRC,
            MAGIC_SAMPLE,
            self.ver & 0xFF,
            self.flags & 0xFF,
            self.seq & 0xFFFFFFFF,
            self.sim_t_us & 0xFFFFFFFF,
            self.host_t_us & 0xFFFFFFFF,
            self.n_ch & 0xFF,
            self.n_samp & 0xFF,
            self.mode & 0xFF,
            0,
            s0, s1, s2, s3,
        )
        crc = crc16_ccitt_false(header_and_payload)
        return header_and_payload + struct.pack("<HH", crc, 0)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SampleFrame":
        if len(raw) != 32:
            raise ValueError(f"Expected 32-byte SAMPLE frame, got {len(raw)} bytes")
        (
            magic, ver, flags, seq, sim_t_us, host_t_us,
            n_ch, n_samp, mode, _rsv, s0, s1, s2, s3, crc, _pad
        ) = struct.unpack(_SAMPLE_FMT_FULL, raw)
        if magic != MAGIC_SAMPLE:
            raise ValueError(f"Bad SAMPLE magic: 0x{magic:04X}")
        if ver != PROTOCOL_VER:
            raise ValueError(f"Protocol version mismatch: got {ver}, expected {PROTOCOL_VER}")
        expected_crc = crc16_ccitt_false(raw[:28])
        if crc != expected_crc:
            raise ValueError(f"CRC mismatch: got 0x{crc:04X}, expected 0x{expected_crc:04X}")
        return cls(
            seq=seq,
            sim_t_us=sim_t_us,
            host_t_us=host_t_us,
            samples=(s0, s1, s2, s3),
            flags=flags,
            n_ch=n_ch,
            n_samp=n_samp,
            mode=mode,
            ver=ver,
        )


@dataclass(frozen=True)
class ResultFrame:
    state: int
    seq: int
    t_rx_us: int
    t_dsp_us: int
    t_dec_us: int
    t_stim_us: int
    cls: int
    conf: int
    phase_q15: int
    R_q15: int
    budget_used: int
    ver: int = PROTOCOL_VER

    def to_bytes(self) -> bytes:
        return struct.pack(
            _RESULT_FMT_FULL,
            MAGIC_RESULT,
            self.ver & 0xFF,
            self.state & 0xFF,
            self.seq & 0xFFFFFFFF,
            self.t_rx_us & 0xFFFFFFFF,
            self.t_dsp_us & 0xFFFFFFFF,
            self.t_dec_us & 0xFFFFFFFF,
            self.t_stim_us & 0xFFFFFFFF,
            self.cls & 0xFF,
            self.conf & 0xFF,
            self.phase_q15,
            self.R_q15,
            self.budget_used & 0xFFFF,
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ResultFrame":
        if len(raw) != 32:
            raise ValueError(f"Expected 32-byte RESULT frame, got {len(raw)} bytes")
        (
            magic, ver, state, seq, t_rx_us, t_dsp_us, t_dec_us, t_stim_us,
            cls_val, conf, phase_q15, R_q15, budget_used
        ) = struct.unpack(_RESULT_FMT_FULL, raw)
        if magic != MAGIC_RESULT:
            raise ValueError(f"Bad RESULT magic: 0x{magic:04X}")
        if ver != PROTOCOL_VER:
            raise ValueError(f"Protocol version mismatch: got {ver}, expected {PROTOCOL_VER}")
        return cls(
            state=state,
            seq=seq,
            t_rx_us=t_rx_us,
            t_dsp_us=t_dsp_us,
            t_dec_us=t_dec_us,
            t_stim_us=t_stim_us,
            cls=cls_val & 0xFF,
            conf=conf & 0xFF,
            phase_q15=phase_q15,
            R_q15=R_q15,
            budget_used=budget_used,
            ver=ver,
        )
