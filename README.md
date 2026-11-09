> **ETHICS & SCOPE DISCLOSURE — READ FIRST:** **This is a simulation testbed.** No human or animal tissue is recorded, stimulated, or involved at any stage. No electrodes are used. The "brain" is a numerical model on a laptop; the "stimulation" is a 3.3 V logic pulse sent back to that laptop over USB. Public, de-identified datasets are used offline for model development only. Nothing here is a medical device, a therapy, or a clinical claim.

# Neural Loop: Closed-Loop Edge Neuromodulation Testbed on ESP32-S3

**One-line pitch:** *A single ESP32-S3-WROOM-1-N16R8 runs a nested two-loop adaptive neuromodulation testbed—predicting pre-ictal transitions seconds in advance (int8 1D-CNN) to arm a sub-5 ms fast loop (61-tap quadrature FIR + 64-neuron int8 LIF Spiking Neural Network + 20 ms artifact blanking) that phase-locks pulses to an empirically identified Phase-Response Curve (PRC).*

---

## Verified Hardware Target (Probed in Week 0)

| Property | Verified Hardware Value |
|---|---|
| **MCU** | `ESP32-S3 (QFN56) rev v0.2` (Dual-Core Xtensa LX7 @ 240 MHz + 128-bit SIMD) |
| **Module Variant** | `ESP32-S3-WROOM-1-N16R8` (`MAC: 84:c7:bb:58:23:ec`) |
| **Flash** | `16 MB Quad SPI (3.3V)` (`Manuf 0x68, Dev 0x4018`) |
| **PSRAM** | `8 MB Octal Embedded PSRAM (AP_3v3)` |
| **Host Interface** | `WCH CH343` High-Speed USB-UART (`COM6`, up to 6 Mbps) + Native USB-OTG (`GPIO19/20`) |

---

## Repository Layout

```text
neural-loop/
├── firmware/                  # ESP-IDF C/C++ project
│   ├── main/                  # FreeRTOS tasks, watchdog, USB transport
│   └── components/
│       ├── link/              # 32-byte wire protocol & CRC16-CCITT-FALSE
│       ├── dsp/               # Q15 quadrature FIR, CORDIC/LUT phase, 20ms artifact blanking
│       └── snn/               # 64-neuron int8 Leaky Integrate-and-Fire (LIF) kernel
├── host/
│   ├── hil/                   # Protocol codec, Windows timeBeginPeriod(1), transport
│   ├── virtualbrain/          # Stage 1 Kuramoto + Stage 2 Jansen-Rit NMM + artifact model
│   ├── replay/                # CHB-MIT 8-channel bipolar montage pipeline
│   └── analysis/              # PRC identification, 5-arm bake-off, bootstrap CI
├── tests/
│   ├── golden/                # Committed byte-exact parity fixtures (CRC16, frames, hashes)
│   └── test_suite.py          # Automated verification of Gates G1, G3, G4
└── experiments/               # Pre-registered experiment configs
```
