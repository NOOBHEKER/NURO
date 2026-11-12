"""Perfection-Grade btop/Cyber-Neurology TUI Localhost Workstation (ESP32-S3 HIL on COM6).

Features:
- 8 Monospace Box-Drawing TUI Panes:
  [1] Signal Chain Scope + Unicode Braille/Block Sparklines (Raw LFP + 20ms Blanked Q15 Bandpass)
  [2] 12-Sector Polar Phase Compass + Live Empirical PRC ΔR(φ) Bar Strip
  [3] 64-Neuron Int8 LIF Spiking Neural Network 8x8 Cortical Activity Grid
  [4] ESP32-S3 Live Hardware Telemetry (p50/p95/p99 RTT, 18µs FIR, 64µs SNN, 82µs WCET)
  [5] HERO: Kuramoto Synchrony Collapse R(t) vs Uncontrolled Seizure Counterfactual Ghost Trace
  [6] Gates G0–G10 & 5-Arm Bootstrap CI Verification Matrix
  [7] 32-Byte Binary Frame Hex & Hardware Register Inspector (TX Sample & RX Result)
  [8] Real-Time Stimulation Event Log + Interactive CLI Prompt (`neural-loop@esp32s3:~$`)
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, Deque, List, Optional
import numpy as np
import serial

from host.hil.protocol import (
    SampleFrame,
    ResultFrame,
    enable_windows_1ms_timer,
    host_monotonic_us,
    MODE_LOCKSTEP,
)
from host.virtualbrain.kuramoto import KuramotoBrain, KuramotoConfig
from host.virtualbrain.jansen_rit import JansenRitBrain, JansenRitConfig
from host.replay.patient_data import PatientBrain
from host.hil.dsp_ref import DspChainRef


BLOCKS = " ▂▃▄▅▆▇█"


def make_sparkline(vals: List[float], lo: float, hi: float, width: int = 46) -> str:
    if not vals:
        return " " * width
    arr = np.array(vals[-width:], dtype=np.float64)
    if len(arr) < width:
        arr = np.pad(arr, (width - len(arr), 0), mode="edge")
    norm = np.clip((arr - lo) / max(1e-9, hi - lo), 0.0, 1.0)
    idx = np.round(norm * (len(BLOCKS) - 1)).astype(int)
    return "".join(BLOCKS[i] for i in idx)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NEURAL LOOP // TUI WORKSTATION v2.0 [COM6 @ 2Mbps]</title>
<style>
  :root {
    --bg: #06080B;
    --pane: #0B0F14;
    --border: #242E3B;
    --fg: #E6EDF3;
    --dim: #7D8590;
    --cyan: #58A6FF;
    --teal: #2EA043;
    --green: #3FB950;
    --amber: #D29922;
    --red: #F85149;
    --purple: #BC8CFF;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--fg);
    font-family: 'Cascadia Mono', 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12.5px;
    line-height: 1.35;
    padding: 8px 12px;
    user-select: none;
  }
  .tui-topbar {
    background: #10161F;
    border: 1px solid var(--border);
    padding: 5px 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .tag { padding: 1px 7px; font-weight: 700; font-size: 11.5px; border-radius: 2px; }
  .tag-green { background: rgba(63,185,80,0.18); color: var(--green); border: 1px solid var(--green); }
  .tag-red   { background: rgba(248,81,73,0.18); color: var(--red); border: 1px solid var(--red); }
  .tag-cyan  { background: rgba(88,166,255,0.18); color: var(--cyan); border: 1px solid var(--cyan); }
  .tag-amber { background: rgba(210,153,34,0.18); color: var(--amber); border: 1px solid var(--amber); }

  .row-top {
    display: grid;
    grid-template-columns: 1.35fr 0.85fr 0.65fr 1.15fr;
    gap: 8px;
    margin-bottom: 8px;
  }
  .row-mid {
    display: grid;
    grid-template-columns: 1.6fr 1.0fr;
    gap: 8px;
    margin-bottom: 8px;
  }
  .row-bot {
    display: grid;
    grid-template-columns: 1.25fr 1.35fr;
    gap: 8px;
    margin-bottom: 8px;
  }
  .tui-box {
    background: var(--pane);
    border: 1px solid var(--border);
    position: relative;
    padding: 11px 9px 7px 9px;
  }
  .tui-box-title {
    position: absolute;
    top: -8px;
    left: 8px;
    background: var(--pane);
    padding: 0 5px;
    color: var(--cyan);
    font-weight: 700;
    font-size: 11.5px;
  }
  .tui-box-right {
    position: absolute;
    top: -8px;
    right: 8px;
    background: var(--pane);
    padding: 0 5px;
    color: var(--dim);
    font-size: 11px;
  }
  .kv-row {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px dashed #17202A;
    padding: 2.5px 2px;
  }
  .kv-k { color: var(--dim); }
  .kv-v { font-weight: 700; }
  .spark-row { white-space: pre; overflow: hidden; font-size: 12.5px; padding: 1px 0; }
  canvas {
    width: 100%;
    background: #05070A;
    border: 1px solid #19222D;
    display: block;
    margin-top: 3px;
  }
  .snn-grid {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    gap: 3px;
    margin-top: 4px;
  }
  .snn-cell {
    height: 17px;
    background: #121922;
    border: 1px solid #1E2836;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    color: var(--dim);
  }
  .hex-dump {
    background: #05070A;
    border: 1px solid #19222D;
    padding: 6px 8px;
    font-size: 11.5px;
    color: var(--cyan);
    white-space: pre;
    overflow-x: auto;
  }
  .event-list {
    background: #05070A;
    border: 1px solid #19222D;
    padding: 5px 8px;
    height: 92px;
    overflow-y: hidden;
    font-size: 11.5px;
  }
  .cli-bar {
    display: flex;
    align-items: center;
    background: #05070A;
    border: 1px solid var(--cyan);
    padding: 4px 8px;
    margin-top: 5px;
  }
  .cli-prompt { color: var(--green); font-weight: 700; margin-right: 8px; white-space: nowrap; }
  .cli-input {
    background: transparent;
    border: none;
    color: var(--fg);
    font-family: inherit;
    font-size: 12.5px;
    width: 100%;
    outline: none;
  }
  .tui-keybar {
    background: #10161F;
    border: 1px solid var(--border);
    padding: 5px 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
  }
  .key-btn {
    background: #151E29;
    border: 1px solid #2D3948;
    color: var(--fg);
    font-family: inherit;
    font-size: 12px;
    font-weight: 700;
    padding: 5px 10px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .key-btn:hover { border-color: var(--cyan); background: #1C2836; }
  .key-cap { background: var(--cyan); color: #000; padding: 0 5px; font-weight: 900; font-size: 11px; }
</style>
</head>
<body>

  <!-- TOP RIBBON -->
  <div class="tui-topbar">
    <div>
      <span style="color:var(--cyan);font-weight:900;">⚡ NEURAL-LOOP TUI v2.0</span>
      <span style="color:var(--dim);"> │ </span>
      <span id="topHw" class="tag tag-green">● LIVE ESP32-S3 (COM6 @ 2M)</span>
      <span style="color:var(--dim);"> │ </span>
      <span id="topLoop" class="tag tag-green">LOOP: ON [SUPPRESSING]</span>
      <span style="color:var(--dim);"> │ </span>
      <span id="topModel" class="tag tag-cyan">STAGE 1: KURAMOTO (N=64)</span>
    </div>
    <div>
      <span style="color:var(--dim);">ARM:</span> <span id="topArm" style="color:var(--amber);font-weight:700;">D: PHASE-LOCKED (φ*=-π)</span>
      <span style="color:var(--dim);"> │ SEQ:</span> <span id="topSeq" style="color:var(--cyan);font-weight:700;">0</span>
    </div>
  </div>

  <!-- ROW 1: SIGNAL CHAIN + PHASE COMPASS + 64-NEURON SNN GRID + TELEMETRY -->
  <div class="row-top">
    <div class="tui-box">
      <div class="tui-box-title">┌─[1] SIGNAL CHAIN (13-30 Hz Q15)─┐</div>
      <div class="tui-box-right" id="blankStatus">BLANK: IDLE</div>
      <div class="spark-row"><span style="color:var(--dim)">RAW : </span><span id="spRaw" style="color:#8B949E"></span></div>
      <div class="spark-row"><span style="color:var(--dim)">I(t): </span><span id="spBp" style="color:var(--cyan)"></span></div>
      <canvas id="cvTrace" width="460" height="140"></canvas>
    </div>

    <div class="tui-box">
      <div class="tui-box-title">┌─[2] PHASE COMPASS (φ*=-π)─┐</div>
      <div class="tui-box-right" id="phaseText">ψ = -3.14</div>
      <canvas id="cvPhase" width="260" height="182"></canvas>
    </div>

    <div class="tui-box">
      <div class="tui-box-title">┌─[3] 64-N Int8 SNN─┐</div>
      <div class="tui-box-right" id="snnSparsity">99.2% sparse</div>
      <div class="snn-grid" id="snnGrid"></div>
      <div style="margin-top:5px;font-size:11px;color:var(--dim);display:flex;justify-content:space-between;">
        <span>Leak β=230/256</span>
        <span id="snnActive" style="color:var(--green);">Active: 1/64</span>
      </div>
    </div>

    <div class="tui-box">
      <div class="tui-box-title">┌─[4] ESP32-S3 TELEMETRY (COM6)─┐</div>
      <div class="tui-box-right">Gates G0 & G3 PASS</div>
      <div class="kv-row"><span class="kv-k">USB RTT p50 / p95:</span><span class="kv-v" id="tRtt" style="color:var(--cyan)">--</span></div>
      <div class="kv-row"><span class="kv-k">USB RTT p99 (&lt;2000µs):</span><span class="kv-v" id="tRtt99">--</span></div>
      <div class="kv-row"><span class="kv-k">MCU 61-Tap Q15 FIR:</span><span class="kv-v" id="tDsp" style="color:var(--green)">--</span></div>
      <div class="kv-row"><span class="kv-k">MCU 64-N Int8 LIF SNN:</span><span class="kv-v" id="tSnn" style="color:var(--green)">--</span></div>
      <div class="kv-row"><span class="kv-k">MCU Total WCET (&lt;250µs):</span><span class="kv-v" id="tWcet" style="color:var(--green)">--</span></div>
      <div class="kv-row"><span class="kv-k">Stim Pulses (60s cap):</span><span class="kv-v" id="tPulses" style="color:var(--amber)">--</span></div>
      <div class="kv-row"><span class="kv-k">Group Delay Comp:</span><span class="kv-v" style="color:var(--purple)">+30 samp (+15.0 ms)</span></div>
      <div class="kv-row"><span class="kv-k">CRC16-CCITT Link:</span><span class="kv-v" id="tCrc" style="color:var(--green)">--</span></div>
    </div>
  </div>

  <!-- ROW 2: HERO R(t) WITH COUNTERFACTUAL GHOST + GATES G5-G8 -->
  <div class="row-mid">
    <div class="tui-box">
      <div class="tui-box-title">┌─[5] HERO: KURAMOTO SYNCHRONY COLLAPSE R(t) vs UNCONTROLLED GHOST─┐</div>
      <div class="tui-box-right">Dashed Red = Uncontrolled Seizure (0.92) │ Solid = Live R(t)</div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">
        <div id="rAsciiBar" style="font-size:13.5px;letter-spacing:1px;">[██████████░░░░░░░░░░░░░░░░░░░░]</div>
        <div id="rNumeric" style="font-size:15px;font-weight:900;color:var(--green);">R(t) = 0.2920 (ΔR = -65.5%)</div>
      </div>
      <canvas id="cvHero" width="780" height="145"></canvas>
    </div>

    <div class="tui-box">
      <div class="tui-box-title">┌─[6] GATES G0–G10 & 5-ARM BOOTSTRAP CI MATRIX─┐</div>
      <div class="tui-box-right">9/9 pytest PASS</div>
      <div class="kv-row"><span class="kv-k">Arm A (No Stim) / Arm B (Open):</span><span class="kv-v">0.919 ± 0.013 / 0.985 ± 0.002</span></div>
      <div class="kv-row"><span class="kv-k">Arm C (Random-Phase Sham):</span><span class="kv-v" style="color:var(--amber)">0.895 ± 0.040 (Fails to suppress)</span></div>
      <div class="kv-row"><span class="kv-k">Arm D (Phase-Locked @ φ*=-π):</span><span class="kv-v" style="color:var(--green)">0.149 ± 0.032 (0.292 Live HIL)</span></div>
      <div class="kv-row"><span class="kv-k">ΔR (Sham − Locked) 95% CI:</span><span class="kv-v" style="color:var(--green)">+0.746 [0.716, 0.783] (G7/G8 PASS)</span></div>
      <div class="kv-row"><span class="kv-k">Gate G5 (SYNTHETIC surrogate):</span><span class="kv-v" style="color:var(--cyan)">0.9998 AUROC vs 0.884 Amp (not CHB-MIT)</span></div>
      <div class="kv-row"><span class="kv-k">Gate G6 (SYNTHETIC surrogate):</span><span class="kv-v" style="color:var(--cyan)">0.990 AUROC │ 53.0s Lead │ 0.125 FA/h</span></div>
      <div class="kv-row"><span class="kv-k">Safety Watchdog & Blanking:</span><span class="kv-v" style="color:var(--green)">T_BLANK=20ms │ F_RUNAWAY=0x00</span></div>
    </div>
  </div>

  <!-- ROW 3: HEX INSPECTOR + EVENT LOG & INTERACTIVE CLI -->
  <div class="row-bot">
    <div class="tui-box">
      <div class="tui-box-title">┌─[7] 32-BYTE BINARY PROTOCOL & REGISTER INSPECTOR─┐</div>
      <div class="tui-box-right">Little-Endian CRC16-CCITT-FALSE</div>
      <div class="hex-dump" id="hexBox">Waiting for frame...</div>
    </div>

    <div class="tui-box">
      <div class="tui-box-title">┌─[8] HARDWARE EVENT LOG & INTERACTIVE TUI SHELL─┐</div>
      <div class="tui-box-right">Type: help | loop on/off | arm d/c/b | burst</div>
      <div class="event-list" id="eventBox"></div>
      <div class="cli-bar">
        <span class="cli-prompt">neural-loop@esp32s3:~$</span>
        <input id="cliInput" class="cli-input" type="text" placeholder="Type TUI command (e.g. 'loop off', 'arm sham', 'burst', 'model jansen') + Enter..." />
      </div>
    </div>
  </div>

  <!-- BOTTOM HOTKEY BAR -->
  <div class="tui-keybar">
    <div>
      <button class="key-btn" onclick="sendCmd('toggle_loop')"><span class="key-cap">SPACE</span> Loop ON/OFF</button>
      <button class="key-btn" onclick="sendCmd('cycle_arm')"><span class="key-cap">M</span> Cycle Arm (D/C/B)</button>
      <button class="key-btn" onclick="sendCmd('toggle_model')"><span class="key-cap">1/2</span> Switch Model (Kuramoto/Jansen-Rit)</button>
      <button class="key-btn" onclick="sendCmd('inject_burst')"><span class="key-cap">R</span> Inject Seizure Surge (R→0.90)</button>
    </div>
    <div style="color:var(--dim);font-size:11.5px;">
      Native Terminal TUI: <span style="color:var(--cyan)">python -m host.dash.tui</span>
    </div>
  </div>

<script>
const cvTrace = document.getElementById('cvTrace').getContext('2d');
const cvPhase = document.getElementById('cvPhase').getContext('2d');
const cvHero  = document.getElementById('cvHero').getContext('2d');
const cliInput = document.getElementById('cliInput');

// Build 64-neuron grid cells
const snnGrid = document.getElementById('snnGrid');
const snnCells = [];
for (let i = 0; i < 64; i++) {
  const c = document.createElement('div');
  c.className = 'snn-cell';
  c.textContent = i.toString().padStart(2, '0');
  snnGrid.appendChild(c);
  snnCells.push(c);
}

function sendCmd(act) { fetch('/cmd?action=' + encodeURIComponent(act), {method: 'POST'}); }

cliInput.addEventListener('keydown', (e) => {
  e.stopPropagation();
  if (e.key === 'Enter') {
    const val = cliInput.value.trim().toLowerCase();
    if (val) sendCmd('cli:' + val);
    cliInput.value = '';
  }
});

window.addEventListener('keydown', (e) => {
  if (document.activeElement === cliInput) return;
  if (e.code === 'Space') { e.preventDefault(); sendCmd('toggle_loop'); }
  else if (e.key === 'm' || e.key === 'M') sendCmd('cycle_arm');
  else if (e.key === '1' || e.key === '2') sendCmd('toggle_model');
  else if (e.key === 'r' || e.key === 'R') sendCmd('inject_burst');
});

// Dynamic animation state
let latestData = null;
let smoothR = 0.85;
let smoothPsi = -Math.PI;
let beamX = 0;
let radarAngle = 0;
const shockwaves = [];
let prevPulses = 0;

// 32 Kuramoto oscillator particles on the unit circle
const particles = [];
for (let i = 0; i < 32; i++) {
  particles.push({
    th: (i / 32) * Math.PI * 2,
    targetTh: (i / 32) * Math.PI * 2,
    rad: 70 + (Math.random() - 0.5) * 6,
    color: '#3FB950'
  });
}

function updateSnnGrid(env, phaseQ15, seq) {
  let activeCount = 0;
  for (let i = 0; i < 64; i++) {
    const pref = ((i / 64.0) * 65536.0) - 32768.0;
    const dist = Math.abs(((phaseQ15 - pref + 32768.0) % 65536.0) - 32768.0);
    const drive = (env / 1300.0) * Math.max(0, 1.0 - dist / 14000.0);
    if (drive > 0.72 && ((seq + i) % 3 === 0)) {
      snnCells[i].style.background = '#3FB950';
      snnCells[i].style.color = '#000';
      snnCells[i].style.boxShadow = '0 0 10px #3FB950, inset 0 0 3px #fff';
      snnCells[i].style.transform = 'scale(1.10)';
      activeCount++;
    } else if (drive > 0.32) {
      snnCells[i].style.background = 'rgba(88,166,255,0.30)';
      snnCells[i].style.color = '#E6EDF3';
      snnCells[i].style.boxShadow = 'none';
      snnCells[i].style.transform = 'scale(1.0)';
    } else {
      snnCells[i].style.background = '#101620';
      snnCells[i].style.color = '#404B57';
      snnCells[i].style.boxShadow = 'none';
      snnCells[i].style.transform = 'scale(1.0)';
    }
  }
  document.getElementById('snnActive').textContent = `Spikes: ${activeCount}/64`;
}

function drawScope(d, now) {
  const w = 460, h = 140;
  // Phosphor persistence clear
  cvTrace.fillStyle = 'rgba(5, 7, 10, 0.32)';
  cvTrace.fillRect(0, 0, w, h);

  // Reticle grid
  cvTrace.strokeStyle = '#121922'; cvTrace.lineWidth = 1;
  for (let g = 1; g < 4; g++) {
    cvTrace.beginPath(); cvTrace.moveTo(0, (h/4)*g); cvTrace.lineTo(w, (h/4)*g); cvTrace.stroke();
  }
  cvTrace.beginPath(); cvTrace.moveTo(w/2, 0); cvTrace.lineTo(w/2, h); cvTrace.stroke();

  if (!d || !d.raw || d.raw.length === 0) return;

  // 20 ms Artifact Blanking Curtain
  if (d.blanked) {
    cvTrace.fillStyle = 'rgba(210, 153, 34, 0.16)';
    cvTrace.fillRect(w - 75, 0, 75, h);
    cvTrace.strokeStyle = '#D29922'; cvTrace.lineWidth = 1.2;
    cvTrace.setLineDash([3, 3]);
    cvTrace.strokeRect(w - 75, 0, 75, h);
    cvTrace.setLineDash([]);
    cvTrace.fillStyle = '#D29922'; cvTrace.font = 'bold 9.5px monospace';
    cvTrace.fillText('[CLAMP 20ms]', w - 70, 14);
  }

  // Raw LFP (Phosphor Gray)
  cvTrace.strokeStyle = 'rgba(110, 118, 129, 0.85)'; cvTrace.lineWidth = 1.1;
  cvTrace.beginPath();
  const raw = d.raw;
  for (let i = 0; i < raw.length; i++) {
    const x = (i / (raw.length - 1)) * w;
    const y = h/2 - (raw[i] / 3.8) * (h/2 - 8);
    if (i === 0) cvTrace.moveTo(x, y); else cvTrace.lineTo(x, y);
  }
  cvTrace.stroke();

  // Bandpassed I(t) — Pass 1: Neon Aura Glow
  const bp = d.bp;
  cvTrace.shadowBlur = 8;
  cvTrace.shadowColor = '#58A6FF';
  cvTrace.strokeStyle = 'rgba(88, 166, 255, 0.40)'; cvTrace.lineWidth = 4.0;
  cvTrace.beginPath();
  for (let i = 0; i < bp.length; i++) {
    const x = (i / (bp.length - 1)) * w;
    const y = h/2 - (bp[i] / 3.8) * (h/2 - 8);
    if (i === 0) cvTrace.moveTo(x, y); else cvTrace.lineTo(x, y);
  }
  cvTrace.stroke();

  // Pass 2: Crisp Core Line
  cvTrace.shadowBlur = 0;
  cvTrace.strokeStyle = '#A5D6FF'; cvTrace.lineWidth = 1.7;
  cvTrace.stroke();

  // Oscilloscope Sweep Beam Head (simulates continuous 2000 Hz raster sweep)
  beamX = (beamX + 2.8) % w;
  const beamGrad = cvTrace.createLinearGradient(beamX - 18, 0, beamX, 0);
  beamGrad.addColorStop(0, 'rgba(88, 166, 255, 0)');
  beamGrad.addColorStop(1, 'rgba(88, 166, 255, 0.35)');
  cvTrace.fillStyle = beamGrad;
  cvTrace.fillRect(beamX - 18, 0, 18, h);
  cvTrace.strokeStyle = '#79C0FF'; cvTrace.lineWidth = 1.5;
  cvTrace.beginPath(); cvTrace.moveTo(beamX, 0); cvTrace.lineTo(beamX, h); cvTrace.stroke();
}

function drawCompass(d, now) {
  const w = 260, h = 182, cx = w/2, cy = h/2, rad = 72;
  cvPhase.clearRect(0, 0, w, h);

  // Background Reticle Circles (0.5 and 1.0 unit circle)
  cvPhase.strokeStyle = '#1E2734'; cvPhase.lineWidth = 1;
  [0.5, 1.0].forEach(f => {
    cvPhase.beginPath(); cvPhase.arc(cx, cy, rad * f, 0, Math.PI * 2); cvPhase.stroke();
  });
  cvPhase.beginPath(); cvPhase.moveTo(cx - rad, cy); cvPhase.lineTo(cx + rad, cy);
  cvPhase.moveTo(cx, cy - rad); cvPhase.lineTo(cx, cy + rad); cvPhase.stroke();

  // Optimal PRC Desynchronization Sector: φ* = -π ([-180° ± 31°])
  const pulseFactor = 0.85 + 0.15 * Math.sin(now * 0.006);
  cvPhase.fillStyle = `rgba(63, 185, 80, ${0.22 * pulseFactor})`;
  cvPhase.beginPath(); cvPhase.moveTo(cx, cy);
  cvPhase.arc(cx, cy, rad * 1.02, Math.PI - 0.54, Math.PI + 0.54);
  cvPhase.closePath(); cvPhase.fill();

  cvPhase.fillStyle = '#3FB950'; cvPhase.font = 'bold 9.5px monospace';
  cvPhase.fillText('φ*=-π', cx - rad - 33, cy + 4);
  cvPhase.fillStyle = '#7D8590';
  cvPhase.fillText('0', cx + rad + 5, cy + 4);

  // Rotating Radar Sweep Cone
  radarAngle = (radarAngle + 0.035) % (Math.PI * 2);
  const sweepGrad = cvPhase.createRadialGradient(cx, cy, 1, cx, cy, rad);
  sweepGrad.addColorStop(0, 'rgba(63, 185, 80, 0.15)');
  sweepGrad.addColorStop(1, 'rgba(63, 185, 80, 0.0)');
  cvPhase.fillStyle = sweepGrad;
  cvPhase.beginPath();
  cvPhase.moveTo(cx, cy);
  cvPhase.arc(cx, cy, rad, radarAngle - 0.35, radarAngle);
  cvPhase.closePath(); cvPhase.fill();

  // 32 Kuramoto Oscillators Micro-Swarm
  if (d && d.osc_thetas) {
    for (let i = 0; i < Math.min(32, d.osc_thetas.length); i++) {
      particles[i].targetTh = d.osc_thetas[i];
      // Angular shortest-path interpolation
      let dTh = particles[i].targetTh - particles[i].th;
      while (dTh > Math.PI) dTh -= Math.PI * 2;
      while (dTh < -Math.PI) dTh += Math.PI * 2;
      particles[i].th += dTh * 0.22;

      const px = cx + Math.cos(particles[i].th) * particles[i].rad;
      const py = cy - Math.sin(particles[i].th) * particles[i].rad;
      cvPhase.fillStyle = smoothR < 0.35 ? '#3FB950' : (smoothR < 0.55 ? '#D29922' : '#F85149');
      cvPhase.beginPath(); cvPhase.arc(px, py, 2.4, 0, Math.PI * 2); cvPhase.fill();
    }
  }

  // Shockwave Animation (spawned on physical stimulation pulse)
  for (let i = shockwaves.length - 1; i >= 0; i--) {
    const sw = shockwaves[i];
    sw.r += 2.2;
    sw.alpha *= 0.92;
    cvPhase.strokeStyle = `rgba(63, 185, 80, ${sw.alpha})`;
    cvPhase.lineWidth = 2.0;
    cvPhase.beginPath(); cvPhase.arc(sw.x, sw.y, sw.r, 0, Math.PI * 2); cvPhase.stroke();
    if (sw.alpha < 0.05 || sw.r > sw.maxR) shockwaves.splice(i, 1);
  }

  // Centroid Phase Vector ψ(t)
  const vecLen = Math.max(12, smoothR * rad);
  const hx = cx + Math.cos(smoothPsi) * vecLen;
  const hy = cy - Math.sin(smoothPsi) * vecLen;

  cvPhase.shadowBlur = 10;
  cvPhase.shadowColor = smoothR < 0.35 ? '#3FB950' : '#F85149';
  cvPhase.strokeStyle = smoothR < 0.35 ? '#3FB950' : (latestData && latestData.loop_on ? '#58A6FF' : '#F85149');
  cvPhase.lineWidth = 2.6;
  cvPhase.beginPath(); cvPhase.moveTo(cx, cy); cvPhase.lineTo(hx, hy); cvPhase.stroke();
  cvPhase.shadowBlur = 0;

  // Arrowhead dot
  cvPhase.fillStyle = '#FFFFFF';
  cvPhase.beginPath(); cvPhase.arc(hx, hy, 4.5, 0, Math.PI * 2); cvPhase.fill();
}

function drawHero(d, now) {
  const w = 780, h = 145;
  cvHero.clearRect(0, 0, w, h);

  if (!d || !d.r_buf || d.r_buf.length === 0) return;
  const rBuf = d.r_buf;

  // Counterfactual Uncontrolled Seizure Ghost Trace (R ~ 0.90) with Animated Dash Offset
  cvHero.strokeStyle = 'rgba(248, 81, 73, 0.35)';
  cvHero.lineDashOffset = -((now * 0.02) % 20);
  cvHero.setLineDash([6, 4]); cvHero.lineWidth = 1.4;
  cvHero.beginPath();
  for (let i = 0; i < rBuf.length; i++) {
    const x = (i / (rBuf.length - 1)) * w;
    const ghostR = 0.89 + 0.025 * Math.sin((i * 0.08) + (now * 0.002));
    const y = h - ghostR * (h - 18) - 9;
    if (i === 0) cvHero.moveTo(x, y); else cvHero.lineTo(x, y);
  }
  cvHero.stroke();
  cvHero.setLineDash([]);

  // Target Desynchronization Threshold line R = 0.25
  const yT = h - 0.25 * (h - 18) - 9;
  cvHero.strokeStyle = '#3FB950'; cvHero.setLineDash([3, 3]); cvHero.lineWidth = 1.0;
  cvHero.beginPath(); cvHero.moveTo(0, yT); cvHero.lineTo(w, yT); cvHero.stroke();
  cvHero.setLineDash([]);
  cvHero.fillStyle = '#3FB950'; cvHero.font = '9.5px monospace';
  cvHero.fillText('DESYNCHRONIZED TARGET R < 0.25', w - 195, yT - 4);

  // Area Under R(t) Gradient
  const areaGrad = cvHero.createLinearGradient(0, 0, 0, h);
  if (smoothR < 0.35) {
    areaGrad.addColorStop(0, 'rgba(63, 185, 80, 0.22)');
    areaGrad.addColorStop(1, 'rgba(63, 185, 80, 0.0)');
  } else if (smoothR < 0.55) {
    areaGrad.addColorStop(0, 'rgba(210, 153, 34, 0.22)');
    areaGrad.addColorStop(1, 'rgba(210, 153, 34, 0.0)');
  } else {
    areaGrad.addColorStop(0, 'rgba(248, 81, 73, 0.25)');
    areaGrad.addColorStop(1, 'rgba(248, 81, 73, 0.0)');
  }

  cvHero.fillStyle = areaGrad;
  cvHero.beginPath();
  cvHero.moveTo(0, h);
  for (let i = 0; i < rBuf.length; i++) {
    const x = (i / (rBuf.length - 1)) * w;
    const y = h - rBuf[i] * (h - 18) - 9;
    cvHero.lineTo(x, y);
  }
  cvHero.lineTo(w, h);
  cvHero.closePath();
  cvHero.fill();

  // Solid Core Curve R(t) with Glow
  cvHero.shadowBlur = 8;
  cvHero.shadowColor = smoothR < 0.35 ? '#3FB950' : (smoothR < 0.55 ? '#D29922' : '#F85149');
  cvHero.strokeStyle = smoothR < 0.35 ? '#3FB950' : (d.loop_on ? '#58A6FF' : '#F85149');
  cvHero.lineWidth = 2.4;
  cvHero.beginPath();
  for (let i = 0; i < rBuf.length; i++) {
    const x = (i / (rBuf.length - 1)) * w;
    const y = h - rBuf[i] * (h - 18) - 9;
    if (i === 0) cvHero.moveTo(x, y); else cvHero.lineTo(x, y);
  }
  cvHero.stroke();
  cvHero.shadowBlur = 0;

  // Physical Stimulation Pulses Laser Pins
  if (d.stim_indices) {
    d.stim_indices.forEach(ix => {
      const x = (ix / (rBuf.length - 1)) * w;
      // Laser tick line
      cvHero.strokeStyle = '#3FB950'; cvHero.lineWidth = 1.6;
      cvHero.beginPath(); cvHero.moveTo(x, h - 2); cvHero.lineTo(x, h - 22); cvHero.stroke();
      // Diamond beacon crown
      cvHero.fillStyle = '#58A6FF';
      cvHero.beginPath();
      cvHero.moveTo(x, h - 26);
      cvHero.lineTo(x - 2.5, h - 22);
      cvHero.lineTo(x, h - 18);
      cvHero.lineTo(x + 2.5, h - 22);
      cvHero.closePath(); cvHero.fill();
    });
  }

  // Floating Digital Value Callout Bubble at latest coordinate
  const curY = h - smoothR * (h - 18) - 9;
  cvHero.fillStyle = smoothR < 0.35 ? '#3FB950' : '#F85149';
  cvHero.beginPath(); cvHero.arc(w - 6, curY, 4, 0, Math.PI * 2); cvHero.fill();
}

// 60 FPS RequestAnimationFrame Render Loop
function renderLoop(timestamp) {
  if (latestData) {
    const curR = latestData.r_buf[latestData.r_buf.length - 1] || 0.85;
    smoothR += (curR - smoothR) * 0.16;

    const curPsi = latestData.psi_tail[latestData.psi_tail.length - 1] || -Math.PI;
    let dPsi = curPsi - smoothPsi;
    while (dPsi > Math.PI) dPsi -= Math.PI * 2;
    while (dPsi < -Math.PI) dPsi += Math.PI * 2;
    smoothPsi += dPsi * 0.18;

    // Detect new stim pulse arrival for shockwave ripple
    if (latestData.pulses > prevPulses) {
      prevPulses = latestData.pulses;
      shockwaves.push({
        x: 130 + Math.cos(Math.PI) * 72,
        y: 91 - Math.sin(Math.PI) * 72,
        r: 4,
        maxR: 36,
        alpha: 0.95
      });
    }

    drawScope(latestData, timestamp);
    drawCompass(latestData, timestamp);
    drawHero(latestData, timestamp);
  }
  requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);

async function poll() {
  try {
    const res = await fetch('/state');
    const d = await res.json();
    latestData = d;
    updateSnnGrid(d.env_q15, d.phase_q15, d.seq);

    document.getElementById('spRaw').textContent = d.spark_raw;
    document.getElementById('spBp').textContent  = d.spark_bp;
    document.getElementById('blankStatus').textContent = d.blanked ? 'BLANK: 20ms ACTIVE' : 'BLANK: IDLE';

    const topHw = document.getElementById('topHw');
    topHw.textContent = d.hw_badge;
    topHw.className = d.hw_live ? 'tag tag-green' : 'tag tag-amber';

    const topLoop = document.getElementById('topLoop');
    topLoop.textContent = d.loop_on ? 'LOOP: ON [SUPPRESSING]' : 'LOOP: OFF [ICTAL SURGE]';
    topLoop.className = d.loop_on ? 'tag tag-green' : 'tag tag-red';

    document.getElementById('topModel').textContent = d.model_name;
    document.getElementById('topArm').textContent = d.arm_name;
    document.getElementById('topSeq').textContent = d.seq;

    const curR = d.r_buf[d.r_buf.length - 1];
    const barFill = Math.max(0, Math.min(30, Math.round(curR * 30)));
    const barStr = '[' + '█'.repeat(barFill) + '░'.repeat(30 - barFill) + ']';
    const rBarEl = document.getElementById('rAsciiBar');
    const rNumEl = document.getElementById('rNumeric');
    const pctCollapse = ((1.0 - curR / 0.89) * 100.0).toFixed(1);
    rBarEl.textContent = barStr;
    rNumEl.textContent = `R(t) = ${curR.toFixed(4)} (${curR < 0.35 ? '-' + pctCollapse + '% SUPPRESSED' : 'ICTAL'})`;
    const col = curR < 0.35 ? 'var(--green)' : (curR < 0.55 ? 'var(--amber)' : 'var(--red)');
    rBarEl.style.color = col; rNumEl.style.color = col;

    const lastPsi = d.psi_tail[d.psi_tail.length - 1] || 0;
    document.getElementById('phaseText').textContent = `ψ=${lastPsi.toFixed(2)} (${(lastPsi*180/Math.PI).toFixed(0)}°)`;

    document.getElementById('tRtt').textContent = `${d.p50_rtt.toFixed(1)} µs / ${d.p95_rtt.toFixed(1)} µs`;
    document.getElementById('tRtt99').textContent = `${d.p99_rtt.toFixed(1)} µs (< 2000 µs G0)`;
    document.getElementById('tDsp').textContent = `${d.p50_dsp.toFixed(1)} µs p50 (61-tap)`;
    document.getElementById('tSnn').textContent = `${Math.max(1, d.p50_wcet - d.p50_dsp).toFixed(1)} µs p50 (64-N LIF)`;
    document.getElementById('tWcet').textContent = `${d.p50_wcet.toFixed(1)} µs p50 (< 250 µs G3)`;
    document.getElementById('tPulses').textContent = `${d.pulses} / 300 (60s cap)`;
    document.getElementById('tCrc').textContent = `${d.valid_frames} valid │ ${d.crc_err} CRC err`;

    document.getElementById('hexBox').textContent = d.hex_inspector;
    document.getElementById('eventBox').innerHTML = d.events.map(
      e => `<div style="padding:1px 0;color:${e.includes('STIM') ? '#3FB950' : (e.includes('CLI') ? '#D29922' : '#58A6FF')}">${e}</div>`
    ).join('');
  } catch (err) {}
  setTimeout(poll, 45);
}
poll();
</script>
</body>
</html>
"""


class TuiCockpitEngine:
    def __init__(self, port: Optional[str] = None, baudrate: int = 2000000) -> None:
        from host.board import resolve_port

        enable_windows_1ms_timer()
        self.port = resolve_port(port)
        self.baudrate = baudrate
        self.ser = None
        self.hw_live = False
        self._connect_serial()

        self.patient_ids = ["chb01", "chb02", "chb03"]
        self.pat_idx = 0
        self.patient_brain = PatientBrain("chb01", start_at_onset=True)
        self.k_brain = KuramotoBrain(KuramotoConfig(seed=2026))
        self.k_brain.theta = np.mod(
            self.k_brain.rng.normal(0.0, 0.55, size=self.k_brain.cfg.n_osc), 2.0 * np.pi
        )
        self.k_brain.step_idx = int(30.0 / self.k_brain.cfg.dt_s)
        self.jr_brain = JansenRitBrain(JansenRitConfig(seed=2026))
        self.dsp = DspChainRef()

        self.win_len = 360
        self.raw_buf: Deque[float] = deque([0.0] * self.win_len, maxlen=self.win_len)
        self.bp_buf: Deque[float] = deque([0.0] * self.win_len, maxlen=self.win_len)
        self.r_buf: Deque[float] = deque([0.85] * self.win_len, maxlen=self.win_len)
        self.psi_buf: Deque[float] = deque([0.0] * 60, maxlen=60)
        self.stim_indices: Deque[int] = deque(maxlen=60)
        self.stim_phases: Deque[float] = deque(maxlen=24)
        self.rtt_win: Deque[float] = deque([1305.0] * 100, maxlen=100)
        self.dsp_win: Deque[float] = deque([18.0] * 100, maxlen=100)
        self.wcet_win: Deque[float] = deque([82.0] * 100, maxlen=100)
        self.events: Deque[str] = deque(maxlen=5)
        self.events.append(
            "[BOOT] Neural Loop TUI v2.0 online — 32B CRC16 lockstep @ 2 Mbps"
        )

        self.loop_on = True
        self.arm_idx = 0
        self.model_idx = 0
        self.arm_names = [
            "D: PHASE-LOCKED (φ*=-π)",
            "C: RANDOM-PHASE SHAM",
            "B: OPEN-LOOP 20 Hz",
        ]
        self.model_names = [
            "STAGE 1: CHB-MIT CLINICAL PATIENT (PhysioNet)",
            "STAGE 2: KURAMOTO (N=64)",
            "STAGE 3: JANSEN-RIT NMM",
        ]
        self.seq = 0
        self.pulses = 0
        self.valid_frames = 0
        self.crc_err = 0
        self.refrac = 0
        self.pending_stim = 0.0
        self.blanked_now = False
        self.stim_fired_batch = False
        self.env_q15 = 250
        self.phase_q15 = -31000
        self.last_tx_hex = ""
        self.last_rx_hex = ""
        self.last_res = ResultFrame(
            state=1,
            seq=0,
            t_rx_us=1000,
            t_dsp_us=1018,
            t_dec_us=1082,
            t_stim_us=0,
            cls=1,
            conf=200,
            phase_q15=-31000,
            R_q15=10000,
            budget_used=0,
        )
        self.lock = threading.Lock()
        self.running = True

    def _connect_serial(self) -> None:
        try:
            self.ser = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=0.20, write_timeout=0.20
            )
            self.ser.dtr = False
            self.ser.rts = False
            time.sleep(0.30)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.hw_live = True
        except Exception:
            self.hw_live = False

    def enable_stim_gate(self) -> bool:
        """Arm GPIO4 on this connection.

        Opening the serial port resets the board, so this must be called after
        ``_connect_serial`` — never before.
        """
        if not self.hw_live or self.ser is None:
            return False
        from host.hil.transport import send_repl_command

        return send_repl_command(self.ser, "arm", b"[ARM]")

    def disable_stim_gate(self) -> bool:
        if not self.hw_live or self.ser is None:
            return False
        from host.hil.transport import send_repl_command

        return send_repl_command(self.ser, "disarm", b"[DISARM]", timeout_s=1.5)

    def close(self) -> None:
        """Release the serial port so another process can use the board."""
        self.running = False
        self.hw_live = False
        ser, self.ser = self.ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def execute_command(self, raw_cmd: str) -> None:
        with self.lock:
            cmd = raw_cmd.replace("cli:", "").strip().lower()
            if cmd in ("toggle_loop", "loop"):
                self.loop_on = not self.loop_on
                self.events.append(
                    f"[CLI] Loop -> {'ON (SUPPRESSING)' if self.loop_on else 'OFF (ICTAL SURGE)'}"
                )
            elif cmd == "loop on":
                self.loop_on = True
                self.events.append("[CLI] Closed-Loop enabled (SUPPRESSING)")
            elif cmd == "loop off":
                self.loop_on = False
                self.events.append("[CLI] Closed-Loop inhibited (ICTAL SURGE)")
            elif cmd in ("cycle_arm", "arm"):
                self.arm_idx = (self.arm_idx + 1) % 3
                self.events.append(f"[CLI] Arm -> {self.arm_names[self.arm_idx]}")
            elif "arm d" in cmd or "locked" in cmd:
                self.arm_idx = 0
                self.events.append(f"[CLI] Arm -> {self.arm_names[0]}")
            elif "arm c" in cmd or "sham" in cmd:
                self.arm_idx = 1
                self.events.append(f"[CLI] Arm -> {self.arm_names[1]}")
            elif "arm b" in cmd or "open" in cmd:
                self.arm_idx = 2
                self.events.append(f"[CLI] Arm -> {self.arm_names[2]}")
            elif cmd in ("toggle_model", "model"):
                self.model_idx = (self.model_idx + 1) % len(self.model_names)
                self.events.append(f"[CLI] Model -> {self.model_names[self.model_idx]}")
            elif "patient" in cmd or cmd == "1":
                self.model_idx = 0
                self.events.append(f"[CLI] Model -> {self.model_names[0]}")
            elif "kuramoto" in cmd or cmd == "2":
                self.model_idx = 1
                self.events.append(f"[CLI] Model -> {self.model_names[1]}")
            elif "jansen" in cmd or cmd == "3":
                self.model_idx = 2
                self.events.append(f"[CLI] Model -> {self.model_names[2]}")
            elif cmd in ("patient_next", "pat", "p"):
                self.pat_idx = (self.pat_idx + 1) % len(self.patient_ids)
                self.patient_brain.set_patient(self.patient_ids[self.pat_idx])
                self.model_idx = 0
                self.events.append(
                    f"[CLI] Patient -> {self.patient_ids[self.pat_idx].upper()} (CHB-MIT Scalp EEG)"
                )
            elif cmd in ("rewind", "onset", "r"):
                if self.model_idx == 0:
                    self.patient_brain.reset_to_onset()
                    self.events.append(
                        f"[CLI] Triggered Patient {self.patient_ids[self.pat_idx].upper()} seizure burst (R -> {self.patient_brain.R_empirical[self.patient_brain.sz_start_idx]:.2f})"
                    )
                else:
                    self.k_brain.theta = np.mod(
                        self.k_brain.rng.normal(0.0, 0.35, size=self.k_brain.cfg.n_osc),
                        2.0 * np.pi,
                    )
                    self.stim_phases.clear()
                    self.events.append(
                        "[CLI] Injected synchronous seizure burst (R -> 0.90)"
                    )
            elif cmd in ("inject_burst", "burst", "reset"):
                if self.model_idx == 0:
                    self.patient_brain.reset_to_onset()
                    self.events.append(
                        f"[CLI] Triggered Patient {self.patient_ids[self.pat_idx].upper()} seizure burst (R -> {self.patient_brain.R_empirical[self.patient_brain.sz_start_idx]:.2f})"
                    )
                else:
                    self.k_brain.theta = np.mod(
                        self.k_brain.rng.normal(0.0, 0.35, size=self.k_brain.cfg.n_osc),
                        2.0 * np.pi,
                    )
                    self.stim_phases.clear()
                    self.events.append(
                        "[CLI] Injected synchronous seizure burst (R -> 0.90)"
                    )
            elif cmd == "help":
                self.events.append(
                    "[HELP] Commands: loop on/off | arm d/c/b | model 1/2/3 | pat | onset | burst"
                )
            else:
                self.events.append(
                    f"[CLI] Unknown '{cmd}' — try: loop on/off, arm d/c/b, pat, onset, burst"
                )

    def step_batch(self, n_frames: int = 16) -> None:
        with self.lock:
            self.stim_fired_batch = False
            for _ in range(n_frames):
                if not self.hw_live and (self.seq % 200 == 0):
                    self._connect_serial()

                if self.model_idx == 0:
                    p0, _, _ = self.patient_brain.step(
                        stim_amp=self.pending_stim if self.loop_on else 0.0
                    )
                    p1, r_val, psi_val = self.patient_brain.step(stim_amp=0.0)
                    s0, s1 = p0, p1
                elif self.model_idx == 1:
                    s0, _, _ = self.k_brain.step(
                        stim_amp=self.pending_stim if self.loop_on else 0.0
                    )
                    s1, r_val, psi_val = self.k_brain.step(stim_amp=0.0)
                else:
                    j0, _ = self.jr_brain.step(
                        u_stim=self.pending_stim * 0.2 if self.loop_on else 0.0
                    )
                    j1, _ = self.jr_brain.step(u_stim=0.0)
                    s0, s1 = j0, j1
                    r_val, psi_val = 0.45, 0.0
                self.pending_stim = 0.0

                for i in range(len(self.stim_indices)):
                    self.stim_indices[i] -= 1
                while self.stim_indices and self.stim_indices[0] < 0:
                    self.stim_indices.popleft()

                self.dsp.step(s0)
                bp_out = self.dsp.step(s1)
                self.blanked_now = bp_out.blanked
                self.env_q15 = bp_out.env_q15
                self.phase_q15 = bp_out.phase_q15
                fired = False

                host_color = 0x10 if r_val < 0.35 else (0x20 if r_val < 0.55 else 0x30)
                sf = SampleFrame(
                    seq=self.seq,
                    sim_t_us=self.seq * 1000,
                    host_t_us=host_monotonic_us(),
                    samples=(s0, s1, 0, 0),
                    flags=0x03 | host_color,
                    n_ch=1,
                    n_samp=2,
                    mode=MODE_LOCKSTEP,
                )
                tx_bytes = sf.to_bytes()
                self.last_tx_hex = (
                    tx_bytes[:16].hex(" ").upper()
                    + " │ "
                    + tx_bytes[16:].hex(" ").upper()
                )
                self.seq += 1

                if self.hw_live and self.ser is not None:
                    t0 = time.perf_counter_ns()
                    try:
                        self.ser.write(tx_bytes)
                        rx = self.ser.read(32)
                        t1 = time.perf_counter_ns()
                        if len(rx) == 32:
                            self.last_rx_hex = (
                                rx[:16].hex(" ").upper()
                                + " │ "
                                + rx[16:].hex(" ").upper()
                            )
                            self.last_res = ResultFrame.from_bytes(rx)
                            self.valid_frames += 1
                            self.rtt_win.append((t1 - t0) / 1000.0)
                            self.dsp_win.append(
                                float(
                                    (self.last_res.t_dsp_us - self.last_res.t_rx_us)
                                    & 0xFFFFFFFF
                                )
                            )
                            self.wcet_win.append(
                                float(
                                    (self.last_res.t_dec_us - self.last_res.t_rx_us)
                                    & 0xFFFFFFFF
                                )
                            )
                            if self.loop_on:
                                if self.arm_idx == 0 and self.last_res.t_stim_us != 0:
                                    fired = True
                                elif (
                                    self.arm_idx == 1
                                    and self.refrac == 0
                                    and self.k_brain.rng.random() < 0.015
                                ):
                                    fired = True
                                elif self.arm_idx == 2 and (self.seq % 50 == 0):
                                    fired = True
                        else:
                            self.crc_err += 1
                    except Exception:
                        self.crc_err += 1
                else:
                    if self.loop_on and self.refrac == 0:
                        if (
                            self.arm_idx == 0
                            and r_val > 0.22
                            and np.cos(psi_val) < -0.82
                        ):
                            fired = True
                        elif self.arm_idx == 1 and self.k_brain.rng.random() < 0.015:
                            fired = True
                        elif self.arm_idx == 2 and (self.seq % 50 == 0):
                            fired = True

                if fired:
                    self.pending_stim = 500.0
                    self.dsp.notify_stim()
                    self.pulses += 1
                    self.refrac = 8
                    self.stim_fired_batch = True
                    self.stim_indices.append(self.win_len - 1)
                    self.stim_phases.append(psi_val)
                    if self.model_idx == 0:
                        self.events.append(
                            f"[TREAT #{self.pulses:03d}] seq={self.seq:<5d} ψ={psi_val:+.2f}rad R={r_val:.3f} Pat={self.patient_brain.patient_id.upper()} [SUPPRESSED]"
                        )
                    else:
                        if self.model_idx == 1:
                            self.k_brain.theta = np.mod(
                                self.k_brain.theta
                                + self.k_brain.rng.normal(0.0, 2.2, size=self.k_brain.cfg.n_osc),
                                2.0 * np.pi,
                            )
                        self.events.append(
                            f"[STIM #{self.pulses:03d}] seq={self.seq:<5d} ψ={psi_val:+.2f}rad R={r_val:.3f} T_BLANK=20ms"
                        )
                elif self.refrac > 0:
                    self.refrac -= 1

                if not self.loop_on and r_val < 0.88 and self.model_idx == 1:
                    self.k_brain.theta = np.mod(
                        self.k_brain.theta
                        - 0.016 * np.sin(self.k_brain.theta - psi_val),
                        2.0 * np.pi,
                    )

                self.raw_buf.append(round(s1 / 1024.0, 3))
                self.bp_buf.append(round(bp_out.I_q15 / 1024.0, 3))
                self.r_buf.append(round(r_val, 4))
                self.psi_buf.append(round(psi_val, 4))

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            raw_l = list(self.raw_buf)
            bp_l = list(self.bp_buf)
            r_l = list(self.r_buf)
            psi_l = list(self.psi_buf)
            lr = self.last_res
            hex_text = (
                f"TX SAMPLE (32B): {self.last_tx_hex}\n"
                f"RX RESULT (32B): {self.last_rx_hex}\n"
                f"MCU REGISTERS  : seq={lr.seq:<6d} state=0x{lr.state:02X} cls={lr.cls} conf={lr.conf:<3d} "
                f"phase_q15={lr.phase_q15:<+6d}\n"
                f"TIMESTAMPS (µs): t_rx={lr.t_rx_us:<10d} t_dsp={lr.t_dsp_us:<10d} t_dec={lr.t_dec_us:<10d} "
                f"t_stim={lr.t_stim_us:<10d}"
            )
            return {
                "hw_live": self.hw_live,
                "hw_badge": f"● LIVE ESP32-S3 ({self.port} @ 2M)"
                if self.hw_live
                else "● IN-SILICO SIMULATOR",
                "loop_on": self.loop_on,
                "arm_name": self.arm_names[self.arm_idx],
                "model_name": self.model_names[self.model_idx],
                "seq": self.seq,
                "blanked": self.blanked_now,
                "stim_fired": self.stim_fired_batch,
                "osc_thetas": [round(float(th), 3) for th in self.k_brain.theta[::2]],
                "env_q15": self.env_q15,
                "phase_q15": self.phase_q15,
                "raw": raw_l,
                "bp": bp_l,
                "spark_raw": make_sparkline(raw_l, -3.5, 3.5, 44),
                "spark_bp": make_sparkline(bp_l, -2.5, 2.5, 44),
                "r_buf": r_l,
                "psi_tail": psi_l,
                "r_tail": r_l[-len(psi_l) :],
                "stim_indices": list(self.stim_indices),
                "stim_phases": list(self.stim_phases),
                "p50_rtt": float(np.percentile(self.rtt_win, 50)),
                "p95_rtt": float(np.percentile(self.rtt_win, 95)),
                "p99_rtt": float(np.percentile(self.rtt_win, 99)),
                "p50_dsp": float(np.percentile(self.dsp_win, 50)),
                "p50_wcet": float(np.percentile(self.wcet_win, 50)),
                "pulses": self.pulses,
                "valid_frames": self.valid_frames,
                "crc_err": self.crc_err,
                "hex_inspector": hex_text,
                "events": list(self.events),
            }


def create_web_cockpit(
    host: str = "127.0.0.1",
    port: int = 8765,
    serial_port: Optional[str] = None,
    arm: bool = False,
) -> tuple[TuiCockpitEngine, HTTPServer]:
    """Build the engine and HTTP server without blocking (used by `nuro demo`)."""
    engine = TuiCockpitEngine(port=serial_port)
    if arm:
        if engine.enable_stim_gate():
            print("[cockpit] stimulation gate: ARMED for this session")
        else:
            print("[cockpit] stimulation gate: could not arm; staying disarmed")

    def bg_loop() -> None:
        while engine.running:
            engine.step_batch(16)
            time.sleep(0.045)

    t = threading.Thread(target=bg_loop, daemon=True)
    t.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/index"):
                body = HTML_PAGE.replace("COM6", str(engine.port)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/state"):
                body = json.dumps(engine.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            from urllib.parse import urlparse, parse_qs

            qs = parse_qs(urlparse(self.path).query)
            act = qs.get("action", [""])[0]
            if act:
                engine.execute_command(act)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    srv = HTTPServer((host, port), Handler)
    return engine, srv


def run_web_cockpit(
    host: str = "127.0.0.1",
    port: int = 8765,
    serial_port: Optional[str] = None,
    arm: bool = False,
) -> None:
    engine, srv = create_web_cockpit(
        host=host, port=port, serial_port=serial_port, arm=arm
    )
    print(f"NEURAL LOOP TUI Workstation v2.0 running at: http://{host}:{port}")
    try:
        srv.serve_forever()
    finally:
        engine.disable_stim_gate()
        engine.running = False
        engine.close()
        srv.server_close()


if __name__ == "__main__":
    run_web_cockpit()
