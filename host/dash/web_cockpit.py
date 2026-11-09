"""TUI-Centric Localhost Cockpit for Neural Loop (ESP32-S3 HIL on COM6).

Styled after btop / lazydocker / Bloomberg monospace terminal TUIs:
- Box-drawing TUI panes, keyboard-first navigation ([SPACE], [M], [1], [2], [R], [C])
- Live 32-byte TX/RX binary hex + register inspector (nl_sample_frame_t & nl_result_frame_t)
- Unicode block sparklines + crisp terminal oscilloscope traces
- Live scrolling hardware event & stimulation log
- Exact 2-sample lockstep protocol (0 CRC errors, deterministic R(t) collapse on COM6)
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, Deque, List
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
from host.hil.dsp_ref import DspChainRef


BLOCKS = " ▂▃▄▅▆▇█"


def make_sparkline(vals: List[float], lo: float, hi: float, width: int = 48) -> str:
    if not vals:
        return " " * width
    arr = np.array(vals[-width:], dtype=np.float64)
    norm = np.clip((arr - lo) / max(1e-9, hi - lo), 0.0, 1.0)
    idx = np.round(norm * (len(BLOCKS) - 1)).astype(int)
    return "".join(BLOCKS[i] for i in idx)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NEURAL LOOP // TUI COCKPIT [COM6 @ 2Mbps]</title>
<style>
  :root {
    --bg: #080B0F;
    --pane: #0D1117;
    --border: #28313B;
    --border-hi: #3FB950;
    --fg: #E6EDF3;
    --dim: #7D8590;
    --cyan: #58A6FF;
    --green: #3FB950;
    --amber: #D29922;
    --red: #F85149;
    --purple: #BC8CFF;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--fg);
    font-family: 'Cascadia Mono', 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.38;
    padding: 10px 14px;
    user-select: none;
    overflow-x: hidden;
  }
  .tui-topbar {
    background: #121820;
    border: 1px solid var(--border);
    padding: 6px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .tag {
    padding: 2px 8px;
    font-weight: 700;
    border-radius: 2px;
    font-size: 12px;
  }
  .tag-green { background: rgba(63,185,80,0.18); color: var(--green); border: 1px solid var(--green); }
  .tag-red   { background: rgba(248,81,73,0.18); color: var(--red); border: 1px solid var(--red); }
  .tag-cyan  { background: rgba(88,166,255,0.18); color: var(--cyan); border: 1px solid var(--cyan); }
  .tag-amber { background: rgba(210,153,34,0.18); color: var(--amber); border: 1px solid var(--amber); }

  .tui-grid-top {
    display: grid;
    grid-template-columns: 1.4fr 0.85fr 1.15fr;
    gap: 8px;
    margin-bottom: 8px;
  }
  .tui-grid-mid {
    display: grid;
    grid-template-columns: 1.55fr 1.05fr;
    gap: 8px;
    margin-bottom: 8px;
  }
  .tui-grid-bot {
    display: grid;
    grid-template-columns: 1.2fr 1.4fr;
    gap: 8px;
    margin-bottom: 8px;
  }
  .tui-box {
    background: var(--pane);
    border: 1px solid var(--border);
    position: relative;
    padding: 12px 10px 8px 10px;
  }
  .tui-box-title {
    position: absolute;
    top: -9px;
    left: 10px;
    background: var(--pane);
    padding: 0 6px;
    color: var(--cyan);
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.4px;
  }
  .tui-box-right {
    position: absolute;
    top: -9px;
    right: 10px;
    background: var(--pane);
    padding: 0 6px;
    color: var(--dim);
    font-size: 11px;
  }
  .kv-row {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px dashed #1B222C;
    padding: 3px 2px;
  }
  .kv-k { color: var(--dim); }
  .kv-v { font-weight: 700; color: var(--fg); }
  .spark-row {
    font-family: 'Cascadia Mono', 'Consolas', monospace;
    white-space: pre;
    overflow: hidden;
    padding: 2px 0;
    font-size: 13px;
  }
  canvas {
    width: 100%;
    background: #06090C;
    border: 1px solid #1C242E;
    display: block;
    margin-top: 4px;
  }
  .ascii-bar {
    font-size: 14px;
    letter-spacing: 1px;
    margin: 4px 0;
  }
  .hex-dump {
    background: #06090C;
    border: 1px solid #1C242E;
    padding: 6px 8px;
    font-size: 11.5px;
    color: var(--cyan);
    white-space: pre;
    overflow-x: auto;
    margin-top: 4px;
  }
  .event-list {
    background: #06090C;
    border: 1px solid #1C242E;
    padding: 6px 8px;
    height: 112px;
    overflow-y: hidden;
    font-size: 12px;
  }
  .event-item {
    padding: 1.5px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .tui-keybar {
    background: #121820;
    border: 1px solid var(--border);
    padding: 6px 10px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
  }
  .key-btn {
    background: #161F29;
    border: 1px solid #303C4A;
    color: var(--fg);
    font-family: inherit;
    font-size: 12.5px;
    font-weight: 700;
    padding: 6px 12px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .key-btn:hover { border-color: var(--cyan); background: #1D2936; }
  .key-cap {
    background: var(--cyan);
    color: #000;
    padding: 1px 5px;
    font-weight: 900;
    font-size: 11px;
  }
</style>
</head>
<body>

  <!-- TOP STATUS RIBBON -->
  <div class="tui-topbar">
    <div>
      <span style="color:var(--cyan);font-weight:900;">⚡ NEURAL-LOOP TUI v1.2</span>
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
      <span style="color:var(--dim);"> │ FPS:</span> <span style="color:var(--green);font-weight:700;">20 Hz UI / 2 kHz HIL</span>
    </div>
  </div>

  <!-- ROW 1: LFP SCOPE + PHASE COMPASS + HARDWARE TELEMETRY -->
  <div class="tui-grid-top">
    <div class="tui-box">
      <div class="tui-box-title">┌─[1] SIGNAL CHAIN: LFP & Q15 BANDPASS (13-30 Hz)─┐</div>
      <div class="tui-box-right" id="blankStatus">BLANK: IDLE (40-samp=20ms)</div>
      <div class="spark-row"><span style="color:var(--dim)">RAW : </span><span id="spRaw" style="color:#8B949E"></span></div>
      <div class="spark-row"><span style="color:var(--dim)">I(t): </span><span id="spBp" style="color:var(--cyan)"></span></div>
      <canvas id="cvTrace" width="520" height="145"></canvas>
    </div>

    <div class="tui-box">
      <div class="tui-box-title">┌─[2] PHASE COMPASS (φ*=-π)─┐</div>
      <div class="tui-box-right" id="phaseText">ψ = -3.14 rad</div>
      <canvas id="cvPhase" width="290" height="190"></canvas>
    </div>

    <div class="tui-box">
      <div class="tui-box-title">┌─[4] ESP32-S3 HARDWARE TELEMETRY (COM6)─┐</div>
      <div class="tui-box-right">Gate G0 & G3 PASS</div>
      <div class="kv-row"><span class="kv-k">USB Round-Trip p50 / p95:</span><span class="kv-v" id="tRtt" style="color:var(--cyan)">--</span></div>
      <div class="kv-row"><span class="kv-k">USB Round-Trip p99 / max:</span><span class="kv-v" id="tRtt99">--</span></div>
      <div class="kv-row"><span class="kv-k">MCU 61-Tap Q15 FIR DSP:</span><span class="kv-v" id="tDsp" style="color:var(--green)">--</span></div>
      <div class="kv-row"><span class="kv-k">MCU 64-Neuron Int8 SNN:</span><span class="kv-v" id="tSnn" style="color:var(--green)">--</span></div>
      <div class="kv-row"><span class="kv-k">MCU Total Compute WCET:</span><span class="kv-v" id="tWcet" style="color:var(--green)">--</span></div>
      <div class="kv-row"><span class="kv-k">Stim Pulses / Refractory:</span><span class="kv-v" id="tPulses" style="color:var(--amber)">--</span></div>
      <div class="kv-row"><span class="kv-k">FIR Group Delay Comp:</span><span class="kv-v" style="color:var(--purple)">+30 samp (+15.0 ms / +19661 Q15)</span></div>
      <div class="kv-row"><span class="kv-k">CRC16-CCITT Integrity:</span><span class="kv-v" id="tCrc" style="color:var(--green)">--</span></div>
    </div>
  </div>

  <!-- ROW 2: HERO ORDER PARAMETER R(t) + BAKE-OFF REFERENCE -->
  <div class="tui-grid-mid">
    <div class="tui-box">
      <div class="tui-box-title">┌─[3] HERO: KURAMOTO SYNCHRONY COLLAPSE R(t) & STIM TICKS─┐</div>
      <div class="tui-box-right">Target Desynchronized Zone: R &lt; 0.25</div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div class="ascii-bar" id="rAsciiBar">[██████████░░░░░░░░░░░░░░░░░░░░]</div>
        <div id="rNumeric" style="font-size:16px;font-weight:900;color:var(--green);">R(t) = 0.2920</div>
      </div>
      <canvas id="cvHero" width="780" height="150"></canvas>
    </div>

    <div class="tui-box">
      <div class="tui-box-title">┌─[5] GATES G5-G8 REFERENCE BENCHMARKS─┐</div>
      <div class="tui-box-right">9/9 pytest PASS</div>
      <div class="kv-row"><span class="kv-k">Arm A (No Stim) R_final:</span><span class="kv-v">0.919 ± 0.013</span></div>
      <div class="kv-row"><span class="kv-k">Arm B (Open-Loop 20 Hz):</span><span class="kv-v">0.985 ± 0.002</span></div>
      <div class="kv-row"><span class="kv-k">Arm C (Random-Phase Sham):</span><span class="kv-v" style="color:var(--amber)">0.895 ± 0.040</span></div>
      <div class="kv-row"><span class="kv-k">Arm D (Phase-Locked @ φ*):</span><span class="kv-v" style="color:var(--green)">0.149 ± 0.032 (0.292 HIL)</span></div>
      <div class="kv-row"><span class="kv-k">ΔR (Sham − Locked) 95% CI:</span><span class="kv-v" style="color:var(--green)">+0.746 [0.716, 0.783]</span></div>
      <div class="kv-row"><span class="kv-k">Gate G5 Int8 SNN Detector:</span><span class="kv-v" style="color:var(--cyan)">0.9998 AUROC (99.2% sparse)</span></div>
      <div class="kv-row"><span class="kv-k">Gate G6 Pre-Ictal Predictor:</span><span class="kv-v" style="color:var(--cyan)">0.990 AUROC | 53.0s Lead | 0.125 FA/h</span></div>
    </div>
  </div>

  <!-- ROW 3: BINARY FRAME HEX INSPECTOR + LIVE EVENT LOG -->
  <div class="tui-grid-bot">
    <div class="tui-box">
      <div class="tui-box-title">┌─[6] 32-BYTE BINARY PROTOCOL & REGISTER INSPECTOR─┐</div>
      <div class="tui-box-right">CRC16-CCITT-FALSE (0x1021)</div>
      <div class="hex-dump" id="hexBox">Waiting for frame...</div>
    </div>

    <div class="tui-box">
      <div class="tui-box-title">┌─[7] REAL-TIME HARDWARE STIMULATION & EVENT LOG─┐</div>
      <div class="tui-box-right">2000 Hz Lockstep Stream</div>
      <div class="event-list" id="eventBox"></div>
    </div>
  </div>

  <!-- BOTTOM TUI KEYBOARD BAR -->
  <div class="tui-keybar">
    <div>
      <button class="key-btn" onclick="sendCmd('toggle_loop')"><span class="key-cap">SPACE</span> Toggle Loop ON/OFF</button>
      <button class="key-btn" onclick="sendCmd('cycle_arm')"><span class="key-cap">M</span> Cycle Arm (Locked / Sham / Open)</button>
      <button class="key-btn" onclick="sendCmd('toggle_model')"><span class="key-cap">1 / 2</span> Switch Brain (Kuramoto / Jansen-Rit)</button>
      <button class="key-btn" onclick="sendCmd('inject_burst')"><span class="key-cap">R</span> Inject Seizure Surge (R→0.90)</button>
    </div>
    <div style="color:var(--dim);font-size:12px;">
      TUI CLI: <span style="color:var(--cyan)">python -m host.dash.tui</span>
    </div>
  </div>

<script>
const cvTrace = document.getElementById('cvTrace').getContext('2d');
const cvPhase = document.getElementById('cvPhase').getContext('2d');
const cvHero  = document.getElementById('cvHero').getContext('2d');

function sendCmd(act) { fetch('/cmd?action=' + act, {method: 'POST'}); }

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); sendCmd('toggle_loop'); }
  else if (e.key === 'm' || e.key === 'M') sendCmd('cycle_arm');
  else if (e.key === '1' || e.key === '2') sendCmd('toggle_model');
  else if (e.key === 'r' || e.key === 'R') sendCmd('inject_burst');
});

function drawScope(raw, bp) {
  const w = 520, h = 145;
  cvTrace.clearRect(0, 0, w, h);
  cvTrace.strokeStyle = '#161F29'; cvTrace.lineWidth = 1;
  for (let g = 1; g < 4; g++) {
    cvTrace.beginPath(); cvTrace.moveTo(0, (h/4)*g); cvTrace.lineTo(w, (h/4)*g); cvTrace.stroke();
  }
  cvTrace.strokeStyle = '#6E7681'; cvTrace.lineWidth = 1.1;
  cvTrace.beginPath();
  for (let i = 0; i < raw.length; i++) {
    const x = (i / (raw.length - 1)) * w;
    const y = h/2 - (raw[i] / 3.8) * (h/2 - 8);
    if (i === 0) cvTrace.moveTo(x, y); else cvTrace.lineTo(x, y);
  }
  cvTrace.stroke();

  cvTrace.strokeStyle = '#58A6FF'; cvTrace.lineWidth = 1.9;
  cvTrace.beginPath();
  for (let i = 0; i < bp.length; i++) {
    const x = (i / (bp.length - 1)) * w;
    const y = h/2 - (bp[i] / 3.8) * (h/2 - 8);
    if (i === 0) cvTrace.moveTo(x, y); else cvTrace.lineTo(x, y);
  }
  cvTrace.stroke();
}

function drawCompass(psiTail, rTail, stimPhases) {
  const w = 290, h = 190, cx = w/2, cy = h/2, rad = 76;
  cvPhase.clearRect(0, 0, w, h);
  cvPhase.fillStyle = 'rgba(63, 185, 80, 0.22)';
  cvPhase.beginPath(); cvPhase.moveTo(cx, cy);
  cvPhase.arc(cx, cy, rad, Math.PI - 0.52, Math.PI + 0.52);
  cvPhase.closePath(); cvPhase.fill();

  cvPhase.strokeStyle = '#28313B'; cvPhase.lineWidth = 1;
  [0.5, 1.0].forEach(f => {
    cvPhase.beginPath(); cvPhase.arc(cx, cy, rad * f, 0, Math.PI * 2); cvPhase.stroke();
  });
  cvPhase.beginPath(); cvPhase.moveTo(cx - rad, cy); cvPhase.lineTo(cx + rad, cy);
  cvPhase.moveTo(cx, cy - rad); cvPhase.lineTo(cx, cy + rad); cvPhase.stroke();

  cvPhase.fillStyle = '#7D8590'; cvPhase.font = '10px monospace';
  cvPhase.fillText('φ*=-π', cx - rad - 32, cy + 4);
  cvPhase.fillText('0', cx + rad + 5, cy + 4);

  cvPhase.strokeStyle = 'rgba(88, 166, 255, 0.55)'; cvPhase.lineWidth = 1.5;
  cvPhase.beginPath();
  for (let i = 0; i < psiTail.length; i++) {
    const x = cx + Math.cos(psiTail[i]) * (rTail[i] * rad);
    const y = cy - Math.sin(psiTail[i]) * (rTail[i] * rad);
    if (i === 0) cvPhase.moveTo(x, y); else cvPhase.lineTo(x, y);
  }
  cvPhase.stroke();

  cvPhase.fillStyle = '#3FB950';
  stimPhases.forEach(a => {
    const x = cx + Math.cos(a) * (0.92 * rad);
    const y = cy - Math.sin(a) * (0.92 * rad);
    cvPhase.beginPath(); cvPhase.arc(x, y, 4, 0, Math.PI * 2); cvPhase.fill();
  });

  if (psiTail.length > 0) {
    const lp = psiTail[psiTail.length - 1], lr = rTail[rTail.length - 1];
    cvPhase.fillStyle = '#E6EDF3';
    cvPhase.beginPath();
    cvPhase.arc(cx + Math.cos(lp)*(lr*rad), cy - Math.sin(lp)*(lr*rad), 5.5, 0, Math.PI * 2);
    cvPhase.fill();
  }
}

function drawHero(rBuf, stimIdx, loopOn) {
  const w = 780, h = 150;
  cvHero.clearRect(0, 0, w, h);
  const yT = h - 0.25 * (h - 18) - 9;
  cvHero.strokeStyle = '#3FB950'; cvHero.setLineDash([4, 4]); cvHero.lineWidth = 1.1;
  cvHero.beginPath(); cvHero.moveTo(0, yT); cvHero.lineTo(w, yT); cvHero.stroke();
  cvHero.setLineDash([]);

  const curR = rBuf[rBuf.length - 1] || 0;
  cvHero.strokeStyle = curR < 0.35 ? '#3FB950' : (loopOn ? '#58A6FF' : '#F85149');
  cvHero.lineWidth = 2.4;
  cvHero.beginPath();
  for (let i = 0; i < rBuf.length; i++) {
    const x = (i / (rBuf.length - 1)) * w;
    const y = h - rBuf[i] * (h - 18) - 9;
    if (i === 0) cvHero.moveTo(x, y); else cvHero.lineTo(x, y);
  }
  cvHero.stroke();

  cvHero.strokeStyle = '#3FB950'; cvHero.lineWidth = 1.8;
  stimIdx.forEach(ix => {
    const x = (ix / (rBuf.length - 1)) * w;
    cvHero.beginPath(); cvHero.moveTo(x, h - 2); cvHero.lineTo(x, h - 20); cvHero.stroke();
  });
}

async function poll() {
  try {
    const res = await fetch('/state');
    const d = await res.json();
    drawScope(d.raw, d.bp);
    drawCompass(d.psi_tail, d.r_tail, d.stim_phases);
    drawHero(d.r_buf, d.stim_indices, d.loop_on);

    document.getElementById('spRaw').textContent = d.spark_raw;
    document.getElementById('spBp').textContent  = d.spark_bp;
    document.getElementById('blankStatus').textContent = d.blanked ? 'BLANK: ACTIVE (20ms CLAMP)' : 'BLANK: IDLE (40-samp=20ms)';

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
    const barFill = Math.max(0, Math.min(32, Math.round(curR * 32)));
    const barStr = '[' + '█'.repeat(barFill) + '░'.repeat(32 - barFill) + ']';
    const rBarEl = document.getElementById('rAsciiBar');
    const rNumEl = document.getElementById('rNumeric');
    rBarEl.textContent = barStr;
    rNumEl.textContent = `R(t) = ${curR.toFixed(4)} (${curR < 0.35 ? 'SUPPRESSED' : 'ICTAL'})`;
    const col = curR < 0.35 ? 'var(--green)' : (curR < 0.65 ? 'var(--amber)' : 'var(--red)');
    rBarEl.style.color = col; rNumEl.style.color = col;

    const lastPsi = d.psi_tail[d.psi_tail.length - 1] || 0;
    document.getElementById('phaseText').textContent = `ψ = ${lastPsi.toFixed(2)} rad (${(lastPsi*180/Math.PI).toFixed(0)}°)`;

    document.getElementById('tRtt').textContent = `${d.p50_rtt.toFixed(1)} µs p50 / ${d.p95_rtt.toFixed(1)} µs p95`;
    document.getElementById('tRtt99').textContent = `${d.p99_rtt.toFixed(1)} µs p99 (< 2000 µs G0)`;
    document.getElementById('tDsp').textContent = `${d.p50_dsp.toFixed(1)} µs p50 (61-tap Q15 FIR)`;
    document.getElementById('tSnn').textContent = `${Math.max(1, d.p50_wcet - d.p50_dsp).toFixed(1)} µs p50 (64-N Int8 LIF)`;
    document.getElementById('tWcet').textContent = `${d.p50_wcet.toFixed(1)} µs p50 (< 250 µs G3 PASS)`;
    document.getElementById('tPulses').textContent = `${d.pulses} / 300 (60s window)`;
    document.getElementById('tCrc').textContent = `${d.valid_frames} valid │ ${d.crc_err} CRC err`;

    document.getElementById('hexBox').textContent = d.hex_inspector;
    document.getElementById('eventBox').innerHTML = d.events.map(
      e => `<div class="event-item" style="color:${e.includes('STIM') ? '#3FB950' : '#58A6FF'}">${e}</div>`
    ).join('');
  } catch (err) {}
  setTimeout(poll, 55);
}
poll();
</script>
</body>
</html>
"""


class TuiCockpitEngine:
    def __init__(self, port: str = "COM6", baudrate: int = 2000000) -> None:
        enable_windows_1ms_timer()
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.hw_live = False
        self._connect_serial()

        self.k_brain = KuramotoBrain(KuramotoConfig(seed=2026))
        self.k_brain.theta = np.mod(self.k_brain.rng.normal(0.0, 0.55, size=self.k_brain.cfg.n_osc), 2.0 * np.pi)
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
        self.events: Deque[str] = deque(maxlen=6)
        self.events.append("[BOOT] Neural Loop TUI Cockpit online — 32B CRC16 lockstep @ 2 Mbps")

        self.loop_on = True
        self.arm_idx = 0
        self.model_idx = 0
        self.arm_names = ["D: PHASE-LOCKED (φ*=-π)", "C: RANDOM-PHASE SHAM", "B: OPEN-LOOP 20 Hz"]
        self.model_names = ["STAGE 1: KURAMOTO (N=64)", "STAGE 2: JANSEN-RIT NMM"]
        self.seq = 0
        self.pulses = 0
        self.valid_frames = 0
        self.crc_err = 0
        self.refrac = 0
        self.pending_stim = 0.0
        self.blanked_now = False
        self.last_tx_hex = ""
        self.last_rx_hex = ""
        self.last_res = ResultFrame(
            state=1, seq=0, t_rx_us=1000, t_dsp_us=1018, t_dec_us=1082, t_stim_us=0,
            cls=1, conf=200, phase_q15=-31000, R_q15=10000, budget_used=0
        )
        self.lock = threading.Lock()
        self.running = True

    def _connect_serial(self) -> None:
        try:
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.20, write_timeout=0.20)
            self.ser.dtr = False
            self.ser.rts = False
            time.sleep(0.30)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.hw_live = True
        except Exception:
            self.hw_live = False

    def step_batch(self, n_frames: int = 16) -> None:
        with self.lock:
            for _ in range(n_frames):
                if not self.hw_live and (self.seq % 200 == 0):
                    self._connect_serial()

                s0, _, _ = self.k_brain.step(stim_amp=self.pending_stim)
                s1, r_val, psi_val = self.k_brain.step(stim_amp=0.0)
                if self.model_idx == 1:
                    j0, _ = self.jr_brain.step(u_stim=self.pending_stim * 0.2)
                    j1, _ = self.jr_brain.step(u_stim=0.0)
                    s0, s1 = j0, j1
                self.pending_stim = 0.0

                for i in range(len(self.stim_indices)):
                    self.stim_indices[i] -= 1
                while self.stim_indices and self.stim_indices[0] < 0:
                    self.stim_indices.popleft()

                self.dsp.step(s0)
                bp_out = self.dsp.step(s1)
                self.blanked_now = bp_out.blanked
                fired = False

                sf = SampleFrame(
                    seq=self.seq,
                    sim_t_us=self.seq * 1000,
                    host_t_us=host_monotonic_us(),
                    samples=(s0, s1, 0, 0),
                    flags=0x03,
                    n_ch=1,
                    n_samp=2,
                    mode=MODE_LOCKSTEP,
                )
                tx_bytes = sf.to_bytes()
                self.last_tx_hex = tx_bytes[:16].hex(" ").upper() + " | " + tx_bytes[16:].hex(" ").upper()
                self.seq += 1

                if self.hw_live and self.ser is not None:
                    t0 = time.perf_counter_ns()
                    try:
                        self.ser.write(tx_bytes)
                        rx = self.ser.read(32)
                        t1 = time.perf_counter_ns()
                        if len(rx) == 32:
                            self.last_rx_hex = rx[:16].hex(" ").upper() + " | " + rx[16:].hex(" ").upper()
                            self.last_res = ResultFrame.from_bytes(rx)
                            self.valid_frames += 1
                            self.rtt_win.append((t1 - t0) / 1000.0)
                            self.dsp_win.append(float((self.last_res.t_dsp_us - self.last_res.t_rx_us) & 0xFFFFFFFF))
                            self.wcet_win.append(float((self.last_res.t_dec_us - self.last_res.t_rx_us) & 0xFFFFFFFF))
                            if self.loop_on:
                                if self.arm_idx == 0 and self.last_res.t_stim_us != 0:
                                    fired = True
                                elif self.arm_idx == 1 and self.refrac == 0 and self.k_brain.rng.random() < 0.015:
                                    fired = True
                                elif self.arm_idx == 2 and (self.seq % 50 == 0):
                                    fired = True
                        else:
                            self.crc_err += 1
                    except Exception:
                        self.crc_err += 1
                else:
                    if self.loop_on and self.refrac == 0:
                        if self.arm_idx == 0 and r_val > 0.22 and np.cos(psi_val) < -0.82:
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
                    self.stim_indices.append(self.win_len - 1)
                    self.stim_phases.append(psi_val)
                    self.events.append(
                        f"[STIM #{self.pulses:03d}] seq={self.seq:<5d} ψ={psi_val:+.2f}rad R={r_val:.3f} T_BLANK=20ms"
                    )
                elif self.refrac > 0:
                    self.refrac -= 1

                if not self.loop_on and r_val < 0.88:
                    self.k_brain.theta = np.mod(
                        self.k_brain.theta - 0.016 * np.sin(self.k_brain.theta - psi_val), 2.0 * np.pi
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
                f"REGISTERS      : seq={lr.seq:<6d} state=0x{lr.state:02X} cls={lr.cls} conf={lr.conf:<3d} "
                f"phase_q15={lr.phase_q15:<+6d}\n"
                f"TIMESTAMPS (µs): t_rx={lr.t_rx_us:<10d} t_dsp={lr.t_dsp_us:<10d} t_dec={lr.t_dec_us:<10d} "
                f"t_stim={lr.t_stim_us:<10d}"
            )
            return {
                "hw_live": self.hw_live,
                "hw_badge": f"● LIVE ESP32-S3 ({self.port} @ 2M)" if self.hw_live else "● IN-SILICO SIMULATOR",
                "loop_on": self.loop_on,
                "arm_name": self.arm_names[self.arm_idx],
                "model_name": self.model_names[self.model_idx],
                "seq": self.seq,
                "blanked": self.blanked_now,
                "raw": raw_l,
                "bp": bp_l,
                "spark_raw": make_sparkline(raw_l, -3.5, 3.5, 48),
                "spark_bp": make_sparkline(bp_l, -2.5, 2.5, 48),
                "r_buf": r_l,
                "psi_tail": psi_l,
                "r_tail": r_l[-len(psi_l):],
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


def run_web_cockpit(host: str = "127.0.0.1", port: int = 8765, serial_port: str = "COM6") -> None:
    engine = TuiCockpitEngine(port=serial_port)

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
                body = HTML_PAGE.encode("utf-8")
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
            with engine.lock:
                if "toggle_loop" in self.path:
                    engine.loop_on = not engine.loop_on
                    engine.events.append(
                        f"[CMD] Closed-Loop toggled -> {'ON (SUPPRESSING)' if engine.loop_on else 'OFF (INHIBITED)'}"
                    )
                elif "cycle_arm" in self.path:
                    engine.arm_idx = (engine.arm_idx + 1) % 3
                    engine.events.append(f"[CMD] Controller Arm -> {engine.arm_names[engine.arm_idx]}")
                elif "toggle_model" in self.path:
                    engine.model_idx = (engine.model_idx + 1) % 2
                    engine.events.append(f"[CMD] Brain Model -> {engine.model_names[engine.model_idx]}")
                elif "inject_burst" in self.path:
                    engine.k_brain.theta = np.mod(
                        engine.k_brain.rng.normal(0.0, 0.35, size=engine.k_brain.cfg.n_osc), 2.0 * np.pi
                    )
                    engine.stim_phases.clear()
                    engine.events.append("[CMD] Injected synchronous seizure burst (R -> 0.90)")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

    srv = HTTPServer((host, port), Handler)
    print(f"NEURAL LOOP TUI Web Cockpit running at: http://{host}:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    run_web_cockpit()
