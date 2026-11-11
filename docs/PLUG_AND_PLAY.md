# Plug & Play & Safety Architecture

## 1. Safety Architecture

The **Neural Loop** closed-loop neuromodulation system implements multi-tier hardware and software safety interlocks:

| Safety Interlock | Mechanism | Parameter / Behavior |
| :--- | :--- | :--- |
| **Boot-Disarmed Gate** | Firmware software gate | Board boots into `g_armed = false` with dim 🔵 **Blue LED** (`0, 0, 15`). GPIO 4 stimulation pulses are physically disabled until explicit host handshake or `arm` command. |
| **Session Pulse Budget** | Hardware pulse counter | Strict limit of **300 pulses** per session. Once consumed, the hardware controller refuses all stimulation requests (`g_budget_used < 300`). |
| **Refractory Period** | Timing interlock | Hard minimum delay of **$\ge 15\text{ ms}$** between consecutive pulses (`(t_dec - g_last_stim_us) >= 15000U`). Prevents rapid strobing or seizure kindling. |
| **Artifact Blanking** | FIR DSP feedback suppression | **$20\text{ ms}$** ($40\text{ samples}$ @ $2000\text{ Hz}$) stimulation artifact blanking window. During blanking, inputs are zero-filled to prevent stimulus self-feedback runaway. |
| **Hardware Watchdog** | ESP-IDF Task Watchdog | **$5.0\text{ s}$** `esp_task_wdt`. Auto-resets the MCU safely if the main processing loop ever hangs. |
| **Hot-Unplug Protection** | Host serial exception handler | If the USB cable is disconnected mid-session, the TUI catches the disconnect, cleanly falls back to safe in-silico simulation, and continuously polls for reconnection without crashing. |
| **Static LED Rest Decay** | WS2812 firmware scheduler | When static (idle/resting/no frames), physical LED decays back to calm 🟢 **Green** (`0, 15, 0`) or 🔵 **Blue** (`0, 0, 15`) once the $250\text{ ms}$ pulse latch expires. Never stays stuck on Red. |
| **Safe Exit Handshake** | Host `finally:` handler | When the TUI exits (pressing `q` or `Ctrl+C`), it transmits `b"rest\r\n"` before closing the COM port, ensuring the ESP32-S3 sits in solid resting Green. |
| **Firmware Integrity** | Dual-image OTA & full backup | Verified 16 MB full-flash backup stored at `backups/20261110_203223/flash.bin` (SHA-256 `9f4f013c0cf6ba85...`). |

---

## 2. Plug & Play Architecture

### Automatic Port Discovery
The project enforces a single-port policy in [`host/board.py`](file:///G:/MIT%20Planner/mit/neural-loop/host/board.py). On any desktop, ports are automatically scored by USB VID:PID:
1. `1A86:55D3` / `1A86:55D4` — WCH CH343 high-speed UART bridge (ESP32-S3 DevKitC-1 v1.3 default) $\to$ **Score 100**
2. `303A:1001` / `303A:4001` — Espressif USB-Serial-JTAG / Native CDC $\to$ **Score 90**
3. `10C4:EA60` — Silicon Labs CP210x $\to$ **Score 60**
4. `1A86:7523` — WCH CH340 $\to$ **Score 50**

No hardcoded `COM6` is required on other desktops: the board is automatically detected regardless of assigned port (`COM3`, `COM4`, `COM7`, `/dev/ttyUSB0`, etc.).

---

### Dynamic Hot-Plug Inside TUI
Inside [`host/dash/tui.py`](file:///G:/MIT%20Planner/mit/neural-loop/host/dash/tui.py):
- **Start Without Board:** The TUI launches immediately in in-silico Kuramoto/Jansen-Rit simulation mode, displaying:
  `LINK: ● IN-SILICO (PnP Auto-Detect)`.
- **Insert Board:** The 1 Hz hotplug probe automatically discovers the ESP32-S3 on whichever USB port it is plugged into, establishes the 2 Mbps lockstep link, and switches to:
  `LINK: ● LIVE ESP32-S3 (COMx @ 2M)` with `[HOTPLUG] Board detected on COMx -> LIVE HIL connected!`.
- **Remove Board:** If unplugged, the TUI catches the disconnection, closes the serial handle, and gracefully reverts to in-silico simulation mode with `[HOTPLUG] Board disconnected -> In-silico fallback`.
- **Re-insert Board:** Automatically reconnects within 1 second.

---

### Windows Plug & Play Auto-Launcher (`nuro watch`)
You can have the TUI automatically pop up the moment you plug in your ESP32-S3 into the computer:

1. **Install into Windows Startup (Runs silently on boot):**
   ```powershell
   nuro watch --install
   ```
   This creates a silent VBScript in `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\NeuralLoop_PlugAndPlay.vbs`.
   Whenever you plug your ESP32-S3 into ANY USB port on your desktop, a console window automatically opens with the live animated 60 FPS TUI!

2. **Run Interactively in Terminal:**
   ```powershell
   nuro watch
   ```

3. **Uninstall from Startup:**
   ```powershell
   nuro watch --uninstall
   ```

4. **One-Click Shortcuts:**
   - Double-click [`Plug_and_Play_TUI.cmd`](file:///G:/MIT%20Planner/mit/neural-loop/Plug_and_Play_TUI.cmd) to start the TUI immediately (connects live or in-silico).
   - Double-click [`Plug_and_Play_Watcher.cmd`](file:///G:/MIT%20Planner/mit/neural-loop/Plug_and_Play_Watcher.cmd) to run the USB watcher.

---

## 3. Running on Any Other Desktop

To run Neural Loop on any fresh desktop or laptop:

1. **Copy or Clone the Repository:**
   ```powershell
   git clone <repo_url> "C:\neural-loop"
   cd "C:\neural-loop"
   ```
2. **Run One-Click Setup:**
   ```cmd
   setup.cmd
   ```
   *(Creates Python virtual environment, installs dependencies, and runs `nuro doctor`).*
3. **Enable Plug & Play Auto-Launch:**
   ```cmd
   nuro watch --install
   ```
4. **Plug in your ESP32-S3:**
   The TUI automatically detects the board and opens the live workstation!
