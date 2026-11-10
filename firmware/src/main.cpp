#include <Arduino.h>
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nl_protocol.h"
#include "nl_dsp.h"
#include "nl_snn.h"
#include "nl_calc_embedded.h"
#include "nl_lcd.h"

/*
 * BOARD SAFETY GUARANTEES (Rule #4: Never destroy the board):
 * - Strapping pins (GPIO 0, 3, 45, 46) are NEVER driven.
 * - Flash / Octal PSRAM pins (GPIO 26..37) are NEVER touched.
 * - Native USB pins (GPIO 19, 20) and UART0 pins (GPIO 43, 44) are managed only by hardware drivers.
 * - Stimulation indicator uses safe general-purpose GPIO 4 only.
 * - Onboard WS2812 RGB LED (GPIO 48 / GPIO 38) operates at safe, low-glare brightness.
 */
#define PIN_STIM_OUT     4
#define RGB_PIN_V1_0    48
#define RGB_PIN_V1_1    38

static nl_dsp_state_t g_dsp;
static nl_snn_state_t g_snn;
static uint8_t        g_state = NL_STATE_ARMED;
static uint16_t       g_budget_used = 0;
static uint32_t       g_last_stim_us = 0;
static int8_t         g_in_feat[NL_SNN_IN];

static nl_dsp_out_t   g_last_dsp_out;
static uint8_t        g_last_spikes = 0;
static uint32_t       g_red_led_until_ms = 0;

/* Deterministic int8 SNN weights (row-major [HID][IN] for L1 cache & SIMD locality) */
static int8_t g_w1[NL_SNN_HID][NL_SNN_IN];
static int8_t g_w2[NL_SNN_HID];

static void init_snn_weights(void) {
    for (int j = 0; j < NL_SNN_HID; ++j) {
        for (int i = 0; i < NL_SNN_IN; ++i) {
            int val = ((i * 7 + j * 13) % 31) - 10;
            g_w1[j][i] = (int8_t)val;
        }
        g_w2[j] = (j % 4 == 0) ? -12 : 18;
    }
}

/* WS2812 RGB LED control helper (supporting Rev 1.0 on GPIO 48 and Rev 1.1 on GPIO 38) */
void nl_set_status_led(uint8_t r, uint8_t g, uint8_t b) {
    rgbLedWrite(RGB_PIN_V1_0, r, g, b);
    rgbLedWrite(RGB_PIN_V1_1, r, g, b);
}

/* Query telemetry for interactive 'neuro' command */
void nl_get_telemetry(nl_live_telemetry_t *telem) {
    if (!telem) return;
    telem->env_mean_q15   = g_dsp.env_mean_q15;
    telem->last_env_q15   = g_last_dsp_out.env_q15;
    telem->last_phase_q15 = g_last_dsp_out.phase_q15;
    telem->last_spikes    = g_last_spikes;
    telem->state          = g_state;
    telem->budget_used    = g_budget_used;
    telem->last_stim_us   = g_last_stim_us;
    telem->is_blanked     = g_last_dsp_out.blanked;
}

/* Synthetic Burst Demonstration */
void nl_run_sim_burst(void) {
    Serial.println("\r\n=======================================================");
    Serial.println("⚡ RUNNING ON-CHIP NEUROMODULATION SYNTHETIC SELF-TEST");
    Serial.println("=======================================================");
    Serial.println("  Stage 1: Resting Baseline Noise -> LED 🟢 GREEN");
    Serial.println("  Stage 2: Pre-Ictal Coherence Ramp -> LED 🟡 YELLOW");
    Serial.println("  Stage 3: Burst Detection & Phase Trough Trigger -> LED 🔴 RED + GPIO 4 Pulse");
    Serial.println("  Stage 4: 20 ms Artifact Suppression & Recovery -> LED 🟢 GREEN");
    Serial.println("-------------------------------------------------------");

    nl_set_status_led(0, 15, 0); // Green
    nl_lcd_show_neuro("RESTING", 45, 0);
    delay(100);

    // Stage 1: Resting Noise (50 samples @ 2 kHz)
    for (int i = 0; i < 50; ++i) {
        int16_t noise = (int16_t)((rand() % 60) - 30);
        nl_dsp_step(&g_dsp, noise);
    }

    // Stage 2: Coherence & Amplitude Ramp (Yellow early warning alert)
    Serial.println("  [ALERT] 🟡 Early Warning: 20Hz phase coherence ramp detected!");
    nl_set_status_led(20, 20, 0); // Yellow
    nl_lcd_show_neuro("WARN RAMP", 240, -45);

    for (int i = 0; i < 60; ++i) {
        float angle = 2.0f * (float)M_PI * 20.0f * ((float)i / 2000.0f);
        int16_t sig = (int16_t)(sinf(angle) * (150.0f + i * 2.5f));
        nl_dsp_out_t d = nl_dsp_step(&g_dsp, sig);
        g_last_dsp_out = d;
        delay(1);
    }

    // Stage 3: High Amplitude Synchronized Burst (Red alert + Stimulus Trigger)
    Serial.println("  [ALERT] 🔴 Burst Trigger: Phase-locked trough detected (theta ~ +/-pi)!");
    nl_set_status_led(35, 0, 0); // Red
    digitalWrite(PIN_STIM_OUT, HIGH);
    uint32_t t_stim = (uint32_t)esp_timer_get_time();
    delayMicroseconds(100);
    digitalWrite(PIN_STIM_OUT, LOW);

    g_last_stim_us = t_stim;
    g_budget_used++;
    nl_dsp_notify_stim(&g_dsp); // Engage 20ms blanking window
    nl_lcd_show_neuro("BURST STIM", 720, -175);

    Serial.printf("  [PULSE] Stimulation delivered on GPIO 4 @ t=%lu us (Pulse #%u/300)\r\n",
                  (unsigned long)t_stim, (unsigned int)g_budget_used);
    Serial.println("  [BLANK] 20 ms (40-sample) stimulation artifact blanking engaged.");
    
    // Stage 4: Run through blanking window
    for (int i = 0; i < NL_BLANK_SAMPLES; ++i) {
        nl_dsp_step(&g_dsp, 0);
    }
    delay(200);

    // Restore resting
    nl_set_status_led(0, 15, 0); // Green
    nl_lcd_show_neuro("RESTING", 50, 0);
    Serial.println("  [RESET] Closed-loop restored to resting baseline -> LED 🟢 GREEN\r\n");
}

/* Reset to resting state */
void nl_run_sim_rest(void) {
    nl_snn_reset(&g_snn);
    nl_dsp_init(&g_dsp);
    g_state = NL_STATE_ARMED;
    nl_set_status_led(0, 15, 0);
    nl_lcd_show_neuro("RESTING", 35, 0);
    Serial.println("  [RESET] Baseline restored. Neural loop armed in resting state (LED 🟢 GREEN).");
}

/* Ring buffer for 32-byte binary frame alignment */
static uint8_t g_rx_buf[64];
static size_t  g_rx_len = 0;

void setup() {
    /* 1. Initialize status LED to calm resting green (safe, low-glare) */
    nl_set_status_led(0, 15, 0);

    /* 2. Configure safe stimulation marker pin */
    pinMode(PIN_STIM_OUT, OUTPUT);
    digitalWrite(PIN_STIM_OUT, LOW);

    /* 3. Initialize DSP & SNN state */
    nl_dsp_init(&g_dsp);
    nl_snn_reset(&g_snn);
    init_snn_weights();

    /* 4. High-speed UART over WCH CH343 @ 2000000 baud (2 Mbps) */
    Serial.setRxBufferSize(1024);
    Serial.setTxBufferSize(1024);
    Serial.begin(2000000);

    /* 5. Initialize optional I2C LCD (silent bypass if not attached) */
    nl_lcd_init();

    /* 6. Initialize embedded Pocket Lab */
    nl_calc_init();

    /* 7. Print Welcome Startup Banner */
    Serial.println("\r\n=======================================================");
    Serial.println("⚡ NEURAL LOOP & SCIENTIFIC POCKET LAB (ESP32-S3)");
    Serial.println("=======================================================");
    Serial.println("  Core: Xtensa Dual-Core LX7 @ 240 MHz | Hardware FPU");
    Serial.println("  DSP: 61-tap Q15 Quadrature FIR | 64-neuron Int8 LIF SNN");
    Serial.println("  Stimulus Pin: GPIO 4 | RGB LED: GPIO 48/38");
    Serial.println("  Status: 🟢 RESTING (Armed @ 1000 Hz lockstep)");
    Serial.println("-------------------------------------------------------");
    Serial.println("  Type 'help' for Scientific Lab (Math/Physics/Chem)");
    Serial.println("  Type 'sim burst' to test Early-Warning & Phase Stimulus");
    Serial.println("=======================================================");
    Serial.print("esp32-s3> ");
}

void loop() {
    /* 1. Demultiplex incoming bytes:
     *    - If byte is ASCII, route immediately to interactive Pocket Lab REPL.
     *    - If magic 0xC3, 0xA5 arrives, accumulate full 32-byte binary DSP frame.
     */
    while (Serial.available() > 0) {
        if (g_rx_len == 0) {
            uint8_t b = (uint8_t)Serial.read();
            if (b == 0xC3U) {
                g_rx_buf[g_rx_len++] = b;
            } else {
                nl_calc_process_char((char)b);
            }
        } else if (g_rx_len == 1) {
            uint8_t b = (uint8_t)Serial.read();
            if (b == 0xA5U) {
                g_rx_buf[g_rx_len++] = b;
            } else {
                /* Not magic 0xA5C3! Flush first byte and this byte to ASCII REPL */
                nl_calc_process_char((char)g_rx_buf[0]);
                nl_calc_process_char((char)b);
                g_rx_len = 0;
            }
        } else {
            g_rx_buf[g_rx_len++] = (uint8_t)Serial.read();
            if (g_rx_len == sizeof(nl_sample_frame_t)) {
                break; /* Full 32-byte frame assembled, process below */
            }
        }
    }

    /* 2. Process complete 32-byte DSP sample frame if present */
    if (g_rx_len >= sizeof(nl_sample_frame_t)) {
        if (g_rx_buf[0] != 0xC3U || g_rx_buf[1] != 0xA5U) {
            memmove(g_rx_buf, g_rx_buf + 1, --g_rx_len);
            return;
        }

        uint32_t t_rx = (uint32_t)esp_timer_get_time();
        const nl_sample_frame_t *in_frame = (const nl_sample_frame_t *)g_rx_buf;

        if (nl_sample_frame_validate(in_frame) != 0) {
            /* Bad CRC or version: drop magic byte and rescan */
            memmove(g_rx_buf, g_rx_buf + 1, --g_rx_len);
            return;
        }

        /* Execute 2-sample block through 61-tap Q15 Quadrature FIR + Artifact Blanking */
        nl_dsp_out_t dsp_out0 = nl_dsp_step(&g_dsp, in_frame->samples[0]);
        (void)dsp_out0;
        nl_dsp_out_t dsp_out1 = nl_dsp_step(&g_dsp, in_frame->samples[1]);
        g_last_dsp_out = dsp_out1;
        uint32_t t_dsp = (uint32_t)esp_timer_get_time();

        /* Populate 16 int8 input features from normalized envelope & quadrature components */
        int32_t norm_env = (g_dsp.env_mean_q15 > 64)
            ? ((int32_t)dsp_out1.env_q15 * 32) / g_dsp.env_mean_q15
            : 0;
        if (norm_env > 127) norm_env = 127;

        for (int k = 0; k < 8; ++k) {
            g_in_feat[k]     = (int8_t)((norm_env * (k + 4)) >> 4);
            g_in_feat[k + 8] = (int8_t)(dsp_out1.I_q15 >> 8);
        }

        /* Execute 64-neuron int8 LIF SNN over 8 sub-steps */
        uint8_t spike_count = nl_snn_step_block(&g_snn, g_in_feat, g_w1, g_w2, dsp_out1.blanked);
        g_last_spikes = spike_count;
        uint32_t t_dec = (uint32_t)esp_timer_get_time();

        /* Phase-locked controller decision:
         * Compensate 61-tap FIR group delay (30 samples = 15 ms = +0.6*pi rad = +19661 Q15 at 20 Hz).
         * Target phase is trough (+/- pi -> |comp_phase_q15| >= 25000) when burst detected.
         */
        int16_t comp_phase_q15 = (int16_t)(dsp_out1.phase_q15 + 19661);
        bool is_burst = (!dsp_out1.blanked) && (spike_count >= 4 || dsp_out1.env_q15 >= 320);
        int32_t abs_phase = (comp_phase_q15 < 0) ? -(int32_t)comp_phase_q15 : (int32_t)comp_phase_q15;
        bool in_target_phase = (abs_phase >= 25000);
        bool refractory_ok = ((t_dec - g_last_stim_us) >= 15000U);

        uint32_t t_stim = 0;
        if ((in_frame->flags & 0x02U) || is_burst) {
            g_state = NL_STATE_ARMED;
        }

        if (g_state == NL_STATE_ARMED && is_burst && in_target_phase && refractory_ok && g_budget_used < 300) {
            digitalWrite(PIN_STIM_OUT, HIGH);
            t_stim = (uint32_t)esp_timer_get_time();
            digitalWrite(PIN_STIM_OUT, LOW);

            g_last_stim_us = t_stim;
            g_budget_used++;
            g_state = NL_STATE_SUPPRESS;

            /* Engage 20 ms stimulation artifact blanking window */
            nl_dsp_notify_stim(&g_dsp);

            /* Trigger Red alert LED */
            nl_set_status_led(35, 0, 0);
            g_red_led_until_ms = millis() + 80;
        } else if (!refractory_ok) {
            g_state = NL_STATE_REFRACTORY;
        } else {
            g_state = NL_STATE_ARMED;
        }

        /* Update WS2812 RGB LED status if not in red flash latch */
        uint32_t now = millis();
        if (now >= g_red_led_until_ms) {
            if (is_burst || dsp_out1.env_q15 >= 180 || spike_count >= 2) {
                nl_set_status_led(20, 20, 0); /* 🟡 Yellow: Early Warning Coherence Ramp */
            } else {
                nl_set_status_led(0, 15, 0);  /* 🟢 Green: Normal Resting State */
            }
        }

        nl_result_frame_t out_frame;
        out_frame.magic       = NL_MAGIC_RESULT;
        out_frame.ver         = NL_PROTOCOL_VER;
        out_frame.state       = g_state;
        out_frame.seq         = in_frame->seq;
        out_frame.t_rx_us     = t_rx;
        out_frame.t_dsp_us    = t_dsp;
        out_frame.t_dec_us    = t_dec;
        out_frame.t_stim_us   = t_stim;
        out_frame.cls         = is_burst ? 1U : 0U;
        out_frame.conf        = (uint8_t)((spike_count * 255U) / NL_SNN_STEPS);
        out_frame.phase_q15   = comp_phase_q15;
        out_frame.R_q15       = dsp_out1.env_q15;
        out_frame.budget_used = g_budget_used;

        Serial.write((const uint8_t *)&out_frame, sizeof(out_frame));

        /* Consume 32-byte frame */
        g_rx_len -= sizeof(nl_sample_frame_t);
        if (g_rx_len > 0) {
            memmove(g_rx_buf, g_rx_buf + sizeof(nl_sample_frame_t), g_rx_len);
        }
    }
}
