#include <Arduino.h>
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nl_protocol.h"
#include "nl_dsp.h"
#include "nl_snn.h"

/*
 * BOARD SAFETY GUARANTEES (Rule #4: Never destroy the board):
 * - Strapping pins (GPIO 0, 3, 45, 46) are NEVER driven.
 * - Flash / Octal PSRAM pins (GPIO 26..37) are NEVER touched.
 * - Native USB pins (GPIO 19, 20) and UART0 pins (GPIO 43, 44) are managed only by hardware drivers.
 * - Stimulation indicator uses safe general-purpose GPIO 4 only.
 * - Onboard WS2812 RGB LED (GPIO 48 / GPIO 38) is turned OFF (0,0,0) on boot to prevent glare & heat.
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

/* Deterministic int8 SNN weights initialized on boot for fast-loop burst detection */
static int8_t g_w1[NL_SNN_IN][NL_SNN_HID];
static int8_t g_w2[NL_SNN_HID];

static void init_snn_weights(void) {
    for (int i = 0; i < NL_SNN_IN; ++i) {
        for (int j = 0; j < NL_SNN_HID; ++j) {
            /* Deterministic structured excitatory/inhibitory receptive fields */
            int val = ((i * 7 + j * 13) % 31) - 10;
            g_w1[i][j] = (int8_t)val;
        }
    }
    for (int j = 0; j < NL_SNN_HID; ++j) {
        g_w2[j] = (j % 4 == 0) ? -12 : 18;
    }
}

/* Ring buffer for 32-byte frame alignment */
static uint8_t g_rx_buf[64];
static size_t  g_rx_len = 0;

void setup() {
    /* 1. Turn off blinding factory white WS2812 RGB LED safely */
    rgbLedWrite(RGB_PIN_V1_0, 0, 0, 0);
    rgbLedWrite(RGB_PIN_V1_1, 0, 0, 0);

    /* 2. Configure safe stimulation marker pin */
    pinMode(PIN_STIM_OUT, OUTPUT);
    digitalWrite(PIN_STIM_OUT, LOW);

    /* 3. Initialize DSP & SNN state */
    nl_dsp_init(&g_dsp);
    nl_snn_reset(&g_snn);
    init_snn_weights();

    /* 4. High-speed UART over WCH CH343 @ 921600 baud (zero framing wait) */
    Serial.setRxBufferSize(1024);
    Serial.setTxBufferSize(1024);
    Serial.begin(921600);
}

void loop() {
    while (Serial.available() > 0 && g_rx_len < sizeof(g_rx_buf)) {
        g_rx_buf[g_rx_len++] = (uint8_t)Serial.read();
    }

    /* Scan for 32-byte SAMPLE frame starting with magic 0xA5C3 (little-endian: 0xC3, 0xA5) */
    while (g_rx_len >= sizeof(nl_sample_frame_t)) {
        if (g_rx_buf[0] != 0xC3U || g_rx_buf[1] != 0xA5U) {
            memmove(g_rx_buf, g_rx_buf + 1, --g_rx_len);
            continue;
        }

        uint32_t t_rx = (uint32_t)esp_timer_get_time();
        const nl_sample_frame_t *in_frame = (const nl_sample_frame_t *)g_rx_buf;

        if (nl_sample_frame_validate(in_frame) != 0) {
            /* Bad CRC or version: drop magic byte and rescan */
            memmove(g_rx_buf, g_rx_buf + 1, --g_rx_len);
            continue;
        }

        /* Execute 2-sample block through 61-tap Q15 Quadrature FIR + Artifact Blanking */
        nl_dsp_out_t dsp_out0 = nl_dsp_step(&g_dsp, in_frame->samples[0]);
        (void)dsp_out0;
        nl_dsp_out_t dsp_out1 = nl_dsp_step(&g_dsp, in_frame->samples[1]);
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

        /* Execute 64-neuron int8 LIF SNN (inhibited automatically if artifact blanking active) */
        uint8_t spike_count = 0;
        for (int s = 0; s < NL_SNN_STEPS; ++s) {
            spike_count += nl_snn_step(&g_snn, g_in_feat, g_w1, g_w2, dsp_out1.blanked);
        }
        uint32_t t_dec = (uint32_t)esp_timer_get_time();

        /* Phase-locked controller decision:
         * Target phase is trough (+/- pi -> |phase_q15| > 26000) when burst detected or armed
         */
        bool is_burst = (!dsp_out1.blanked) && (spike_count >= 5 || norm_env >= 48);
        int32_t abs_phase = (dsp_out1.phase_q15 < 0) ? -(int32_t)dsp_out1.phase_q15 : (int32_t)dsp_out1.phase_q15;
        bool in_target_phase = (abs_phase >= 25000); /* within ~42 deg of +/- pi */
        bool refractory_ok = ((t_dec - g_last_stim_us) >= 20000U); /* 20 ms hard refractory (max 50 Hz) */

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

            /* DEC-007: Engage 20 ms (40-sample) stimulation artifact blanking window */
            nl_dsp_notify_stim(&g_dsp);
        } else if (!refractory_ok) {
            g_state = NL_STATE_REFRACTORY;
        } else {
            g_state = NL_STATE_ARMED;
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
        out_frame.phase_q15   = dsp_out1.phase_q15;
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
