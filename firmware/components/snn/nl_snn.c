#include "nl_snn.h"
#include <string.h>

static inline int16_t clamp_v_q12(int32_t v) {
    if (v > NL_SNN_MAX_V) return NL_SNN_MAX_V;
    if (v < -NL_SNN_MAX_V) return -NL_SNN_MAX_V;
    return (int16_t)v;
}

void nl_snn_reset(nl_snn_state_t *st) {
    if (!st) return;
    memset(st, 0, sizeof(*st));
}

uint8_t nl_snn_step_block(
    nl_snn_state_t *st,
    const int8_t in_features[NL_SNN_IN],
    const int8_t w1[NL_SNN_HID][NL_SNN_IN],
    const int8_t w2[NL_SNN_HID],
    bool inhibit
) {
    if (inhibit) {
        /* DEC-007: Freeze SNN during stimulation artifact blanking window */
        st->total_hid_spikes = 0;
        return 0;
    }

    /* Precompute synaptic current I_syn[j] once for the 1 ms block (exact invariance across the 8 sub-steps) */
    int32_t i_syn[NL_SNN_HID];
    for (int j = 0; j < NL_SNN_HID; ++j) {
        const int8_t *w_row = w1[j];
        int32_t sum = 0;
        for (int i = 0; i < NL_SNN_IN; ++i) {
            sum += (int32_t)in_features[i] * (int32_t)w_row[i];
        }
        i_syn[j] = sum;
    }

    uint8_t out_spike_count = 0;
    uint8_t last_hid_spikes = 0;

    for (int step = 0; step < NL_SNN_STEPS; ++step) {
        uint8_t hid_spikes = 0;
        int32_t out_acc = ((int32_t)st->v_out_q12 * NL_SNN_BETA_M) >> 12;

        for (int j = 0; j < NL_SNN_HID; ++j) {
            int32_t acc = (((int32_t)st->v_hid_q12[j] * NL_SNN_BETA_M) >> 12) + i_syn[j];
            if (acc >= NL_SNN_TH_Q12) {
                acc -= NL_SNN_TH_Q12;
                hid_spikes++;
                out_acc += ((int32_t)w2[j]) << 4;
            }
            st->v_hid_q12[j] = clamp_v_q12(acc);
        }

        last_hid_spikes = hid_spikes;
        if (out_acc >= NL_SNN_TH_Q12) {
            out_acc -= NL_SNN_TH_Q12;
            out_spike_count++;
        }
        st->v_out_q12 = clamp_v_q12(out_acc);
    }

    st->total_hid_spikes = last_hid_spikes;
    return out_spike_count;
}
