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

uint8_t nl_snn_step(
    nl_snn_state_t *st,
    const int8_t in_features[NL_SNN_IN],
    const int8_t w1[NL_SNN_IN][NL_SNN_HID],
    const int8_t w2[NL_SNN_HID],
    bool inhibit
) {
    if (inhibit) {
        /* DEC-007: Freeze SNN during stimulation artifact blanking window */
        st->total_hid_spikes = 0;
        return 0;
    }

    uint8_t hid_spikes = 0;
    int32_t out_acc = ((int32_t)st->v_out_q12 * NL_SNN_BETA_M) >> 12;

    for (int j = 0; j < NL_SNN_HID; ++j) {
        int32_t acc = ((int32_t)st->v_hid_q12[j] * NL_SNN_BETA_M) >> 12;
        for (int i = 0; i < NL_SNN_IN; ++i) {
            acc += (int32_t)in_features[i] * (int32_t)w1[i][j];
        }
        if (acc >= NL_SNN_TH_Q12) {
            acc -= NL_SNN_TH_Q12;
            hid_spikes++;
            out_acc += ((int32_t)w2[j]) << 4;
        }
        st->v_hid_q12[j] = clamp_v_q12(acc);
    }

    st->total_hid_spikes = hid_spikes;
    uint8_t out_spike = 0;
    if (out_acc >= NL_SNN_TH_Q12) {
        out_acc -= NL_SNN_TH_Q12;
        out_spike = 1;
    }
    st->v_out_q12 = clamp_v_q12(out_acc);
    return out_spike;
}
