#include "nl_dsp.h"
#include <string.h>

/* 61-tap Hamming-windowed quadrature bandpass FIR pair (13-30 Hz @ 2000 Hz) in Q15 */
static const int16_t k_hc_q15[NL_FIR_TAPS] = {
    -53, -47, -43, -39, -34, -26, -13, 5, 31, 65, 107, 160, 222, 293, 374, 464,
    560, 663, 770, 880, 990, 1098, 1202, 1301, 1391, 1471, 1539, 1594, 1634, 1658,
    1666,
    1658, 1634, 1594, 1539, 1471, 1391, 1301, 1202, 1098, 990, 880, 770, 663, 560,
    464, 374, 293, 222, 160, 107, 65, 31, 5, -13, -26, -34, -39, -43, -47, -53
};

static const int16_t k_hs_q15[NL_FIR_TAPS] = {
    -172, -184, -208, -243, -290, -349, -418, -497, -583, -676, -771, -866, -959, -1046, -1125, -1192,
    -1245, -1281, -1299, -1296, -1272, -1225, -1157, -1067, -957, -829, -684, -525, -356, -180,
    0,
    180, 356, 525, 684, 829, 957, 1067, 1157, 1225, 1272, 1296, 1299, 1281, 1245,
    1192, 1125, 1046, 959, 866, 771, 676, 583, 497, 418, 349, 290, 243, 208, 184, 172
};

static inline int16_t clamp_i16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

static uint32_t isqrt32(uint32_t val) {
    uint32_t res = 0;
    uint32_t bit = 1UL << 30;
    while (bit > val) {
        bit >>= 2;
    }
    while (bit != 0) {
        if (val >= res + bit) {
            val -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return res;
}

static int16_t atan2_q15(int16_t q, int16_t i) {
    if (i == 0 && q == 0) return 0;

    int32_t abs_i = (i < 0) ? -(int32_t)i : (int32_t)i;
    int32_t abs_q = (q < 0) ? -(int32_t)q : (int32_t)q;
    int32_t ang = 0;

    if (abs_i >= abs_q) {
        int32_t z_q15 = (abs_i > 0) ? ((abs_q << 15) / abs_i) : 0;
        int32_t z2_q15 = (z_q15 * z_q15) >> 15;
        int32_t poly = 10430 - ((2245 * z2_q15) >> 15);
        ang = (z_q15 * poly) >> 15;
    } else {
        int32_t z_q15 = (abs_i << 15) / abs_q;
        int32_t z2_q15 = (z_q15 * z_q15) >> 15;
        int32_t poly = 10430 - ((2245 * z2_q15) >> 15);
        ang = 16384 - ((z_q15 * poly) >> 15);
    }

    if (i < 0) ang = 32768 - ang;
    if (q < 0) ang = -ang;

    return clamp_i16(ang);
}

void nl_dsp_init(nl_dsp_state_t *st) {
    if (!st) return;
    memset(st, 0, sizeof(*st));
    st->env_mean_q15 = 256;
}

void nl_dsp_notify_stim(nl_dsp_state_t *st) {
    if (!st) return;
    st->blank_rem = NL_BLANK_SAMPLES;
}

nl_dsp_out_t nl_dsp_step(nl_dsp_state_t *st, int16_t raw_sample) {
    nl_dsp_out_t out = {0, 0, 0, 0, false};
    if (!st) return out;

    int16_t x = raw_sample;
    if (st->blank_rem > 0) {
        st->blank_rem--;
        x = st->last_clean_samp;
        out.blanked = true;
    } else {
        st->last_clean_samp = x;
    }

    st->hist[st->head] = x;

    int32_t acc_i = 0;
    int32_t acc_q = 0;
    uint16_t idx = st->head;

    for (int k = 0; k < NL_FIR_TAPS; ++k) {
        int32_t s = st->hist[idx];
        acc_i += s * (int32_t)k_hc_q15[k];
        acc_q += s * (int32_t)k_hs_q15[k];
        idx = (idx > 0) ? (idx - 1) : (NL_FIR_TAPS - 1);
    }

    st->head = (uint16_t)((st->head + 1) % NL_FIR_TAPS);

    out.I_q15 = clamp_i16(acc_i >> 15);
    out.Q_q15 = clamp_i16(acc_q >> 15);

    uint32_t mag_sq = (uint32_t)((int32_t)out.I_q15 * out.I_q15 + (int32_t)out.Q_q15 * out.Q_q15);
    uint32_t env = isqrt32(mag_sq);
    out.env_q15 = (env > 32767U) ? 32767 : (int16_t)env;
    out.phase_q15 = atan2_q15(out.Q_q15, out.I_q15);

    st->env_mean_q15 += ((int32_t)out.env_q15 - st->env_mean_q15) >> 8;

    return out;
}
