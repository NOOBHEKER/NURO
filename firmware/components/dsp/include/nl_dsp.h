#ifndef NL_DSP_H
#define NL_DSP_H

#include <stdint.h>
#include <stdbool.h>

#define NL_FIR_TAPS       61
#define NL_BLANK_SAMPLES  40  /* 20 ms @ 2 kHz stimulation artifact blanking window */

typedef struct {
    int16_t  hist[NL_FIR_TAPS];
    uint16_t head;
    uint16_t blank_rem;       /* remaining samples in artifact blanking window */
    int16_t  last_clean_samp; /* sample held during blanking clamp */
    int32_t  env_mean_q15;    /* 200 ms exponential moving average of envelope */
} nl_dsp_state_t;

typedef struct {
    int16_t I_q15;
    int16_t Q_q15;
    int16_t env_q15;
    int16_t phase_q15; /* [-32768, 32767] mapping [-pi, pi) */
    bool    blanked;
} nl_dsp_out_t;

void nl_dsp_init(nl_dsp_state_t *st);
void nl_dsp_notify_stim(nl_dsp_state_t *st);
nl_dsp_out_t nl_dsp_step(nl_dsp_state_t *st, int16_t raw_sample);

#endif /* NL_DSP_H */
