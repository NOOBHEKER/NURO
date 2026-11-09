#ifndef NL_SNN_H
#define NL_SNN_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NL_SNN_IN      16
#define NL_SNN_HID     64
#define NL_SNN_STEPS   8
#define NL_SNN_TH_Q12  4096   /* theta = 1.0 in Q12 */
#define NL_SNN_BETA_M  3686   /* beta = 0.9 in Q12 (3686 / 4096 = 0.8999) */
#define NL_SNN_MAX_V   28672  /* Clamp at 7.0 * theta to prevent int16 overflow */

typedef struct {
    int16_t v_hid_q12[NL_SNN_HID];
    int16_t v_out_q12;
    uint8_t total_hid_spikes;
} nl_snn_state_t;

void nl_snn_reset(nl_snn_state_t *st);

/* Executes 1 timestep across 64 hidden LIF neurons + 1 readout neuron */
uint8_t nl_snn_step(
    nl_snn_state_t *st,
    const int8_t in_features[NL_SNN_IN],
    const int8_t w1[NL_SNN_IN][NL_SNN_HID],
    const int8_t w2[NL_SNN_HID],
    bool inhibit
);

#ifdef __cplusplus
}
#endif

#endif /* NL_SNN_H */
