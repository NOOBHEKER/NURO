#ifndef NL_CALC_EMBEDDED_H
#define NL_CALC_EMBEDDED_H

#include <Arduino.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t  env_mean_q15;
    int16_t  last_env_q15;
    int16_t  last_phase_q15;
    uint8_t  last_spikes;
    uint8_t  state;
    uint16_t budget_used;
    uint32_t last_stim_us;
    bool     is_blanked;
} nl_live_telemetry_t;

/* Initialize embedded calculator & flash databases */
void nl_calc_init(void);

/* Process single incoming character from serial */
void nl_calc_process_char(char c);

/* Execute full command line directly */
void nl_calc_exec_command(const char *cmd_line);

/* Query whether interactive calculator is actively printing */
bool nl_calc_is_active(void);

/* Hardware RGB LED control: safe low-glare levels */
void nl_set_status_led(uint8_t r, uint8_t g, uint8_t b);

/* Explicit arm gate for GPIO4 stimulation (boot default: disarmed) */
bool nl_is_armed(void);
void nl_set_armed(bool armed);

/* One-shot boot self-test: chip, flash, PSRAM, reset reason, MAC */
void nl_run_boot_selftest(void);

/* Query live neuromodulation DSP / SNN statistics */
void nl_get_telemetry(nl_live_telemetry_t *telem);

/* Run synthetic 20Hz burst self-test (Early Warning Yellow -> Phase Trough Red + GPIO 4 Pulse) */
void nl_run_sim_burst(void);

/* Restore baseline resting state */
void nl_run_sim_rest(void);

#ifdef __cplusplus
}
#endif

#endif /* NL_CALC_EMBEDDED_H */
