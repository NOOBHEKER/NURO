#ifndef NL_PROTOCOL_H
#define NL_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NL_MAGIC_SAMPLE  0xA5C3U
#define NL_MAGIC_RESULT  0x5C3AU
#define NL_PROTOCOL_VER  1U

/* Operating modes */
#define NL_MODE_LOCKSTEP    0U
#define NL_MODE_REALTIME    1U
#define NL_MODE_STANDALONE  2U

/* Controller states */
#define NL_STATE_IDLE        0U
#define NL_STATE_ARMED       1U
#define NL_STATE_SUPPRESS    2U
#define NL_STATE_REFRACTORY  3U
#define NL_STATE_FAULT       4U

#pragma pack(push, 1)

typedef struct {
    uint16_t magic;       /* 0: 0xA5C3 */
    uint8_t  ver;         /* 2: 1 */
    uint8_t  flags;       /* 3: bit0=lockstep, bit1=arm-hint, bit2=last-block */
    uint32_t seq;         /* 4: increments per frame */
    uint32_t sim_t_us;    /* 8: simulation time of first sample */
    uint32_t host_t_us;   /* 12: host monotonic clock, us */
    uint8_t  n_ch;        /* 16: channels in payload (1..4) */
    uint8_t  n_samp;      /* 17: samples per channel (1..4) */
    uint8_t  mode;        /* 18: 0=lockstep, 1=realtime, 2=standalone */
    uint8_t  rsv;         /* 19: 0 */
    int16_t  samples[4];  /* 20..27: 4x int16 interleaved samples */
    uint16_t crc16;       /* 28: CRC16-CCITT-FALSE over bytes 0..27 */
    uint16_t pad;         /* 30: 0 */
} nl_sample_frame_t;

typedef struct {
    uint16_t magic;       /* 0: 0x5C3A */
    uint8_t  ver;         /* 2: 1 */
    uint8_t  state;       /* 3: IDLE/ARMED/SUPPRESS/REFRACTORY/FAULT */
    uint32_t seq;         /* 4: echo of sample seq */
    uint32_t t_rx_us;     /* 8: device clock at ISR entry */
    uint32_t t_dsp_us;    /* 12: device clock after DSP */
    uint32_t t_dec_us;    /* 16: device clock after SNN decision */
    uint32_t t_stim_us;   /* 20: pulse edge timestamp (0 if none) */
    uint8_t  cls;         /* 24: 0=none, 1=burst, 2=pre-ictal */
    uint8_t  conf;        /* 25: 0..255 confidence */
    int16_t  phase_q15;   /* 26: instantaneous phase, Q15 of [-pi, pi] */
    int16_t  R_q15;       /* 28: running order/envelope proxy in Q15 */
    uint16_t budget_used; /* 30: pulses in rolling 60 s window */
} nl_result_frame_t;

#pragma pack(pop)

_Static_assert(sizeof(nl_sample_frame_t) == 32, "nl_sample_frame_t must be exactly 32 bytes");
_Static_assert(sizeof(nl_result_frame_t) == 32, "nl_result_frame_t must be exactly 32 bytes");

uint16_t nl_crc16_ccitt_false(const uint8_t *data, size_t len);
int nl_sample_frame_validate(const nl_sample_frame_t *frame);

#ifdef __cplusplus
}
#endif

#endif /* NL_PROTOCOL_H */
