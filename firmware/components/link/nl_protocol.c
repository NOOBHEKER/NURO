#include "nl_protocol.h"

uint16_t nl_crc16_ccitt_false(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFFU;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; ++b) {
            if (crc & 0x8000U) {
                crc = (uint16_t)((crc << 1) ^ 0x1021U);
            } else {
                crc = (uint16_t)(crc << 1);
            }
        }
    }
    return crc;
}

int nl_sample_frame_validate(const nl_sample_frame_t *frame) {
    if (!frame) return -1;
    if (frame->magic != NL_MAGIC_SAMPLE) return -2;
    if (frame->ver != NL_PROTOCOL_VER) return -3;
    uint16_t expected = nl_crc16_ccitt_false((const uint8_t *)frame, 28);
    if (frame->crc16 != expected) return -4;
    return 0;
}
