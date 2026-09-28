#ifndef RUIP_SHA256_H
#define RUIP_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    uint8_t buffer[64];
    size_t buffer_size;
} ruip_sha256_t;

void ruip_sha256_init(ruip_sha256_t *context);
void ruip_sha256_update(ruip_sha256_t *context, const void *data, size_t size);
void ruip_sha256_final(ruip_sha256_t *context, uint8_t digest[32]);
void ruip_sha256_hex(const void *data, size_t size, char output[65]);

#ifdef __cplusplus
}
#endif

#endif
