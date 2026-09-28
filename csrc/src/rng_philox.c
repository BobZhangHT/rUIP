#include "ruip/rng.h"

#include <math.h>
#include <stddef.h>

#define PHILOX_M0 UINT32_C(0xD2511F53)
#define PHILOX_M1 UINT32_C(0xCD9E8D57)
#define PHILOX_W0 UINT32_C(0x9E3779B9)
#define PHILOX_W1 UINT32_C(0xBB67AE85)

static void multiply_high_low(uint32_t multiplier, uint32_t value,
                              uint32_t *high, uint32_t *low) {
    const uint64_t product = (uint64_t)multiplier * (uint64_t)value;
    *low = (uint32_t)product;
    *high = (uint32_t)(product >> 32);
}

void ruip_philox4x32_10(const uint32_t counter[4], const uint32_t key[2],
                        uint32_t output[4]) {
    uint32_t c0 = counter[0], c1 = counter[1];
    uint32_t c2 = counter[2], c3 = counter[3];
    uint32_t k0 = key[0], k1 = key[1];
    for (int round = 0; round < 10; ++round) {
        uint32_t hi0, lo0, hi1, lo1;
        multiply_high_low(PHILOX_M0, c0, &hi0, &lo0);
        multiply_high_low(PHILOX_M1, c2, &hi1, &lo1);
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += PHILOX_W0;
        k1 += PHILOX_W1;
    }
    output[0] = c0; output[1] = c1;
    output[2] = c2; output[3] = c3;
}

void ruip_rng_init(ruip_rng_t *rng, uint64_t phase_root,
                   uint32_t scenario_index, uint32_t replicate_id) {
    if (rng == NULL) return;
    rng->key[0] = (uint32_t)phase_root;
    rng->key[1] = (uint32_t)(phase_root >> 32);
    rng->counter_prefix[0] = replicate_id;
    rng->counter_prefix[1] = scenario_index;
    rng->counter_prefix[2] = 0u;
    rng->block_index = 0u;
    rng->output_index = 4u;
    rng->spare_normal = 0.0;
    rng->has_spare_normal = 0;
}

uint32_t ruip_rng_u32(ruip_rng_t *rng) {
    if (rng->output_index >= 4u) {
        const uint32_t counter[4] = {
            rng->counter_prefix[0], rng->counter_prefix[1],
            rng->counter_prefix[2], (uint32_t)rng->block_index
        };
        ruip_philox4x32_10(counter, rng->key, rng->output);
        ++rng->block_index;
        rng->output_index = 0u;
    }
    return rng->output[rng->output_index++];
}

double ruip_rng_uniform_open(ruip_rng_t *rng) {
    /* 27 high bits followed by 26 high bits form j in [0, 2^53). */
    const uint64_t high = (uint64_t)(ruip_rng_u32(rng) >> 5);
    const uint64_t low = (uint64_t)(ruip_rng_u32(rng) >> 6);
    const uint64_t j = (high << 26) | low;
    return ((double)j + 0.5) * 0x1.0p-53;
}

double ruip_rng_normal(ruip_rng_t *rng) {
    if (rng->has_spare_normal) {
        rng->has_spare_normal = 0;
        return rng->spare_normal;
    }
    const double u1 = ruip_rng_uniform_open(rng);
    const double u2 = ruip_rng_uniform_open(rng);
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 6.283185307179586476925286766559 * u2;
    rng->spare_normal = radius * sin(angle);
    rng->has_spare_normal = 1;
    return radius * cos(angle);
}

uint32_t ruip_rng_binomial_bernoulli_sum(ruip_rng_t *rng, uint32_t n,
                                        double probability) {
    uint32_t events = 0u;
    if (!(probability > 0.0)) return 0u;
    if (probability >= 1.0) return n;
    for (uint32_t i = 0; i < n; ++i)
        events += (uint32_t)(ruip_rng_uniform_open(rng) < probability);
    return events;
}

