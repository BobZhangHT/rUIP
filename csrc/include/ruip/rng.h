#ifndef RUIP_RNG_H
#define RUIP_RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t key[2];
    uint32_t counter_prefix[3];
    uint64_t block_index;
    uint32_t output[4];
    uint32_t output_index;
    double spare_normal;
    int has_spare_normal;
} ruip_rng_t;

void ruip_philox4x32_10(const uint32_t counter[4], const uint32_t key[2],
                        uint32_t output[4]);
void ruip_rng_init(ruip_rng_t *rng, uint64_t phase_root,
                   uint32_t scenario_index, uint32_t replicate_id);
uint32_t ruip_rng_u32(ruip_rng_t *rng);
double ruip_rng_uniform_open(ruip_rng_t *rng);
double ruip_rng_normal(ruip_rng_t *rng);
uint32_t ruip_rng_binomial_bernoulli_sum(ruip_rng_t *rng, uint32_t n,
                                        double probability);

#ifdef __cplusplus
}
#endif

#endif

