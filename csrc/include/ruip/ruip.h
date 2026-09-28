#ifndef RUIP_RUIP_H
#define RUIP_RUIP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RUIP_HISTORY_COUNT 3u
#define RUIP_SCENARIO_COUNT 12u

typedef enum {
    RUIP_CONTINUOUS = 0,
    RUIP_BINARY = 1
} ruip_outcome_t;

typedef enum {
    RUIP_CALIBRATION = 0,
    RUIP_NULL_VALIDATION = 1,
    RUIP_ALTERNATIVE = 2
} ruip_phase_t;

typedef struct {
    uint32_t n;
    double mean;
    double known_variance;
} ruip_continuous_summary_t;

typedef struct {
    uint32_t n;
    uint32_t events;
} ruip_binomial_t;

typedef struct {
    uint32_t n;
    double estimate;
    double unit_info;
} ruip_history_t;

typedef ruip_history_t ruip_history_summary_t;
typedef ruip_continuous_summary_t ruip_continuous_group_t;

typedef struct {
    ruip_continuous_summary_t control;
    ruip_continuous_summary_t treatment;
} ruip_continuous_current_t;

typedef struct {
    ruip_binomial_t control;
    ruip_binomial_t treatment;
} ruip_binary_current_t;

typedef struct {
    ruip_outcome_t outcome;
    ruip_phase_t phase;
    uint32_t scenario_index;
    uint32_t replicate_id;
    double historical_shifts[RUIP_HISTORY_COUNT];
    double true_primary_effect;
    double true_risk_difference;
    union {
        ruip_continuous_current_t continuous;
        ruip_binary_current_t binary;
    } current;
    ruip_history_summary_t histories[RUIP_HISTORY_COUNT];
} ruip_dataset_t;

/* Generate one complete, coordinate-addressed data set. Returns 0 on success. */
int ruip_generate_dataset(ruip_outcome_t outcome, ruip_phase_t phase,
                          uint32_t scenario_index, uint32_t replicate_id,
                          ruip_dataset_t *output);

#ifdef __cplusplus
}
#endif

#endif
