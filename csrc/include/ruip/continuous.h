#ifndef RUIP_CONTINUOUS_H
#define RUIP_CONTINUOUS_H

#include <stddef.h>

#include "ruip/priors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RUIP_CONTINUOUS_NIP = 0,
    RUIP_CONTINUOUS_RMAP = 1,
    RUIP_CONTINUOUS_COMMENSURATE = 2,
    RUIP_CONTINUOUS_STANDARD_UIP = 3,
    RUIP_CONTINUOUS_RUIP = 4,
    RUIP_CONTINUOUS_POWER_PRIOR = 5,
    RUIP_CONTINUOUS_PROMOTED_RUIP = 6
} ruip_continuous_method_t;

typedef struct {
    double mu_c_mean;
    double mu_c_variance;
    double delta_mean;
    double delta_variance;
    double probability_delta_positive;
    double delta_q025;
    double delta_q975;
    double expected_j_g;
    double expected_m_eff;
    double expected_tau;
    double robust_component_probability;
    double expected_borrowing_m;
    double expected_source_weight[RUIP_MAX_HISTORIES];
    double state_mass[1u << RUIP_MAX_HISTORIES];
    double retention_probability[RUIP_MAX_HISTORIES];
    ruip_promoted_diagnostics_t promoted;
    size_t component_count;
    int numerical_ok;
} ruip_continuous_result_t;

int ruip_continuous_analyze(ruip_continuous_method_t method,
                            const ruip_history_t *histories,
                            size_t history_count,
                            const ruip_continuous_group_t *control,
                            const ruip_continuous_group_t *treatment,
                            double planned_n_control,
                            ruip_continuous_result_t *result);

int ruip_continuous_analyze_promoted_variant(
                            const ruip_history_t *histories,
                            size_t history_count,
                            const ruip_continuous_group_t *control,
                            const ruip_continuous_group_t *treatment,
                            int no_local, int no_global, double cap_ratio,
                            ruip_continuous_result_t *result);

#ifdef __cplusplus
}
#endif

#endif
