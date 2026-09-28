#include "ruip/priors.h"

#include <math.h>
#include <stdlib.h>

int ruip_power_prior_mixture(const ruip_history_t *histories,
                             size_t history_count,
                             ruip_normal_mixture_t *mixture) {
    double total_information = 0.0;
    double weighted_summary = 0.0;
    size_t index;
    ruip_normal_component_t *component;

    if (mixture == NULL || histories == NULL || history_count == 0u ||
        history_count > RUIP_MAX_HISTORIES) {
        return -1;
    }
    mixture->items = NULL;
    mixture->count = 0u;
    for (index = 0u; index < history_count; ++index) {
        double information;
        if (histories[index].n == 0u || !isfinite(histories[index].estimate) ||
            !isfinite(histories[index].unit_info) ||
            histories[index].unit_info <= 0.0) {
            return -1;
        }
        information = (double)histories[index].n * histories[index].unit_info;
        total_information += information;
        weighted_summary += information * histories[index].estimate;
    }
    if (!isfinite(total_information) || total_information <= 0.0 ||
        !isfinite(weighted_summary)) {
        return -1;
    }
    component = (ruip_normal_component_t *)calloc(1u, sizeof(*component));
    if (component == NULL) {
        return -2;
    }
    component->mean = weighted_summary / total_information;
    component->variance = 1.0 / (RUIP_POWER_PRIOR_A0 * total_information);
    component->log_weight = 0.0;
    mixture->items = component;
    mixture->count = 1u;
    return 0;
}
