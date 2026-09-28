#include "ruip/priors.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "ruip/mathutil.h"
#include "quadrature_tables.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static void store_component(ruip_normal_component_t *component,
                            double precision, double weighted,
                            double quadratic, double sum_log_variance,
                            double log_mass, size_t history_count) {
    component->mean = weighted / precision;
    component->variance = 1.0 / precision;
    component->log_weight = log_mass - 0.5 * (
        (double)(history_count - 1u) * log(2.0 * M_PI)
        + sum_log_variance + log(precision) + quadratic
        - weighted * weighted / precision);
}

int ruip_commensurate_source_product_mixture(const ruip_history_t *h, size_t k,
                                              ruip_normal_mixture_t *out) {
    double kappa[RUIP_COMMENSURATE_KAPPA_ORDER];
    double logmass[RUIP_COMMENSURATE_KAPPA_ORDER];
    double precision[RUIP_MAX_HISTORIES][RUIP_COMMENSURATE_KAPPA_ORDER];
    double weighted[RUIP_MAX_HISTORIES][RUIP_COMMENSURATE_KAPPA_ORDER];
    double quadratic[RUIP_MAX_HISTORIES][RUIP_COMMENSURATE_KAPPA_ORDER];
    double log_variance[RUIP_MAX_HISTORIES][RUIP_COMMENSURATE_KAPPA_ORDER];
    double *logs = NULL, lse, jsum = 0.0, nsum = 0.0, reference_sd_squared;
    size_t count = 1u, index = 0u, i, d0, d1, d2;
    if (out == NULL || h == NULL || k == 0u ||
        k > RUIP_COMPARATOR_MAX_HISTORIES) return -1;
    out->items = NULL;
    out->count = 0u;
    for (i = 0u; i < k; ++i) {
        if (h[i].n == 0u || !isfinite(h[i].estimate)
            || !isfinite(h[i].unit_info) || h[i].unit_info <= 0.0) return -1;
        jsum += (double)h[i].n * h[i].unit_info;
        nsum += (double)h[i].n;
    }
    reference_sd_squared = nsum / jsum;
    for (i = 0u; i < RUIP_COMMENSURATE_KAPPA_ORDER - 1u; ++i) {
        const double kappa_star =
            0.5 * (RUIP_COMMENSURATE_KAPPA_STAR_UPPER
                   - RUIP_COMMENSURATE_KAPPA_STAR_LOWER) * ruip_gl80_nodes[i]
            + 0.5 * (RUIP_COMMENSURATE_KAPPA_STAR_UPPER
                     + RUIP_COMMENSURATE_KAPPA_STAR_LOWER);
        kappa[i] = kappa_star / reference_sd_squared;
        logmass[i] = log(RUIP_COMMENSURATE_UNIFORM_WEIGHT
                         * 0.5 * ruip_gl80_weights[i]);
    }
    kappa[RUIP_COMMENSURATE_KAPPA_ORDER - 1u] =
        RUIP_COMMENSURATE_KAPPA_STAR_SPIKE / reference_sd_squared;
    logmass[RUIP_COMMENSURATE_KAPPA_ORDER - 1u] =
        log(RUIP_COMMENSURATE_SPIKE_WEIGHT);
    for (i = 0u; i < k; ++i) {
        const double sampling_variance = 1.0 / ((double)h[i].n * h[i].unit_info);
        for (d0 = 0u; d0 < RUIP_COMMENSURATE_KAPPA_ORDER; ++d0) {
            const double variance = sampling_variance + 1.0 / kappa[d0];
            const double p = 1.0 / variance;
            precision[i][d0] = p;
            weighted[i][d0] = p * h[i].estimate;
            quadratic[i][d0] = weighted[i][d0] * h[i].estimate;
            log_variance[i][d0] = log(variance);
        }
    }
    for (i = 0u; i < k; ++i) count *= RUIP_COMMENSURATE_KAPPA_ORDER;
    out->items = (ruip_normal_component_t*)malloc(count * sizeof(*out->items));
    logs = (double*)malloc(count * sizeof(*logs));
    if (out->items == NULL || logs == NULL) {
        free(logs);
        ruip_normal_mixture_free(out);
        return -3;
    }
    if (k == 1u) {
        for (d0 = 0u; d0 < RUIP_COMMENSURATE_KAPPA_ORDER; ++d0) {
            store_component(&out->items[index], precision[0][d0], weighted[0][d0],
                            quadratic[0][d0], log_variance[0][d0], logmass[d0], k);
            logs[index] = out->items[index].log_weight;
            ++index;
        }
    } else if (k == 2u) {
        for (d1 = 0u; d1 < RUIP_COMMENSURATE_KAPPA_ORDER; ++d1) {
            for (d0 = 0u; d0 < RUIP_COMMENSURATE_KAPPA_ORDER; ++d0) {
                store_component(&out->items[index],
                    precision[0][d0] + precision[1][d1],
                    weighted[0][d0] + weighted[1][d1],
                    quadratic[0][d0] + quadratic[1][d1],
                    log_variance[0][d0] + log_variance[1][d1],
                    logmass[d0] + logmass[d1], k);
                logs[index] = out->items[index].log_weight;
                ++index;
            }
        }
    } else {
        for (d2 = 0u; d2 < RUIP_COMMENSURATE_KAPPA_ORDER; ++d2) {
            for (d1 = 0u; d1 < RUIP_COMMENSURATE_KAPPA_ORDER; ++d1) {
                const double p12 = precision[1][d1] + precision[2][d2];
                const double q12 = weighted[1][d1] + weighted[2][d2];
                const double r12 = quadratic[1][d1] + quadratic[2][d2];
                const double lv12 = log_variance[1][d1] + log_variance[2][d2];
                const double lm12 = logmass[d1] + logmass[d2];
                for (d0 = 0u; d0 < RUIP_COMMENSURATE_KAPPA_ORDER; ++d0) {
                    store_component(&out->items[index], precision[0][d0] + p12,
                        weighted[0][d0] + q12, quadratic[0][d0] + r12,
                        log_variance[0][d0] + lv12, logmass[d0] + lm12, k);
                    logs[index] = out->items[index].log_weight;
                    ++index;
                }
            }
        }
    }
    lse = ruip_logsumexp(logs, count);
    for (index = 0u; index < count; ++index) out->items[index].log_weight -= lse;
    free(logs);
    out->count = count;
    return 0;
}

int ruip_commensurate_mixture(const ruip_history_t *h, size_t k,
                              ruip_normal_mixture_t *out) {
    double jsum = 0.0, weighted = 0.0, nsum = 0.0;
    double reference_sd_squared, historical_mean, historical_variance;
    double logs[RUIP_COMMENSURATE_KAPPA_ORDER], lse;
    size_t i;
    if (out == NULL || h == NULL || k == 0u || k > RUIP_MAX_HISTORIES)
        return -1;
    out->items = NULL;
    out->count = 0u;
    for (i = 0u; i < k; ++i) {
        const double information = (double)h[i].n * h[i].unit_info;
        if (h[i].n == 0u || !isfinite(h[i].estimate) ||
            !isfinite(h[i].unit_info) || h[i].unit_info <= 0.0)
            return -1;
        jsum += information;
        weighted += information * h[i].estimate;
        nsum += (double)h[i].n;
    }
    historical_mean = weighted / jsum;
    historical_variance = 1.0 / jsum;
    reference_sd_squared = nsum / jsum;
    out->items = (ruip_normal_component_t*)calloc(
        RUIP_COMMENSURATE_KAPPA_ORDER, sizeof(*out->items));
    if (out->items == NULL) return -3;
    for (i = 0u; i < RUIP_COMMENSURATE_KAPPA_ORDER - 1u; ++i) {
        const double kappa_star =
            0.5 * (RUIP_COMMENSURATE_KAPPA_STAR_UPPER -
                   RUIP_COMMENSURATE_KAPPA_STAR_LOWER) * ruip_gl80_nodes[i]
            + 0.5 * (RUIP_COMMENSURATE_KAPPA_STAR_UPPER +
                     RUIP_COMMENSURATE_KAPPA_STAR_LOWER);
        const double kappa = kappa_star / reference_sd_squared;
        out->items[i].mean = historical_mean;
        out->items[i].variance = historical_variance + 1.0 / kappa;
        logs[i] = log(RUIP_COMMENSURATE_UNIFORM_WEIGHT *
                      0.5 * ruip_gl80_weights[i]);
        out->items[i].log_weight = logs[i];
    }
    i = RUIP_COMMENSURATE_KAPPA_ORDER - 1u;
    out->items[i].mean = historical_mean;
    out->items[i].variance = historical_variance +
        reference_sd_squared / RUIP_COMMENSURATE_KAPPA_STAR_SPIKE;
    logs[i] = log(RUIP_COMMENSURATE_SPIKE_WEIGHT);
    out->items[i].log_weight = logs[i];
    lse = ruip_logsumexp(logs, RUIP_COMMENSURATE_KAPPA_ORDER);
    for (i = 0u; i < RUIP_COMMENSURATE_KAPPA_ORDER; ++i)
        out->items[i].log_weight -= lse;
    out->count = RUIP_COMMENSURATE_KAPPA_ORDER;
    return 0;
}
