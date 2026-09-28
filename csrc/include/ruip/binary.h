#ifndef RUIP_BINARY_H
#define RUIP_BINARY_H

#include <stddef.h>

#include "ruip/ruip.h"
#include "ruip/priors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  RUIP_BINARY_NIP = 0,
  RUIP_BINARY_RMAP = 1,
  RUIP_BINARY_COMMENSURATE = 2,
  RUIP_BINARY_STANDARD_UIP = 3,
  RUIP_BINARY_RUIP = 4,
  RUIP_BINARY_POWER_PRIOR = 5,
  RUIP_BINARY_PROMOTED_RUIP = 6
} ruip_binary_method_t;

typedef enum {
  RUIP_BINARY_STATUS_OK = 0,
  RUIP_BINARY_STATUS_IMPROPER_FLAT_LOGIT = 1,
  RUIP_BINARY_STATUS_TREATMENT_ENDPOINT = 2,
  RUIP_BINARY_STATUS_QUADRATURE_WARNING = 3,
  RUIP_BINARY_STATUS_INVALID_ARGUMENT = 4,
  RUIP_BINARY_STATUS_NUMERICAL_FAILURE = 5
} ruip_binary_status_t;

typedef struct {
  ruip_binary_method_t method;
  ruip_binary_status_t status;
  int effect_available;
  double eta_c_mean;
  double eta_c_variance;
  double log_or_mean;
  double log_or_variance;
  double probability_log_or_positive;
  double log_or_q025;
  double log_or_q975;
  double p_c_mean;
  double p_t_mean;
  double risk_difference_mean;
  double risk_difference_q025;
  double risk_difference_q975;
  double quadrature_evidence_relative_difference;
  double integration_error;
  size_t posterior_node_count;
  size_t prior_component_count;
  double diagnostic_tau;
  double diagnostic_robust_component;
  double diagnostic_j_g;
  double diagnostic_m_eff;
  double diagnostic_retention[RUIP_HISTORY_COUNT];
  double diagnostic_borrowing_m;
  double diagnostic_source_weight[RUIP_HISTORY_COUNT];
  ruip_promoted_diagnostics_t promoted;
} ruip_binary_result_t;

/* Construct the frozen 0.5-corrected historical logit summary. */
int ruip_binary_history(uint32_t events, uint32_t n, ruip_history_t *output);

/* Full exact Binomial log likelihood, including the combinatorial constant. */
double ruip_binary_log_likelihood(double eta, ruip_binomial_t data);

/*
 * Analyze a current two-arm binary data set.  NIP deliberately ignores
 * histories and planned_n_control.  All borrowing priors are history-only;
 * planned_n_control is used solely by the frozen standard-UIP comparator.
 */
int ruip_binary_analyze(ruip_binary_method_t method,
                        const ruip_history_t *histories, size_t history_count,
                        ruip_binomial_t control, ruip_binomial_t treatment,
                        double planned_n_control, ruip_binary_result_t *output);

/* Calibration-only path: computes the complete control posterior and S, while
 * deliberately leaving estimators and intervals unavailable. */
int ruip_binary_analyze_statistic(ruip_binary_method_t method,
                                  const ruip_history_t *histories,
                                  size_t history_count,
                                  ruip_binomial_t control,
                                  ruip_binomial_t treatment,
                                  double planned_n_control,
                                  ruip_binary_result_t *output);

/* Fixed-a0 normalized power-prior comparator using the historical Binomial
 * likelihoods themselves. With a0=0.5 and a flat-logit baseline, the current
 * control posterior is a conjugate Beta distribution. */
int ruip_binary_analyze_power_prior_exact(
    const ruip_binomial_t *historical, size_t history_count,
    ruip_binomial_t control, ruip_binomial_t treatment, int statistic_only,
    ruip_binary_result_t *output);

int ruip_binary_analyze_promoted_variant(
                        const ruip_history_t *histories, size_t history_count,
                        ruip_binomial_t control, ruip_binomial_t treatment,
                        int no_local, int no_global, double cap_ratio,
                        int statistic_only, ruip_binary_result_t *output);

/* Deterministic validation hooks for the integer-parameter beta CDF. */
double ruip_binary_beta_cdf_finite_sum(uint32_t a, uint32_t b, double x);
double ruip_binary_beta_cdf_lentz(uint32_t a, uint32_t b, double x);

const char *ruip_binary_status_string(ruip_binary_status_t status);

#ifdef __cplusplus
}
#endif

#endif
