#ifndef RUIP_PRIORS_H
#define RUIP_PRIORS_H

#include <stddef.h>

#include "ruip/ruip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RUIP_MAX_HISTORIES 8u
#define RUIP_COMPARATOR_MAX_HISTORIES 3u
#define RUIP_RMAP_TAU_ORDER 256u
#define RUIP_RUIP_LAMBDA_ORDER 96u
#define RUIP_STANDARD_UIP_WEIGHT_ORDER 24u
#define RUIP_STANDARD_UIP_M_ORDER 32u
#define RUIP_COMMENSURATE_KAPPA_ORDER 81u

#define RUIP_RMAP_ROBUST_WEIGHT 0.20
#define RUIP_COMMENSURATE_UNIFORM_WEIGHT 0.99
#define RUIP_COMMENSURATE_SPIKE_WEIGHT 0.01
/* Hobbs hyperprior values for kappa_star on theta_star = theta / s_ref.
 * The analysis-scale precision is kappa = kappa_star / s_ref^2, where
 * s_ref^2 = sum(n_k) / sum(n_k I_Uk). */
#define RUIP_COMMENSURATE_KAPPA_STAR_LOWER 0.005
#define RUIP_COMMENSURATE_KAPPA_STAR_UPPER 2.0
#define RUIP_COMMENSURATE_KAPPA_STAR_SPIKE 200.0
#define RUIP_POWER_PRIOR_A0 0.5

typedef struct {
    double mean;
    double variance;
    double log_weight;
    double j_g;
    double m_eff;
    unsigned retention_mask;
    double tau;
    double robust_component;
    double borrowing_m;
    double source_weight[RUIP_MAX_HISTORIES];
} ruip_normal_component_t;

typedef struct {
    ruip_normal_component_t *items;
    size_t count;
} ruip_normal_mixture_t;

typedef struct {
    double t_g;
    double gate;
    double local_mean;
    double local_variance;
    double local_weight_sum;
    double expected_borrowed_precision;
    double borrowed_precision_ratio;
    double cap_binding_mass;
    double fisher_cross_covariance_error;
    double source_contribution[RUIP_MAX_HISTORIES];
} ruip_promoted_diagnostics_t;

typedef struct {
    double local_mean;
    double local_variance;
    double local_precision;
    double global_discount;
    /* For design-stage capped-medoid rUIP, this is 1/R_k, where R_k is
     * source k's compatibility-discounted retained information. */
    double working_variance[RUIP_MAX_HISTORIES];
    /* Normalized retained-information weights q_k=R_k/sum(R). */
    double local_weight[RUIP_MAX_HISTORIES];
} ruip_design_stage_diagnostics_t;

void ruip_normal_mixture_free(ruip_normal_mixture_t *mixture);

int ruip_rmap_mixture(const ruip_history_t *histories,
                      size_t history_count,
                      ruip_normal_mixture_t *mixture);

int ruip_ruip_mixture(const ruip_history_t *histories,
                      size_t history_count,
                      ruip_normal_mixture_t *mixture);

/* History-only design-stage capped-medoid rUIP.  The source medoid minimizes
 * total L1 distance (ties: larger information, then input order); each source
 * retains min(J_k,J_m) times its Gaussian compatibility with that medoid.
 * The prior is N(sum(q_k h_k), 1/sum(R_k)+delta_clin^2). */
int ruip_design_stage_ruip_prior(
                      const ruip_history_t *histories,
                      size_t history_count,
                      double delta_clin,
                      ruip_normal_mixture_t *prior,
                      ruip_design_stage_diagnostics_t *diagnostics);

/* Promoted rUIP: Fisher-residual IG(3/2,6) local mixture, q50 global
 * compatibility gate, and a componentwise 0.5-current-information cap.
 * current_variance is the sampling variance of the current control estimate.
 * The closed gate returns the proper N(0,1e10) branch. */
int ruip_promoted_ruip_mixture(const ruip_history_t *histories,
                               size_t history_count,
                               double current_estimate,
                               double current_variance,
                               ruip_normal_mixture_t *mixture,
                               ruip_promoted_diagnostics_t *diagnostics);

/* Prespecified mechanism variants. no_local uses one information-weighted
 * historical component; no_global forces the gate open; cap_ratio may be
 * INFINITY for the no-cap variant. */
int ruip_promoted_ruip_mixture_variant(const ruip_history_t *histories,
                               size_t history_count,
                               double current_estimate,
                               double current_variance,
                               int no_local,
                               int no_global,
                               double cap_ratio,
                               ruip_normal_mixture_t *mixture,
                               ruip_promoted_diagnostics_t *diagnostics);

int ruip_standard_uip_mixture(const ruip_history_t *histories,
                              size_t history_count,
                              double planned_n_control,
                              ruip_normal_mixture_t *mixture);

/* Fixed common-power prior with a0=0.5 applied to every historical
 * summary-data likelihood. */
int ruip_power_prior_mixture(const ruip_history_t *histories,
                             size_t history_count,
                             ruip_normal_mixture_t *mixture);

/* Hobbs-style pooled-history commensurate prior.  A flat-baseline posterior
 * for one common historical parameter is linked to the current parameter by
 * one commensurability precision.  The result has exactly 81 components. */
int ruip_commensurate_mixture(const ruip_history_t *histories,
                              size_t history_count,
                              ruip_normal_mixture_t *mixture);

/* Legacy source-product comparator: one independent commensurability
 * precision per historical source, producing 81^K components for K<=3.
 * Retained for reproducibility only; v5 CP does not call this function. */
int ruip_commensurate_source_product_mixture(
                              const ruip_history_t *histories,
                              size_t history_count,
                              ruip_normal_mixture_t *mixture);

#ifdef __cplusplus
}
#endif

#endif
