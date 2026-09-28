#include "ruip/protocol.h"
#include "ruip/mathutil.h"
#include "ruip/rng.h"
#include "quadrature_tables.h"

#include <math.h>
#include <stddef.h>

const char *const ruip_scenario_names[RUIP_SCENARIO_COUNT] = {
    "compatible", "local_positive", "local_negative", "global_positive",
    "global_negative", "reinforcing_positive", "reinforcing_negative",
    "bridge_positive", "bridge_negative", "crossover_positive",
    "crossover_negative", "random_heterogeneity"
};

static int validate_symmetric_rule(const double *nodes, const double *weights,
                                   size_t count, double expected_sum) {
    double total = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(nodes[i]) || !isfinite(weights[i]) || !(weights[i] > 0.0)) return -1;
        if (fabs(nodes[i] + nodes[count - 1u - i]) > 5e-14
            || fabs(weights[i] - weights[count - 1u - i]) > 5e-14) return -1;
        total += weights[i];
    }
    return fabs(total - expected_sum) <= 5e-13 ? 0 : -1;
}

int ruip_validate_frozen_tables(void) {
    const double sqrt_two_pi = 2.506628274631000502415765284811;
    return validate_symmetric_rule(ruip_gl32_nodes, ruip_gl32_weights, 32u, 2.0)
        || validate_symmetric_rule(ruip_gl80_nodes, ruip_gl80_weights, 80u, 2.0)
        || validate_symmetric_rule(ruip_gl96_nodes, ruip_gl96_weights, 96u, 2.0)
        || validate_symmetric_rule(ruip_gl128_nodes, ruip_gl128_weights, 128u, 2.0)
        || validate_symmetric_rule(ruip_gl256_nodes, ruip_gl256_weights, 256u, 2.0)
        || validate_symmetric_rule(ruip_ghn12_nodes, ruip_ghn12_weights, 12u, sqrt_two_pi)
        || validate_symmetric_rule(ruip_ghn16_nodes, ruip_ghn16_weights, 16u, sqrt_two_pi)
        ? -1 : 0;
}

uint64_t ruip_phase_root(ruip_outcome_t outcome, ruip_phase_t phase) {
    if ((unsigned)outcome > 1u || (unsigned)phase > 2u) return UINT64_MAX;
    return ruip_protocol_roots[(unsigned)outcome][(unsigned)phase];
}

uint64_t ruip_shift_root(ruip_outcome_t outcome, ruip_phase_t phase) {
    if ((unsigned)outcome > 1u || (unsigned)phase > 2u) return UINT64_MAX;
    return ruip_protocol_shift_roots[(unsigned)outcome][(unsigned)phase];
}

int ruip_fixed_shifts(ruip_outcome_t outcome, uint32_t scenario_index,
                      double shifts[RUIP_HISTORY_COUNT]) {
    if ((unsigned)outcome > 1u || shifts == NULL) return -1;
    if (scenario_index == RUIP_RANDOM_HETEROGENEITY_SCENARIO) return 1;
    if (scenario_index >= RUIP_RANDOM_HETEROGENEITY_SCENARIO) return -1;
    for (size_t i = 0; i < RUIP_HISTORY_COUNT; ++i)
        shifts[i] = ruip_protocol_fixed_shifts[(unsigned)outcome][scenario_index][i];
    return 0;
}

static void random_shifts(ruip_outcome_t outcome, ruip_phase_t phase,
                          uint32_t scenario_index, uint32_t replicate_id,
                          double shifts[RUIP_HISTORY_COUNT]) {
    ruip_rng_t rng;
    ruip_rng_init(&rng, ruip_shift_root(outcome, phase), scenario_index, replicate_id);
    for (size_t i = 0; i < RUIP_HISTORY_COUNT; ++i)
        shifts[i] = RUIP_RANDOM_HETEROGENEITY_SD * ruip_rng_normal(&rng);
}

static ruip_continuous_summary_t continuous_group(ruip_rng_t *rng, uint32_t n,
                                                   double mean, double variance) {
    const double sd = sqrt(variance);
    double sample_mean = 0.0, sum_squares = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        const double value = mean + sd * ruip_rng_normal(rng);
        const double difference = value - sample_mean;
        sample_mean += difference / (double)(i + 1u);
        sum_squares += difference * (value - sample_mean);
    }
    (void)sum_squares;
    return (ruip_continuous_summary_t){n, sample_mean, variance};
}

static ruip_history_t continuous_history(ruip_rng_t *rng, uint32_t n,
                                         double mean, double variance) {
    const double sd = sqrt(variance);
    double sample_mean = 0.0, sum_squares = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        const double value = mean + sd * ruip_rng_normal(rng);
        const double difference = value - sample_mean;
        sample_mean += difference / (double)(i + 1u);
        sum_squares += difference * (value - sample_mean);
    }
    const double sample_variance = sum_squares / (double)(n - 1u);
    return (ruip_history_t){n, sample_mean, 1.0 / sample_variance};
}

static ruip_history_t binary_history(ruip_rng_t *rng, uint32_t n,
                                     double probability) {
    const uint32_t events = ruip_rng_binomial_bernoulli_sum(rng, n, probability);
    const double corrected = ((double)events + 0.5) / ((double)n + 1.0);
    return (ruip_history_t){n, ruip_logit(corrected), corrected * (1.0 - corrected)};
}

int ruip_generate_dataset(ruip_outcome_t outcome, ruip_phase_t phase,
                          uint32_t scenario_index, uint32_t replicate_id,
                          ruip_dataset_t *output) {
    if (output == NULL || (unsigned)outcome > 1u || (unsigned)phase > 2u
        || scenario_index >= RUIP_SCENARIO_COUNT
        || replicate_id >= RUIP_REPLICATES_PER_PHASE) return -1;
    *output = (ruip_dataset_t){0};
    output->outcome = outcome; output->phase = phase;
    output->scenario_index = scenario_index; output->replicate_id = replicate_id;
    int shift_status = ruip_fixed_shifts(outcome, scenario_index, output->historical_shifts);
    if (shift_status < 0) return -1;
    if (shift_status > 0)
        random_shifts(outcome, phase, scenario_index, replicate_id, output->historical_shifts);

    ruip_rng_t rng;
    ruip_rng_init(&rng, ruip_phase_root(outcome, phase), scenario_index, replicate_id);
    if (outcome == RUIP_CONTINUOUS) {
        const double delta = phase == RUIP_ALTERNATIVE ? 0.35 : 0.0;
        output->true_primary_effect = delta;
        output->true_risk_difference = NAN;
        /* Frozen draw order: current control, treatment, H1, H2, H3. */
        output->current.continuous.control = continuous_group(&rng, 100u, 0.0, 1.0);
        output->current.continuous.treatment = continuous_group(&rng, 100u, delta, 1.0);
        for (size_t i = 0; i < RUIP_HISTORY_COUNT; ++i)
            output->histories[i] = continuous_history(&rng, 150u,
                                                      output->historical_shifts[i], 1.0);
    } else {
        const double p_control = 0.40;
        const double p_treatment = phase == RUIP_ALTERNATIVE ? 0.58 : 0.40;
        const double baseline_logit = ruip_logit(p_control);
        output->true_primary_effect = ruip_logit(p_treatment) - baseline_logit;
        output->true_risk_difference = p_treatment - p_control;
        /* Bernoulli-sum draw order: C, T, H1, H2, H3. */
        output->current.binary.control = (ruip_binomial_t){100u,
            ruip_rng_binomial_bernoulli_sum(&rng, 100u, p_control)};
        output->current.binary.treatment = (ruip_binomial_t){100u,
            ruip_rng_binomial_bernoulli_sum(&rng, 100u, p_treatment)};
        for (size_t i = 0; i < RUIP_HISTORY_COUNT; ++i)
            output->histories[i] = binary_history(&rng, 150u,
                ruip_logistic(baseline_logit + output->historical_shifts[i]));
    }
    return 0;
}
