#ifndef RUIP_PROTOCOL_H
#define RUIP_PROTOCOL_H

#include <stdint.h>
#include "ruip/ruip.h"
#include "protocol_generated.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RUIP_COMPUTE_PROTOCOL "ruip-compute-c17-v1"
#define RUIP_RNG_PROTOCOL "ruip-rng-philox4x32-10-v1"
#define RUIP_REPLICATES_PER_PHASE 1000u
#define RUIP_CURRENT_CONTROL_N 100u
#define RUIP_CURRENT_TREATMENT_N 100u
#define RUIP_HISTORICAL_N 150u
#define RUIP_RANDOM_HETEROGENEITY_SCENARIO 11u
#define RUIP_RANDOM_HETEROGENEITY_SD 0.20
#define RUIP_CALIBRATION_QUANTILE 0.95
#define RUIP_CALIBRATION_QUANTILE_HIGHER 1
#define RUIP_REJECTION_STRICT_GREATER 1

/* Main power is always evaluated on the independent alternative phase using
 * each method's max-over-scenarios calibrated cutoff.  The literal 0.95 is
 * never a main-report decision cutoff. */

extern const char *const ruip_scenario_names[RUIP_SCENARIO_COUNT];

uint64_t ruip_phase_root(ruip_outcome_t outcome, ruip_phase_t phase);
uint64_t ruip_shift_root(ruip_outcome_t outcome, ruip_phase_t phase);
int ruip_fixed_shifts(ruip_outcome_t outcome, uint32_t scenario_index,
                      double shifts[RUIP_HISTORY_COUNT]);
/* Runner startup gate: verifies generated quadrature normalization/symmetry. */
int ruip_validate_frozen_tables(void);

#ifdef __cplusplus
}
#endif

#endif
