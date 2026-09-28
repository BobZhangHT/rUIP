#include "ruip/binary.h"

#include "ruip/mathutil.h"
#include "ruip/priors.h"
#include "quadrature_tables.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#define RUIP_LOG_2PI 1.83787706640934548356065947281123528
#define RUIP_GK_MAX_DEPTH 24
#define RUIP_GK_MAX_INTERVALS 65536u
#define RUIP_POSTERIOR_INITIAL_CAPACITY 512u
#define RUIP_BETA_CF_MAX_ITERATIONS 256

typedef struct {
  double *x;
  double *mass;
  size_t count;
  size_t capacity;
  double log_evidence;
  double integration_error;
} node_posterior_t;

typedef struct {
  const ruip_history_t *histories;
  size_t history_count;
  ruip_binomial_t control;
  double peak;
  double kappa[81];
  double log_weight[81];
  double source_log_norm[RUIP_HISTORY_COUNT][81];
  double source_inverse_two_variance[RUIP_HISTORY_COUNT][81];
} commensurate_context_t;

typedef struct {
  commensurate_context_t *kernel;
  node_posterior_t *posterior;
  int direction;
  double mode;
  size_t intervals;
  int failed;
} transformed_context_t;

static const double gh12_x[12] = {
    -5.500901704467748,  -4.2718258479322815, -3.2237098287700974,
    -2.2594644510007993, -1.3403751971516167, -0.44440300194413895,
    0.44440300194413895, 1.3403751971516167,  2.2594644510007993,
    3.2237098287700974,  4.2718258479322815,  5.500901704467748};
static const double gh12_w[12] = {
    3.7597598482865016e-7, 1.2125024496584412e-4, 5.5230563311466068e-3,
    7.2984713184738717e-2, 3.6839175806947677e-1, 8.0629298350918743e-1,
    8.0629298350918743e-1, 3.6839175806947677e-1, 7.2984713184738717e-2,
    5.5230563311466068e-3, 1.2125024496584412e-4, 3.7597598482865016e-7};
static const double gh16_x[16] = {
    -6.6308781983931286, -5.4722257059493433,  -4.4929553025200111,
    -3.6008736241715482, -2.7602450476307014,  -1.9519803457163336,
    -1.1638291005549648, -0.38676060450055733, 0.38676060450055733,
    1.1638291005549648,  1.9519803457163336,   2.7602450476307014,
    3.6008736241715482,  4.4929553025200111,   5.4722257059493433,
    6.6308781983931286};
static const double gh16_w[16] = {
    3.7544647352359996e-10, 3.282362588816467e-7,  3.8351493221254921e-5,
    1.3184486889798951e-3,  1.8215511261108784e-2, 1.1852529720949612e-1,
    3.9689544209659317e-1,  7.1832075795439576e-1, 7.1832075795439576e-1,
    3.9689544209659317e-1,  1.1852529720949612e-1, 1.8215511261108784e-2,
    1.3184486889798951e-3,  3.8351493221254921e-5, 3.282362588816467e-7,
    3.7544647352359996e-10};
static const double gh32_x[32] = {
    -10.077422674229467,-9.0643992107024065,-8.2197287653822464,-7.4607557541215197,
    -6.7559308305407049,-6.0889643090769878,-5.4500332736234292,-4.8326046132444889,
    -4.2320211099954097,-3.6447812498808334,-3.0681351690131216,-2.4998404151873954,
    -1.9380049059257174,-1.3809801992721442,-0.82728490377976516,-0.27554641923027584,
    0.27554641923027584,0.82728490377976516,1.3809801992721442,1.9380049059257174,
    2.4998404151873954,3.0681351690131216,3.6447812498808334,4.2320211099954097,
    4.8326046132444889,5.4500332736234292,6.0889643090769878,6.7559308305407049,
    7.4607557541215197,8.2197287653822464,9.0643992107024065,10.077422674229467};
static const double gh32_w[32] = {
    1.0338857753727874e-22,1.3055647014199345e-18,1.693300147798992e-15,
    5.9609246063989027e-13,8.3909412570480885e-11,5.7966240373100508e-09,
    2.2262094416688876e-07,5.1627070008246583e-06,7.6602066419597791e-05,
    0.00075839799560131044,0.0051687954689409212,0.02482429711976207,
    0.085500708753580387,0.21392770950775913,0.39238506783508209,
    0.53066716735927877,0.53066716735927877,0.39238506783508209,
    0.21392770950775913,0.085500708753580387,0.02482429711976207,
    0.0051687954689409212,0.00075839799560131044,7.6602066419597791e-05,
    5.1627070008246583e-06,2.2262094416688876e-07,5.7966240373100508e-09,
    8.3909412570480885e-11,5.9609246063989027e-13,1.693300147798992e-15,
    1.3055647014199345e-18,1.0338857753727874e-22};
static const double gh64_x[64] = {
-14.886186143339454,-13.994049908876471,-13.255649357397287,-12.596752480605721,-11.989036605128156,-11.417918056820673,-10.874651988398705,-10.353475921269004,-9.8503384578821507,-9.3622525462523072,-8.8869339058381804,-8.422584092328588,-7.9677529819412216,-7.5212476617873092,-7.0820698308048415,-6.6493714563417212,-6.2224225326264557,-5.8005871018292101,-5.3833050611646138,-4.9700781116024269,-4.5604587281440505,-4.1540413713409086,-3.7504553852256111,-3.3493591797524949,-2.9504354015399628,-2.5533868709840828,-2.1579331167626106,-1.7638073769428102,-1.3707539637008053,-0.97852590898502922,-0.58688282330529251,-0.19558891056727554,0.19558891056727554,0.58688282330529251,0.97852590898502922,1.3707539637008053,1.7638073769428102,2.1579331167626106,2.5533868709840828,2.9504354015399628,3.3493591797524949,3.7504553852256111,4.1540413713409086,4.5604587281440505,4.9700781116024269,5.3833050611646138,5.8005871018292101,6.2224225326264557,6.6493714563417212,7.0820698308048415,7.5212476617873092,7.9677529819412216,8.422584092328588,8.8869339058381804,9.3622525462523072,9.8503384578821507,10.353475921269004,10.874651988398705,11.417918056820673,11.989036605128156,12.596752480605721,13.255649357397287,13.994049908876471,14.886186143339454};
static const double gh64_w[64] = {
7.828671260325933e-49,2.3755223889798603e-43,4.838219774267789e-39,2.2024829432641773e-35,3.6057650229778912e-32,2.7281644679292966e-29,1.1118261057733848e-26,2.7035618016457601e-24,4.2184050042265271e-22,4.4579611600021508e-20,3.3260672549756856e-18,1.8103254351123549e-16,7.3802484510667485e-15,2.3028215455341994e-13,5.599122893863478e-12,1.0769543515573169e-10,1.6597447140192743e-09,2.0720000931366819e-08,2.1150029620953363e-07,1.7795618490882624e-06,1.2428814805174092e-05,7.2491585035923482e-05,0.00035494455037173315,0.001465590667605509,0.0051231116359196039,0.015211338366505487,0.03847103390529108,0.083070879147180435,0.15343983708390277,0.24280044671752321,0.32950438641205726,0.38378563487387862,0.38378563487387862,0.32950438641205726,0.24280044671752321,0.15343983708390277,0.083070879147180435,0.03847103390529108,0.015211338366505487,0.0051231116359196039,0.001465590667605509,0.00035494455037173315,7.2491585035923482e-05,1.2428814805174092e-05,1.7795618490882624e-06,2.1150029620953363e-07,2.0720000931366819e-08,1.6597447140192743e-09,1.0769543515573169e-10,5.599122893863478e-12,2.3028215455341994e-13,7.3802484510667485e-15,1.8103254351123549e-16,3.3260672549756856e-18,4.4579611600021508e-20,4.2184050042265271e-22,2.7035618016457601e-24,1.1118261057733848e-26,2.7281644679292966e-29,3.6057650229778912e-32,2.2024829432641773e-35,4.838219774267789e-39,2.3755223889798603e-43,7.828671260325933e-49};

/* Kronrod 15 / embedded Gauss 7 rule on [-1,1]. */
static const double gk_x[8] = {
    0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
    0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
    0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
    0.207784955007898467600689403773245, 0.0};
static const double gk_w[8] = {
    0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
    0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
    0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
    0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
static const double g7_w[4] = {
    0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
    0.381830050505118944950369775488975, 0.417959183673469387755102040816327};

static double nan_value(void) { return NAN; }

static int valid_binomial(ruip_binomial_t data) {
  return data.n > 0u && data.events <= data.n;
}

static void initialize_result(ruip_binary_method_t method,
                              ruip_binary_result_t *result) {
  memset(result, 0, sizeof(*result));
  result->method = method;
  result->status = RUIP_BINARY_STATUS_NUMERICAL_FAILURE;
  result->effect_available = 0;
  result->eta_c_mean = nan_value();
  result->eta_c_variance = nan_value();
  result->log_or_mean = nan_value();
  result->log_or_variance = nan_value();
  result->probability_log_or_positive = nan_value();
  result->log_or_q025 = nan_value();
  result->log_or_q975 = nan_value();
  result->p_c_mean = nan_value();
  result->p_t_mean = nan_value();
  result->risk_difference_mean = nan_value();
  result->risk_difference_q025 = nan_value();
  result->risk_difference_q975 = nan_value();
  result->quadrature_evidence_relative_difference = nan_value();
  result->integration_error = nan_value();
}

int ruip_binary_history(uint32_t events, uint32_t n, ruip_history_t *output) {
  double p;
  if (output == NULL || n == 0u || events > n)
    return -1;
  p = ((double)events + 0.5) / ((double)n + 1.0);
  output->n = n;
  output->estimate = log(p / (1.0 - p));
  output->unit_info = p * (1.0 - p);
  return 0;
}

double ruip_binary_log_likelihood(double eta, ruip_binomial_t data) {
  double constant;
  if (!valid_binomial(data) || !isfinite(eta))
    return NAN;
  constant = lgamma((double)data.n + 1.0) - lgamma((double)data.events + 1.0) -
             lgamma((double)(data.n - data.events) + 1.0);
  return constant + (double)data.events * eta -
         (double)data.n * ruip_log1pexp(eta);
}

static double logadd(double a, double b) {
  double m;
  if (a == -INFINITY)
    return b;
  if (b == -INFINITY)
    return a;
  m = a > b ? a : b;
  return m + log(exp(a - m) + exp(b - m));
}

/* Regularized I_x(a,b), for positive integer a,b, as a stable finite sum. */
typedef struct {
  uint32_t a, b, n;
  double mean;
  double log_combination;
  double log_beta_inverse;
  double switch_point;
  double direct_initial;
  double complement_initial;
  double direct_first[RUIP_BETA_CF_MAX_ITERATIONS];
  double direct_second[RUIP_BETA_CF_MAX_ITERATIONS];
  double complement_first[RUIP_BETA_CF_MAX_ITERATIONS];
  double complement_second[RUIP_BETA_CF_MAX_ITERATIONS];
} integer_beta_context_t;

static int initialize_integer_beta(uint32_t a, uint32_t b,
                                   integer_beta_context_t *context) {
  int m;
  if (context == NULL || a == 0u || b == 0u)
    return -1;
  context->a = a;
  context->b = b;
  context->n = a + b - 1u;
  context->mean = (double)a / (double)(a + b);
  context->log_combination =
      lgamma((double)context->n + 1.0) - lgamma((double)a + 1.0) -
      lgamma((double)(context->n - a) + 1.0);
  context->log_beta_inverse =
      lgamma((double)a + (double)b) - lgamma((double)a) - lgamma((double)b);
  context->switch_point = ((double)a + 1.0) / ((double)a + (double)b + 2.0);
  context->direct_initial = ((double)a + (double)b) / ((double)a + 1.0);
  context->complement_initial =
      ((double)a + (double)b) / ((double)b + 1.0);
  for (m = 1; m <= RUIP_BETA_CF_MAX_ITERATIONS; ++m) {
    const double dm = (double)m, m2 = 2.0 * dm;
    context->direct_first[m - 1] =
        dm * ((double)b - dm) /
        (((double)a - 1.0 + m2) * ((double)a + m2));
    context->direct_second[m - 1] =
        -((double)a + dm) * ((double)a + (double)b + dm) /
        (((double)a + m2) * ((double)a + 1.0 + m2));
    context->complement_first[m - 1] =
        dm * ((double)a - dm) /
        (((double)b - 1.0 + m2) * ((double)b + m2));
    context->complement_second[m - 1] =
        -((double)b + dm) * ((double)a + (double)b + dm) /
        (((double)b + m2) * ((double)b + 1.0 + m2));
  }
  return 0;
}

static double integer_beta_cdf_reference(const integer_beta_context_t *context,
                                         double x) {
  uint32_t k, n;
  double term, total, correction = 0.0;
  uint32_t a;
  if (context == NULL || isnan(x))
    return NAN;
  a = context->a;
  if (x <= 0.0)
    return 0.0;
  if (x >= 1.0)
    return 1.0;
  n = context->n;
  if (x > context->mean) {
    /* I_x=1-P(Binomial(n,x)<=a-1); sum the small complementary tail. */
    term = exp((double)n * log1p(-x));
    total = term;
    for (k = 0u; k + 1u < a; ++k) {
      double adjusted, next;
      term *= ((double)(n - k) / (double)(k + 1u)) * x / (1.0 - x);
      adjusted = term - correction;
      next = total + adjusted;
      correction = (next - total) - adjusted;
      total = next;
    }
    return fmin(1.0, fmax(0.0, 1.0 - total));
  }
  term = exp(context->log_combination + (double)a * log(x) +
             (double)(n - a) * log1p(-x));
  total = term;
  correction = 0.0;
  for (k = a; k < n; ++k) {
    double adjusted, next;
    term *= ((double)(n - k) / (double)(k + 1u)) * x / (1.0 - x);
    adjusted = term - correction;
    next = total + adjusted;
    correction = (next - total) - adjusted;
    total = next;
  }
  return fmin(1.0, fmax(0.0, total));
}

static double beta_continued_fraction(const double *first,
                                      const double *second,
                                      double initial, double x) {
  const double epsilon = 2.0 * DBL_EPSILON;
  const double minimum = DBL_MIN / epsilon;
  double c = 1.0;
  double d = 1.0 - initial * x;
  double h;
  int m;
  if (fabs(d) < minimum)
    d = minimum;
  d = 1.0 / d;
  h = d;
  for (m = 0; m < RUIP_BETA_CF_MAX_ITERATIONS; ++m) {
    double aa = first[m] * x;
    double delta;
    d = 1.0 + aa * d;
    if (fabs(d) < minimum)
      d = minimum;
    c = 1.0 + aa / c;
    if (fabs(c) < minimum)
      c = minimum;
    d = 1.0 / d;
    h *= d * c;
    aa = second[m] * x;
    d = 1.0 + aa * d;
    if (fabs(d) < minimum)
      d = minimum;
    c = 1.0 + aa / c;
    if (fabs(c) < minimum)
      c = minimum;
    d = 1.0 / d;
    delta = d * c;
    h *= delta;
    if (fabs(delta - 1.0) <= epsilon)
      return h;
  }
  return NAN;
}

static double integer_beta_cdf_lentz_context(
    const integer_beta_context_t *context, double x) {
  double front, fraction, result;
  if (context == NULL || isnan(x))
    return NAN;
  if (x <= 0.0)
    return 0.0;
  if (x >= 1.0)
    return 1.0;
  /* The frozen finite-sum oracle intentionally defines the extreme-tail
   * rounding contract.  This rare branch has negligible production cost. */
  if (x < 1e-3 || x > 1.0 - 1e-3)
    return integer_beta_cdf_reference(context, x);
  /* Preserve the frozen finite-tail endpoint behavior when its initial term
   * underflows exactly to zero. */
  if (x > context->mean &&
      (double)context->n * log1p(-x) < log(DBL_TRUE_MIN))
    return 1.0;
  if (x <= context->mean &&
      context->log_combination + (double)context->a * log(x) +
              (double)(context->n - context->a) * log1p(-x) <
          log(DBL_TRUE_MIN))
    return 0.0;
  front = exp(context->log_beta_inverse + (double)context->a * log(x) +
              (double)context->b * log1p(-x));
  if (x < context->switch_point) {
    fraction = beta_continued_fraction(
        context->direct_first, context->direct_second,
        context->direct_initial, x);
    result = front * fraction / (double)context->a;
  } else {
    fraction = beta_continued_fraction(
        context->complement_first, context->complement_second,
        context->complement_initial, 1.0 - x);
    result = 1.0 - front * fraction / (double)context->b;
  }
  return isfinite(result) ? fmin(1.0, fmax(0.0, result)) : NAN;
}

double ruip_binary_beta_cdf_finite_sum(uint32_t a, uint32_t b, double x) {
  integer_beta_context_t context;
  return initialize_integer_beta(a, b, &context) == 0
             ? integer_beta_cdf_reference(&context, x)
             : NAN;
}

double ruip_binary_beta_cdf_lentz(uint32_t a, uint32_t b, double x) {
  integer_beta_context_t context;
  return initialize_integer_beta(a, b, &context) == 0
             ? integer_beta_cdf_lentz_context(&context, x)
             : NAN;
}

static int posterior_reserve(node_posterior_t *posterior, size_t additional) {
  size_t need, capacity;
  double *new_x, *new_mass;
  if (additional > SIZE_MAX - posterior->count)
    return -1;
  need = posterior->count + additional;
  if (need <= posterior->capacity)
    return 0;
  capacity = posterior->capacity ? posterior->capacity
                                 : RUIP_POSTERIOR_INITIAL_CAPACITY;
  while (capacity < need) {
    if (capacity > SIZE_MAX / 2u)
      return -1;
    capacity *= 2u;
  }
  new_x = (double *)realloc(posterior->x, capacity * sizeof(double));
  if (new_x == NULL)
    return -1;
  posterior->x = new_x;
  new_mass = (double *)realloc(posterior->mass, capacity * sizeof(double));
  if (new_mass == NULL)
    return -1;
  posterior->mass = new_mass;
  posterior->capacity = capacity;
  return 0;
}

static void posterior_free(node_posterior_t *posterior) {
  free(posterior->x);
  free(posterior->mass);
  memset(posterior, 0, sizeof(*posterior));
}

static int posterior_normalize(node_posterior_t *posterior) {
  size_t i;
  double sum = 0.0;
  for (i = 0u; i < posterior->count; ++i)
    sum += posterior->mass[i];
  if (!(sum > 0.0) || !isfinite(sum))
    return -1;
  for (i = 0u; i < posterior->count; ++i)
    posterior->mass[i] /= sum;
  return 0;
}

static double harmonic(uint32_t n, int power) {
  uint32_t k;
  double total = 0.0;
  for (k = 1u; k <= n; ++k) {
    double d = (double)k;
    total += power == 1 ? 1.0 / d : 1.0 / (d * d);
  }
  return total;
}

typedef struct {
  double events;
  double trials;
  double log_binomial_coefficient;
  double adjusted_gh12[12];
  double adjusted_gh16[16];
  double adjusted_gh32[32];
  double adjusted_gh64[64];
} aghq_control_context_t;

static void initialize_aghq_control(ruip_binomial_t control,
                                    aghq_control_context_t *context) {
  size_t j;
  context->events = (double)control.events;
  context->trials = (double)control.n;
  context->log_binomial_coefficient =
      lgamma((double)control.n + 1.0) -
      lgamma((double)control.events + 1.0) -
      lgamma((double)(control.n - control.events) + 1.0);
  for (j = 0u; j < 12u; ++j)
    context->adjusted_gh12[j] = log(gh12_w[j]) + 0.5 * gh12_x[j] * gh12_x[j];
  for (j = 0u; j < 16u; ++j)
    context->adjusted_gh16[j] = log(gh16_w[j]) + 0.5 * gh16_x[j] * gh16_x[j];
  for (j = 0u; j < 32u; ++j)
    context->adjusted_gh32[j] = log(gh32_w[j]) + 0.5 * gh32_x[j] * gh32_x[j];
  for (j = 0u; j < 64u; ++j)
    context->adjusted_gh64[j] = log(gh64_w[j]) + 0.5 * gh64_x[j] * gh64_x[j];
}

static int mixture_component_mode(const ruip_normal_component_t *part,
                                  const aghq_control_context_t *control,
                                  double *mode, double *scale) {
  size_t j;
  double theta = part->mean;
  for (j = 0u; j < 60u; ++j) {
    double p = ruip_logistic(theta);
    double score = control->events - control->trials * p -
                   (theta - part->mean) / part->variance;
    double curvature = control->trials * p * (1.0 - p) +
                       1.0 / part->variance;
    double step = score / curvature;
    theta += step;
    if (fabs(step) < 2e-12)
      break;
  }
  {
    const double p = ruip_logistic(theta);
    *scale = 1.0 / sqrt(control->trials * p * (1.0 - p) +
                        1.0 / part->variance);
  }
  *mode = theta;
  return *scale > 0.0 && isfinite(*scale) ? 0 : -1;
}

static double mixture_component_log_evidence(
    const ruip_normal_component_t *part,
    const aghq_control_context_t *control, const double *roots,
    const double *adjusted_weights, size_t order, int keep,
    double mode, double scale, node_posterior_t *posterior) {
  size_t j;
  double component_log_z = -INFINITY;
  const double normal_log_norm = -0.5 * (RUIP_LOG_2PI + log(part->variance));
  const double inverse_two_variance = 0.5 / part->variance;
  const double log_scale = log(scale);
  if (!(scale > 0.0) || !isfinite(scale))
    return NAN;
  for (j = 0u; j < order; ++j) {
    double x = mode + scale * roots[j];
    double difference = x - part->mean;
    double log_term = adjusted_weights[j] + control->events * x -
                      control->trials * ruip_log1pexp(x) + normal_log_norm -
                      difference * difference * inverse_two_variance + log_scale;
    component_log_z = logadd(component_log_z, log_term);
    if (keep) {
      posterior->x[posterior->count] = x;
      posterior->mass[posterior->count] = part->log_weight + log_term;
      posterior->count++;
    }
  }
  return component_log_z;
}

static int aghq_posterior(const ruip_normal_mixture_t *mixture,
                          ruip_binomial_t control, node_posterior_t *posterior,
                          double *relative_difference,
                          ruip_binary_result_t *result, int high_order) {
  size_t i, base;
  double log_z16 = -INFINITY, log_z12 = -INFINITY, max_log_mass = -INFINITY;
  double *component_log_mass;
  aghq_control_context_t control_context;
  if (mixture == NULL || mixture->items == NULL || mixture->count == 0u)
    return -1;
  if (posterior_reserve(posterior, mixture->count * (high_order?64u:16u)) != 0)
    return -1;
  component_log_mass = (double *)malloc(mixture->count * sizeof(double));
  if (component_log_mass == NULL)
    return -1;
  base = posterior->count;
  initialize_aghq_control(control, &control_context);
  for (i = 0u; i < mixture->count; ++i) {
    double mode, scale;
    if (mixture_component_mode(&mixture->items[i], &control_context,
                               &mode, &scale) != 0) {
      free(component_log_mass);
      return -1;
    }
    double z16 = mixture_component_log_evidence(
        &mixture->items[i], &control_context, high_order?gh64_x:gh16_x,
        high_order?control_context.adjusted_gh64:control_context.adjusted_gh16,
        high_order?64u:16u, 1, mode, scale, posterior);
    double z12 = mixture_component_log_evidence(
        &mixture->items[i], &control_context, high_order?gh32_x:gh12_x,
        high_order?control_context.adjusted_gh32:control_context.adjusted_gh12,
        high_order?32u:12u, 0, mode, scale, posterior);
    if (!isfinite(z16) || !isfinite(z12)) {
      free(component_log_mass);
      return -1;
    }
    component_log_mass[i] = mixture->items[i].log_weight + z16;
    log_z16 = logadd(log_z16, mixture->items[i].log_weight + z16);
    log_z12 = logadd(log_z12, mixture->items[i].log_weight + z12);
  }
  for (i = base; i < posterior->count; ++i)
    if (posterior->mass[i] > max_log_mass)
      max_log_mass = posterior->mass[i];
  for (i = base; i < posterior->count; ++i)
    posterior->mass[i] = exp(posterior->mass[i] - max_log_mass);
  if (posterior_normalize(posterior) != 0)
    return -1;
  posterior->log_evidence = log_z16 + control_context.log_binomial_coefficient;
  posterior->integration_error = fabs(exp(fmin(0.0, log_z12 - log_z16)) -
                                      exp(fmin(0.0, log_z16 - log_z12)));
  *relative_difference = posterior->integration_error;
  for (i = 0u; i < mixture->count; ++i) {
    size_t source;
    double probability = exp(component_log_mass[i] - log_z16);
    const ruip_normal_component_t *part = &mixture->items[i];
    result->diagnostic_tau += probability * part->tau;
    result->diagnostic_robust_component += probability * part->robust_component;
    result->diagnostic_j_g += probability * part->j_g;
    result->diagnostic_m_eff += probability * part->m_eff;
    result->diagnostic_borrowing_m += probability * part->borrowing_m;
    for (source = 0u; source < RUIP_HISTORY_COUNT; ++source) {
      result->diagnostic_retention[source] +=
          probability * (double)((part->retention_mask >> source) & 1u);
      result->diagnostic_source_weight[source] +=
          probability * part->source_weight[source];
    }
  }
  free(component_log_mass);
  return 0;
}

static int nip_posterior(ruip_binomial_t control, node_posterior_t *posterior) {
  size_t j;
  double a = (double)control.events;
  double b = (double)(control.n - control.events);
  double mode, scale, max_log = -INFINITY;
  if (!(a > 0.0) || !(b > 0.0))
    return -1;
  if (posterior_reserve(posterior, 16u) != 0)
    return -1;
  mode = log(a / b);
  scale = 1.0 / sqrt((double)control.n * ruip_logistic(mode) *
                     (1.0 - ruip_logistic(mode)));
  for (j = 0u; j < 16u; ++j) {
    double x = mode + scale * gh16_x[j];
    double lm = log(gh16_w[j]) + ruip_binary_log_likelihood(x, control) +
                0.5 * gh16_x[j] * gh16_x[j] + log(scale);
    posterior->x[j] = x;
    posterior->mass[j] = lm;
    if (lm > max_log)
      max_log = lm;
  }
  posterior->count = 16u;
  for (j = 0u; j < 16u; ++j)
    posterior->mass[j] = exp(posterior->mass[j] - max_log);
  posterior->integration_error = 0.0;
  return posterior_normalize(posterior);
}

static double beta_logit_aghq(double a, double b, const double *roots,
                              const double *weights, size_t order, int keep,
                              node_posterior_t *posterior) {
  const double mode = log(a / b);
  const double p_mode = a / (a + b);
  const double scale = 1.0 / sqrt((a + b) * p_mode * (1.0 - p_mode));
  double maximum = -INFINITY, sum = 0.0;
  size_t j, base = posterior->count;
  if (!(a > 0.0) || !(b > 0.0) || !isfinite(scale)) return NAN;
  if (keep && posterior_reserve(posterior, order) != 0) return NAN;
  for (j = 0u; j < order; ++j) {
    const double x = mode + scale * roots[j];
    const double log_mass = log(weights[j]) + 0.5 * roots[j] * roots[j] +
                            a * x - (a + b) * ruip_log1pexp(x) + log(scale);
    if (keep) {
      posterior->x[posterior->count] = x;
      posterior->mass[posterior->count++] = log_mass;
    }
    if (log_mass > maximum) maximum = log_mass;
  }
  for (j = 0u; j < order; ++j) {
    const double x = mode + scale * roots[j];
    const double log_mass = log(weights[j]) + 0.5 * roots[j] * roots[j] +
                            a * x - (a + b) * ruip_log1pexp(x) + log(scale);
    sum += exp(log_mass - maximum);
  }
  if (keep)
    for (j = base; j < posterior->count; ++j)
      posterior->mass[j] = exp(posterior->mass[j] - maximum);
  return maximum + log(sum);
}

static int exact_power_prior_posterior(const ruip_binomial_t *historical,
                                       size_t history_count,
                                       ruip_binomial_t control,
                                       node_posterior_t *posterior,
                                       double *relative_difference) {
  double a = (double)control.events;
  double b = (double)(control.n - control.events);
  double log_z64, log_z32;
  size_t i;
  if (historical == NULL || history_count == 0u ||
      history_count > RUIP_HISTORY_COUNT) return -1;
  for (i = 0u; i < history_count; ++i) {
    if (!valid_binomial(historical[i])) return -1;
    a += RUIP_POWER_PRIOR_A0 * (double)historical[i].events;
    b += RUIP_POWER_PRIOR_A0 *
         (double)(historical[i].n - historical[i].events);
  }
  if (!(a > 0.0) || !(b > 0.0)) return -1;
  log_z64 = beta_logit_aghq(a, b, gh64_x, gh64_w, 64u, 1, posterior);
  log_z32 = beta_logit_aghq(a, b, gh32_x, gh32_w, 32u, 0, posterior);
  if (!isfinite(log_z64) || !isfinite(log_z32) ||
      posterior_normalize(posterior) != 0) return -1;
  posterior->log_evidence = log_z64;
  posterior->integration_error =
      fabs(exp(fmin(0.0, log_z32 - log_z64)) -
           exp(fmin(0.0, log_z64 - log_z32)));
  *relative_difference = posterior->integration_error;
  return 0;
}

static double commensurate_log_prior(double eta,
                                     const commensurate_context_t *context) {
  size_t source, node;
  double total = 0.0;
  for (source = 0u; source < context->history_count; ++source) {
    double source_sum = -INFINITY;
    double difference = eta - context->histories[source].estimate;
    for (node = 0u; node < 81u; ++node) {
      double value = context->log_weight[node] +
                     context->source_log_norm[source][node] -
                     difference * difference *
                         context->source_inverse_two_variance[source][node];
      source_sum = logadd(source_sum, value);
    }
    total += source_sum;
  }
  return total;
}

static double commensurate_log_kernel(double eta,
                                      commensurate_context_t *context) {
  return commensurate_log_prior(eta, context) +
         ruip_binary_log_likelihood(eta, context->control);
}

static double find_commensurate_peak(commensurate_context_t *context,
                                     double *mode) {
  int i, iteration;
  double best_x = 0.0, best = -INFINITY;
  double candidates[RUIP_HISTORY_COUNT + 1u];
  size_t count = context->history_count + 1u;
  candidates[0] =
      log(((double)context->control.events + 0.5) /
          ((double)(context->control.n - context->control.events) + 0.5));
  for (i = 0; i < (int)context->history_count; ++i)
    candidates[i + 1] = context->histories[i].estimate;
  for (i = 0; i < (int)count; ++i) {
    double left = candidates[i] - 12.0, right = candidates[i] + 12.0;
    const double phi = 0.618033988749894848204586834365638;
    double c = right - phi * (right - left);
    double d = left + phi * (right - left);
    double fc = commensurate_log_kernel(c, context);
    double fd = commensurate_log_kernel(d, context);
    for (iteration = 0; iteration < 160; ++iteration) {
      if (fc > fd) {
        right = d;
        d = c;
        fd = fc;
        c = right - phi * (right - left);
        fc = commensurate_log_kernel(c, context);
      } else {
        left = c;
        c = d;
        fc = fd;
        d = left + phi * (right - left);
        fd = commensurate_log_kernel(d, context);
      }
      if (right - left < 1e-11)
        break;
    }
    if (fc > best) {
      best = fc;
      best_x = c;
    }
    if (fd > best) {
      best = fd;
      best_x = d;
    }
  }
  *mode = best_x;
  return best;
}

static double transformed_density(double t, transformed_context_t *context,
                                  double *eta_out) {
  double eta, one_minus_t, log_value;
  if (t <= 0.0) {
    eta = context->mode;
    *eta_out = eta;
    return exp(commensurate_log_kernel(eta, context->kernel) -
               context->kernel->peak);
  }
  if (t >= 1.0) {
    *eta_out = context->direction > 0 ? INFINITY : -INFINITY;
    return 0.0;
  }
  one_minus_t = 1.0 - t;
  eta = context->mode + (double)context->direction * t / one_minus_t;
  *eta_out = eta;
  log_value = commensurate_log_kernel(eta, context->kernel) -
              context->kernel->peak - 2.0 * log(one_minus_t);
  return log_value < log(DBL_MIN) ? 0.0 : exp(log_value);
}

static int gk_integrate_interval(transformed_context_t *context, double lower,
                                 double upper, double abs_tolerance,
                                 double rel_tolerance, int depth,
                                 double *integral, double *error) {
  double midpoint = 0.5 * (lower + upper), half = 0.5 * (upper - lower);
  double values_left[7], values_right[7], eta_left[7], eta_right[7];
  double center_eta, center_value, kronrod, gauss;
  int i;
  if (++context->intervals > RUIP_GK_MAX_INTERVALS)
    return -1;
  center_value = transformed_density(midpoint, context, &center_eta);
  kronrod = gk_w[7] * center_value;
  gauss = g7_w[3] * center_value;
  for (i = 0; i < 7; ++i) {
    double offset = half * gk_x[i];
    values_left[i] =
        transformed_density(midpoint - offset, context, &eta_left[i]);
    values_right[i] =
        transformed_density(midpoint + offset, context, &eta_right[i]);
    kronrod += gk_w[i] * (values_left[i] + values_right[i]);
  }
  gauss += g7_w[0] * (values_left[1] + values_right[1]);
  gauss += g7_w[1] * (values_left[3] + values_right[3]);
  gauss += g7_w[2] * (values_left[5] + values_right[5]);
  kronrod *= half;
  gauss *= half;
  *error = fabs(kronrod - gauss);
  if (*error <= fmax(abs_tolerance, rel_tolerance * fabs(kronrod))) {
    if (posterior_reserve(context->posterior, 15u) != 0)
      return -1;
    for (i = 0; i < 7; ++i) {
      context->posterior->x[context->posterior->count] = eta_left[i];
      context->posterior->mass[context->posterior->count++] =
          half * gk_w[i] * values_left[i];
      context->posterior->x[context->posterior->count] = eta_right[i];
      context->posterior->mass[context->posterior->count++] =
          half * gk_w[i] * values_right[i];
    }
    context->posterior->x[context->posterior->count] = center_eta;
    context->posterior->mass[context->posterior->count++] =
        half * gk_w[7] * center_value;
    *integral = kronrod;
    return 0;
  }
  if (depth >= RUIP_GK_MAX_DEPTH)
    return -1;
  {
    double left_value, right_value, left_error, right_error;
    if (gk_integrate_interval(context, lower, midpoint, 0.5 * abs_tolerance,
                              rel_tolerance, depth + 1, &left_value,
                              &left_error) != 0)
      return -1;
    if (gk_integrate_interval(context, midpoint, upper, 0.5 * abs_tolerance,
                              rel_tolerance, depth + 1, &right_value,
                              &right_error) != 0)
      return -1;
    *integral = left_value + right_value;
    *error = left_error + right_error;
  }
  return 0;
}

static int initialize_commensurate(commensurate_context_t *context,
                                   const ruip_history_t *histories,
                                   size_t history_count,
                                   ruip_binomial_t control) {
  size_t source, i;
  double jsum = 0.0, nsum = 0.0, reference_sd_squared;
  if (history_count == 0u || history_count > RUIP_HISTORY_COUNT)
    return -1;
  memset(context, 0, sizeof(*context));
  context->histories = histories;
  context->history_count = history_count;
  context->control = control;
  for (source = 0u; source < history_count; ++source) {
    if (!(histories[source].n > 0u) || !(histories[source].unit_info > 0.0) ||
        !isfinite(histories[source].estimate))
      return -1;
    jsum += (double)histories[source].n * histories[source].unit_info;
    nsum += (double)histories[source].n;
  }
  reference_sd_squared = nsum / jsum;
  for (i = 0u; i < 80u; ++i) {
    double kappa_star =
        0.5 * (RUIP_COMMENSURATE_KAPPA_STAR_UPPER -
               RUIP_COMMENSURATE_KAPPA_STAR_LOWER) * ruip_gl80_nodes[i] +
        0.5 * (RUIP_COMMENSURATE_KAPPA_STAR_UPPER +
               RUIP_COMMENSURATE_KAPPA_STAR_LOWER);
    context->kappa[i] = kappa_star / reference_sd_squared;
    context->log_weight[i] = log(0.99 * 0.5 * ruip_gl80_weights[i]);
  }
  context->kappa[80] =
      RUIP_COMMENSURATE_KAPPA_STAR_SPIKE / reference_sd_squared;
  context->log_weight[80] = log(0.01);
  for (source = 0u; source < history_count; ++source) {
    for (i = 0u; i < 81u; ++i) {
      double variance =
          1.0 / ((double)histories[source].n * histories[source].unit_info) +
          1.0 / context->kappa[i];
      context->source_log_norm[source][i] =
          -0.5 * (RUIP_LOG_2PI + log(variance));
      context->source_inverse_two_variance[source][i] = 0.5 / variance;
    }
  }
  return 0;
}

static int commensurate_posterior(const ruip_history_t *histories,
                                  size_t history_count, ruip_binomial_t control,
                                  node_posterior_t *posterior) {
  commensurate_context_t kernel;
  transformed_context_t transformed;
  double mode, left, right, left_error, right_error;
  if (initialize_commensurate(&kernel, histories, history_count, control) != 0)
    return -1;
  kernel.peak = find_commensurate_peak(&kernel, &mode);
  memset(&transformed, 0, sizeof(transformed));
  transformed.kernel = &kernel;
  transformed.posterior = posterior;
  transformed.mode = mode;
  transformed.direction = -1;
  if (gk_integrate_interval(&transformed, 0.0, 1.0, 1e-10, 2e-10, 0, &left,
                            &left_error) != 0)
    return -1;
  transformed.direction = 1;
  if (gk_integrate_interval(&transformed, 0.0, 1.0, 1e-10, 2e-10, 0, &right,
                            &right_error) != 0)
    return -1;
  posterior->integration_error = left_error + right_error;
  posterior->log_evidence = kernel.peak + log(left + right);
  return posterior_normalize(posterior);
}

typedef struct {
  const node_posterior_t *posterior;
  const double *p_control;
  integer_beta_context_t beta;
  int risk_difference;
} effect_cdf_context_t;

static double effect_cdf(double value, void *opaque) {
  effect_cdf_context_t *context = (effect_cdf_context_t *)opaque;
  size_t i;
  double total = 0.0;
  double odds_scale = context->risk_difference ? 0.0 : exp(-fabs(value));
  for (i = 0u; i < context->posterior->count; ++i) {
    double p_control = context->p_control[i];
    double argument;
    if (context->risk_difference)
      argument = p_control + value;
    else if (value >= 0.0)
      argument = p_control /
                 (p_control + (1.0 - p_control) * odds_scale);
    else
      argument = p_control * odds_scale /
                 (1.0 - p_control + p_control * odds_scale);
    double cdf = argument <= 0.0 ? 0.0
                 : argument >= 1.0
                     ? 1.0
                     : integer_beta_cdf_lentz_context(&context->beta, argument);
    total += context->posterior->mass[i] * cdf;
  }
  return total;
}

typedef struct {
  effect_cdf_context_t *cdf;
  double probability;
} root_context_t;
static double cdf_root(double value, void *opaque) {
  root_context_t *context = (root_context_t *)opaque;
  return effect_cdf(value, context->cdf) - context->probability;
}

static int summarize_posterior(ruip_binary_method_t method,
                               node_posterior_t *posterior,
                               ruip_binomial_t treatment,
                               int statistic_only,
                               ruip_binary_result_t *result) {
  size_t i;
  double eta_mean = 0.0, eta_variance = 0.0;
  double *p_control;
  effect_cdf_context_t cdf_context;
  root_context_t root_context;
  int ok;
  p_control = (double *)malloc(posterior->count * sizeof(*p_control));
  if (p_control == NULL)
    return -1;
  result->p_c_mean = 0.0;
  for (i = 0u; i < posterior->count; ++i) {
    p_control[i] = ruip_logistic(posterior->x[i]);
    eta_mean += posterior->mass[i] * posterior->x[i];
    result->p_c_mean += posterior->mass[i] * p_control[i];
  }
  for (i = 0u; i < posterior->count; ++i) {
    double d = posterior->x[i] - eta_mean;
    eta_variance += posterior->mass[i] * d * d;
  }
  result->eta_c_mean = eta_mean;
  result->eta_c_variance = eta_variance;
  result->posterior_node_count = posterior->count;
  result->integration_error = posterior->integration_error;
  if (treatment.events == 0u || treatment.events == treatment.n) {
    result->status = RUIP_BINARY_STATUS_TREATMENT_ENDPOINT;
    result->effect_available = 0;
    free(p_control);
    return 0;
  }
  cdf_context.posterior = posterior;
  cdf_context.p_control = p_control;
  if (initialize_integer_beta(treatment.events,
                              treatment.n - treatment.events,
                              &cdf_context.beta) != 0)
    goto failure;
  cdf_context.risk_difference = 0;
  root_context.cdf = &cdf_context;
  result->effect_available = 1;
  result->p_t_mean =
      (double)cdf_context.beta.a /
      (double)(cdf_context.beta.a + cdf_context.beta.b);
  result->probability_log_or_positive =
      fmin(1.0, fmax(0.0, 1.0 - effect_cdf(0.0, &cdf_context)));
  if (statistic_only) {
    result->status = result->quadrature_evidence_relative_difference > 5e-5
                         ? RUIP_BINARY_STATUS_QUADRATURE_WARNING
                         : RUIP_BINARY_STATUS_OK;
    free(p_control);
    return 0;
  }
  result->log_or_mean = harmonic(cdf_context.beta.a - 1u, 1) -
                        harmonic(cdf_context.beta.b - 1u, 1) - eta_mean;
  result->log_or_variance = (M_PI * M_PI / 3.0) -
                            harmonic(cdf_context.beta.a - 1u, 2) -
                            harmonic(cdf_context.beta.b - 1u, 2) + eta_variance;
  root_context.probability = 0.025;
  result->log_or_q025 =
      ruip_brent(cdf_root, &root_context, -40.0, 40.0, 1e-12, 200, &ok);
  if (!ok)
    goto failure;
  root_context.probability = 0.975;
  result->log_or_q975 =
      ruip_brent(cdf_root, &root_context, -40.0, 40.0, 1e-12, 200, &ok);
  if (!ok)
    goto failure;
  result->risk_difference_mean = result->p_t_mean - result->p_c_mean;
  cdf_context.risk_difference = 1;
  root_context.probability = 0.025;
  result->risk_difference_q025 =
      ruip_brent(cdf_root, &root_context, -1.0, 1.0, 1e-12, 200, &ok);
  if (!ok)
    goto failure;
  root_context.probability = 0.975;
  result->risk_difference_q975 =
      ruip_brent(cdf_root, &root_context, -1.0, 1.0, 1e-12, 200, &ok);
  if (!ok)
    goto failure;
  result->status = result->quadrature_evidence_relative_difference > 5e-5
                       ? RUIP_BINARY_STATUS_QUADRATURE_WARNING
                       : RUIP_BINARY_STATUS_OK;
  (void)method;
  free(p_control);
  return 0;
failure:
  free(p_control);
  return -1;
}

static int binary_analyze_impl(ruip_binary_method_t method,
                               const ruip_history_t *histories,
                               size_t history_count,
                               ruip_binomial_t control,
                               ruip_binomial_t treatment,
                               double planned_n_control,
                               int statistic_only,
                               ruip_binary_result_t *output) {
  node_posterior_t posterior;
  ruip_normal_mixture_t mixture;
  int status = -1;
  if (output == NULL)
    return -1;
  initialize_result(method, output);
  memset(&posterior, 0, sizeof(posterior));
  memset(&mixture, 0, sizeof(mixture));
  if (method < RUIP_BINARY_NIP || method > RUIP_BINARY_PROMOTED_RUIP ||
      !valid_binomial(control) || !valid_binomial(treatment)) {
    output->status = RUIP_BINARY_STATUS_INVALID_ARGUMENT;
    return -1;
  }
  if (method == RUIP_BINARY_NIP) {
    if (control.events == 0u || control.events == control.n) {
      output->status = RUIP_BINARY_STATUS_IMPROPER_FLAT_LOGIT;
      return 0;
    }
    status = nip_posterior(control, &posterior);
    output->quadrature_evidence_relative_difference = 0.0;
  } else if (histories == NULL || history_count == 0u ||
             history_count > RUIP_HISTORY_COUNT) {
    output->status = RUIP_BINARY_STATUS_INVALID_ARGUMENT;
    return -1;
  } else {
    if (method == RUIP_BINARY_RMAP)
      status = ruip_rmap_mixture(histories, history_count, &mixture);
    else if (method == RUIP_BINARY_COMMENSURATE)
      status = ruip_commensurate_mixture(histories, history_count, &mixture);
    else if (method == RUIP_BINARY_RUIP)
      status = ruip_ruip_mixture(histories, history_count, &mixture);
    else if (method == RUIP_BINARY_POWER_PRIOR)
      status = ruip_power_prior_mixture(histories, history_count, &mixture);
    else if (method == RUIP_BINARY_PROMOTED_RUIP) {
      const double y=(double)control.events+0.5;
      const double n_y=(double)(control.n-control.events)+0.5;
      const double eta=log(y/n_y);
      const double variance=1.0/y+1.0/n_y;
      status=ruip_promoted_ruip_mixture(histories,history_count,eta,variance,
                                        &mixture,&output->promoted);
    } else
      status = ruip_standard_uip_mixture(histories, history_count,
                                         planned_n_control, &mixture);
    if (status == 0) {
      output->prior_component_count = mixture.count;
      status = aghq_posterior(&mixture, control, &posterior,
                              &output->quadrature_evidence_relative_difference,
                              output,method==RUIP_BINARY_PROMOTED_RUIP);
      if(status==0&&method==RUIP_BINARY_PROMOTED_RUIP){
        const double y=(double)control.events+0.5;
        const double n_y=(double)(control.n-control.events)+0.5;
        const double j_current=1.0/(1.0/y+1.0/n_y);
        output->promoted.expected_borrowed_precision=output->diagnostic_j_g;
        output->promoted.borrowed_precision_ratio=output->diagnostic_j_g/j_current;
        output->promoted.cap_binding_mass=output->diagnostic_robust_component;
        for(size_t source=0u;source<history_count;++source)
          output->promoted.source_contribution[source]=output->diagnostic_source_weight[source];
      }
    }
  }
  if (status == 0)
    status = summarize_posterior(method, &posterior, treatment, statistic_only,
                                 output);
  if (status != 0)
    output->status = RUIP_BINARY_STATUS_NUMERICAL_FAILURE;
  ruip_normal_mixture_free(&mixture);
  posterior_free(&posterior);
  return status;
}

int ruip_binary_analyze(ruip_binary_method_t method,
                        const ruip_history_t *histories, size_t history_count,
                        ruip_binomial_t control, ruip_binomial_t treatment,
                        double planned_n_control,
                        ruip_binary_result_t *output) {
  return binary_analyze_impl(method, histories, history_count, control,
                             treatment, planned_n_control, 0, output);
}

int ruip_binary_analyze_statistic(ruip_binary_method_t method,
                                  const ruip_history_t *histories,
                                  size_t history_count,
                                  ruip_binomial_t control,
                                  ruip_binomial_t treatment,
                                  double planned_n_control,
                                  ruip_binary_result_t *output) {
  return binary_analyze_impl(method, histories, history_count, control,
                             treatment, planned_n_control, 1, output);
}

int ruip_binary_analyze_power_prior_exact(
    const ruip_binomial_t *historical, size_t history_count,
    ruip_binomial_t control, ruip_binomial_t treatment, int statistic_only,
    ruip_binary_result_t *output) {
  node_posterior_t posterior;
  int status;
  if (output == NULL) return -1;
  initialize_result(RUIP_BINARY_POWER_PRIOR, output);
  memset(&posterior, 0, sizeof(posterior));
  if (!valid_binomial(control) || !valid_binomial(treatment) ||
      historical == NULL || history_count == 0u ||
      history_count > RUIP_HISTORY_COUNT) {
    output->status = RUIP_BINARY_STATUS_INVALID_ARGUMENT;
    return -1;
  }
  status = exact_power_prior_posterior(
      historical, history_count, control, &posterior,
      &output->quadrature_evidence_relative_difference);
  if (status == 0) {
    output->prior_component_count = 1u;
    status = summarize_posterior(RUIP_BINARY_POWER_PRIOR, &posterior, treatment,
                                 statistic_only, output);
  }
  if (status != 0) output->status = RUIP_BINARY_STATUS_NUMERICAL_FAILURE;
  posterior_free(&posterior);
  return status;
}

int ruip_binary_analyze_promoted_variant(
                        const ruip_history_t *histories,size_t history_count,
                        ruip_binomial_t control,ruip_binomial_t treatment,
                        int no_local,int no_global,double cap_ratio,
                        int statistic_only,ruip_binary_result_t *output) {
  node_posterior_t posterior;ruip_normal_mixture_t mixture;int status;
  double y,n_y,eta,variance;
  if(output==NULL||histories==NULL||history_count<2u||
     history_count>RUIP_HISTORY_COUNT||!valid_binomial(control)||
     !valid_binomial(treatment))return -1;
  initialize_result(RUIP_BINARY_PROMOTED_RUIP,output);
  memset(&posterior,0,sizeof(posterior));memset(&mixture,0,sizeof(mixture));
  y=(double)control.events+0.5;n_y=(double)(control.n-control.events)+0.5;
  eta=log(y/n_y);variance=1.0/y+1.0/n_y;
  status=ruip_promoted_ruip_mixture_variant(histories,history_count,eta,variance,
      no_local,no_global,cap_ratio,&mixture,&output->promoted);
  if(status==0){output->prior_component_count=mixture.count;
    status=aghq_posterior(&mixture,control,&posterior,
        &output->quadrature_evidence_relative_difference,output,1);}
  if(status==0){
    const double j_current=1.0/variance;
    output->promoted.expected_borrowed_precision=output->diagnostic_j_g;
    output->promoted.borrowed_precision_ratio=output->diagnostic_j_g/j_current;
    output->promoted.cap_binding_mass=output->diagnostic_robust_component;
    for(size_t source=0u;source<history_count;++source)
      output->promoted.source_contribution[source]=output->diagnostic_source_weight[source];
  }
  if(status==0)status=summarize_posterior(RUIP_BINARY_PROMOTED_RUIP,&posterior,
                                          treatment,statistic_only,output);
  if(status!=0)output->status=RUIP_BINARY_STATUS_NUMERICAL_FAILURE;
  ruip_normal_mixture_free(&mixture);posterior_free(&posterior);return status;
}

const char *ruip_binary_status_string(ruip_binary_status_t status) {
  switch (status) {
  case RUIP_BINARY_STATUS_OK:
    return "ok";
  case RUIP_BINARY_STATUS_IMPROPER_FLAT_LOGIT:
    return "improper_flat_logit_posterior";
  case RUIP_BINARY_STATUS_TREATMENT_ENDPOINT:
    return "treatment_endpoint_effect_unavailable";
  case RUIP_BINARY_STATUS_QUADRATURE_WARNING:
    return "quadrature_accuracy_warning";
  case RUIP_BINARY_STATUS_INVALID_ARGUMENT:
    return "invalid_argument";
  case RUIP_BINARY_STATUS_NUMERICAL_FAILURE:
    return "numerical_failure";
  default:
    return "unknown";
  }
}
