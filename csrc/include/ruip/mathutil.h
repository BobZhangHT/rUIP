#ifndef RUIP_MATHUTIL_H
#define RUIP_MATHUTIL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*ruip_scalar_function)(double x, void *context);

double ruip_normal_cdf(double x);
double ruip_normal_pdf(double x);
double ruip_normal_logpdf(double x, double mean, double variance);
double ruip_logsumexp(const double *values, size_t count);
double ruip_log1pexp(double x);
double ruip_logistic(double x);
double ruip_logit(double probability);
int ruip_gauss_legendre(int order, double *nodes, double *weights);
double ruip_brent(ruip_scalar_function function, void *context,
                  double lower, double upper, double absolute_tolerance,
                  int max_iterations, int *ok);

#ifdef __cplusplus
}
#endif

#endif

