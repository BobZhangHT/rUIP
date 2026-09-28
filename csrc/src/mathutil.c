#include "ruip/mathutil.h"

#include <float.h>
#include <math.h>

double ruip_normal_cdf(double x) {
    return 0.5 * erfc(-x * 0.707106781186547524400844362105);
}

double ruip_normal_pdf(double x) {
    return exp(-0.5 * x * x) * 0.398942280401432677939946059934;
}

double ruip_normal_logpdf(double x, double mean, double variance) {
    if (!(variance > 0.0) || !isfinite(variance)) return NAN;
    const double difference = x - mean;
    return -0.5 * (1.837877066409345483560659472811 + log(variance)
                   + difference * difference / variance);
}

double ruip_logsumexp(const double *values, size_t count) {
    if (values == NULL || count == 0u) return -INFINITY;
    double maximum = values[0];
    for (size_t i = 1; i < count; ++i)
        if (values[i] > maximum) maximum = values[i];
    if (isinf(maximum)) return maximum;
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) sum += exp(values[i] - maximum);
    return maximum + log(sum);
}

double ruip_log1pexp(double x) {
    if (x > 0.0) return x + log1p(exp(-x));
    return log1p(exp(x));
}

double ruip_logistic(double x) {
    if (x >= 0.0) {
        const double z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = exp(x);
    return z / (1.0 + z);
}

double ruip_logit(double probability) {
    if (!(probability > 0.0 && probability < 1.0)) return NAN;
    return log(probability) - log1p(-probability);
}

int ruip_gauss_legendre(int order, double *nodes, double *weights) {
    if (order < 1 || nodes == NULL || weights == NULL) return -1;
    const int half = (order + 1) / 2;
    const double pi = 3.1415926535897932384626433832795;
    for (int i = 0; i < half; ++i) {
        double z = cos(pi * ((double)i + 0.75) / ((double)order + 0.5));
        double previous;
        double derivative = 0.0;
        int iteration = 0;
        do {
            double p0 = 1.0, p1 = z;
            for (int degree = 2; degree <= order; ++degree) {
                const double p2 = (((2.0 * degree - 1.0) * z * p1)
                                  - (degree - 1.0) * p0) / degree;
                p0 = p1; p1 = p2;
            }
            derivative = order * (z * p1 - p0) / (z * z - 1.0);
            previous = z;
            z = previous - p1 / derivative;
            ++iteration;
        } while (fabs(z - previous) > 4.0 * DBL_EPSILON && iteration < 100);
        if (iteration >= 100) return -2;
        const double weight = 2.0 / ((1.0 - z * z) * derivative * derivative);
        nodes[i] = -z;
        nodes[order - 1 - i] = z;
        weights[i] = weight;
        weights[order - 1 - i] = weight;
    }
    return 0;
}

double ruip_brent(ruip_scalar_function function, void *context,
                  double lower, double upper, double absolute_tolerance,
                  int max_iterations, int *ok) {
    double a = lower, b = upper, c = upper;
    double fa = function(a, context), fb = function(b, context), fc = fb;
    double d = 0.0, e = 0.0;
    if (ok != NULL) *ok = 0;
    if (!isfinite(fa) || !isfinite(fb) || fa * fb > 0.0) return NAN;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            c = a; fc = fa; d = b - a; e = d;
        }
        if (fabs(fc) < fabs(fb)) {
            const double old_b = b, old_fb = fb;
            a = b; fa = fb; b = c; fb = fc; c = old_b; fc = old_fb;
        }
        const double tolerance = 2.0 * DBL_EPSILON * fabs(b)
                               + 0.5 * absolute_tolerance;
        const double midpoint = 0.5 * (c - b);
        if (fabs(midpoint) <= tolerance || fb == 0.0) {
            if (ok != NULL) *ok = 1;
            return b;
        }
        if (fabs(e) >= tolerance && fabs(fa) > fabs(fb)) {
            double p, q;
            const double s = fb / fa;
            if (a == c) {
                p = 2.0 * midpoint * s;
                q = 1.0 - s;
            } else {
                const double q1 = fa / fc, r = fb / fc;
                p = s * (2.0 * midpoint * q1 * (q1 - r)
                    - (b - a) * (r - 1.0));
                q = (q1 - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q; else p = -p;
            if (2.0 * p < fmin(3.0 * midpoint * q - fabs(tolerance * q),
                               fabs(e * q))) {
                e = d; d = p / q;
            } else { d = midpoint; e = d; }
        } else { d = midpoint; e = d; }
        a = b; fa = fb;
        b += fabs(d) > tolerance ? d : copysign(tolerance, midpoint);
        fb = function(b, context);
        if (!isfinite(fb)) return NAN;
    }
    return b;
}
