#include "ruip/priors.h"

#include <math.h>
#include <stdlib.h>

#include "ruip/mathutil.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static int valid_histories(const ruip_history_t *h, size_t k) {
    size_t i;
    if (h == NULL || k == 0u || k > RUIP_MAX_HISTORIES) return 0;
    for (i = 0u; i < k; ++i) {
        if (h[i].n == 0u || !isfinite(h[i].estimate) ||
            !isfinite(h[i].unit_info) || h[i].unit_info <= 0.0) return 0;
    }
    return 1;
}

/* Acklam's inverse-standard-Normal approximation, followed by one Halley step. */
static double inv_normal(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    double q, r, x, e, u;
    if (!(p > 0.0 && p < 1.0)) return NAN;
    if (p < 0.02425) {
        q = sqrt(-2.0 * log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p > 1.0 - 0.02425) {
        q = sqrt(-2.0 * log1p(-p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else {
        q = p - 0.5; r = q*q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    }
    e = ruip_normal_cdf(x) - p;
    u = e / ruip_normal_pdf(x);
    return x - u / (1.0 + 0.5*x*u);
}

static void conditional_predictive(const ruip_history_t *h, size_t k,
                                   double tau, double *mean, double *variance) {
    size_t i; double psum = 0.0, qsum = 0.0;
    for (i = 0u; i < k; ++i) {
        const double v = 1.0 / ((double)h[i].n * h[i].unit_info) + tau*tau;
        const double p = 1.0 / v;
        psum += p; qsum += p*h[i].estimate;
    }
    *mean = qsum/psum;
    *variance = tau*tau + 1.0/psum;
}

static double log_marginal(const ruip_history_t *h, size_t k, double tau) {
    size_t i; double psum = 0.0, qsum = 0.0, logv = 0.0, residual = 0.0;
    for (i = 0u; i < k; ++i) {
        const double v = 1.0 / ((double)h[i].n*h[i].unit_info) + tau*tau;
        const double p = 1.0/v;
        psum += p; qsum += p*h[i].estimate; logv += log(v);
    }
    qsum /= psum;
    for (i = 0u; i < k; ++i) {
        const double v = 1.0 / ((double)h[i].n*h[i].unit_info) + tau*tau;
        const double d = h[i].estimate-qsum;
        residual += d*d/v;
    }
    return -0.5*((double)(k-1u)*log(2.0*M_PI) + logv + log(psum) + residual);
}

void ruip_normal_mixture_free(ruip_normal_mixture_t *mixture) {
    if (mixture != NULL) {
        free(mixture->items); mixture->items = NULL; mixture->count = 0u;
    }
}

int ruip_rmap_mixture(const ruip_history_t *h, size_t k,
                      ruip_normal_mixture_t *out) {
    double x[RUIP_RMAP_TAU_ORDER], w[RUIP_RMAP_TAU_ORDER];
    double logw[RUIP_RMAP_TAU_ORDER], lse, jsum = 0.0, hsum = 0.0,
           nsum = 0.0, tau_scale;
    size_t i;
    if (out == NULL || !valid_histories(h,k)) return -1;
    out->items = NULL; out->count = 0u;
    if (ruip_gauss_legendre((int)RUIP_RMAP_TAU_ORDER,x,w) != 0) return -2;
    out->items = (ruip_normal_component_t*)calloc(RUIP_RMAP_TAU_ORDER+1u,
                                                  sizeof(*out->items));
    if (out->items == NULL) return -3;
    for (i = 0u; i < k; ++i) {
        const double j = (double)h[i].n * h[i].unit_info;
        jsum += j;
        hsum += j * h[i].estimate;
        nsum += (double)h[i].n;
    }
    /* RBesT-style scale: tau ~ HalfNormal(s_ref / 2). */
    tau_scale = 0.5 * sqrt(nsum / jsum);
    for (i=0u;i<RUIP_RMAP_TAU_ORDER;++i) {
        const double p=0.5*(x[i]+1.0);
        const double tau=tau_scale * inv_normal(0.5*(p+1.0));
        logw[i]=log(0.5*w[i])+log_marginal(h,k,tau);
        conditional_predictive(h,k,tau,&out->items[i].mean,&out->items[i].variance);
        out->items[i].log_weight=logw[i];
        out->items[i].tau=tau;
    }
    lse=ruip_logsumexp(logw,RUIP_RMAP_TAU_ORDER);
    for (i=0u;i<RUIP_RMAP_TAU_ORDER;++i)
        out->items[i].log_weight += log(1.0-RUIP_RMAP_ROBUST_WEIGHT)-lse;
    out->items[RUIP_RMAP_TAU_ORDER].mean=hsum/jsum;
    out->items[RUIP_RMAP_TAU_ORDER].variance = nsum / jsum;
    out->items[RUIP_RMAP_TAU_ORDER].log_weight=log(RUIP_RMAP_ROBUST_WEIGHT);
    out->items[RUIP_RMAP_TAU_ORDER].robust_component=1.0;
    out->count=RUIP_RMAP_TAU_ORDER+1u;
    return 0;
}
