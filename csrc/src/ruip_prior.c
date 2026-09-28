#include "ruip/priors.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ruip/mathutil.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

int ruip_design_stage_ruip_prior(
                      const ruip_history_t *h, size_t k, double delta_clin,
                      ruip_normal_mixture_t *out,
                      ruip_design_stage_diagnostics_t *diagnostics) {
    double info[RUIP_MAX_HISTORIES];
    long double retained[RUIP_MAX_HISTORIES], score[RUIP_MAX_HISTORIES];
    long double total_retained = 0.0L, weighted = 0.0L;
    double mean, local_variance, variance;
    size_t i, j, medoid = 0u;
    if (out == NULL || diagnostics == NULL || h == NULL || k == 0u ||
        k > RUIP_MAX_HISTORIES || !(delta_clin > 0.0) ||
        !isfinite(delta_clin)) return -1;
    out->items = NULL; out->count = 0u;
    memset(diagnostics, 0, sizeof(*diagnostics));
    for (i = 0u; i < k; ++i) {
        if (h[i].n == 0u || !isfinite(h[i].estimate) ||
            !(h[i].unit_info > 0.0) || !isfinite(h[i].unit_info)) return -1;
        info[i] = (double)h[i].n * h[i].unit_info;
        if (!(info[i] > 0.0) || !isfinite(info[i])) return -1;
    }
    /* The L1 source medoid is selected by minimum total distance.  Exact
     * score ties use the larger information source, then input order. */
    for (i = 0u; i < k; ++i) {
        score[i] = 0.0L;
        for (j = 0u; j < k; ++j)
            score[i] += fabsl((long double)h[i].estimate -
                              (long double)h[j].estimate);
        if (i > 0u && (score[i] < score[medoid] ||
            (score[i] == score[medoid] && info[i] > info[medoid])))
            medoid = i;
    }

    for (i = 0u; i < k; ++i) {
        long double compatibility = 1.0L;
        if (i != medoid) {
            const long double difference = (long double)h[i].estimate -
                                           (long double)h[medoid].estimate;
            const long double denominator = 2.0L *
                (1.0L/(long double)info[i] + 1.0L/(long double)info[medoid]);
            compatibility = expl(-(difference*difference)/denominator);
        }
        retained[i] = fminl((long double)info[i], (long double)info[medoid]) *
                      compatibility;
        total_retained += retained[i];
    }
    if (!(total_retained > 0.0L) || !isfinite(total_retained)) return -2;
    for (i = 0u; i < k; ++i) {
        const long double source_weight = retained[i] / total_retained;
        diagnostics->working_variance[i] = (double)(1.0L/retained[i]);
        diagnostics->local_weight[i] = (double)source_weight;
        weighted += source_weight * (long double)h[i].estimate;
    }
    local_variance = (double)(1.0L/total_retained);
    mean = (double)weighted;
    variance = local_variance + delta_clin * delta_clin;
    if (!(local_variance > 0.0) || !isfinite(local_variance) ||
        !isfinite(mean) || !(variance > 0.0) || !isfinite(variance)) return -2;
    out->items = (ruip_normal_component_t*)calloc(1u, sizeof(*out->items));
    if (out->items == NULL) return -3;
    out->count = 1u;
    out->items[0].mean = mean;
    out->items[0].variance = variance;
    out->items[0].log_weight = 0.0;
    out->items[0].j_g = 1.0 / variance;
    out->items[0].m_eff = out->items[0].j_g;
    out->items[0].borrowing_m = out->items[0].j_g;
    for (i = 0u; i < k; ++i)
        out->items[0].source_weight[i] = diagnostics->local_weight[i];
    diagnostics->local_mean = mean;
    diagnostics->local_variance = local_variance;
    diagnostics->local_precision = (double)total_retained;
    diagnostics->global_discount = local_variance / variance;
    return 0;
}

int ruip_ruip_mixture(const ruip_history_t *h, size_t k,
                      ruip_normal_mixture_t *out) {
    double x[RUIP_RUIP_LAMBDA_ORDER], w[RUIP_RUIP_LAMBDA_ORDER];
    size_t mask, q, i, states, at=0u;
    if (out==NULL || h==NULL || k==0u || k>RUIP_MAX_HISTORIES) return -1;
    out->items=NULL; out->count=0u;
    for (i=0u;i<k;++i)
        if (h[i].n==0u || !isfinite(h[i].estimate) ||
            !isfinite(h[i].unit_info) || h[i].unit_info<=0.0) return -1;
    if (ruip_gauss_legendre((int)RUIP_RUIP_LAMBDA_ORDER,x,w)!=0) return -2;
    states=((size_t)1u)<<k;
    out->items=(ruip_normal_component_t*)calloc(states*RUIP_RUIP_LAMBDA_ORDER,
                                                sizeof(*out->items));
    if (out->items==NULL) return -3;
    for (mask=0u;mask<states;++mask) {
        double j_l=0.0,q_l=0.0,n_units=0.0;
        unsigned r=0u;
        for (i=0u;i<k;++i) {
            const unsigned z=(unsigned)((mask>>i)&1u);
            const double m=1.0+(double)z*((double)h[i].n-1.0);
            const double j=m*h[i].unit_info;
            r+=z; j_l+=j; q_l+=j*h[i].estimate; n_units+=m;
        }
        for (q=0u;q<RUIP_RUIP_LAMBDA_ORDER;++q) {
            const double u=0.5*(x[q]+1.0);
            const double lambda=tan(0.5*M_PI*u);
            const double s_u2=n_units/j_l;
            const double variance=1.0/j_l+s_u2*lambda*lambda;
            ruip_normal_component_t *c=&out->items[at++];
            c->mean=q_l/j_l;
            c->variance=variance;
            c->log_weight=lgamma(1.0+(double)r)+
                lgamma(1.0+(double)k-(double)r)-lgamma(2.0+(double)k)+
                log(0.5*w[q]);
            c->j_g=1.0/variance;
            c->m_eff=1.0/(1.0/n_units+lambda*lambda);
            c->retention_mask=(unsigned)mask;
        }
    }
    out->count=at;
    return 0;
}

#define PROMOTED_IG_ORDER 32u
#define PROMOTED_Q50 0.67448975019608174320
#define PROMOTED_C_B 0.5
#define PROMOTED_CLOSED_VARIANCE 1e10

static const double promoted_laguerre_x[PROMOTED_IG_ORDER] = {
0.075352743443543216,0.30158462588090856,0.67921881231965597,1.2091346748887941,1.8925791809476384,2.7311833356258357,3.7269842127381612,4.8824533138672512,6.2005322481470238,7.6846770442172057,9.3389128175502307,11.167901058389647,13.177022531899675,15.372479772098687,17.76142452338247,20.352117419061983,23.154129967825632,26.178602997323242,29.438581817741753,32.949457762150693,36.729560601284263,40.80097052984285,45.190659342164295,49.93214260420902,55.067958508579224,60.653552758173035,66.763707770537906,73.503954785587069,81.032823137771004,89.611390797633504,99.739004352949962,112.69994395176784};
static const double promoted_laguerre_w[PROMOTED_IG_ORDER] = {
0.04330037596877457,0.13833307611395859,0.21387023125858701,0.22457449310405506,0.17793850499132621,0.11137313296232554,0.056364861411899302,0.023359054091903024,0.0079816514052177354,0.002255957637966891,0.00052784736509895087,0.00010212986981805506,1.6297426896982076e-05,2.1362609312867177e-06,2.2877843261127566e-07,1.9881435729730512e-08,1.3903893082058256e-09,7.7467146074315017e-11,3.3974701934897382e-12,1.1560317168572354e-13,2.9991163292016522e-15,5.8081755194907602e-17,8.1814114112064546e-19,8.1151191997330707e-21,5.4394879332553822e-23,2.3350587781047049e-25,5.9714668244962476e-28,8.2153828936029937e-31,5.2194907156924615e-34,1.1924129957232271e-37,6.094621708342168e-42,2.0986861253178063e-47};

static double promoted_state_mass(size_t mask, size_t k) {
    size_t i, retained = 0u;
    for (i = 0u; i < k; ++i) retained += (mask >> i) & 1u;
    return lgamma(1.0 + (double)retained) +
           lgamma(1.0 + (double)(k - retained)) - lgamma(2.0 + (double)k);
}

/* Deterministic Fisher-orthogonal basis. Columns span the complement of
 * sqrt(info)/sqrt(sum(info)); modified Gram-Schmidt is sufficient for K<=8. */
static int promoted_basis(const double *info, size_t k, double *omega,
                          double *basis, double *cross_error) {
    double u[RUIP_MAX_HISTORIES], total = 0.0, maximum = 0.0;
    size_t i, col, previous;
    for (i = 0u; i < k; ++i) total += info[i];
    if (!(total > 0.0) || !isfinite(total)) return -1;
    for (i = 0u; i < k; ++i) {
        omega[i] = info[i] / total;
        u[i] = sqrt(info[i] / total);
    }
    for (col = 0u; col + 1u < k; ++col) {
        double norm, dot = u[col];
        for (i = 0u; i < k; ++i)
            basis[i*(k-1u)+col] = (i == col ? 1.0 : 0.0) - u[i]*dot;
        for (previous = 0u; previous < col; ++previous) {
            dot = 0.0;
            for (i = 0u; i < k; ++i)
                dot += basis[i*(k-1u)+previous]*basis[i*(k-1u)+col];
            for (i = 0u; i < k; ++i)
                basis[i*(k-1u)+col] -= dot*basis[i*(k-1u)+previous];
        }
        norm = 0.0;
        for (i = 0u; i < k; ++i) {
            const double v = basis[i*(k-1u)+col]; norm += v*v;
        }
        if (!(norm > 1e-24)) return -1;
        norm = sqrt(norm);
        for (i = 0u; i < k; ++i) basis[i*(k-1u)+col] /= norm;
    }
    for (col = 0u; col + 1u < k; ++col) {
        double value = 0.0;
        for (i = 0u; i < k; ++i)
            value += omega[i]*basis[i*(k-1u)+col]/sqrt(info[i]);
        if (fabs(value) > maximum) maximum = fabs(value);
    }
    *cross_error = maximum;
    return 0;
}

static int promoted_cholesky(double *matrix, size_t n, double *logdet) {
    size_t i, j, p;
    *logdet = 0.0;
    for (i = 0u; i < n; ++i) {
        for (j = 0u; j <= i; ++j) {
            double value = matrix[i*n+j];
            for (p = 0u; p < j; ++p) value -= matrix[i*n+p]*matrix[j*n+p];
            if (i == j) {
                if (!(value > 0.0) || !isfinite(value)) return -1;
                matrix[i*n+j] = sqrt(value);
                *logdet += 2.0*log(matrix[i*n+j]);
            } else matrix[i*n+j] = value/matrix[j*n+j];
        }
        for (j = i+1u; j < n; ++j) matrix[i*n+j] = 0.0;
    }
    return 0;
}

static void promoted_cholesky_solve(const double *lower, size_t n,
                                    const double *rhs, double *solution) {
    size_t i, j;
    for (i = 0u; i < n; ++i) {
        double value = rhs[i];
        for (j = 0u; j < i; ++j) value -= lower[i*n+j]*solution[j];
        solution[i] = value/lower[i*n+i];
    }
    for (i = n; i-- > 0u;) {
        double value = solution[i];
        for (j = i+1u; j < n; ++j) value -= lower[j*n+i]*solution[j];
        solution[i] = value/lower[i*n+i];
    }
}

int ruip_promoted_ruip_mixture_variant(const ruip_history_t *h, size_t k,
                               double current_estimate, double current_variance,
                               int no_local, int no_global, double cap_ratio,
                               ruip_normal_mixture_t *out,
                               ruip_promoted_diagnostics_t *diagnostics) {
    const size_t states = ((size_t)1u) << k;
    size_t count = states*PROMOTED_IG_ORDER;
    double info[RUIP_MAX_HISTORIES], omega[RUIP_MAX_HISTORIES];
    double basis[RUIP_MAX_HISTORIES*(RUIP_MAX_HISTORIES-1u)] = {0.0};
    double maxlog = -INFINITY, sum = 0.0, local_mean = 0.0, local_variance = 0.0;
    double cross_error = 0.0, t_g, gate, j_current, borrowed = 0.0, cap_mass = 0.0;
    size_t i, j, mask, q, at = 0u;
    if (out == NULL || diagnostics == NULL || h == NULL || k < 2u ||
        k > RUIP_MAX_HISTORIES || !isfinite(current_estimate) ||
        !(current_variance > 0.0) || !isfinite(current_variance) ||
        !(cap_ratio > 0.0)) return -1;
    memset(diagnostics, 0, sizeof(*diagnostics)); out->items = NULL; out->count = 0u;
    for (i = 0u; i < k; ++i) {
        if (h[i].n == 0u || !isfinite(h[i].estimate) ||
            !(h[i].unit_info > 0.0) || !isfinite(h[i].unit_info)) return -1;
        info[i] = (double)h[i].n*h[i].unit_info;
    }
    if (promoted_basis(info, k, omega, basis, &cross_error) != 0) return -2;
    out->items = (ruip_normal_component_t*)calloc(count, sizeof(*out->items));
    if (out->items == NULL) return -3;
    for (mask = 0u; mask < states; ++mask) for (q = 0u; q < PROMOTED_IG_ORDER; ++q) {
        double retained[RUIP_MAX_HISTORIES], dinv[RUIP_MAX_HISTORIES];
        double y[RUIP_MAX_HISTORIES-1u] = {0.0};
        double cov[RUIP_MAX_HISTORIES-1u] = {0.0};
        double matrix[(RUIP_MAX_HISTORIES-1u)*(RUIP_MAX_HISTORIES-1u)] = {0.0};
        double solved_y[RUIP_MAX_HISTORIES-1u] = {0.0};
        double solved_cov[RUIP_MAX_HISTORIES-1u] = {0.0};
        double a = 0.0, y_global = 0.0, precision = 0.0, weighted = 0.0;
        double quadratic = 0.0, adjustment = 0.0, logdet = 0.0;
        const double tau2 = 6.0/promoted_laguerre_x[q];
        ruip_normal_component_t *component = &out->items[at++];
        for (i = 0u; i < k; ++i) {
            const int z = (int)((mask >> i) & 1u);
            /* tau2 is measured in unit-information variance units: the
             * additional source variance is tau2 / I_Uk. */
            retained[i] = info[i]/(1.0 + info[i]*(double)(1-z)*tau2/
                                   h[i].unit_info);
            dinv[i] = 1.0/retained[i]; precision += retained[i];
            weighted += retained[i]*h[i].estimate;
            y_global += omega[i]*h[i].estimate;
            a += omega[i]*omega[i]*dinv[i];
            for (j = 0u; j + 1u < k; ++j) {
                const double qij = basis[i*(k-1u)+j];
                y[j] += qij*sqrt(info[i])*h[i].estimate;
                cov[j] += qij*sqrt(info[i])*dinv[i]*omega[i];
            }
        }
        for (i = 0u; i + 1u < k; ++i) for (j = 0u; j + 1u < k; ++j)
            for (size_t source = 0u; source < k; ++source)
                matrix[i*(k-1u)+j] += basis[source*(k-1u)+i]*info[source]*
                    dinv[source]*basis[source*(k-1u)+j];
        if (promoted_cholesky(matrix, k-1u, &logdet) != 0) goto numerical_failure;
        promoted_cholesky_solve(matrix, k-1u, y, solved_y);
        promoted_cholesky_solve(matrix, k-1u, cov, solved_cov);
        for (i = 0u; i + 1u < k; ++i) {
            quadratic += y[i]*solved_y[i]; adjustment += cov[i]*solved_y[i];
        }
        component->mean = y_global-adjustment;
        component->variance = a;
        adjustment = 0.0;
        for (i = 0u; i + 1u < k; ++i) adjustment += cov[i]*solved_cov[i];
        component->variance -= adjustment;
        if (!(component->variance > 0.0) ||
            fabs(component->mean-weighted/precision) > 2e-10) goto numerical_failure;
        component->log_weight = promoted_state_mass(mask,k)+log(promoted_laguerre_w[q])-
            0.5*((double)(k-1u)*log(2.0*M_PI)+logdet+quadratic);
        component->retention_mask = (unsigned)mask;
        for (i = 0u; i < k; ++i) component->source_weight[i]=retained[i]/precision;
        if (component->log_weight > maxlog) maxlog=component->log_weight;
    }
    for (i = 0u; i < count; ++i) sum += exp(out->items[i].log_weight-maxlog);
    for (i = 0u; i < count; ++i) {
        out->items[i].log_weight -= maxlog+log(sum);
        local_mean += exp(out->items[i].log_weight)*out->items[i].mean;
    }
    for (i = 0u; i < count; ++i) {
        const double weight=exp(out->items[i].log_weight), d=out->items[i].mean-local_mean;
        local_variance += weight*(out->items[i].variance+d*d);
        for (j = 0u; j < k; ++j)
            diagnostics->source_contribution[j] += weight*out->items[i].source_weight[j];
    }
    if (no_local) {
        double total=0.0, weighted=0.0;
        for(i=0u;i<k;++i){total+=info[i];weighted+=info[i]*h[i].estimate;}
        free(out->items);out->items=(ruip_normal_component_t*)calloc(1u,sizeof(*out->items));
        if(out->items==NULL)return -3;
        count=1u;local_mean=weighted/total;local_variance=1.0/total;
        out->items[0].mean=local_mean;out->items[0].variance=local_variance;
        out->items[0].log_weight=0.0;
        for(i=0u;i<k;++i){out->items[0].source_weight[i]=info[i]/total;
            diagnostics->source_contribution[i]=info[i]/total;}
    }
    t_g=fabs(current_estimate-local_mean)/sqrt(current_variance+local_variance);
    gate=no_global || t_g<=PROMOTED_Q50 ? 1.0 : 0.0; j_current=1.0/current_variance;
    diagnostics->t_g=t_g;diagnostics->gate=gate;diagnostics->local_mean=local_mean;
    diagnostics->local_variance=local_variance;diagnostics->local_weight_sum=1.0;
    diagnostics->fisher_cross_covariance_error=cross_error;
    if (!gate) {
        free(out->items); out->items=(ruip_normal_component_t*)calloc(1u,sizeof(*out->items));
        if (out->items==NULL) return -3;
        out->count=1u;out->items[0].mean=0.0;out->items[0].variance=PROMOTED_CLOSED_VARIANCE;
        out->items[0].log_weight=0.0;return 0;
    }
    for (i = 0u; i < count; ++i) {
        const double weight=exp(out->items[i].log_weight);
        const double raw=1.0/out->items[i].variance;
        const double admitted=fmin(raw,cap_ratio*j_current);
        if (raw>cap_ratio*j_current) cap_mass+=weight;
        out->items[i].robust_component = raw>cap_ratio*j_current ? 1.0 : 0.0;
        out->items[i].variance=1.0/admitted;out->items[i].j_g=admitted;
        out->items[i].m_eff=admitted;out->items[i].borrowing_m=admitted;
        borrowed+=weight*admitted;
    }
    out->count=count;diagnostics->expected_borrowed_precision=borrowed;
    diagnostics->borrowed_precision_ratio=borrowed/j_current;
    diagnostics->cap_binding_mass=cap_mass;return 0;
numerical_failure:
    ruip_normal_mixture_free(out);return -4;
}

int ruip_promoted_ruip_mixture(const ruip_history_t *h, size_t k,
                               double current_estimate, double current_variance,
                               ruip_normal_mixture_t *out,
                               ruip_promoted_diagnostics_t *diagnostics) {
    return ruip_promoted_ruip_mixture_variant(h,k,current_estimate,current_variance,
                                              0,0,PROMOTED_C_B,out,diagnostics);
}
