#include "ruip/continuous.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ruip/mathutil.h"

typedef struct {
    const double *weights;
    const double *delta_means;
    const double *standard_deviations;
    size_t count;
    double target;
} cdf_context_t;

static double delta_cdf(double value,void *raw) {
    const cdf_context_t *ctx=(const cdf_context_t*)raw;
    double sum=0.0; size_t i;
    for(i=0u;i<ctx->count;++i){
        sum+=ctx->weights[i]*ruip_normal_cdf(
            (value-ctx->delta_means[i])/ctx->standard_deviations[i]);
    }
    return sum-ctx->target;
}

static int valid_group(const ruip_continuous_group_t *g) {
    return g!=NULL&&g->n>0u&&isfinite(g->mean)&&isfinite(g->known_variance)&&
           g->known_variance>0.0;
}

static void reset_promoted_posterior_diagnostics(ruip_promoted_diagnostics_t *d){
    d->expected_borrowed_precision=0.0;d->borrowed_precision_ratio=0.0;
    d->cap_binding_mass=0.0;
    memset(d->source_contribution,0,sizeof(d->source_contribution));
}

static int posterior_from_mixture(ruip_continuous_method_t method,
                                  ruip_normal_mixture_t *mix,
                                  const ruip_continuous_group_t *control,
                                  const ruip_continuous_group_t *treatment,
                                  ruip_continuous_result_t *result) {
    double *weights,*delta_means,*standard_deviations;
    double maxlog=-INFINITY,sumexp=0.0;
    double lse,mu=0.0,delta=0.0;
    double vc=control->known_variance/(double)control->n;
    double vt=treatment->known_variance/(double)treatment->n;
    double varmu=0.0,vardelta=0.0,p=0.0,spread,lo,hi;
    size_t i,j; int ok=0; cdf_context_t ctx;
    weights=(double*)malloc(mix->count*sizeof(*weights));
    delta_means=(double*)malloc(mix->count*sizeof(*delta_means));
    standard_deviations=(double*)malloc(mix->count*sizeof(*standard_deviations));
    if(weights==NULL||delta_means==NULL||standard_deviations==NULL){
        free(weights);free(delta_means);free(standard_deviations);return -1;
    }
    for(i=0u;i<mix->count;++i){
        ruip_normal_component_t *c=&mix->items[i];
        const double prior_var=c->variance;
        const double post_var=1.0/(1.0/prior_var+1.0/vc);
        const double post_mean=post_var*(c->mean/prior_var+control->mean/vc);
        weights[i]=c->log_weight+ruip_normal_logpdf(control->mean,c->mean,prior_var+vc);
        if(weights[i]>maxlog) maxlog=weights[i];
        c->mean=post_mean;
        c->variance=post_var;
    }
    for(i=0u;i<mix->count;++i)sumexp+=exp(weights[i]-maxlog);
    lse=maxlog+log(sumexp);
    for(i=0u;i<mix->count;++i){
        const double w=exp(weights[i]-lse); ruip_normal_component_t *c=&mix->items[i];
        weights[i]=w;mu+=w*c->mean;delta+=w*(treatment->mean-c->mean);
        if(method==RUIP_CONTINUOUS_RMAP){
            result->expected_tau+=w*c->tau;
            result->robust_component_probability+=w*c->robust_component;
        } else if(method==RUIP_CONTINUOUS_STANDARD_UIP){
            result->expected_borrowing_m+=w*c->borrowing_m;
            for(j=0u;j<RUIP_MAX_HISTORIES;++j)
                result->expected_source_weight[j]+=w*c->source_weight[j];
        } else if(method==RUIP_CONTINUOUS_RUIP ||
                  method==RUIP_CONTINUOUS_PROMOTED_RUIP) {
            result->expected_j_g+=w*c->j_g;result->expected_m_eff+=w*c->m_eff;
            result->state_mass[c->retention_mask]+=w;
            for(j=0u;j<RUIP_MAX_HISTORIES;++j)
                if((c->retention_mask>>j)&1u)result->retention_probability[j]+=w;
            if(method==RUIP_CONTINUOUS_PROMOTED_RUIP){
                result->promoted.cap_binding_mass+=w*c->robust_component;
                for(j=0u;j<RUIP_MAX_HISTORIES;++j)
                    result->promoted.source_contribution[j]+=w*c->source_weight[j];
            }
        }
    }
    for(i=0u;i<mix->count;++i){
        const double w=weights[i];
        const double dm=mix->items[i].mean-mu;
        const double dd=(treatment->mean-mix->items[i].mean)-delta;
        varmu+=w*(mix->items[i].variance+dm*dm);
        vardelta+=w*(vt+mix->items[i].variance+dd*dd);
        standard_deviations[i]=sqrt(vt+mix->items[i].variance);
        delta_means[i]=treatment->mean-mix->items[i].mean;
        p+=w*ruip_normal_cdf(delta_means[i]/standard_deviations[i]);
    }
    result->mu_c_mean=mu;result->mu_c_variance=varmu;
    result->delta_mean=delta;result->delta_variance=vardelta;
    result->probability_delta_positive=p;result->component_count=mix->count;
    if(method==RUIP_CONTINUOUS_PROMOTED_RUIP){
        result->promoted.expected_borrowed_precision=result->expected_j_g;
        result->promoted.borrowed_precision_ratio=result->expected_j_g*vc;
    }
    spread=sqrt(fmax(vardelta,1e-15));lo=delta-12.0*spread;hi=delta+12.0*spread;
    ctx.weights=weights;ctx.delta_means=delta_means;
    ctx.standard_deviations=standard_deviations;ctx.count=mix->count;ctx.target=0.025;
    while(delta_cdf(lo,&ctx)>0.0)lo-=12.0*spread;
    result->delta_q025=ruip_brent(delta_cdf,&ctx,lo,hi,5e-13,100,&ok);
    if(!ok){free(weights);free(delta_means);free(standard_deviations);return -2;}
    ctx.target=0.975;
    while(delta_cdf(hi,&ctx)<0.0)hi+=12.0*spread;
    result->delta_q975=ruip_brent(delta_cdf,&ctx,lo,hi,5e-13,100,&ok);
    if(!ok){free(weights);free(delta_means);free(standard_deviations);return -2;}
    free(weights);free(delta_means);free(standard_deviations);
    result->numerical_ok=isfinite(mu)&&isfinite(varmu)&&isfinite(p)&&p>=0.0&&p<=1.0;
    return result->numerical_ok?0:-3;
}

int ruip_continuous_analyze(ruip_continuous_method_t method,
                            const ruip_history_t *histories,size_t history_count,
                            const ruip_continuous_group_t *control,
                            const ruip_continuous_group_t *treatment,
                            double planned_n_control,
                            ruip_continuous_result_t *result) {
    ruip_normal_mixture_t mix={0}; int rc;
    if(result==NULL||!valid_group(control)||!valid_group(treatment)||
       control->known_variance!=treatment->known_variance)return -1;
    memset(result,0,sizeof(*result));
    if(method==RUIP_CONTINUOUS_NIP){
        const double vc=control->known_variance/(double)control->n;
        const double vt=treatment->known_variance/(double)treatment->n;
        const double sd=sqrt(vc+vt),d=treatment->mean-control->mean;
        result->mu_c_mean=control->mean;result->mu_c_variance=vc;
        result->delta_mean=d;result->delta_variance=vc+vt;
        result->probability_delta_positive=ruip_normal_cdf(d/sd);
        result->delta_q025=d-1.959963984540054*sd;
        result->delta_q975=d+1.959963984540054*sd;
        result->component_count=1u;result->numerical_ok=1;return 0;
    }
    if(histories==NULL||history_count==0u||history_count>RUIP_MAX_HISTORIES)return -1;
    switch(method){
        case RUIP_CONTINUOUS_RMAP:rc=ruip_rmap_mixture(histories,history_count,&mix);break;
        case RUIP_CONTINUOUS_COMMENSURATE:rc=ruip_commensurate_mixture(histories,history_count,&mix);break;
        case RUIP_CONTINUOUS_STANDARD_UIP:rc=ruip_standard_uip_mixture(histories,history_count,planned_n_control,&mix);break;
        case RUIP_CONTINUOUS_RUIP:rc=ruip_ruip_mixture(histories,history_count,&mix);break;
        case RUIP_CONTINUOUS_POWER_PRIOR:rc=ruip_power_prior_mixture(histories,history_count,&mix);break;
        case RUIP_CONTINUOUS_PROMOTED_RUIP:
            rc=ruip_promoted_ruip_mixture(histories,history_count,control->mean,
                control->known_variance/(double)control->n,&mix,&result->promoted);break;
        default:return -1;
    }
    if(rc!=0)return rc;
    if(method==RUIP_CONTINUOUS_PROMOTED_RUIP)
        reset_promoted_posterior_diagnostics(&result->promoted);
    rc=posterior_from_mixture(method,&mix,control,treatment,result);
    ruip_normal_mixture_free(&mix);return rc;
}

int ruip_continuous_analyze_promoted_variant(
                            const ruip_history_t *histories,size_t history_count,
                            const ruip_continuous_group_t *control,
                            const ruip_continuous_group_t *treatment,
                            int no_local,int no_global,double cap_ratio,
                            ruip_continuous_result_t *result) {
    ruip_normal_mixture_t mix={0};int rc;
    if(result==NULL||!valid_group(control)||!valid_group(treatment)||
       control->known_variance!=treatment->known_variance||histories==NULL||
       history_count<2u||history_count>RUIP_MAX_HISTORIES)return -1;
    memset(result,0,sizeof(*result));
    rc=ruip_promoted_ruip_mixture_variant(histories,history_count,control->mean,
       control->known_variance/(double)control->n,no_local,no_global,cap_ratio,
       &mix,&result->promoted);
    if(rc==0){reset_promoted_posterior_diagnostics(&result->promoted);
        rc=posterior_from_mixture(RUIP_CONTINUOUS_PROMOTED_RUIP,&mix,
                                      control,treatment,result);
    }
    ruip_normal_mixture_free(&mix);return rc;
}
