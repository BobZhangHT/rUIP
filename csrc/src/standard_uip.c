#include "ruip/priors.h"

#include <math.h>
#include <stdlib.h>

#include "ruip/mathutil.h"

#define JMAX RUIP_STANDARD_UIP_WEIGHT_ORDER

static void identity(double v[JMAX][JMAX], size_t n) {
    size_t i,j;
    for(i=0u;i<n;++i) for(j=0u;j<n;++j) v[i][j]=(i==j)?1.0:0.0;
}

/* Symmetric Jacobi eigensolver; n=24 makes the O(n^3) setup negligible. */
static int symmetric_eigen(double a[JMAX][JMAX], double v[JMAX][JMAX], size_t n) {
    size_t sweep,p,q,k; identity(v,n);
    for(sweep=0u;sweep<100u*n*n;++sweep) {
        double largest=0.0; p=0u; q=0u;
        for(size_t i=0u;i<n;++i) for(size_t j=i+1u;j<n;++j)
            if(fabs(a[i][j])>largest){largest=fabs(a[i][j]);p=i;q=j;}
        if(largest<2e-15) return 0;
        {
            const double app=a[p][p], aqq=a[q][q], apq=a[p][q];
            const double phi=0.5*atan2(2.0*apq,aqq-app);
            const double c=cos(phi),s=sin(phi);
            for(k=0u;k<n;++k) if(k!=p && k!=q) {
                const double akp=a[k][p],akq=a[k][q];
                a[k][p]=a[p][k]=c*akp-s*akq;
                a[k][q]=a[q][k]=s*akp+c*akq;
            }
            a[p][p]=c*c*app-2.0*s*c*apq+s*s*aqq;
            a[q][q]=s*s*app+2.0*s*c*apq+c*c*aqq;
            a[p][q]=a[q][p]=0.0;
            for(k=0u;k<n;++k) {
                const double vkp=v[k][p],vkq=v[k][q];
                v[k][p]=c*vkp-s*vkq; v[k][q]=s*vkp+c*vkq;
            }
        }
    }
    return -1;
}

static int gauss_jacobi_probability(size_t n,double alpha,double beta,
                                    double *nodes,double *weights) {
    double a[JMAX][JMAX]={{0}},v[JMAX][JMAX]; size_t i,j,best;
    if(n==0u||n>JMAX||alpha<=-1.0||beta<=-1.0) return -1;
    for(i=0u;i<n;++i) {
        const double t=2.0*(double)i+alpha+beta;
        a[i][i]=(i==0u)?(beta-alpha)/(alpha+beta+2.0):
            (beta*beta-alpha*alpha)/(t*(t+2.0));
        if(i>0u) {
            const double ii=(double)i;
            const double den=t*t*(t-1.0)*(t+1.0);
            const double num=4.0*ii*(ii+alpha)*(ii+beta)*(ii+alpha+beta);
            if(!(num/den>0.0)) return -2;
            a[i][i-1]=a[i-1][i]=sqrt(num/den);
        }
    }
    if(symmetric_eigen(a,v,n)!=0) return -3;
    for(i=0u;i<n;++i){nodes[i]=a[i][i];weights[i]=v[0][i]*v[0][i];}
    for(i=0u;i<n;++i) {
        best=i; for(j=i+1u;j<n;++j) if(nodes[j]<nodes[best]) best=j;
        if(best!=i){double z=nodes[i];nodes[i]=nodes[best];nodes[best]=z;
                    z=weights[i];weights[i]=weights[best];weights[best]=z;}
    }
    return 0;
}

int ruip_standard_uip_mixture(const ruip_history_t *h,size_t k,
                              double planned_n_control,
                              ruip_normal_mixture_t *out) {
    double xm[RUIP_STANDARD_UIP_M_ORDER],wm[RUIP_STANDARD_UIP_M_ORDER];
    double gamma[RUIP_MAX_HISTORIES],x1[JMAX],p1[JMAX],x2[JMAX],p2[JMAX];
    double weights[JMAX*JMAX][RUIP_MAX_HISTORIES];
    double masses[JMAX*JMAX];
    size_t nw=0u,i,j,q,at=0u,total_n=0u;
    double mmax;
    if(out==NULL||h==NULL||k==0u||k>RUIP_COMPARATOR_MAX_HISTORIES||
       !isfinite(planned_n_control)||planned_n_control<=0.0) return -1;
    out->items=NULL;out->count=0u;
    for(i=0u;i<k;++i){
        if(h[i].n==0u||!isfinite(h[i].estimate)||
           !isfinite(h[i].unit_info)||h[i].unit_info<=0.0)return -1;
        gamma[i]=fmin(1.0,(double)h[i].n/planned_n_control); total_n+=h[i].n;
    }
    mmax=fmin(planned_n_control,(double)total_n);
    if(k==1u){weights[0][0]=1.0;masses[0]=1.0;nw=1u;}
    else if(k==2u){
        if(gauss_jacobi_probability(JMAX,gamma[1]-1.0,gamma[0]-1.0,x1,p1)!=0)return -2;
        for(i=0u;i<JMAX;++i){weights[i][0]=0.5*(x1[i]+1.0);
            weights[i][1]=1.0-weights[i][0];masses[i]=p1[i];} nw=JMAX;
    } else {
        if(gauss_jacobi_probability(JMAX,gamma[1]+gamma[2]-1.0,gamma[0]-1.0,x1,p1)!=0||
           gauss_jacobi_probability(JMAX,gamma[2]-1.0,gamma[1]-1.0,x2,p2)!=0)return -2;
        for(i=0u;i<JMAX;++i)for(j=0u;j<JMAX;++j){
            const double a0=0.5*(x1[i]+1.0),b0=0.5*(x2[j]+1.0);
            weights[nw][0]=a0;weights[nw][1]=(1.0-a0)*b0;
            weights[nw][2]=(1.0-a0)*(1.0-b0);masses[nw]=p1[i]*p2[j];++nw;
        }
    }
    if(ruip_gauss_legendre((int)RUIP_STANDARD_UIP_M_ORDER,xm,wm)!=0)return -3;
    out->items=(ruip_normal_component_t*)calloc(nw*RUIP_STANDARD_UIP_M_ORDER,
                                                sizeof(*out->items));
    if(out->items==NULL)return -4;
    for(i=0u;i<nw;++i){
        double mean=0.0,unit=0.0;
        for(j=0u;j<k;++j){mean+=weights[i][j]*h[j].estimate;
                           unit+=weights[i][j]*h[j].unit_info;}
        for(q=0u;q<RUIP_STANDARD_UIP_M_ORDER;++q){
            const double u=0.5*(xm[q]+1.0),m=mmax*u*u;
            ruip_normal_component_t *c=&out->items[at++];
            c->mean=mean;c->variance=1.0/(m*unit);
            c->log_weight=log(masses[i])+log(wm[q]*u);
            c->borrowing_m=m;
            for(j=0u;j<k;++j)c->source_weight[j]=weights[i][j];
        }
    }
    out->count=at;return 0;
}
