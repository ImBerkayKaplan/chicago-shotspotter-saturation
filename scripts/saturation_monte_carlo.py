import numpy as np, statsmodels.api as sm, warnings
warnings.filterwarnings("ignore")
rng=np.random.default_rng(7)
N=77;T=15;g0=8  # treat from index 8 (2017); 2009..2023
lam=np.ones(T); lam[g0:]=1.32   # common proportional trend, R=1.32
def sim(k,effect,rng):
    x=rng.normal(size=N)
    mu=np.exp(0.9*x+0.9)           # risk-driven level
    # placement: sharp logistic in risk; center set so ~2/3 treated
    c=np.quantile(x,0.34)
    p=1/(1+np.exp(-k*(x-c)))
    D=(rng.uniform(size=N)<p).astype(int)
    if D.sum()<5 or D.sum()>N-5: return None
    lamD=np.outer(1+0*mu,lam)
    m=mu[:,None]*lam[None,:]
    m=m*np.where((D[:,None]==1)&(np.arange(T)[None,:]>=g0),1+effect,1)
    Y=rng.poisson(m)
    return x,D,Y
def est(x,D,Y):
    pre=Y[:,:g0].mean(1); post=Y[:,g0:].mean(1)
    dy=post-pre
    X=sm.add_constant(D.astype(float))
    r=sm.OLS(dy,X).fit(cov_type="HC1")
    ols=(r.params[1],r.pvalues[1])
    # PPML TWFE collapsed to pre/post
    def ppml(idx):
        n=idx.sum(); ids=np.arange(n)
        Yl=np.r_[Y[idx,:g0].sum(1),Y[idx,g0:].sum(1)]
        expo=np.r_[np.full(n,g0),np.full(n,T-g0)].astype(float)
        U=np.tile(np.eye(n),(2,1))[:,1:]
        post=np.r_[np.zeros(n),np.ones(n)]
        Dp=np.r_[np.zeros(n),D[idx]]
        Xm=np.column_stack([np.ones(2*n),U,post,Dp])
        try:
            f=sm.GLM(Yl,Xm,family=sm.families.Poisson(),offset=np.log(expo)).fit(cov_type="cluster",cov_kwds={"groups":np.r_[ids,ids]})
            return f.params[-1],f.pvalues[-1]
        except Exception: return (np.nan,np.nan)
    allidx=np.ones(N,bool)
    pp=ppml(allidx)
    # propensity
    try:
        lg=sm.Logit(D,sm.add_constant(x)).fit(disp=0,maxiter=200); e=lg.predict(sm.add_constant(x))
    except Exception: return None
    ceil=e[D==0].max(); floor=e[D==1].min()
    omega=np.mean(e[D==1]>ceil)
    cs=(e<=ceil)&(e>=floor)
    tr=pp_t=(np.nan,np.nan)
    if D[cs].sum()>=3 and (1-D[cs]).sum()>=3:
        tr=ppml(cs)
    return ols,pp,tr,omega,ceil
res={}
for k in [0.5,1,2,4,8]:
  for eff in [0.0,-0.2]:
    rows=[];tries=0
    while len(rows)<300 and tries<2000:
        tries+=1
        s=sim(k,eff,rng)
        if s is None: continue
        r=est(*s)
        if r is None: continue
        rows.append(r)
    a=np.array([[r[0][0],r[0][1],r[1][0],r[1][1],r[2][0],r[2][1],r[3]] for r in rows])
    print(f"k={k} eff={eff} n={len(rows)} Omega={np.nanmean(a[:,6]):.2f} "
          f"OLSbias={np.nanmean(a[:,0]):.2f} OLSrej={np.nanmean(a[:,1]<.05):.2f} "
          f"PPMLlogcoef={np.nanmean(a[:,2]):.3f} PPMLrej={np.nanmean(a[:,3]<.05):.2f} "
          f"TRIMlogcoef={np.nanmean(a[:,4]):.3f} TRIMrej={np.nanmean(a[:,5]<.05):.2f} TRIMsd={np.nanstd(a[:,4]):.3f} PPMLsd={np.nanstd(a[:,2]):.3f}",flush=True)