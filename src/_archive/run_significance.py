#!/usr/bin/env python
"""run_significance.py -- DeLong + paired bootstrap on the CONTIGUOUS split.

Comparisons (all on contiguous OOF predictions):
  1. top sequence-only cell        vs Identity one-hot floor
  2. top predicted-structure cell  vs Identity one-hot floor
  3. top predicted-structure cell  vs top sequence-only cell
Writes results/tables/significance.csv. Requires cached contiguous OOF predictions
(run_benchmark.py caches them to data/interim/oof_contiguous.npz).
"""
import sys
from pathlib import Path
_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'.projectroot').exists())
sys.path.insert(0, str(_root)); from paths import INTERIM, PROCESSED, TABLES
import numpy as np, pandas as pd
from scipy import stats

# --- DeLong (fast implementation, Sun & Xu 2014) ---
def _compute_midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N)
    i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2
def delong_var(y, p1, p2):
    # returns z, pvalue for AUC(p1)-AUC(p2), paired
    order=np.argsort(-y)  # positives first? build via label
    pos=y==1; neg=y==0
    def structural(preds):
        m=pos.sum(); n=neg.sum()
        tx=np.vstack([preds[pos]]); ty=np.vstack([preds[neg]])
        k=1; tz=np.hstack([tx,ty])
        v01=np.empty((k,m)); v10=np.empty((k,n))
        for r in range(k):
            v01[r]=(_compute_midrank(tz[r])[:m]-_compute_midrank(tx[r]))/n
            v10[r]=1.0-(_compute_midrank(tz[r])[m:]-_compute_midrank(ty[r]))/m
        auc=(_compute_midrank(np.r_[tx[0],ty[0]])[:m].sum()/m - (m+1)/2)/n
        return auc,v01,v10,m,n
    a1,v01_1,v10_1,m,n=structural(p1)
    a2,v01_2,v10_2,_,_=structural(p2)
    v01=np.vstack([v01_1,v01_2]); v10=np.vstack([v10_1,v10_2])
    sx=np.cov(v01); sy=np.cov(v10)
    s=sx/m+sy/n
    var=s[0,0]+s[1,1]-2*s[0,1]
    if var<=0: return 0.0,1.0,a1,a2
    z=(a1-a2)/np.sqrt(var)
    pval=2*stats.norm.sf(abs(z))
    return float(z),float(pval),float(a1),float(a2)

def paired_bootstrap(y,p1,p2,n=5000,seed=42):
    rng=np.random.default_rng(seed); N=len(y); d=[]
    from sklearn.metrics import roc_auc_score
    for _ in range(n):
        idx=rng.integers(0,N,N)
        if len(set(y[idx]))<2: continue
        d.append(roc_auc_score(y[idx],p1[idx])-roc_auc_score(y[idx],p2[idx]))
    d=np.array(d); lo,hi=np.percentile(d,[2.5,97.5])
    p=2*min((d<=0).mean(),(d>=0).mean())
    return float(d.mean()),float(lo),float(hi),float(p)

df=pd.read_parquet(PROCESSED/'tem1_firnberg_processed.parquet')
pool=df[~df.excluded_wetlab_validation].reset_index(drop=True)
y=pool.DMS_score_bin.values.astype(int)
oof=np.load(INTERIM/'oof_contiguous.npz', allow_pickle=True)   # {name: pred vector}
res=pd.read_csv(TABLES/'plm_benchmark_results.csv')
cont=res[res.split=='contiguous']

def top(tier):
    sub=cont[cont.tier==tier].sort_values('auroc',ascending=False)
    return sub.iloc[0] if len(sub) else None
floor_key=[k for k in oof.files if 'Identity one-hot' in k]
top_seq=top('seq_plm'); top_str=top('pred_struct')

def keyfor(row):
    return f"{row.classifier}|{row.feature_set}"
rows=[]
def compare(label,a_row,b_row):
    ka,kb=keyfor(a_row),keyfor(b_row)
    if ka not in oof.files or kb not in oof.files:
        print('  missing OOF for',ka,'or',kb); return
    z,pv,aa,ab=delong_var(y,oof[ka],oof[kb])
    md,lo,hi,pb=paired_bootstrap(y,oof[ka],oof[kb])
    rows.append(dict(comparison=label,model_a=ka,auroc_a=aa,model_b=kb,auroc_b=ab,
                     delong_z=z,delong_p=pv,boot_delta=md,boot_ci_lo=lo,boot_ci_hi=hi,boot_p=pb))
    print(f'  {label}: {aa:.4f} vs {ab:.4f}  DeLong p={pv:.2e}  bootΔ={md:+.4f}[{lo:+.4f},{hi:+.4f}] p={pb:.2e}')

# floor row
floor_row=cont[cont.feature_set=='Identity one-hot'].sort_values('auroc',ascending=False).iloc[0]
if top_seq is not None: compare('top sequence-only vs Identity one-hot', top_seq, floor_row)
if top_str is not None: compare('top predicted-structure vs Identity one-hot', top_str, floor_row)
if top_seq is not None and top_str is not None: compare('top predicted-structure vs top sequence-only', top_str, top_seq)
pd.DataFrame(rows).to_csv(TABLES/'significance.csv',index=False)
print('wrote significance.csv')
