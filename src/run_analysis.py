#!/usr/bin/env python
"""run_analysis.py -- significance test, zero-shot AUROC, and feature-weight analysis.

Reproduces exactly the analysis behind docs/05_benchmark_results.md:
  1. Zero-shot AUROC per architecture (raw masked-marginal surprisal, no training)
  2. DeLong + paired-bootstrap significance: best PLM vs identity floor (contiguous)
  3. All-features LR feature-weight ranking (per-block + top individual)
Outputs: results/tables/{zeroshot_vs_classifier,significance_plm_vs_floor,
         feature_weights_by_block,feature_weights_top20}.csv
"""
import sys, os, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
_root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'.projectroot').exists())
sys.path.insert(0,str(_root))
from paths import INTERIM, PROCESSED, TABLES
import numpy as np, pandas as pd
from scipy import stats
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[v]='4'
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
SEED=42
df=pd.read_parquet(PROCESSED/'tem1_firnberg_processed.parquet')
pool=df[~df.excluded_wetlab_validation].reset_index(drop=True)
y=pool.DMS_score_bin.values.astype(int); mi={m:i for i,m in enumerate(pool.mutant.values)}
def keyed(npz,key):
    d=np.load(INTERIM/npz,allow_pickle=True); M=d[key]; M=M[:,None] if M.ndim==1 else M
    o=[mi[m] for m in d['mutant']]; out=np.full((len(pool),M.shape[1]),np.nan,np.float32); out[o]=M
    assert not np.isnan(out).any(),npz; return out

# ---------- 1. Zero-shot ----------
zrows=[]
for tag,name in [("feat_esm2_650m.npz","ESM-2 650M"),("feat_esm1v.npz","ESM-1v"),("feat_esmc.npz","ESM C 600M")]:
    s=keyed(tag,'rep1').ravel(); au=roc_auc_score(y,-s); au=au if au>=0.5 else roc_auc_score(y,s)
    zrows.append(dict(architecture=name,zeroshot_auroc=round(au,4)))
zdf=pd.DataFrame(zrows)
# Bias-free classifier selection: pick classifier per feature_set via MODULO-split AUROC
# (not the contiguous score being reported), matching build_architecture_comparison.py.
seq=pd.read_csv(TABLES/'plm_benchmark_sequence_only.csv')
c=seq[seq.split=='contiguous']; m=seq[seq.split=='modulo']
def bias_free_contiguous(feature_set):
    con=c[c.feature_set==feature_set]; mod=m[m.feature_set==feature_set]
    common=set(con.classifier)&set(mod.classifier)
    mod_c=mod[mod.classifier.isin(common)]
    pre_clf=mod_c.loc[mod_c.auroc.idxmax(),'classifier']
    return con[con.classifier==pre_clf].auroc.iloc[0]
rep1={'ESM-2 650M':'ESM-2 650M Rep1 masked-marginal scalar','ESM-1v':'ESM-1v Rep1 masked-marginal scalar','ESM C 600M':'ESM C 600M Rep1 masked-marginal scalar'}
for i,r in zdf.iterrows():
    a=r.architecture
    aset_names=[fs for fs in c[c.feature_set.str.startswith(a)].feature_set.unique()]
    r1_auroc=bias_free_contiguous(rep1[a])
    aset_auroc=max(bias_free_contiguous(fs) for fs in aset_names)
    zdf.loc[i,'best_clf_surprisal_only']=round(r1_auroc,4)
    zdf.loc[i,'best_clf_richest']=round(aset_auroc,4)
    zdf.loc[i,'gain_zs_to_best']=round(aset_auroc-r.zeroshot_auroc,4)
zdf.to_csv(TABLES/'zeroshot_vs_classifier.csv',index=False)
print('[1] zero-shot:', dict(zip(zdf.architecture,zdf.zeroshot_auroc)))

# ---------- 2. Significance (best PLM vs identity floor, contiguous) ----------
def make_clf(kind):
    if kind=='LR': return Pipeline([('sc',StandardScaler()),('clf',LogisticRegression(max_iter=2000,random_state=SEED))]),{'clf__C':[0.01,0.1,1,10]}
    if kind=='RF': return RandomForestClassifier(random_state=SEED,n_jobs=1),{'n_estimators':[300],'max_depth':[None,8,16],'min_samples_leaf':[1,3]}
    if kind=='XGB': return XGBClassifier(random_state=SEED,n_jobs=1,eval_metric='logloss',verbosity=0),{'n_estimators':[300],'max_depth':[3,6],'learning_rate':[0.05,0.1]}
def oof(X,kind):
    folds=df.loc[pool.index,'fold_contiguous'].values; o=np.full(len(pool),np.nan)
    for f in sorted(set(folds)):
        tr,te=(folds!=f),(folds==f); base,grid=make_clf(kind)
        gs=GridSearchCV(base,grid,scoring='roc_auc',cv=StratifiedKFold(3,shuffle=True,random_state=SEED),n_jobs=1).fit(X[tr],y[tr])
        o[te]=gs.best_estimator_.predict_proba(X[te])[:,1]
    return o
def midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2
def delong(y,p1,p2):
    pos=y==1; neg=y==0; m=pos.sum(); n=neg.sum(); aucs=[]; V10=[]; V01=[]
    for p in (p1,p2):
        tx=midrank(p[pos]); ty=midrank(p[neg]); tz=midrank(p)
        aucs.append((tz[pos].sum()-m*(m+1)/2)/(m*n)); V10.append((tz[pos]-tx)/n); V01.append(1-(tz[neg]-ty)/m)
    V10=np.array(V10); V01=np.array(V01); S=np.cov(V10)/m+np.cov(V01)/n
    d=aucs[0]-aucs[1]; var=S[0,0]+S[1,1]-2*S[0,1]; z=d/np.sqrt(var) if var>0 else np.inf
    return aucs[0],aucs[1],z,2*stats.norm.sf(abs(z))
def pboot(y,p1,p2,n=5000):
    rng=np.random.default_rng(SEED); N=len(y); d=[]
    for _ in range(n):
        idx=rng.integers(0,N,N)
        if len(set(y[idx]))<2: continue
        d.append(roc_auc_score(y[idx],p1[idx])-roc_auc_score(y[idx],p2[idx]))
    d=np.array(d); return d.mean(),np.percentile(d,2.5),np.percentile(d,97.5),(d<=0).mean()
# Classifier selection is bias-free: both PLM and floor classifiers are the ones picked
# by build_architecture_comparison.py via MODULO-split AUROC, never the contiguous score
# being reported here. (ESM C Rep4 -> XGBoost; Identity one-hot floor -> XGBoost, not RF.)
plm=oof(keyed('feat_esmc.npz','rep4'),'XGB'); flo=oof(keyed('feat_rep8_identity.npz','X'),'XGB')
a1,a2,z,pv=delong(y,plm,flo); md,lo,hi,pb=pboot(y,plm,flo)
sig=pd.DataFrame([dict(comparison='ESM C Rep4 (XGBoost, bias-free selected) vs Identity floor (XGBoost, bias-free selected)',
                       auroc_plm=round(a1,4),auroc_floor=round(a2,4),
                       delta=round(a1-a2,4),delong_z=round(z,2),delong_p=f'{pv:.2e}',boot_delta=round(md,4),
                       boot_ci=f'[{lo:.4f},{hi:.4f}]',boot_p_le0=f'{pb:.4f}',
                       selection_note='Classifier selected via modulo-split AUROC, not the contiguous score being reported.')])
sig.to_csv(TABLES/'significance_plm_vs_floor.csv',index=False)
print(f'[2] significance: delta={a1-a2:.4f} DeLong p={pv:.2e}')

# ---------- 3. Feature weights (all-features LR) ----------
blocks={'ESM2_Rep1_surprisal':keyed('feat_esm2_650m.npz','rep1'),'ESM2_Rep3_surprisal20':keyed('feat_esm2_650m.npz','rep3'),
 'ESM2_Rep4_site+onehot':keyed('feat_esm2_650m.npz','rep4'),'ESM1v_Rep1_surprisal':keyed('feat_esm1v.npz','rep1'),
 'ESMC_Rep1_surprisal':keyed('feat_esmc.npz','rep1'),'ESM2_site_emb':keyed('emb3_esm2_650m.npz','emb_site'),
 'ESMC_site_emb':keyed('emb3_esmc.npz','emb_site'),'Physicochemical':keyed('feat_rep7_physico.npz','X'),
 'ESMFold_struct':keyed('feat_esmfold_struct.npz','X'),'Identity_onehot':keyed('feat_rep8_identity.npz','X')}
names=[]; mats=[]
for b,M in blocks.items():
    mats.append(M); names+= [f'{b}[{i}]' for i in range(M.shape[1])]
X=np.hstack(mats); block_of=np.array([n.rsplit('[',1)[0] for n in names])
coef=np.abs(LogisticRegression(max_iter=5000,C=1.0).fit(StandardScaler().fit_transform(X),y).coef_[0])
imp=pd.DataFrame({'feature':names,'block':block_of,'abscoef':coef})
agg=imp.groupby('block').agg(n_features=('abscoef','size'),total_absweight=('abscoef','sum'),mean_absweight=('abscoef','mean'),max_absweight=('abscoef','max')).reset_index().sort_values('mean_absweight',ascending=False)
for col in ['total_absweight','mean_absweight','max_absweight']: agg[col]=agg[col].round(4)
agg.to_csv(TABLES/'feature_weights_by_block.csv',index=False)
imp.sort_values('abscoef',ascending=False).head(20).assign(abscoef=lambda d:d.abscoef.round(4)).to_csv(TABLES/'feature_weights_top20.csv',index=False)
print(f'[3] feature weights: top block by mean|coef| = {agg.iloc[0].block} ({agg.iloc[0].mean_absweight})')
print('run_analysis.py DONE')
