#!/usr/bin/env python
"""run_benchmark_distilled.py -- FEATURE-IMPORTANCE-GUIDED direction-finding benchmark.

Instead of throwing raw 2500-dim stacks at tree models (slow + wrong inductive bias), this:
  1. builds a grand matrix of all named feature blocks,
  2. INSIDE each CV fold: fits LR on the TRAIN fold, ranks features by |coef|, selects top-K,
  3. trains LR / RandomForest / XGBoost on the distilled top-K (low-dim -> trees get a fast, fair shot),
  4. evaluates on the held-out fold -> pooled OOF AUROC + bootstrap CI, across all 3 splits.

Selection is done per-fold on train only => no leakage. Tests directly whether a tree model,
given the DISTILLED signal, beats LR -- the 'what if RF/XGBoost is actually best' question.
Output: plm_benchmark_distilled.csv  (feature_set = 'Distilled top-{K}')
"""
import sys, time, warnings, os
from pathlib import Path
_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'.projectroot').exists())
sys.path.insert(0, str(_root))
from paths import INTERIM, PROCESSED, TABLES
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[v]='4'
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
SEED=42
df=pd.read_parquet(PROCESSED/'tem1_firnberg_processed.parquet')
pool=df[~df.excluded_wetlab_validation].reset_index(drop=True)
y=pool.DMS_score_bin.values.astype(int)
mi={m:i for i,m in enumerate(pool.mutant.values)}
def keyed(npz,key):
    d=np.load(INTERIM/npz,allow_pickle=True); M=d[key]; M=M[:,None] if M.ndim==1 else M
    o=[mi[m] for m in d['mutant']]; out=np.full((len(pool),M.shape[1]),np.nan,np.float32); out[o]=M
    assert not np.isnan(out).any(),npz; return out

blocks={
 'ESM2_Rep1':keyed('feat_esm2_650m.npz','rep1'),'ESM2_Rep3':keyed('feat_esm2_650m.npz','rep3'),
 'ESM2_Rep4':keyed('feat_esm2_650m.npz','rep4'),'ESM1v_Rep1':keyed('feat_esm1v.npz','rep1'),
 'ESMC_Rep1':keyed('feat_esmc.npz','rep1'),'ESMC_Rep4':keyed('feat_esmc.npz','rep4'),
 'ESM2_site_emb':keyed('emb3_esm2_650m.npz','emb_site'),'ESMC_site_emb':keyed('emb3_esmc.npz','emb_site'),
 'Physicochemical':keyed('feat_rep7_physico.npz','X'),'ESMFold_struct':keyed('feat_esmfold_struct.npz','X'),
 'Identity':keyed('feat_rep8_identity.npz','X'),
}
X=np.hstack(list(blocks.values()))
print(f'grand matrix: {X.shape}',flush=True)

def make_clf(kind):
    if kind=='Logistic Regression': return Pipeline([('sc',StandardScaler()),('clf',LogisticRegression(max_iter=3000,random_state=SEED))]),{'clf__C':[0.01,0.1,1,10]}
    if kind=='Random Forest': return RandomForestClassifier(random_state=SEED,n_jobs=1),{'n_estimators':[300],'max_depth':[None,8,16],'min_samples_leaf':[1,3]}
    if kind=='XGBoost': return XGBClassifier(random_state=SEED,n_jobs=1,eval_metric='logloss',verbosity=0),{'n_estimators':[300],'max_depth':[3,6],'learning_rate':[0.05,0.1]}
    if kind=='SVM': return Pipeline([('sc',StandardScaler()),('clf',SVC(probability=True,random_state=SEED))]),{'clf__C':[0.1,1,10],'clf__gamma':['scale']}

def distilled_oof(splitcol,kind,K):
    folds=df.loc[pool.index,splitcol].values; oof=np.full(len(pool),np.nan)
    for f in sorted(set(folds)):
        tr,te=(folds!=f),(folds==f)
        # per-fold feature selection: LR on TRAIN only, rank |coef|, take top-K
        sel=Pipeline([('sc',StandardScaler()),('clf',LogisticRegression(max_iter=2000,C=1.0,random_state=SEED))]).fit(X[tr],y[tr])
        idx=np.argsort(-np.abs(sel.named_steps['clf'].coef_[0]))[:K]
        Xtr,Xte=X[tr][:,idx],X[te][:,idx]
        base,grid=make_clf(kind)
        gs=GridSearchCV(base,grid,scoring='roc_auc',cv=StratifiedKFold(3,shuffle=True,random_state=SEED),n_jobs=1).fit(Xtr,y[tr])
        m=gs.best_estimator_
        p=m.predict_proba(Xte)[:,1] if hasattr(m,'predict_proba') else m.decision_function(Xte)
        oof[te]=p
    return oof
def boot_ci(yt,pr,n=2000):
    rng=np.random.default_rng(SEED);N=len(yt);s=[]
    for _ in range(n):
        i=rng.integers(0,N,N)
        if len(set(yt[i]))<2:continue
        s.append(roc_auc_score(yt[i],pr[i]))
    return float(np.percentile(s,2.5)),float(np.percentile(s,97.5))

rows=[];t0=time.time()
CK=INTERIM/'benchmark_partial_distilled.csv'
done=set()
if CK.exists():
    prev=pd.read_csv(CK);rows=prev.to_dict('records');done={(r['feature_set'],r['classifier'],r['split']) for r in rows}
oof_contig={}
for K in [30,60,100]:
    for kind in ['Logistic Regression','Random Forest','XGBoost','SVM']:
        for scol,sname in [('fold_random','random'),('fold_modulo','modulo'),('fold_contiguous','contiguous')]:
            fs=f'Distilled top-{K}'
            if (fs,kind,sname) in done: continue
            oof=distilled_oof(scol,kind,K)
            au=roc_auc_score(y,oof);lo,hi=boot_ci(y,oof)
            rows.append(dict(feature_set=fs,classifier=kind,split=sname,tier='distilled',auroc=au,ci_lo=lo,ci_hi=hi,n_features=K))
            pd.DataFrame(rows).to_csv(CK,index=False)
            if sname=='contiguous': oof_contig[f'{kind}|{fs}']=oof
            print(f'  K={K:3} {sname:10} {kind:20} AUROC {au:.4f} [{lo:.4f},{hi:.4f}] ({time.time()-t0:.0f}s)',flush=True)
pd.DataFrame(rows).to_csv(TABLES/'plm_benchmark_distilled.csv',index=False)
if oof_contig: np.savez_compressed(INTERIM/'oof_contiguous_distilled.npz',**oof_contig)
print(f'DISTILLED DONE: {len(rows)} cells in {time.time()-t0:.0f}s',flush=True)
