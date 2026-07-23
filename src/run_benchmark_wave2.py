#!/usr/bin/env python
"""run_benchmark_wave2.py -- run the 38 WAVE-2 feature sets and APPEND them into the two arm tables.

Reuses the primary harness (run_benchmark.py) classifier/OOF/CI functions by importing them, and the
wave-2 set definitions from run_benchmark_wave2_defs.py. Sequence-tier sets append to
plm_benchmark_sequence_only.csv; pred_struct sets append to plm_benchmark_esmfold.csv. Per-cell
checkpointing to benchmark_partial_sequence.csv / _esmfold.csv (same files the primary run uses),
so wave-2 cells are added idempotently and a resumed run skips finished cells.
"""
import sys, time, warnings, os, importlib.util
from pathlib import Path
_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'.projectroot').exists())
sys.path.insert(0, str(_root)); sys.path.insert(0, str(_root/'src'))
from paths import INTERIM, PROCESSED, TABLES
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[v]='4'

SEED=42
df = pd.read_parquet(PROCESSED/'tem1_firnberg_processed.parquet')
pool = df[~df.excluded_wetlab_validation].reset_index(drop=True)
y = pool.DMS_score_bin.values.astype(int)

# --- import the primary harness's classifier functions (reuse, don't duplicate) ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def make_clf(kind):
    if kind=='Logistic Regression':
        return Pipeline([('sc',StandardScaler()),('clf',LogisticRegression(max_iter=2000,random_state=SEED))]), {'clf__C':[0.01,0.1,1,10]}
    if kind=='Random Forest':
        return RandomForestClassifier(random_state=SEED,n_jobs=1), {'n_estimators':[300],'max_depth':[None,8,16],'min_samples_leaf':[1,3]}
    if kind=='XGBoost':
        return XGBClassifier(random_state=SEED,n_jobs=1,eval_metric='logloss',verbosity=0), {'n_estimators':[300],'max_depth':[3,6],'learning_rate':[0.05,0.1]}
    if kind=='SVM':
        return Pipeline([('sc',StandardScaler()),('clf',SVC(probability=True,random_state=SEED))]), {'clf__C':[0.1,1,10],'clf__gamma':['scale']}

def oof_auroc(X, splitcol, kind):
    folds=df.loc[pool.index, splitcol].values
    oof=np.full(len(pool), np.nan)
    for f in sorted(set(folds)):
        tr,te=(folds!=f),(folds==f)
        base,grid=make_clf(kind)
        gs=GridSearchCV(base,grid,scoring='roc_auc',cv=StratifiedKFold(3,shuffle=True,random_state=SEED),n_jobs=1)
        gs.fit(X[tr],y[tr]); m=gs.best_estimator_
        p=m.predict_proba(X[te])[:,1] if hasattr(m,'predict_proba') else m.decision_function(X[te])
        oof[te]=p
    return oof

def boot_ci(yt,pr,n=2000):
    rng=np.random.default_rng(SEED); N=len(yt); s=[]
    for _ in range(n):
        idx=rng.integers(0,N,N)
        if len(set(yt[idx]))<2: continue
        s.append(roc_auc_score(yt[idx],pr[idx]))
    return float(np.percentile(s,2.5)), float(np.percentile(s,97.5))

# --- load wave-2 set definitions ---
spec=importlib.util.spec_from_file_location('w2defs', str(_root/'src'/'run_benchmark_wave2_defs.py'))
w2=importlib.util.module_from_spec(spec)
# the defs module prints its enumeration on import; suppress
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(w2)
sets=w2.sets   # {name: (matrix, tier)}
print(f'WAVE-2: {len(sets)} feature sets to benchmark', flush=True)

CLFS=['Logistic Regression','Random Forest','XGBoost','SVM']
SPLITS=[('fold_random','random'),('fold_modulo','modulo'),('fold_contiguous','contiguous')]
SVM_MAXDIM=200
TREE_MAXDIM=400  # direction-finding: RF/XGBoost only on <=400-dim sets; LR-only on wide stacks

# route each set to its arm's checkpoint + final table
ARM_OF={'seq_plm':'sequence','seq_plmfree':'sequence','pred_struct':'esmfold'}
CK={'sequence':INTERIM/'benchmark_partial_sequence.csv','esmfold':INTERIM/'benchmark_partial_esmfold.csv'}
OUT={'sequence':TABLES/'plm_benchmark_sequence_only.csv','esmfold':TABLES/'plm_benchmark_esmfold.csv'}
OOF={'sequence':INTERIM/'oof_contiguous_sequence.npz','esmfold':INTERIM/'oof_contiguous_esmfold.npz'}

# load existing rows per arm (so we append, and skip already-done cells)
rows={a:[] for a in ('sequence','esmfold')}; done={a:set() for a in ('sequence','esmfold')}
for a in rows:
    if CK[a].exists():
        prev=pd.read_csv(CK[a]); rows[a]=prev.to_dict('records')
        done[a]={(r['feature_set'],r['classifier'],r['split']) for r in rows[a]}

t0=time.time()
oof_contig={a:{} for a in ('sequence','esmfold')}   # cache contiguous OOF as we go (no recompute)
for fname,(X,tier) in sets.items():
    arm=ARM_OF[tier]
    for kind in CLFS:
        if kind=='SVM' and X.shape[1]>SVM_MAXDIM: continue
        if kind in ('Random Forest','XGBoost') and X.shape[1]>TREE_MAXDIM: continue
        for scol,sname in SPLITS:
            if (fname,kind,sname) in done[arm]: continue
            oof=oof_auroc(X,scol,kind)
            au=roc_auc_score(y,oof); lo,hi=boot_ci(y,oof)
            rows[arm].append(dict(feature_set=fname,classifier=kind,split=sname,tier=tier,
                                  auroc=au,ci_lo=lo,ci_hi=hi,n_features=X.shape[1]))
            pd.DataFrame(rows[arm]).to_csv(CK[arm],index=False)
            if sname=='contiguous': oof_contig[arm][f'{kind}|{fname}']=oof
            print(f'  [{arm:8}] {sname:10} {kind:20} {fname[:44]:44} AUROC {au:.4f} ({time.time()-t0:.0f}s)',flush=True)

# write final tables + merge cached contiguous OOF vectors for significance
for a in rows:
    if rows[a]: pd.DataFrame(rows[a]).to_csv(OUT[a],index=False)
    oofd={}
    if OOF[a].exists():
        z=np.load(OOF[a]); oofd={k:z[k] for k in z.files}
    oofd.update(oof_contig[a])
    if oofd: np.savez_compressed(OOF[a], **oofd)
print(f'WAVE-2 DONE: sequence {len(rows["sequence"])} rows, esmfold {len(rows["esmfold"])} rows in {time.time()-t0:.0f}s',flush=True)
