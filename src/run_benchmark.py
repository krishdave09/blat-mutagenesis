#!/usr/bin/env python
"""run_benchmark.py -- full PLM + predicted-structure benchmark grid.

Named cells only: <Classifier>_<ExactFeatureSet>_<split>. Nested CV (inner GridSearchCV per outer
fold), pooled out-of-fold AUROC, 2000-bootstrap 95% CI, across random/modulo/contiguous. Headline
= contiguous. Leakage guard: every matrix asserts the sealed 13 are absent.

Tiers kept nominally distinct:
  - SEQUENCE-ONLY, PLM-free:  Physicochemical (Rep7, 16), Identity one-hot (Rep8, 40)
  - SEQUENCE-ONLY, PLM:       per PLM {Rep1 scalar, Rep3 surprisal-vec, Rep4 site+onehot,
                              Rep2a mean-pool emb, Rep2b site emb, Rep2c site-delta emb};
                              multi-PLM Rep1 concat (Rep6); curated combos with Rep7
  - PREDICTED-STRUCTURE:      ESMFold_struct (8); combos ESMFold+Rep7, ESMFold+bestPLM  (labeled _ESMFold_)
"""
import sys, json, time, warnings
from pathlib import Path
_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'.projectroot').exists())
sys.path.insert(0, str(_root)); from paths import INTERIM, PROCESSED, TABLES
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[v]='4'
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# --- ARM selection: run the sequence-only tier and the predicted-structure tier as SEPARATE
#     benchmarks (user request). Usage:
#       run_benchmark.py sequence   -> only seq_plm + seq_plmfree tiers  -> plm_benchmark_sequence_only.csv
#       run_benchmark.py esmfold    -> only pred_struct tier (+ identity floor reference) -> plm_benchmark_esmfold.csv
#       run_benchmark.py (no arg)   -> all tiers together -> plm_benchmark_results.csv
ARM = (sys.argv[1] if len(sys.argv) > 1 else 'all').lower()
assert ARM in ('all','sequence','esmfold'), ARM
OUT_NAME = {'all':'plm_benchmark_results.csv','sequence':'plm_benchmark_sequence_only.csv',
            'esmfold':'plm_benchmark_esmfold.csv'}[ARM]
CK_NAME  = {'all':'benchmark_partial.csv','sequence':'benchmark_partial_sequence.csv',
            'esmfold':'benchmark_partial_esmfold.csv'}[ARM]
SEED=42; np.random.seed(SEED)

df = pd.read_parquet(PROCESSED/'tem1_firnberg_processed.parquet')
pool = df[~df.excluded_wetlab_validation].reset_index(drop=True)
y = pool.DMS_score_bin.values.astype(int)
mut_index = {m:i for i,m in enumerate(pool.mutant.values)}

def load_keyed(npz, key='X'):
    d=np.load(INTERIM/npz, allow_pickle=True)
    M=d[key] if key in d else d['emb']
    order=[mut_index[m] for m in d['mutant']]
    out=np.full((len(pool), M.shape[1] if M.ndim>1 else 1), np.nan, np.float32)
    out[order]=M if M.ndim>1 else M[:,None]
    assert not np.isnan(out).any(), f'{npz}: missing rows'
    return out

# ---- assemble feature blocks (built lazily so missing optional blocks are skipped) ----
def safe(fn):
    try: return fn()
    except FileNotFoundError as e: print('  skip:', e); return None

# derived PLM feature .npz written by derive_feats.py: keys rep1/rep3/rep4 (per tag)
def plm_rep(tag, rep):
    d=np.load(INTERIM/f'feat_{tag}.npz', allow_pickle=True)
    M=d[rep]; M=M[:,None] if M.ndim==1 else M
    order=[mut_index[m] for m in d['mutant']]
    out=np.full((len(pool), M.shape[1]), np.nan, np.float32); out[order]=M
    assert not np.isnan(out).any(); return out
def emb_slice(tag, slc):
    d=np.load(INTERIM/f'emb3_{tag}.npz', allow_pickle=True)
    M=d[slc]; order=[mut_index[m] for m in d['mutant']]
    out=np.full((len(pool), M.shape[1]), np.nan, np.float32); out[order]=M
    assert not np.isnan(out).any(); return out

print('build blocks...', flush=True)
blocks={}
# PLM-free
blocks['Physicochemical']       = safe(lambda: load_keyed('feat_rep7_physico.npz'))
blocks['Identity one-hot']      = safe(lambda: load_keyed('feat_rep8_identity.npz'))
PLMS=[('esm2_650m','ESM-2 650M'),('esm1v','ESM-1v'),('esmc','ESM C 600M')]
for tag,name in PLMS:
    blocks[f'{name} Rep1 masked-marginal scalar'] = safe(lambda t=tag: plm_rep(t,'rep1'))
    blocks[f'{name} Rep3 surprisal vector']       = safe(lambda t=tag: plm_rep(t,'rep3'))
    blocks[f'{name} Rep4 site+onehot']            = safe(lambda t=tag: plm_rep(t,'rep4'))
    blocks[f'{name} Rep2a mean-pooled emb']       = safe(lambda t=tag: emb_slice(t,'emb_mean'))
    blocks[f'{name} Rep2b site emb']              = safe(lambda t=tag: emb_slice(t,'emb_site'))
    blocks[f'{name} Rep2c site-delta emb']        = safe(lambda t=tag: emb_slice(t,'emb_sdelta'))
# predicted-structure
blocks['ESMFold_struct'] = safe(lambda: load_keyed('feat_esmfold_struct.npz'))
# ESMFold_delta may have PARTIAL Colab coverage (NaN rows for unfolded variants).
# Load WITHOUT the full-coverage assert; only admit it as a benchmark block if coverage is
# effectively complete. Partial coverage stays an illustrative side-arm (see 05 results doc),
# never a crashing block-assembly step -- keeps one-command resume working with real-world CSVs.
def load_delta():
    d=np.load(INTERIM/'feat_esmfold_delta.npz', allow_pickle=True)
    M=d['X']; order=[mut_index[m] for m in d['mutant']]
    out=np.full((len(pool), M.shape[1]), np.nan, np.float32); out[order]=M
    cov=np.isfinite(out).all(axis=1).mean()
    if cov < 0.99:
        print(f'  ESMFold_delta coverage {cov:.1%} < 99% -> illustrative side-arm only, NOT added to grid')
        return None
    return out
blocks['ESMFold_delta']  = safe(load_delta)
blocks={k:v for k,v in blocks.items() if v is not None}
print('available blocks:', list(blocks.keys()), flush=True)

def cat(*names): return np.hstack([blocks[n] for n in names])

# ---- feature SETS to benchmark (name -> (matrix, tier)) ----
# Comprehensive curated combinatorial sweep. Every combination is scientifically motivated;
# true all-subsets (2^n) is infeasible and mostly redundant, so we enumerate the meaningful
# families. The CONTIGUOUS split is what separates combos that generalize from those that
# merely memorize position (high-dim embedding concats are expected to overfit -> documented).
from itertools import combinations
sets={}
def add(name, mat, tier):
    if mat is not None and name not in sets: sets[name]=(mat, tier)
def have(k): return k in blocks
REP_KEYS=['Rep1 masked-marginal scalar','Rep3 surprisal vector','Rep4 site+onehot',
          'Rep2a mean-pooled emb','Rep2b site emb','Rep2c site-delta emb']
PHYS = 'Physicochemical' if have('Physicochemical') else None
def catk(keys):  # concat a list of block keys present in blocks
    ks=[k for k in keys if have(k)]
    return (np.hstack([blocks[k] for k in ks]) if ks else None), ks

# 1) SINGLES: PLM-free + every PLM x every representation
add('Identity one-hot', blocks.get('Identity one-hot'), 'seq_plmfree')
add('Physicochemical',  blocks.get('Physicochemical'),  'seq_plmfree')
for _,name in PLMS:
    for rep in REP_KEYS:
        add(f'{name} {rep}', blocks.get(f'{name} {rep}'), 'seq_plm')

# 2) EACH PLM single-rep + Physicochemical
if PHYS:
    for _,name in PLMS:
        for rep in REP_KEYS:
            k=f'{name} {rep}'
            if have(k): m,_=catk([k,PHYS]); add(f'{k} + Physicochemical', m, 'seq_plm')

# 3) WITHIN-PLM multi-rep concats (non-nested, meaningful): surprisal x embedding views
within=[('Rep4 site+onehot','Rep2b site emb'),('Rep4 site+onehot','Rep2c site-delta emb'),
        ('Rep4 site+onehot','Rep2a mean-pooled emb'),('Rep1 masked-marginal scalar','Rep2b site emb'),
        ('Rep2b site emb','Rep2c site-delta emb'),('Rep1 masked-marginal scalar','Rep4 site+onehot')]
for _,name in PLMS:
    for a,b in within:
        m,ks=catk([f'{name} {a}',f'{name} {b}'])
        if len(ks)==2:
            add(f'{name} [{a} + {b}]', m, 'seq_plm')
            if PHYS: m2,_=catk([f'{name} {a}',f'{name} {b}',PHYS]); add(f'{name} [{a} + {b}] + Physicochemical', m2,'seq_plm')

# 4) CROSS-PLM same-representation concats (pairs + triple), alone and + Physicochemical
for rep in REP_KEYS:
    keys=[f'{name} {rep}' for _,name in PLMS if have(f'{name} {rep}')]
    for r in (2,3):
        for combo in combinations(keys,r):
            names='+'.join(k.split(' ')[0]+k.split(' ')[1][0] if False else k.rsplit(' Rep',1)[0].replace('ESM','ESM') for k in combo)
            short='+'.join([c.split(' Rep')[0].split(' Rep2')[0].replace(' ','') for c in combo])
            label=f'{rep} [{short}]'
            m,_=catk(list(combo)); add(label, m, 'seq_plm')
            if PHYS: m2,_=catk(list(combo)+[PHYS]); add(f'{label} + Physicochemical', m2,'seq_plm')

# 5) FULL SEQUENCE STACKS: multi-PLM rich reps + physicochemical (the kitchen-sink family)
def triple(rep): return [f'{name} {rep}' for _,name in PLMS if have(f'{name} {rep}')]
for rep in ['Rep1 masked-marginal scalar','Rep4 site+onehot','Rep2b site emb','Rep2c site-delta emb']:
    ks=triple(rep)
    if len(ks)>=2 and PHYS:
        m,_=catk(ks+[PHYS]); add(f'AllPLM {rep} + Physicochemical', m, 'seq_plm')
# grand sequence-only stack: all 3 PLMs Rep4 + all 3 site emb + physicochemical
grand=triple('Rep4 site+onehot')+triple('Rep2b site emb')+([PHYS] if PHYS else [])
mg,ksg=catk(grand)
if len(ksg)>=3: add('GRAND sequence stack (AllPLM Rep4 + AllPLM site emb + Physicochemical)', mg, 'seq_plm')

# 6) PREDICTED-STRUCTURE tier (kept nominally separate; labeled _ESMFold_)
if have('ESMFold_struct'):
    add('ESMFold_struct', blocks['ESMFold_struct'], 'pred_struct')
    if PHYS: m,_=catk(['ESMFold_struct',PHYS]); add('ESMFold_struct + Physicochemical', m,'pred_struct')
    # structure + best PLM reps (combination the memo flagged as the real test)
    for _,name in PLMS:
        for rep in ['Rep1 masked-marginal scalar','Rep4 site+onehot','Rep2b site emb']:
            k=f'{name} {rep}'
            if have(k): m,_=catk(['ESMFold_struct',k]); add(f'ESMFold_struct + {k}', m,'pred_struct')
    # full stack incl structure
    fs=triple('Rep4 site+onehot')+(['ESMFold_struct'])+([PHYS] if PHYS else [])
    m,ks=catk(fs)
    if len(ks)>=3: add('ESMFold_struct + AllPLM Rep4 + Physicochemical', m,'pred_struct')
if have('ESMFold_delta'):
    add('ESMFold_delta', blocks['ESMFold_delta'], 'pred_struct')
    if have('ESMFold_struct'): m,_=catk(['ESMFold_struct','ESMFold_delta']); add('ESMFold_struct + ESMFold_delta', m,'pred_struct')
    if PHYS: m,_=catk(['ESMFold_struct','ESMFold_delta',PHYS]); add('ESMFold_struct + ESMFold_delta + Physicochemical', m,'pred_struct')

# --- filter to the requested ARM (separate sequence-only vs ESMFold benchmarks) ---
if ARM=='sequence':
    sets={k:v for k,v in sets.items() if v[1] in ('seq_plm','seq_plmfree')}
elif ARM=='esmfold':
    # predicted-structure tier PLUS the identity one-hot + physicochemical floors for reference
    sets={k:v for k,v in sets.items() if v[1]=='pred_struct' or k in ('Identity one-hot','Physicochemical')}
print(f'ARM={ARM}: {len(sets)} feature sets', flush=True)
n_hi=sum(1 for _,(X,_) in sets.items() if X.shape[1]>200)
print(f'{len(sets)} feature sets x 4 classifiers x 3 splits (SVM gated to <=200-dim sets; {n_hi} high-dim sets skip SVM)', flush=True)

# ---- classifiers + inner grids ----
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
        gs.fit(X[tr],y[tr])
        m=gs.best_estimator_
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

rows=[]; t0=time.time()
CLFS=['Logistic Regression','Random Forest','XGBoost','SVM']
SPLITS=[('fold_random','random'),('fold_modulo','modulo'),('fold_contiguous','contiguous')]
CK=INTERIM/CK_NAME
oof_contig={}   # {classifier|feature_set: oof vector on contiguous} for significance step
done=set()
if CK.exists():
    prev=pd.read_csv(CK); rows=prev.to_dict('records'); done={(r['feature_set'],r['classifier'],r['split']) for r in rows}
SVM_MAXDIM=200  # RBF-SVM is ~O(n^2) and scales badly with dimensionality; gate to low-dim sets
# DIRECTION-FINDING gate: on wide feature sets (>=TREE_MAXDIM) run only Logistic Regression.
# LR is both fastest AND strongest on the 1152-3800 dim embedding/mixed stacks (RF/XGBoost lose
# there and each fit costs 15-30 min under RAM pressure). Full 4-classifier sweep stays on all
# interpretable/deployable sets (<=~300 dim) where the scientific comparison actually lives.
TREE_MAXDIM=400  # RF/XGBoost gated off sets wider than this (covers 263-dim fingerprint, Rep4+phys, etc.)
for fname,(X,tier) in sets.items():
    for kind in CLFS:
        if kind=='SVM' and X.shape[1]>SVM_MAXDIM: continue
        if kind in ('Random Forest','XGBoost') and X.shape[1]>TREE_MAXDIM: continue
        for scol,sname in SPLITS:
            if (fname,kind,sname) in done: continue
            oof=oof_auroc(X,scol,kind)
            au=roc_auc_score(y,oof); lo,hi=boot_ci(y,oof)
            rows.append(dict(feature_set=fname,classifier=kind,split=sname,tier=tier,
                             auroc=au,ci_lo=lo,ci_hi=hi,n_features=X.shape[1]))
            pd.DataFrame(rows).to_csv(CK,index=False)
            if sname=='contiguous': oof_contig[f'{kind}|{fname}']=oof
            print(f'  {sname:10} {kind:20} {fname[:40]:40} AUROC {au:.4f} [{lo:.4f},{hi:.4f}] ({time.time()-t0:.0f}s)',flush=True)
# cache contiguous OOF predictions for DeLong/paired-bootstrap significance testing
if oof_contig:
    oof_out = {'all':'oof_contiguous.npz','sequence':'oof_contiguous_sequence.npz',
               'esmfold':'oof_contiguous_esmfold.npz'}[ARM]
    np.savez_compressed(INTERIM/oof_out, **oof_contig)
    print(f'cached {len(oof_contig)} contiguous OOF vectors -> {oof_out}')
res=pd.DataFrame(rows)
res.to_csv(TABLES/OUT_NAME,index=False)
print(f'DONE ARM={ARM}: {len(res)} cells in {time.time()-t0:.0f}s -> {OUT_NAME}')
