
import numpy as np, pandas as pd, json, time, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

# --- path portability (code-only reproducibility) ---
import sys as _sys
from pathlib import Path as _Path
_root = next(p for p in [_Path.cwd(), *_Path.cwd().parents] if (p/'.projectroot').exists())
_sys.path.insert(0, str(_root))
from paths import INTERIM as _INTERIM, PROCESSED as _PROCESSED, FIGURES as _FIGURES, TABLES as _TABLES
_INTERIM.mkdir(parents=True, exist_ok=True)
def _P(rel):
    # map legacy workspace paths -> project paths
    if rel.startswith("phase2_out/"): return str(_PROCESSED / rel.split("/",1)[1])
    if rel.startswith("phase3_out/"): return str(_INTERIM  / rel.split("/",1)[1])
    return rel
# --- end path portability ---
warnings.filterwarnings("ignore")
SEED=42; np.random.seed(SEED)

df = pd.read_parquet(_P("phase2_out/tem1_firnberg_processed.parquet"))
pool = df[~df.excluded_wetlab_validation].reset_index(drop=True)

AAS=list("ACDEFGHIKLMNPQRSTVWY"); aa_idx={a:i for i,a in enumerate(AAS)}
def rep8(f):
    X=np.zeros((len(f),40),np.float32)
    for i,(w,m) in enumerate(zip(f.wt_aa,f.mut_aa)): X[i,aa_idx[w]]=1; X[i,20+aa_idx[m]]=1
    return X
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
VOL={'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0}
CHG={a:0 for a in AAS}; CHG.update({'D':-1,'E':-1,'K':1,'R':1,'H':0.1})
MW={'A':89.1,'R':174.2,'N':132.1,'D':133.1,'C':121.2,'Q':146.2,'E':147.1,'G':75.1,'H':155.2,'I':131.2,'L':131.2,'K':146.2,'M':149.2,'F':165.2,'P':115.1,'S':105.1,'T':119.1,'W':204.2,'Y':181.2,'V':117.1}
PI={'A':6.0,'R':10.8,'N':5.4,'D':2.8,'C':5.1,'Q':5.7,'E':3.2,'G':6.0,'H':7.6,'I':6.0,'L':6.0,'K':9.7,'M':5.7,'F':5.5,'P':6.3,'S':5.7,'T':5.6,'W':5.9,'Y':5.7,'V':6.0}
def rep7(f):
    cols=[];
    for sc in (KD,VOL,CHG,MW,PI):
        wt=f.wt_aa.map(sc).values.astype(np.float32); mu=f.mut_aa.map(sc).values.astype(np.float32)
        cols+=[wt,mu,mu-wt]
    cols.append((f.mut_aa.map(CHG)-f.wt_aa.map(CHG)).abs().values.astype(np.float32))
    return np.vstack(cols).T

FEATS={"Identity one-hot":rep8(pool),"Physicochemical":rep7(pool)}
y=pool.DMS_score_bin.values.astype(int)

def make_clf(name):
    if name=="Logistic Regression":
        return Pipeline([("sc",StandardScaler()),("clf",LogisticRegression(max_iter=2000,random_state=SEED))]),{"clf__C":[0.01,0.1,1,10]}
    if name=="Random Forest":
        return RandomForestClassifier(random_state=SEED,n_jobs=1),{"n_estimators":[300],"max_depth":[None,8,16],"min_samples_leaf":[1,5]}
    if name=="XGBoost":
        return XGBClassifier(random_state=SEED,n_jobs=1,eval_metric="logloss",verbosity=0,tree_method="hist"),{"n_estimators":[300],"max_depth":[3,6],"learning_rate":[0.05,0.1]}
    if name=="SVM":
        return Pipeline([("sc",StandardScaler()),("clf",SVC(probability=True,random_state=SEED))]),{"clf__C":[0.1,1,10],"clf__gamma":["scale"]}

def boot_ci(yt,yp,n=2000,seed=SEED):
    rng=np.random.default_rng(seed); idx=np.arange(len(yt)); aucs=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if len(np.unique(yt[b]))<2: continue
        aucs.append(roc_auc_score(yt[b],yp[b]))
    return float(np.percentile(aucs,2.5)),float(np.percentile(aucs,97.5))

CLASSIFIERS=["Logistic Regression","Random Forest","XGBoost","SVM"]
SPLITS={"random":"fold_random","modulo":"fold_modulo","contiguous":"fold_contiguous"}
results=[]; oof_store={}
t0=time.time()
for fname,X in FEATS.items():
    for cname in CLASSIFIERS:
        for sname,scol in SPLITS.items():
            folds=pool[scol].values
            oof=np.zeros(len(y))*np.nan
            for k in sorted(np.unique(folds)):
                tr=folds!=k; te=folds==k
                est,grid=make_clf(cname)
                inner=StratifiedKFold(3,shuffle=True,random_state=SEED)
                gs=GridSearchCV(est,grid,scoring="roc_auc",cv=inner,n_jobs=1)
                gs.fit(X[tr],y[tr])
                oof[te]=gs.predict_proba(X[te])[:,1]
            auc=roc_auc_score(y,oof); lo,hi=boot_ci(y,oof)
            results.append({"feature":fname,"classifier":cname,"split":sname,
                            "auroc":round(auc,4),"ci_lo":round(lo,4),"ci_hi":round(hi,4),"n":len(y)})
            oof_store[f"{fname}|{cname}|{sname}"]=oof
            print(f"{fname:16} {cname:20} {sname:11} AUROC={auc:.4f} [{lo:.4f},{hi:.4f}]  ({time.time()-t0:.0f}s)")
res=pd.DataFrame(results)
res.to_csv(_P("phase2_out/baseline_results.csv"),index=False)
np.savez_compressed(_P("phase2_out/baseline_oof.npz"),y=y,**{k.replace("|","__"):v for k,v in oof_store.items()})
print("DONE",time.time()-t0,"s")
