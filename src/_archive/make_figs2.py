
import numpy as np, pandas as pd, os
import matplotlib as mpl, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc as sk_auc
from sklearn.calibration import calibration_curve


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
def apply_style():
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
        "axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":9,"axes.labelsize":8,
        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"figure.dpi":110,"savefig.dpi":300,"axes.linewidth":0.8})
def frame(ax): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
apply_style()

# Distinct hue per classifier (project palette: blues/purples/greens/blacks/dark-pinks)
CLF_COLOR={"Logistic Regression":"#2c5aa0","Random Forest":"#2a7f62","XGBoost":"#6a4c93","SVM":"#a63a6b"}
GREY="#9aa0a6"
CLFS=["Logistic Regression","Random Forest","XGBoost","SVM"]
FEATS=["Identity one-hot","Physicochemical"]
SPLITS=["random","modulo","contiguous"]

res=pd.read_csv(_P("phase2_out/baseline_results.csv"))
d=np.load(_P("phase2_out/baseline_oof.npz")); y=d["y"]
def oof(feat,clf,split): return d[f"{feat}__{clf}__{split}"]
FIGD="/Users/kdave2/Beta-Lactamase Mutagenesis/Beta Lactam ML v2/results/figures"; os.makedirs(FIGD,exist_ok=True); os.makedirs("fig_out",exist_ok=True)
def save(fig,name):
    for dst in (FIGD,"fig_out"): fig.savefig(f"{dst}/{name}",dpi=300,bbox_inches="tight")
    plt.close(fig)

# ---- FIG 1: ROC — contiguous only, 2 panels (per feature), 4 classifier lines ----
fig,axes=plt.subplots(1,2,figsize=(9,4.4))
for ax,feat in zip(axes,FEATS):
    for clf in CLFS:
        p=oof(feat,clf,"contiguous"); fpr,tpr,_=roc_curve(y,p); a=sk_auc(fpr,tpr)
        ax.plot(fpr,tpr,color=CLF_COLOR[clf],lw=1.9,label=f"{clf} ({a:.3f})")
    ax.plot([0,1],[0,1],ls="--",color=GREY,lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(feat,fontsize=8.5); ax.legend(frameon=False,fontsize=6.5,loc="lower right",title="AUROC"); frame(ax)
fig.suptitle("ROC — contiguous split, by classifier",fontsize=9.5); fig.tight_layout(); save(fig,"fig1_roc.png"); print("fig1")

# ---- FIG 2: PR — contiguous only, 2 panels, 4 classifier lines ----
fig,axes=plt.subplots(1,2,figsize=(9,4.4)); br=y.mean()
for ax,feat in zip(axes,FEATS):
    for clf in CLFS:
        p=oof(feat,clf,"contiguous"); pr,rc,_=precision_recall_curve(y,p); a=sk_auc(rc,pr)
        ax.plot(rc,pr,color=CLF_COLOR[clf],lw=1.9,label=f"{clf} ({a:.3f})")
    ax.axhline(br,ls="--",color=GREY,lw=1,label=f"random ({br:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0,1.02)
    ax.set_title(feat,fontsize=8.5); ax.legend(frameon=False,fontsize=6.5,loc="lower left",title="Avg precision"); frame(ax)
fig.suptitle("Precision-Recall — contiguous split, by classifier",fontsize=9.5); fig.tight_layout(); save(fig,"fig2_precision_recall.png"); print("fig2")

# ---- FIG 3: Confusion matrix — 2x2 grid, one per classifier (contiguous, Physicochemical) ----
FEAT_CM="Physicochemical"
fig,axes=plt.subplots(2,2,figsize=(7.2,6.8))
for ax,clf in zip(axes.ravel(),CLFS):
    p=oof(FEAT_CM,clf,"contiguous"); yhat=(p>=0.5).astype(int); cm=confusion_matrix(y,yhat)
    im=ax.imshow(cm,cmap="Blues",vmin=0,vmax=cm.max())
    for i in range(2):
        for j in range(2):
            v=cm[i,j]; ax.text(j,i,f"{v}\n({v/cm.sum():.0%})",ha="center",va="center",
                fontsize=8,color="white" if v>cm.max()*0.5 else "#222")
    ax.set_xticks([0,1]); ax.set_xticklabels(["non-func","func"],fontsize=6.5)
    ax.set_yticks([0,1]); ax.set_yticklabels(["non-func","func"],fontsize=6.5)
    ax.set_xlabel("Predicted",fontsize=7); ax.set_ylabel("Actual (DMS)",fontsize=7)
    ax.set_title(clf,fontsize=8,color=CLF_COLOR[clf])
fig.suptitle(f"Confusion matrices — {FEAT_CM}, contiguous split, threshold 0.5",fontsize=9.5)
fig.tight_layout(); save(fig,"fig3_confusion_matrix.png"); print("fig3")

# ---- FIG 4: Gains & Lift — contiguous, 2x2 (rows: gains/lift; cols: feature), 4 model lines ----
fig,axes=plt.subplots(2,2,figsize=(9,7.4))
for col,feat in enumerate(FEATS):
    axg=axes[0,col]; axl=axes[1,col]
    for clf in CLFS:
        p=oof(feat,clf,"contiguous"); order=np.argsort(-p); ys=y[order]
        cum=np.cumsum(ys)/ys.sum(); frac=np.arange(1,len(ys)+1)/len(ys); lift=cum/frac
        axg.plot(frac,cum,color=CLF_COLOR[clf],lw=1.8,label=clf)
        axl.plot(frac,lift,color=CLF_COLOR[clf],lw=1.8,label=clf)
    axg.plot([0,1],[0,1],ls="--",color=GREY,lw=1); axg.set_title(f"Cumulative gains — {feat}",fontsize=8)
    axg.set_xlabel("Fraction ranked by score"); axg.set_ylabel("Frac functional found"); axg.legend(frameon=False,fontsize=6); frame(axg)
    axl.axhline(1,ls="--",color=GREY,lw=1); axl.set_title(f"Lift — {feat}",fontsize=8)
    axl.set_xlabel("Fraction ranked by score"); axl.set_ylabel("Lift over random"); axl.legend(frameon=False,fontsize=6); frame(axl)
fig.suptitle("Gains & Lift — contiguous split, by classifier",fontsize=9.5); fig.tight_layout(); save(fig,"fig4_gains_lift.png"); print("fig4")


# ---- FIG 5: Learning curve — contiguous, 2 panels (per feature), 4 model CV lines ----
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
AAS=list("ACDEFGHIKLMNPQRSTVWY"); aa_idx={a:i for i,a in enumerate(AAS)}
dfp=pd.read_parquet(_P("phase2_out/tem1_firnberg_processed.parquet")); pool=dfp[~dfp.excluded_wetlab_validation].reset_index(drop=True)
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
    cols=[]
    for sc in (KD,VOL,CHG,MW,PI):
        wt=f.wt_aa.map(sc).values.astype(np.float32); mu=f.mut_aa.map(sc).values.astype(np.float32); cols+=[wt,mu,mu-wt]
    cols.append((f.mut_aa.map(CHG)-f.wt_aa.map(CHG)).abs().values.astype(np.float32)); return np.vstack(cols).T
FEATFN={"Identity one-hot":rep8,"Physicochemical":rep7}; yv=pool.DMS_score_bin.values.astype(int)
def mk(name):
    if name=="Logistic Regression": return Pipeline([("sc",StandardScaler()),("clf",LogisticRegression(max_iter=2000,C=1))])
    if name=="Random Forest": return RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=1)
    if name=="XGBoost": return XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,n_jobs=1,eval_metric="logloss",verbosity=0,tree_method="hist")
    if name=="SVM": return Pipeline([("sc",StandardScaler()),("clf",SVC(C=1,probability=False))])
fig,axes=plt.subplots(1,2,figsize=(9,4.4))
for ax,feat in zip(axes,FEATS):
    X=FEATFN[feat](pool)
    for clf in CLFS:
        sizes,tr,te=learning_curve(mk(clf),X,yv,cv=StratifiedKFold(5,shuffle=True,random_state=42),
            scoring="roc_auc",train_sizes=np.linspace(0.15,1.0,5),n_jobs=1)
        ax.plot(sizes,te.mean(1),"o-",color=CLF_COLOR[clf],lw=1.7,ms=3.5,label=clf)
    ax.set_xlabel("Training examples"); ax.set_ylabel("Cross-validation AUROC")
    ax.set_title(feat,fontsize=8.5); ax.legend(frameon=False,fontsize=6.5); frame(ax)
fig.suptitle("Learning curve (cross-validation AUROC) — by classifier",fontsize=9.5); fig.tight_layout(); save(fig,"fig5_learning_curve.png"); print("fig5")

# ---- FIG 6: Calibration — contiguous, 2 panels (per feature), 4 model lines ----
fig,axes=plt.subplots(1,2,figsize=(9,4.6))
for ax,feat in zip(axes,FEATS):
    for clf in CLFS:
        p=oof(feat,clf,"contiguous"); fp,mp=calibration_curve(y,p,n_bins=10,strategy="quantile")
        ax.plot(mp,fp,"o-",color=CLF_COLOR[clf],lw=1.7,ms=4,label=clf)
    ax.plot([0,1],[0,1],ls="--",color=GREY,lw=1,label="perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed freq functional")
    ax.set_title(feat,fontsize=8.5); ax.legend(frameon=False,fontsize=6.5,loc="upper left"); frame(ax)
fig.suptitle("Calibration — contiguous split, by classifier",fontsize=9.5); fig.tight_layout(); save(fig,"fig6_calibration.png"); print("fig6")

# ---- FIG 7: AUROC bars — colored by model, 2 panels (per feature), x=split ----
fig,axes=plt.subplots(1,2,figsize=(9.5,4.6),sharey=True)
xpos=np.arange(len(SPLITS)); w=0.19
for ax,feat in zip(axes,FEATS):
    for ci,clf in enumerate(CLFS):
        vals=[res[(res.feature==feat)&(res.classifier==clf)&(res.split==s)].auroc.values[0] for s in SPLITS]
        los=[res[(res.feature==feat)&(res.classifier==clf)&(res.split==s)].ci_lo.values[0] for s in SPLITS]
        his=[res[(res.feature==feat)&(res.classifier==clf)&(res.split==s)].ci_hi.values[0] for s in SPLITS]
        err=[np.array(vals)-np.array(los),np.array(his)-np.array(vals)]
        ax.bar(xpos+(ci-1.5)*w,vals,w,yerr=err,capsize=2,color=CLF_COLOR[clf],
               edgecolor="#222",linewidth=0.4,label=clf)
    ax.axhline(0.5,ls=":",color=GREY,lw=1)
    ax.set_xticks(xpos); ax.set_xticklabels(SPLITS); ax.set_ylim(0.45,0.82)
    ax.set_title(feat,fontsize=8.5); ax.set_ylabel("AUROC (OOF, 95% CI)"); frame(ax)
axes[0].legend(frameon=False,fontsize=6.5,ncol=2,loc="upper right")
fig.suptitle("Sequence-only baseline AUROC — by classifier and split (contiguous = honest)",fontsize=9.5)
fig.tight_layout(); save(fig,"fig7_auroc_comparison.png"); print("fig7")
print("ALL DONE")
