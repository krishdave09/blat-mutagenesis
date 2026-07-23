#!/usr/bin/env python
"""make_benchmark_figs.py -- the 5 document figures, exactly as in docs/05_benchmark_results.md.

Reads results/tables/{architecture_comparison,esmfold_arm_summary,zeroshot_vs_classifier,
feature_weights_by_block,feature_weights_top20}.csv and writes results/figures/fig_*.png.

Figures 1/2/4 read the classifier PRE-SPECIFIED by build_architecture_comparison.py /
build_esmfold_summary.py (chosen via modulo-split AUROC, not the contiguous score being
plotted) -- NOT idxmax() on the raw per-classifier benchmark tables, which would
reintroduce the same selection bias those generator scripts were written to fix.
"""
import sys, os
from pathlib import Path
_root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'.projectroot').exists())
sys.path.insert(0,str(_root))
from paths import TABLES, FIGURES
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
os.makedirs(FIGURES,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})
CLF_COL={'Logistic Regression':'#2c5aa0','Random Forest':'#2a7f62','XGBoost':'#6a4c93','SVM':'#a63a6b'}
ARCH_COL={'ESM-2 650M':'#1b6ca8','ESM-1v':'#e08214','ESM C 600M':'#762a83'}

# architecture_comparison.csv: pre-specified classifier per feature_set (bias-free, modulo-selected)
at=pd.read_csv(TABLES/'architecture_comparison.csv')
IDEN=at[at.feature_set=='Identity one-hot'].contiguous_auroc.iloc[0]

# ---- Fig 1: PLM vs floor ----
order=[('Identity one-hot','Identity one-hot'),('Physicochemical','Physicochemical'),
 ('ESM-1v Rep1 masked-marginal scalar','ESM-1v surprisal'),('ESM-2 650M Rep1 masked-marginal scalar','ESM-2 surprisal'),
 ('ESM C 600M Rep1 masked-marginal scalar','ESM C surprisal'),('ESM-2 650M Rep3 surprisal vector','ESM-2 surprisal-20'),
 ('ESM-2 650M Rep2b site emb','ESM-2 site-emb'),('ESM C 600M Rep2a mean-pooled emb','ESM C mean-emb'),
 ('ESM-1v Rep4 site+onehot','ESM-1v Rep4'),('ESM-2 650M Rep4 site+onehot','ESM-2 Rep4'),('ESM C 600M Rep4 site+onehot','ESM C Rep4')]
rows=[]
for fs,lab in order:
    sub=at[at.feature_set==fs]
    if len(sub):
        r=sub.iloc[0]
        rows.append((lab,r.contiguous_auroc,float(r.contiguous_ci.strip('[]').split(',')[0]),float(r.contiguous_ci.strip('[]').split(',')[1]),r.selected_classifier,fs in('Identity one-hot','Physicochemical')))
d=pd.DataFrame(rows,columns=['lab','a','lo','hi','clf','floor'])
fig,ax=plt.subplots(figsize=(7.6,4.8)); yy=np.arange(len(d))[::-1]
for i,(_,r) in zip(yy,d.iterrows()):
    col=CLF_COL[r.clf]
    ax.errorbar(r.a,i,xerr=[[r.a-r.lo],[r.hi-r.a]],fmt='o',ms=7,color=col,ecolor=col,elinewidth=1.5,capsize=3,
                mfc='white' if r.floor else col,mec=col,mew=1.5,alpha=0.55 if r.floor else 1.0)
ax.axvline(IDEN,ls='--',lw=1.2,color='#888',zorder=0)
ax.text(IDEN-0.004,0.15,f'identity floor {IDEN:.2f}',fontsize=6,color='#666',ha='right',va='bottom',rotation=90)
ax.set_yticks(yy); ax.set_yticklabels(d.lab); ax.set_xlabel('AUROC (contiguous split, out-of-fold)'); ax.set_xlim(0.65,0.94); ax.margins(y=0.04)
leg=[Line2D([0],[0],marker='o',color='w',mfc=CLF_COL[k],label=k,ms=7) for k in CLF_COL]+[Line2D([0],[0],marker='o',color='w',mfc='white',mec='#888',label='floor (open)',ms=7,mew=1.5)]
ax.legend(handles=leg,loc='center left',bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=6.5)
ax.set_title('AUROC by feature representation (contiguous split)',fontsize=8,loc='left')
fig.tight_layout(); fig.savefig(FIGURES/'fig_plm_vs_floor.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# ---- Fig 2: architecture ----
reps=[('Rep1 masked-marginal scalar','surprisal\n(Rep1)'),('Rep3 surprisal vector','surprisal-20\n(Rep3)'),
 ('Rep4 site+onehot','Rep4\n(site+onehot)'),('Rep2a mean-pooled emb','mean-emb\n(Rep2a)'),('Rep2b site emb','site-emb\n(Rep2b)')]
ARCH=['ESM-2 650M','ESM-1v','ESM C 600M']
fig,ax=plt.subplots(figsize=(7.6,4.2)); x=np.arange(len(reps)); w=0.26
for j,a in enumerate(ARCH):
    vals=[at[at.feature_set==f'{a} {rk}'].contiguous_auroc.iloc[0] if len(at[at.feature_set==f'{a} {rk}']) else np.nan for rk,_ in reps]
    ax.bar(x+(j-1)*w,vals,w,label=a,color=ARCH_COL[a],edgecolor='white',lw=0.5)
ax.axhline(IDEN,ls='--',lw=1.2,color='#888',zorder=0)
ax.text(len(reps)-0.5,IDEN+0.003,f'identity floor {IDEN:.2f}',fontsize=6,color='#666',ha='right',va='bottom')
ax.set_xticks(x); ax.set_xticklabels([l for _,l in reps],fontsize=6.5); ax.set_ylabel('AUROC (contiguous, pre-specified classifier)'); ax.set_ylim(0.68,0.92)
ax.legend(frameon=False,loc='upper left',fontsize=6.5,ncol=3); ax.margins(x=0.02)
ax.set_title('AUROC by representation and architecture (contiguous split)',fontsize=7.5,loc='left')
fig.tight_layout(); fig.savefig(FIGURES/'fig_architecture_comparison.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# ---- Fig 3: zero-shot ----
zt=pd.read_csv(TABLES/'zeroshot_vs_classifier.csv')
fig,ax=plt.subplots(figsize=(7.0,4.2)); x=np.arange(len(zt)); w=0.26
for j,(col,hatch,al) in enumerate([('zeroshot_auroc','//',0.45),('best_clf_surprisal_only','',0.7),('best_clf_richest','',1.0)]):
    ax.bar(x+(j-1)*w,zt[col],w,color=[ARCH_COL[a] for a in zt.architecture],alpha=al,edgecolor='white',lw=0.5,hatch=hatch)
ax.axhline(IDEN,ls='--',lw=1.2,color='#888',zorder=0)
ax.text(len(zt)-0.5,IDEN-0.008,f'identity floor {IDEN:.2f}',fontsize=6,color='#666',ha='right',va='top')
ax.set_xticks(x); ax.set_xticklabels(zt.architecture); ax.set_ylabel('AUROC (contiguous)'); ax.set_ylim(0.70,0.92)
ax.legend(handles=[Patch(fc='#777',alpha=0.45,hatch='//',label='zero-shot (no training)'),Patch(fc='#777',alpha=0.7,label='classifier on surprisal only'),Patch(fc='#777',alpha=1.0,label='classifier on richest features')],frameon=False,loc='upper left',fontsize=6)
ax.set_title('Zero-shot vs. trained-classifier AUROC by architecture',fontsize=7.5,loc='left')
fig.tight_layout(); fig.savefig(FIGURES/'fig_zeroshot_vs_classifier.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# ---- Fig 4: ESMFold ----
efs=pd.read_csv(TABLES/'esmfold_arm_summary.csv')
def esh(fs): return (fs.replace('ESMFold_struct','Struct').replace(' masked-marginal scalar',' surprisal').replace(' site+onehot',' Rep4').replace(' site emb',' site-emb').replace('600M ','').replace('650M ','').replace('ESM-2 Rep4 Rep4','ESM-2 Rep4').replace('ESM C Rep4 Rep4','ESM C Rep4'))
rows=[]
for _,r in efs.iterrows():
    fs=r.feature_set
    lo,hi=[float(v) for v in r.contiguous_ci.strip('[]').split(',')]
    rows.append((esh(fs),r.contiguous_auroc,lo,hi,r.selected_classifier,fs in('Identity one-hot','Physicochemical'),fs=='ESMFold_struct'))
de=pd.DataFrame(rows,columns=['lab','a','lo','hi','clf','floor','struct'])
fig,ax=plt.subplots(figsize=(7.8,4.6)); yy=np.arange(len(de))[::-1]
for i,(_,r) in zip(yy,de.iterrows()):
    col=CLF_COL[r.clf]; op=r.floor or r.struct
    ax.errorbar(r.a,i,xerr=[[r.a-r.lo],[r.hi-r.a]],fmt='o',ms=7,color=col,ecolor=col,elinewidth=1.5,capsize=3,mfc='white' if op else col,mec=col,mew=1.5,alpha=0.6 if r.floor else 1.0)
ax.axvline(IDEN,ls='--',lw=1.2,color='#888',zorder=0)
ax.text(IDEN-0.004,0.1,f'identity floor {IDEN:.2f}',fontsize=6,color='#666',ha='right',va='bottom',rotation=90)
ax.set_yticks(yy); ax.set_yticklabels(de.lab,fontsize=6.5); ax.set_xlabel('AUROC (contiguous, out-of-fold)'); ax.set_xlim(0.65,0.94); ax.margins(y=0.06)
leg=[Line2D([0],[0],marker='o',color='w',mfc=CLF_COL[k],label=k,ms=6) for k in CLF_COL]+[Line2D([0],[0],marker='o',color='w',mfc='white',mec='#888',label='floor / structure-only (open)',ms=6,mew=1.5)]
ax.legend(handles=leg,loc='center left',bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=6)
ax.set_title('AUROC by feature representation, predicted-structure arm (contiguous split)',fontsize=7,loc='left')
fig.tight_layout(); fig.savefig(FIGURES/'fig_esmfold_arm.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# ---- Fig 5: feature weights (unaffected by classifier-selection fix -- single all-features LR) ----
wb=pd.read_csv(TABLES/'feature_weights_by_block.csv'); top=pd.read_csv(TABLES/'feature_weights_top20.csv')
def fam(b):
    if 'surprisal' in b and 'Rep3' not in b and 'Rep4' not in b: return 'surprisal'
    if 'site_emb' in b: return 'embedding'
    if 'Physico' in b: return 'physchem'
    if 'ESMFold' in b: return 'structure'
    if 'Identity' in b: return 'identity'
    return 'other'
FC={'surprisal':'#2166ac','embedding':'#762a83','physchem':'#1a9850','structure':'#d6604d','identity':'#888','other':'#bbb'}
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.4,4.6))
wb2=wb.sort_values('mean_absweight'); yy=np.arange(len(wb2))
ax1.barh(yy,wb2.mean_absweight,color=[FC[fam(b)] for b in wb2.block],edgecolor='white',lw=0.4)
ax1.set_yticks(yy); ax1.set_yticklabels([b.replace('_',' ') for b in wb2.block],fontsize=6); ax1.set_xlabel('mean |LR coef| per feature'); ax1.margins(y=0.02); ax1.set_title('Per-feature signal density',fontsize=7.5,loc='left')
t12=top.head(12).iloc[::-1]
def fl(f): return f.replace('ESMC_Rep1_surprisal[0]','ESM C surprisal').replace('ESM2_Rep1_surprisal[0]','ESM-2 surprisal').replace('ESM1v_Rep1_surprisal[0]','ESM-1v surprisal').replace('ESMC_site_emb[','ESM C emb dim ').replace(']','')
yy2=np.arange(len(t12))
ax2.barh(yy2,t12.abscoef,color=[FC[fam(b)] for b in t12.block],edgecolor='white',lw=0.4)
ax2.set_yticks(yy2); ax2.set_yticklabels([fl(f) for f in t12.feature],fontsize=6); ax2.set_xlabel('|LR coef|'); ax2.margins(y=0.02); ax2.set_title('Top individual features',fontsize=7.5,loc='left')
fig.legend(handles=[Patch(fc=FC[k],label=k) for k in ['surprisal','embedding','physchem','structure','identity']],frameon=False,loc='lower center',ncol=5,fontsize=6.5,bbox_to_anchor=(0.5,-0.02))
fig.suptitle('Feature importance by block and by individual feature (LR coefficients)',fontsize=8)
fig.tight_layout(rect=[0,0.03,1,0.97]); fig.savefig(FIGURES/'fig_feature_weights.png',dpi=300,bbox_inches='tight'); plt.close(fig)
print('make_benchmark_figs.py DONE: 5 figures ->',FIGURES)
