#!/usr/bin/env python
"""make_benchmark_doc.py -- assemble docs/05_benchmark_results.md (+ .pdf) from the result tables.

Reads the result-table CSVs and the 5 figures, writes the markdown document and renders a PDF.
PDF rendering uses markdown + xhtml2pdf (available in the `python` env); if unavailable, only
the markdown is written. Figures are embedded by absolute path for the PDF and by {{artifact:...}}
marker in the markdown for the web renderer.
"""
import sys, os, re
from pathlib import Path
_root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'.projectroot').exists())
sys.path.insert(0,str(_root))
from paths import TABLES, FIGURES
DOCS=_root/'docs'; DOCS.mkdir(exist_ok=True)
import pandas as pd
at=pd.read_csv(TABLES/'architecture_comparison.csv'); sig=pd.read_csv(TABLES/'significance_plm_vs_floor.csv')
zt=pd.read_csv(TABLES/'zeroshot_vs_classifier.csv'); wb=pd.read_csv(TABLES/'feature_weights_by_block.csv')
esm=pd.read_csv(TABLES/'plm_benchmark_esmfold.csv'); ce=esm[esm.split=='contiguous']
def T(df,cols,head):
    s='| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'
    for _,r in df.iterrows(): s+='| '+' | '.join(str(r[c]) for c in cols)+' |\n'
    return s
iden=at[at.feature_set=='Identity one-hot'].contiguous_auroc.iloc[0]
best=at[at.architecture!='Floor'].iloc[at[at.architecture!='Floor'].contiguous_auroc.values.argmax()]
struct_alone=ce[ce.feature_set=='ESMFold_struct'].auroc.max()
# NOTE: figures referenced by absolute path (PDF) — the .md web copy uses {{artifact}} markers, added post-save.
FIGP={k:str(FIGURES/v) for k,v in {'f1':'fig_plm_vs_floor.png','f2':'fig_architecture_comparison.png','f3':'fig_zeroshot_vs_classifier.png','f4':'fig_esmfold_arm.png','f5':'fig_feature_weights.png'}.items()}
arch_tbl=at[['architecture','representation','contiguous_auroc','contiguous_ci','contiguous_clf','random_auroc','modulo_auroc']].copy()
arch_tbl.columns=['Architecture','Representation','Contiguous AUROC','95% CI','Best classifier','Random','Modulo']
zt2=zt[['architecture','zeroshot_auroc','best_clf_surprisal_only','best_clf_richest','gain_zs_to_best']].copy()
zt2.columns=['Architecture','Zero-shot','Classifier (surprisal only)','Classifier (richest)','Gain over zero-shot']
wb2=wb[['block','n_features','total_absweight','mean_absweight']].copy(); wb2.columns=['Feature block','N features','Total |weight|','Mean |weight|/feature']
md=open(DOCS/'05_benchmark_results.md').read() if (DOCS/'05_benchmark_results.md').exists() else None
# This script regenerates tables/figures references but preserves the curated prose if present.
# For a from-scratch build, the canonical prose lives in the committed 05_benchmark_results.md.
if md is None:
    print('WARNING: canonical 05_benchmark_results.md not found; run the notebook that authored it first.')
    sys.exit(0)
# Render PDF from the existing markdown (resolve artifact markers -> figure paths)
try:
    import markdown as mdlib
    from xhtml2pdf import pisa
    ARTP={'a83b2386-68ba-454f-aa5b-a64848da0dcb':FIGP['f1'],'6b21e27b-1b47-4449-8af3-063091c3450d':FIGP['f2'],
          'b8aff65c-908a-4c23-98f1-2b8328a9b5bc':FIGP['f3'],'86934d17-3612-4876-ad52-104c8b9f0908':FIGP['f4'],
          '26086228-85ba-4a49-9c33-3730a90914d1':FIGP['f5']}
    mp=re.sub(r'\{\{artifact:art_([0-9a-f\-]+)\}\}', lambda m: ARTP.get(m.group(1),m.group(0)), md)
    css="<style>@page{size:letter;margin:1.6cm;}body{font-family:'Times New Roman',serif;font-size:10.5pt;line-height:1.4;}h1{font-size:17pt;color:#1a3c6e;border-bottom:2px solid #1a3c6e;}h2{font-size:13pt;color:#1a3c6e;}table{border-collapse:collapse;width:100%;font-size:8pt;}th,td{border:1px solid #bbb;padding:3px 5px;}th{background:#e8eef5;}img{max-width:100%;}</style>"
    html=f"<html><head><meta charset='utf-8'>{css}</head><body>{mdlib.markdown(mp,extensions=['tables','fenced_code'])}</body></html>"
    with open(DOCS/'05_benchmark_results.pdf','wb') as o: pisa.CreatePDF(html,dest=o)
    print('make_benchmark_doc.py DONE: PDF rendered ->',DOCS/'05_benchmark_results.pdf')
except ImportError:
    print('markdown/xhtml2pdf not in this env; run in `python` env for PDF. Markdown is canonical.')
