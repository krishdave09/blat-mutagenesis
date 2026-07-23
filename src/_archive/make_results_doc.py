#!/usr/bin/env python
"""make_results_doc.py -- assemble docs/05_benchmark_results.md (+ .pdf).

Reads results/tables/plm_benchmark_results.csv and significance.csv, writes the benchmark
results document: contiguous-split thesis verdict, feature-representation sub-study, and the
sequence-only vs predicted-structure comparison. No model locked/saved; sealed holdout NOT scored.
"""
import sys
from pathlib import Path
_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'.projectroot').exists())
sys.path.insert(0, str(_root)); from paths import TABLES
import pandas as pd, numpy as np
DOCS=_root/'docs'; DOCS.mkdir(exist_ok=True)

res=pd.read_csv(TABLES/'plm_benchmark_results.csv')
sig=pd.read_csv(TABLES/'significance.csv') if (TABLES/'significance.csv').exists() else pd.DataFrame()
cont=res[res.split=='contiguous'].copy()

def fmt(r): return f"{r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
floor=cont[cont.feature_set=='Identity one-hot'].sort_values('auroc',ascending=False).iloc[0]
seq=cont[cont.tier.isin(['seq_plm','seq_plmfree'])].sort_values('auroc',ascending=False)
strc=cont[cont.tier=='pred_struct'].sort_values('auroc',ascending=False)
top_seq=seq.iloc[0]

md=[]
md.append("# TEM-1 β-lactamase v2 — sequence-only PLM + predicted-structure benchmark: results\n")
md.append("_Body text: Times New Roman. Headline metric = contiguous (region-holdout) split. "
          "Sealed 13-variant wet-lab holdout NOT scored. No final model locked or saved._\n")
md.append("## 1. Thesis verdict (contiguous split)\n")
md.append(f"**Identity one-hot floor** (patterns, not biology): {fmt(floor)} "
          f"({floor.classifier}).\n")
md.append(f"**Best sequence-only model**: {top_seq.feature_set} ({top_seq.classifier}): {fmt(top_seq)}.\n")
delta=top_seq.auroc-floor.auroc
md.append(f"Gain over the identity floor: **{delta:+.3f} AUROC**.\n")
if len(sig):
    row=sig[sig.comparison.str.contains('sequence-only vs Identity')]
    if len(row):
        r=row.iloc[0]
        md.append(f"Significance (DeLong): p={r.delong_p:.2e}; paired bootstrap Δ={r.boot_delta:+.3f} "
                  f"[{r.boot_ci_lo:+.3f}, {r.boot_ci_hi:+.3f}], p={r.boot_p:.2e}.\n")
md.append("## 2. Feature-representation sub-study (contiguous)\n")
md.append(seq.head(15)[['feature_set','classifier','auroc','ci_lo','ci_hi']]
          .to_markdown(index=False))
md.append("\n## 3. Sequence-only vs predicted-structure (contiguous)\n")
if len(strc):
    md.append(f"Best predicted-structure model: {strc.iloc[0].feature_set} "
              f"({strc.iloc[0].classifier}): {fmt(strc.iloc[0])}.\n")
    md.append(strc.head(10)[['feature_set','classifier','auroc','ci_lo','ci_hi']].to_markdown(index=False))
else:
    md.append("_(predicted-structure tier pending)_\n")
md.append("\n## 4. Leakage tax (random vs modulo vs contiguous)\n")
piv=res.pivot_table(index=['tier','feature_set','classifier'],columns='split',values='auroc')
md.append(piv.round(3).to_markdown())
md.append("\n## 5. Full results table\n")
md.append(res.sort_values(['split','auroc'],ascending=[True,False])
          [['tier','feature_set','classifier','split','auroc','ci_lo','ci_hi','n_features']]
          .to_markdown(index=False))

text="\n".join(md)
(DOCS/'05_benchmark_results.md').write_text(text)
print("wrote docs/05_benchmark_results.md")
# PDF via xhtml2pdf (Times New Roman body) -- same toolchain as Phase-0 docs
try:
    import markdown as mdlib
    from xhtml2pdf import pisa
    html=('<html><head><style>body{font-family:"Times New Roman",serif;font-size:11pt;margin:1in;}'
          'table{border-collapse:collapse;font-size:9pt;}td,th{border:1px solid #999;padding:3px;}'
          'h1,h2{color:#1a3c6e;}</style></head><body>'
          + mdlib.markdown(text, extensions=['tables']) + '</body></html>')
    with open(DOCS/'05_benchmark_results.pdf','wb') as f:
        pisa.CreatePDF(html, dest=f)
    print("wrote docs/05_benchmark_results.pdf")
except Exception as e:
    print("PDF step skipped:", e)
