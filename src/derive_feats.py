
# Derive Rep 1/3/4 from a masked-marginal logprob matrix (L x 20) for a given PLM tag.
import numpy as np, pandas as pd, json, sys

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
TAG=sys.argv[1]  # e.g. esm2_650m
d=np.load(_P(f"phase3_out/mm_{TAG}.npz"),allow_pickle=True)
logprobs=d["logprobs"]; AAS=list(str(d["aas"])); aa_idx={a:i for i,a in enumerate(AAS)}
wt_seq=str(d["wt_seq"]); L=len(wt_seq)
assert not np.isnan(logprobs).any(), "incomplete logprob matrix"

df=pd.read_parquet(_P("phase2_out/tem1_firnberg_processed.parquet"))
pool=df[~df.excluded_wetlab_validation].reset_index(drop=True)
positions=sorted(df.drop_duplicates('position_linear').set_index('position_linear').index)
pos2idx={p:i for i,p in enumerate(positions)}

n=len(pool)
rep1=np.zeros(n,np.float32)                 # masked-marginal scalar: lp(mut)-lp(wt)
rep3=np.zeros((n,20),np.float32)            # per-position surprisal vector (all 20 logprobs at the site)
rep4=np.zeros((n,40),np.float32)            # rep3 (20) + one-hot substitution (20 mut)
for k,(p,w,m) in enumerate(zip(pool.position_linear,pool.wt_aa,pool.mut_aa)):
    i=pos2idx[p]; lp=logprobs[i]
    rep1[k]=lp[aa_idx[m]]-lp[aa_idx[w]]
    rep3[k]=lp
    rep4[k,:20]=lp; rep4[k,20+aa_idx[m]]=1.0
np.savez_compressed(_P(f"phase3_out/feat_{TAG}.npz"),rep1=rep1,rep3=rep3,rep4=rep4,
                    mutant=pool.mutant.values)
print(f"{TAG}: rep1{rep1.shape} rep3{rep3.shape} rep4{rep4.shape}")
print(f"  rep1 range [{rep1.min():.3f},{rep1.max():.3f}] mean {rep1.mean():.3f}")
# quick sanity: rep1 (surprisal) should correlate with DMS (negative = destabilizing predicted)
from scipy.stats import spearmanr
rho=spearmanr(rep1,pool.DMS_score).correlation
print(f"  Spearman(rep1 scalar, DMS_score) = {rho:.3f}  (zero-shot signal)")
