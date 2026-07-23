
# Rep 2 (three slices): mean-pooled, site, site-delta. From the SAME forward passes.
import torch, esm, numpy as np, pandas as pd, json, time, os, sys

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
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","6")))
MODEL_NAME=sys.argv[1]; LAYER=int(sys.argv[2]); TAG=sys.argv[3]
BATCH=int(os.environ.get("EMB_BATCH","8"))
OUT=_P(f"phase3_out/emb3_{TAG}.npz"); CKPT=_P(f"phase3_out/emb3_{TAG}_ckpt.npz")
d=json.load(open(_P("phase3_out/wt_seq.json"))); wt_seq=d["wt_seq"]; positions=d["positions"]
pos2idx={p:i for i,p in enumerate(positions)}
df=pd.read_parquet(_P("phase2_out/tem1_firnberg_processed.parquet"))
pool=df[~df.excluded_wetlab_validation].reset_index(drop=True)
mut_idx=np.array([pos2idx[p] for p in pool.position_linear])  # 0-based residue idx per variant
def mutseq(p,m):
    s=list(wt_seq); s[pos2idx[p]]=m; return "".join(s)
seqs=[mutseq(p,m) for p,m in zip(pool.position_linear,pool.mut_aa)]; n=len(seqs)

print(f"[{TAG}] loading {MODEL_NAME}",flush=True); t0=time.time()
model,alphabet=getattr(esm.pretrained,MODEL_NAME)(); model=model.eval()
bc=alphabet.get_batch_converter()
# WT per-position embeddings (one pass) for site-delta reference
_,_,wt_toks=bc([("wt",wt_seq)])
with torch.no_grad():
    wt_out=model(wt_toks,repr_layers=[LAYER])
wt_res=wt_out["representations"][LAYER][0,1:1+len(wt_seq)].numpy()  # (L,D)
D=wt_res.shape[1]; print(f"[{TAG}] loaded {time.time()-t0:.0f}s D={D}",flush=True)

emb_mean=np.full((n,D),np.nan,np.float32)
emb_site=np.full((n,D),np.nan,np.float32)
emb_sdelta=np.full((n,D),np.nan,np.float32)
start=0
if os.path.exists(CKPT):
    s=np.load(CKPT); k=s["emb_mean"].shape[0]
    emb_mean[:k]=s["emb_mean"]; emb_site[:k]=s["emb_site"]; emb_sdelta[:k]=s["emb_sdelta"]
    done=(~np.isnan(emb_mean[:,0])); start=int(np.where(done)[0].max())+1 if done.any() else 0
    print(f"[{TAG}] resume {start}",flush=True)
t1=time.time()
for b in range(start,n,BATCH):
    chunk=[(str(j),seqs[j]) for j in range(b,min(b+BATCH,n))]
    _,_,toks=bc(chunk)
    with torch.no_grad():
        rep=model(toks,repr_layers=[LAYER])["representations"][LAYER]  # (B,Lmax+2,D)
    for bi,(j,_) in enumerate(chunk):
        jj=int(j); Lj=len(seqs[jj]); i=mut_idx[jj]
        res=rep[bi,1:1+Lj].numpy()
        emb_mean[jj]=res.mean(0)
        emb_site[jj]=res[i]
        emb_sdelta[jj]=res[i]-wt_res[i]
    if (b//BATCH)%10==0 or b+BATCH>=n:
        k=min(b+BATCH,n)
        np.savez_compressed(CKPT,emb_mean=emb_mean[:k],emb_site=emb_site[:k],emb_sdelta=emb_sdelta[:k])
        el=time.time()-t1; r=(k-start)/max(el,1e-9)
        print(f"[{TAG}] {k}/{n} {el:.0f}s {r:.1f}/s ETA {(n-k)/max(r,1e-9):.0f}s",flush=True)
np.savez_compressed(OUT,emb_mean=emb_mean,emb_site=emb_site,emb_sdelta=emb_sdelta,mutant=pool.mutant.values)
print(f"[{TAG}] DONE {time.time()-t0:.0f}s",flush=True)
