
import torch, numpy as np, pandas as pd, json, time, os

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
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
TAG="esmc"; OUT=_P(f"phase3_out/emb3_{TAG}.npz"); CKPT=_P(f"phase3_out/emb3_{TAG}_ckpt.npz")
d=json.load(open(_P("phase3_out/wt_seq.json"))); wt_seq=d["wt_seq"]; positions=d["positions"]
pos2idx={p:i for i,p in enumerate(positions)}
df=pd.read_parquet(_P("phase2_out/tem1_firnberg_processed.parquet"))
pool=df[~df.excluded_wetlab_validation].reset_index(drop=True)
mut_idx=np.array([pos2idx[p] for p in pool.position_linear])
def mutseq(p,m):
    s=list(wt_seq); s[pos2idx[p]]=m; return "".join(s)
seqs=[mutseq(p,m) for p,m in zip(pool.position_linear,pool.mut_aa)]; n=len(seqs)

print("[esmc-emb3] loading esmc_600m",flush=True); t0=time.time()
model=ESMC.from_pretrained("esmc_600m").eval()
enc=model.encode(ESMProtein(sequence=wt_seq))
with torch.no_grad():
    lo=model.logits(enc,LogitsConfig(sequence=True,return_embeddings=True))
wt_res=lo.embeddings[0,1:1+len(wt_seq)].numpy(); D=wt_res.shape[1]
print(f"[esmc-emb3] loaded {time.time()-t0:.0f}s D={D}",flush=True)

emb_mean=np.full((n,D),np.nan,np.float32); emb_site=np.full((n,D),np.nan,np.float32); emb_sdelta=np.full((n,D),np.nan,np.float32)
start=0
if os.path.exists(CKPT):
    s=np.load(CKPT); k=s["emb_mean"].shape[0]
    emb_mean[:k]=s["emb_mean"]; emb_site[:k]=s["emb_site"]; emb_sdelta[:k]=s["emb_sdelta"]
    done=(~np.isnan(emb_mean[:,0])); start=int(np.where(done)[0].max())+1 if done.any() else 0
    print(f"[esmc-emb3] resume {start}",flush=True)
t1=time.time()
for j in range(start,n):
    enc=model.encode(ESMProtein(sequence=seqs[j]))
    with torch.no_grad():
        lo=model.logits(enc,LogitsConfig(sequence=True,return_embeddings=True))
    res=lo.embeddings[0,1:1+len(seqs[j])].numpy(); i=mut_idx[j]
    emb_mean[j]=res.mean(0); emb_site[j]=res[i]; emb_sdelta[j]=res[i]-wt_res[i]
    if (j+1)%50==0 or j==n-1:
        np.savez_compressed(CKPT,emb_mean=emb_mean[:j+1],emb_site=emb_site[:j+1],emb_sdelta=emb_sdelta[:j+1])
        el=time.time()-t1; r=(j+1-start)/max(el,1e-9)
        print(f"[esmc-emb3] {j+1}/{n} {el:.0f}s {r:.1f}/s ETA {(n-j-1)/max(r,1e-9):.0f}s",flush=True)
np.savez_compressed(OUT,emb_mean=emb_mean,emb_site=emb_site,emb_sdelta=emb_sdelta,mutant=pool.mutant.values)
print(f"[esmc-emb3] DONE {time.time()-t0:.0f}s",flush=True)
