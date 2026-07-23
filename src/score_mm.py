
import torch, esm, numpy as np, json, time, os, sys

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
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","4")))
MODEL_NAME = sys.argv[1] if len(sys.argv)>1 else "esm2_t33_650M_UR50D"
LAYER = int(sys.argv[2]) if len(sys.argv)>2 else 33
TAG = sys.argv[3] if len(sys.argv)>3 else "esm2_650m"
OUT = _P(f"phase3_out/mm_{TAG}.npz")
CKPT = _P(f"phase3_out/mm_{TAG}_ckpt.npy")

d = json.load(open(_P("phase3_out/wt_seq.json"))); wt_seq=d["wt_seq"]; L=len(wt_seq)
AAS=list("ACDEFGHIKLMNPQRSTVWY")

print(f"[{TAG}] loading {MODEL_NAME} ...",flush=True); t0=time.time()
model,alphabet = getattr(esm.pretrained, MODEL_NAME)()
model=model.eval()
bc=alphabet.get_batch_converter()
mask_idx=alphabet.mask_idx
aa_tok=[alphabet.get_idx(a) for a in AAS]   # vocab indices for the 20 AAs
print(f"[{TAG}] loaded in {time.time()-t0:.0f}s; L={L}, mask_idx={mask_idx}",flush=True)

# resume from checkpoint if present
logprobs=np.full((L,20),np.nan,dtype=np.float32)
start=0
if os.path.exists(CKPT):
    saved=np.load(CKPT)
    logprobs[:saved.shape[0]]=saved
    start=int(np.where(~np.isnan(logprobs[:,0]))[0].max())+1 if (~np.isnan(logprobs[:,0])).any() else 0
    print(f"[{TAG}] resumed from position index {start}",flush=True)

_,_,base_toks = bc([("wt",wt_seq)])   # (1, L+2)
t1=time.time()
for i in range(start,L):
    toks=base_toks.clone(); toks[0,i+1]=mask_idx   # +1 for BOS
    with torch.no_grad():
        out=model(toks,repr_layers=[])
    lg=torch.log_softmax(out["logits"][0,i+1],dim=-1)   # (vocab,)
    logprobs[i]=lg[aa_tok].numpy()
    if (i+1)%20==0 or i==L-1:
        np.save(CKPT,logprobs[:i+1])
        el=time.time()-t1; rate=(i+1-start)/max(el,1e-9)
        print(f"[{TAG}] pos {i+1}/{L}  {el:.0f}s  {rate:.2f} pos/s  ETA {(L-i-1)/max(rate,1e-9):.0f}s",flush=True)

np.savez_compressed(OUT, logprobs=logprobs, aas="".join(AAS), wt_seq=wt_seq)
print(f"[{TAG}] DONE total {time.time()-t0:.0f}s -> {OUT}",flush=True)
