
import torch, numpy as np, json, time, os, sys

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
MODEL=os.environ.get("ESMC_MODEL","esmc_600m")
TAG="esmc"; OUT=_P(f"phase3_out/mm_{TAG}.npz"); CKPT=_P(f"phase3_out/mm_{TAG}_ckpt.npy")
d=json.load(open(_P("phase3_out/wt_seq.json"))); wt_seq=d["wt_seq"]; L=len(wt_seq)
AAS=list("ACDEFGHIKLMNPQRSTVWY")

print(f"[esmc] loading {MODEL}",flush=True); t0=time.time()
model=ESMC.from_pretrained(MODEL).eval()
tok=model.tokenizer
mask_tok=tok.mask_token   # '<mask>'
aa_ids=[tok.convert_tokens_to_ids(a) for a in AAS]
print(f"[esmc] loaded {time.time()-t0:.0f}s mask={mask_tok}",flush=True)

logprobs=np.full((L,20),np.nan,np.float32); start=0
if os.path.exists(CKPT):
    s=np.load(CKPT); logprobs[:s.shape[0]]=s
    start=int(np.where(~np.isnan(logprobs[:,0]))[0].max())+1 if (~np.isnan(logprobs[:,0])).any() else 0
    print(f"[esmc] resumed {start}",flush=True)

t1=time.time()
for i in range(start,L):
    s=list(wt_seq); s[i]=mask_tok
    # ESMProtein sequence is a plain string; mask token embedded as '<mask>'
    prot=ESMProtein(sequence="".join(s))
    enc=model.encode(prot)
    with torch.no_grad():
        lo=model.logits(enc,LogitsConfig(sequence=True))
    lg=lo.logits.sequence[0]   # (Lseq, vocab) incl special tokens
    # position i in seq -> account for BOS offset (ESMC adds BOS): find via +1
    row=torch.log_softmax(lg[i+1],dim=-1)
    logprobs[i]=np.array([row[a].item() for a in aa_ids],np.float32)
    if (i+1)%20==0 or i==L-1:
        np.save(CKPT,logprobs[:i+1]); el=time.time()-t1; r=(i+1-start)/max(el,1e-9)
        print(f"[esmc] {i+1}/{L} {el:.0f}s {r:.2f}/s ETA {(L-i-1)/max(r,1e-9):.0f}s",flush=True)
np.savez_compressed(OUT,logprobs=logprobs,aas="".join(AAS),wt_seq=wt_seq)
print(f"[esmc] DONE {time.time()-t0:.0f}s",flush=True)
