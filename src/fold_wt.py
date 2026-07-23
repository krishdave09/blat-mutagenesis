
# Fold WT TEM-1 once with HF ESMFold (CPU), extract per-residue structural features.
import torch, numpy as np, json, time, os

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
from transformers import EsmForProteinFolding, AutoTokenizer

d=json.load(open(_P("phase3_out/wt_seq.json"))); wt=d["wt_seq"]; L=len(wt)
print(f"folding WT L={L}",flush=True); t0=time.time()
tok=AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model=EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",low_cpu_mem_usage=True)
model=model.eval()
# CPU: use float32; chunk to limit memory
model.esm = model.esm.float()
model.trunk.set_chunk_size(64)
print(f"loaded {time.time()-t0:.0f}s",flush=True)

inp=tok([wt],return_tensors="pt",add_special_tokens=False)["input_ids"]
t1=time.time()
with torch.no_grad():
    out=model(inp)
print(f"folded {time.time()-t1:.0f}s",flush=True)
# outputs: positions (coords), plddt, etc.
plddt=out["plddt"][0].numpy()   # (L, 37?) per-atom or (L,) — inspect
pos=out["positions"]            # (n_recycle?, 1, L, 14/37, 3)
print("plddt shape",plddt.shape,"positions shape",tuple(pos.shape),flush=True)
np.savez_compressed(_P("phase3_out/esmfold_wt_raw.npz"),
    plddt=plddt, atom_positions=out["positions"][-1,0].numpy(),
    atom_mask=out["atom37_atom_exists"][0].numpy() if "atom37_atom_exists" in out else np.array([]),
    aatype=out["aatype"][0].numpy() if "aatype" in out else np.array([]))
# also dump PDB for SASA/DSSP downstream
pdb=model.output_to_pdb(out)[0]
open(_P("phase3_out/esmfold_wt.pdb"),"w").write(pdb)
print(f"WT fold DONE {time.time()-t0:.0f}s; keys={list(out.keys())}",flush=True)
