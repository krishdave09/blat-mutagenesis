# ESMFold per-variant structural feature extraction — TEM-1 beta-lactamase v2
# Run on Google Colab with GPU (Runtime > Change runtime type > GPU; A100/L4 ideal).
# Upload the two companion files first: colab_variant_manifest.csv, colab_wt_spec.json
# Outputs: esmfold_variant_features.csv  (keyed by `mutant`, drops into the v2 benchmark)

# ============ CELL 1: install ============
!pip -q install transformers accelerate biotite scipy pandas

# ============ CELL 2: load model + inputs ============
import torch, numpy as np, pandas as pd, json, time
from transformers import EsmForProteinFolding, AutoTokenizer
import biotite.structure as struc, biotite.structure.io.pdb as pdb
from scipy.spatial.distance import cdist
from io import StringIO
from google.colab import files

# upload colab_variant_manifest.csv and colab_wt_spec.json when prompted
up = files.upload()
spec = json.load(open("colab_wt_spec.json")); wt = spec["wt_seq"]; positions = spec["positions"]
pos2idx = {p:i for i,p in enumerate(positions)}; L=len(wt)
man = pd.read_csv("colab_variant_manifest.csv")
print(f"WT len {L}, variants {len(man)}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
model = model.to(dev).eval()
if dev=="cuda": model.esm = model.esm.half()   # fp16 on GPU for speed/memory
model.trunk.set_chunk_size(64)
print("device:", dev)

# ============ CELL 3: feature helpers ============
MAXASA={'A':129,'R':274,'N':195,'D':193,'C':167,'E':223,'Q':225,'G':104,'H':224,'I':197,
        'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
def struct_feats(pdb_str, idx, aa):
    f=pdb.PDBFile.read(StringIO(pdb_str)); arr=f.get_structure(model=1)
    arr=arr[struc.filter_amino_acids(arr)]
    sasa=struc.apply_residue_wise(arr, struc.sasa(arr,vdw_radii="Single"), np.nansum)
    rel=sasa[idx]/MAXASA.get(aa,200)
    ca=arr[arr.atom_name=="CA"]; Dm=cdist(ca.coord,ca.coord)
    cn=((Dm[idx]<8.0)&(Dm[idx]>0)).sum()
    lo,hi=max(0,idx-2),min(L,idx+3); cn-=((Dm[idx,lo:hi]<8.0)&(Dm[idx,lo:hi]>0)).sum()
    return float(sasa[idx]), float(rel), 1.0-min(max(rel,0),1), float(cn)

def fold_seq(seq):
    inp=tok([seq],return_tensors="pt",add_special_tokens=False)["input_ids"].to(dev)
    with torch.no_grad(): out=model(inp)
    plddt=out["plddt"][0,:,1].cpu().numpy()*100.0   # per-residue CA pLDDT (0-100)
    return out, plddt

# ============ CELL 4: fold WT reference ============
wt_out, wt_plddt = fold_seq(wt)
wt_pdb = model.output_to_pdb(wt_out)[0]
open("esmfold_wt_colab.pdb","w").write(wt_pdb)
wt_feat = {}
for p in positions:
    i=pos2idx[p]; s,rel,bur,cn = struct_feats(wt_pdb, i, wt[i])
    wt_feat[p] = dict(plddt=wt_plddt[i], sasa=s, rel_sasa=rel, burial=bur, contact=cn)
print("WT folded, mean pLDDT %.1f" % wt_plddt.mean())

# ============ CELL 5: fold all variants (checkpointed) ============
import os
CK="esmfold_variant_features.csv"
rows = pd.read_csv(CK).to_dict("records") if os.path.exists(CK) else []
done = {r["mutant"] for r in rows}
t0=time.time()
for n,(_,r) in enumerate(man.iterrows()):
    if r.mutant in done: continue
    i=pos2idx[int(r.position_linear)]; m=r.mut_aa
    s=list(wt); s[i]=m; out,plddt=fold_seq("".join(s))
    pdbs=model.output_to_pdb(out)[0]
    sasa,rel,bur,cn=struct_feats(pdbs,i,m)
    w=wt_feat[int(r.position_linear)]
    rows.append(dict(mutant=r.mutant,
        v_plddt=plddt[i], v_sasa=sasa, v_rel_sasa=rel, v_burial=bur, v_contact=cn,
        d_plddt=plddt[i]-w["plddt"], d_sasa=sasa-w["sasa"], d_rel_sasa=rel-w["rel_sasa"],
        d_burial=bur-w["burial"], d_contact=cn-w["contact"]))
    if len(rows)%25==0 or n==len(man)-1:
        pd.DataFrame(rows).to_csv(CK,index=False)
        el=time.time()-t0; rate=(len(rows)-len(done))/max(el,1e-9)
        print(f"{len(rows)}/{len(man)} {el/60:.1f}min {rate:.2f}/s ETA {(len(man)-len(rows))/max(rate,1e-9)/60:.0f}min")
pd.DataFrame(rows).to_csv(CK,index=False)
print("DONE ->", CK)

# ============ CELL 6: download ============
files.download("esmfold_variant_features.csv")
files.download("esmfold_wt_colab.pdb")
