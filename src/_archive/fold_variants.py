
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
from transformers import EsmForProteinFolding, AutoTokenizer
import biotite.structure as struc, biotite.structure.io.pdb as pdb
from scipy.spatial.distance import cdist
from io import StringIO

d=json.load(open(_P("phase3_out/wt_seq.json"))); wt=d["wt_seq"]; positions=d["positions"]; L=len(wt)
pos2idx={p:i for i,p in enumerate(positions)}
sub=pd.read_csv(_P("phase3_out/delta_subset.csv"))
CKPT=_P("phase3_out/delta_folds.csv")
maxasa={'A':129,'R':274,'N':195,'D':193,'C':167,'E':223,'Q':225,'G':104,'H':224,'I':197,
        'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}

def feats_at(pdb_str, idx, aa):
    f=pdb.PDBFile.read(StringIO(pdb_str)); arr=f.get_structure(model=1)
    arr=arr[struc.filter_amino_acids(arr)]
    sasa_atom=struc.sasa(arr,vdw_radii="Single")
    sasa_res=struc.apply_residue_wise(arr,sasa_atom,np.nansum)
    rel=sasa_res[idx]/maxasa.get(aa,200)
    ca=arr[arr.atom_name=="CA"]; Dm=cdist(ca.coord,ca.coord)
    cn=((Dm[idx]<8.0)&(Dm[idx]>0)).sum()
    lo,hi=max(0,idx-2),min(L,idx+3); cn-=((Dm[idx,lo:hi]<8.0)&(Dm[idx,lo:hi]>0)).sum()
    return sasa_res[idx], rel, cn

print("loading ESMFold",flush=True); t0=time.time()
tok=AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model=EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",low_cpu_mem_usage=True).eval()
model.esm=model.esm.float(); model.trunk.set_chunk_size(64)
print(f"loaded {time.time()-t0:.0f}s",flush=True)

# WT reference at each position from prior run
wtd=np.load(_P("phase3_out/esmfold_wt_perres.npz"),allow_pickle=True); wtf=wtd["feat"]; wtcols=list(wtd["cols"])
wt_plddt=wtf[:,wtcols.index("plddt")]; wt_sasa=wtf[:,wtcols.index("sasa")]; wt_rel=wtf[:,wtcols.index("rel_sasa")]; wt_cn=wtf[:,wtcols.index("contact_number")]

done=set()
if os.path.exists(CKPT):
    prev=pd.read_csv(CKPT); done=set(prev.mutant); print(f"resume: {len(done)} done",flush=True)
rows=[] if not done else pd.read_csv(CKPT).to_dict("records")

t1=time.time(); cnt=0
for _,r in sub.iterrows():
    if r.mutant in done: continue
    p=int(r.position_linear); idx=pos2idx[p]; m=r.mut_aa
    s=list(wt); s[idx]=m; mseq="".join(s)
    inp=tok([mseq],return_tensors="pt",add_special_tokens=False)["input_ids"]
    with torch.no_grad(): out=model(inp)
    plddt_v=out["plddt"][0,idx,1].item()*100.0
    pdb_str=model.output_to_pdb(out)[0]
    sasa_v,rel_v,cn_v=feats_at(pdb_str,idx,m)
    rows.append(dict(mutant=r.mutant,position_linear=p,mut_aa=m,DMS_score=r.DMS_score,DMS_score_bin=r.DMS_score_bin,
        v_plddt=plddt_v,v_sasa=sasa_v,v_rel_sasa=rel_v,v_contact=cn_v,
        d_plddt=plddt_v-wt_plddt[idx], d_sasa=sasa_v-wt_sasa[idx], d_rel_sasa=rel_v-wt_rel[idx], d_contact=cn_v-wt_cn[idx]))
    pd.DataFrame(rows).to_csv(CKPT,index=False); cnt+=1
    el=time.time()-t1; rate=cnt/max(el,1e-9)
    print(f"{cnt} {r.mutant} plddt {plddt_v:.0f} dplddt {plddt_v-wt_plddt[idx]:+.1f} | {el/60:.1f}min ETA {(len(sub)-len(done)-cnt)/max(rate,1e-9)/60:.0f}min",flush=True)
print(f"DELTA FOLD DONE {cnt} variants {(time.time()-t0)/60:.0f}min",flush=True)
