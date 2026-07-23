
import numpy as np, json
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from scipy.spatial.distance import cdist


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
d=json.load(open(_P("phase3_out/wt_seq.json"))); wt_seq=d["wt_seq"]; positions=d["positions"]; L=len(wt_seq)
raw=np.load(_P("phase3_out/esmfold_wt_raw.npz")); ca_plddt=raw["plddt"][:,1]*100.0

f=pdb.PDBFile.read(_P("phase3_out/esmfold_wt.pdb")); arr=f.get_structure(model=1)
arr=arr[struc.filter_amino_acids(arr)]
res_ids=np.unique(arr.res_id); assert len(res_ids)==L,(len(res_ids),L)

sasa_atom=struc.sasa(arr,vdw_radii="Single")
sasa_res=struc.apply_residue_wise(arr,sasa_atom,np.nansum)
maxasa={'A':129,'R':274,'N':195,'D':193,'C':167,'E':223,'Q':225,'G':104,'H':224,'I':197,
        'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
rel_sasa=np.array([sasa_res[i]/maxasa.get(wt_seq[i],200) for i in range(L)])
burial=1.0-np.clip(rel_sasa,0,1)

try:
    sse=struc.annotate_sse(arr)
    if len(sse)==L:
        sse_helix=(sse=='a').astype(float); sse_sheet=(sse=='b').astype(float); sse_coil=(sse=='c').astype(float)
    else:
        sse_helix=np.zeros(L); sse_sheet=np.zeros(L); sse_coil=np.ones(L); print("SSE len mismatch",len(sse))
except Exception as e:
    print("SSE fallback:",e); sse_helix=np.zeros(L); sse_sheet=np.zeros(L); sse_coil=np.ones(L)

ca=arr[arr.atom_name=="CA"]; coords=ca.coord; D=cdist(coords,coords)
contacts=((D<8.0)&(D>0)).sum(1).astype(float)
for i in range(L):
    lo,hi=max(0,i-2),min(L,i+3)
    contacts[i]-=((D[i,lo:hi]<8.0)&(D[i,lo:hi]>0)).sum()

feat=np.vstack([ca_plddt,sasa_res,rel_sasa,burial,sse_helix,sse_sheet,sse_coil,contacts]).T
cols=["plddt","sasa","rel_sasa","burial","sse_helix","sse_sheet","sse_coil","contact_number"]
np.savez_compressed(_P("phase3_out/esmfold_wt_perres.npz"),feat=feat,cols=cols,positions=positions)
print("per-residue structural features:",feat.shape,cols)
print("pLDDT mean %.1f | mean rel_sasa %.2f | buried(rel<0.25) %d | helix %d sheet %d"%(
    ca_plddt.mean(),rel_sasa.mean(),int((rel_sasa<0.25).sum()),int(sse_helix.sum()),int(sse_sheet.sum())))
