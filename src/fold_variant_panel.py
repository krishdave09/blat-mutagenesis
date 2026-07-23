
# Fold 9 real single-mutant sequences with HF ESMFold (CPU), extract per-residue
# structural features + global deltas vs the WT fold (already computed; reference
# feature array loaded from esmfold_wt_perres.npz, reference structure from esmfold_wt.pdb).
import torch, numpy as np, json, time, os, sys
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","6")))
from transformers import EsmForProteinFolding, AutoTokenizer
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from scipy.spatial.distance import cdist

d = json.load(open("wt_seq.json")); wt_seq = d["wt_seq"]; positions = d["positions"]; L = len(wt_seq)
variants = json.load(open("variant_panel.json"))  # list of {mutant, position_linear, mut_aa}
offset = positions[0]

print(f"loading ESMFold, L={L}, n_variants={len(variants)}", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
model = model.eval()
model.esm = model.esm.float()
model.trunk.set_chunk_size(64)
print(f"model loaded {time.time()-t0:.0f}s", flush=True)

maxasa={'A':129,'R':274,'N':195,'D':193,'C':167,'E':223,'Q':225,'G':104,'H':224,'I':197,
        'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}

def extract_struct(seq, arr, plddt_ca):
    Lc = len(seq)
    sasa_atom = struc.sasa(arr, vdw_radii="Single")
    sasa_res = struc.apply_residue_wise(arr, sasa_atom, np.nansum)
    rel_sasa = np.array([sasa_res[i]/maxasa.get(seq[i],200) for i in range(Lc)])
    burial = 1.0 - np.clip(rel_sasa,0,1)
    try:
        sse = struc.annotate_sse(arr)
        if len(sse)==Lc:
            sse_helix=(sse=='a').astype(float); sse_sheet=(sse=='b').astype(float); sse_coil=(sse=='c').astype(float)
        else:
            sse_helix=np.zeros(Lc); sse_sheet=np.zeros(Lc); sse_coil=np.ones(Lc)
    except Exception:
        sse_helix=np.zeros(Lc); sse_sheet=np.zeros(Lc); sse_coil=np.ones(Lc)
    ca = arr[arr.atom_name=="CA"]; coords = ca.coord; D = cdist(coords,coords)
    contacts = ((D<8.0)&(D>0)).sum(1).astype(float)
    for i in range(Lc):
        lo,hi = max(0,i-2), min(Lc,i+3)
        contacts[i] -= ((D[i,lo:hi]<8.0)&(D[i,lo:hi]>0)).sum()
    feat = np.vstack([plddt_ca, sasa_res, rel_sasa, burial, sse_helix, sse_sheet, sse_coil, contacts]).T
    return feat, coords

def tm_score(coords_a, coords_b, Lref):
    d0 = 1.24*(Lref-15)**(1/3) - 1.8 if Lref>21 else 0.5
    di = np.linalg.norm(coords_a-coords_b, axis=1)
    return float(np.mean(1.0/(1.0+(di/d0)**2)))

# WT reference: feature array already extracted (esmfold_wt_perres.npz), structure from esmfold_wt.pdb
wtp = np.load("esmfold_wt_perres.npz", allow_pickle=True)
wt_feat_ref = wtp["feat"]  # (L, 8): plddt, sasa, rel_sasa, burial, sse_helix, sse_sheet, sse_coil, contact_number
wt_plddt_ca = wt_feat_ref[:,0]
f = pdb.PDBFile.read("esmfold_wt.pdb"); wt_arr = f.get_structure(model=1)
wt_arr = wt_arr[struc.filter_amino_acids(wt_arr)]
assert len(np.unique(wt_arr.res_id))==L
wt_ca_only = wt_arr[wt_arr.atom_name=="CA"]
wt_feat = wt_feat_ref  # reuse precomputed WT features directly (cols match extract_struct order)

results = []
for i, v in enumerate(variants):
    mutant = v["mutant"]; pos_lin = v["position_linear"]; mut_aa = v["mut_aa"]
    idx = pos_lin - offset
    mseq = wt_seq[:idx] + mut_aa + wt_seq[idx+1:]
    assert len(mseq)==L
    print(f"[{i+1}/{len(variants)}] folding {mutant} (idx={idx})", flush=True)
    t1 = time.time()
    inp = tok([mseq], return_tensors="pt", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(inp)
    dt = time.time()-t1
    plddt = out["plddt"][0].numpy(); m_plddt_ca = plddt[:,1]*100.0
    pdb_str = model.output_to_pdb(out)[0]
    pdb_path = f"mut_{mutant}.pdb"
    open(pdb_path,"w").write(pdb_str)

    fm = pdb.PDBFile.read(pdb_path); m_arr = fm.get_structure(model=1)
    m_arr = m_arr[struc.filter_amino_acids(m_arr)]
    m_feat, m_coords = extract_struct(mseq, m_arr, m_plddt_ca)
    m_ca_only = m_arr[m_arr.atom_name=="CA"]

    # superimpose mutant CA onto WT CA, then compute RMSD + TM-score on the SAME superposition
    fitted, transform = struc.superimpose(wt_ca_only, m_ca_only)
    rmsd = float(struc.rmsd(wt_ca_only, fitted))
    tm = tm_score(wt_ca_only.coord, fitted.coord, L)

    d_plddt_global = float(m_plddt_ca.mean() - wt_plddt_ca.mean())
    d_plddt_site = float(m_plddt_ca[idx] - wt_plddt_ca[idx])
    d_sasa_site = float(m_feat[idx,1] - wt_feat[idx,1])
    d_burial_site = float(m_feat[idx,3] - wt_feat[idx,3])

    results.append(dict(mutant=mutant, position_linear=pos_lin, mut_aa=mut_aa,
        fold_time_s=dt, d_plddt_global=d_plddt_global, d_plddt_site=d_plddt_site,
        d_sasa_site=d_sasa_site, d_burial_site=d_burial_site,
        global_rmsd=rmsd, tm_score=tm,
        m_plddt_global=float(m_plddt_ca.mean()), wt_plddt_global=float(wt_plddt_ca.mean())))
    print(f"  done {dt:.0f}s | d_plddt_global={d_plddt_global:.2f} d_plddt_site={d_plddt_site:.2f} rmsd={rmsd:.2f} tm={tm:.3f}", flush=True)
    json.dump(results, open("variant_fold_results_partial.json","w"), indent=2)

json.dump(results, open("variant_fold_results.json","w"), indent=2)
print("ALL DONE", len(results), "variants folded")
