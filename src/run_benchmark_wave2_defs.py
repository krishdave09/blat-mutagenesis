#!/usr/bin/env python
"""run_benchmark_wave2.py -- SECOND WAVE feature sets (additions after the 139-set primary grid).

Adds the representations from the feature-design doc that the primary grid missed, PLUS
mixed-FAMILY multi-way combos (surprisal + embedding + physicochemical + structural, incl.
cross-PLM mixes) -- the "1+1+1+1 across different feature TYPES" sets, not just same-rep stacks.

Run: run_benchmark_wave2.py            -> appends to plm_benchmark_sequence_only.csv / _esmfold.csv
Reuses the same nested-CV harness by importing the primary module's functions.
"""
import sys, time, warnings, os
from pathlib import Path
_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'.projectroot').exists())
sys.path.insert(0, str(_root)); sys.path.insert(0, str(_root/'src'))
from paths import INTERIM, PROCESSED, TABLES
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[v]='4'

df = pd.read_parquet(PROCESSED/'tem1_firnberg_processed.parquet')
pool = df[~df.excluded_wetlab_validation].reset_index(drop=True)
mut_index={m:i for i,m in enumerate(pool.mutant.values)}
def keyed(npz, key):
    d=np.load(INTERIM/npz, allow_pickle=True); M=d[key]; M=M[:,None] if M.ndim==1 else M
    order=[mut_index[m] for m in d['mutant']]
    out=np.full((len(pool),M.shape[1]),np.nan,np.float32); out[order]=M
    assert not np.isnan(out).any(), npz; return out

PLMS=[('esm2_650m','ESM-2 650M'),('esm1v','ESM-1v'),('esmc','ESM C 600M')]
B={}
# --- NEW representations from the design doc ---
for tag,name in PLMS:
    B[f'{name} Rep-global surprisal (sum/mean/max/delta)'] = keyed(f'feat_newreps_{tag}.npz','rep_global')
    B[f'{name} Rep-binary surprisal fingerprint (263)']    = keyed(f'feat_newreps_{tag}.npz','rep_binary')
    B[f'{name} Rep5 sliding-window surprisal (i +/-2,+/-5)']= keyed(f'feat_newreps_{tag}.npz','rep_window')
    B[f'{name} Rep-sensitive surprisal (top-15 sites)']    = keyed(f'feat_newreps_{tag}.npz','rep_sensitive')
B['Physicochemical sliding-window (hydro/charge/vol, i +/-2,+/-5)'] = keyed('feat_physicowindow.npz','X')
# --- existing blocks reused for mixed-family combos ---
def plm(tag,rep): return keyed(f'feat_{tag}.npz',rep)
def emb(tag,slc): return keyed(f'emb3_{tag}.npz',slc)
PHYS = keyed('feat_rep7_physico.npz','X')
STRUCT = keyed('feat_esmfold_struct.npz','X')
PW = B['Physicochemical sliding-window (hydro/charge/vol, i +/-2,+/-5)']

sets={}
def add(name,mat,tier): 
    if name not in sets: sets[name]=(mat,tier)
# 1) the new representations as singles + each with physicochemical
for k,v in B.items():
    tier='seq_plm' if 'Physicochemical sliding' not in k else 'seq_plmfree'
    add(k,v,tier)
    if 'Physicochemical sliding' not in k:
        add(f'{k} + Physicochemical', np.hstack([v,PHYS]),'seq_plm')

# 2) MIXED-FAMILY multi-way combos (one component per family) -- the real gap.
#    families: surprisal(Rep4 site+onehot) / embedding(Rep2b site) / physchem / structure
def r4(tag): return plm(tag,'rep4')
def site(tag): return emb(tag,'emb_site')
# best-guess strong single-PLM mixed stacks (surprisal + embedding + physchem [+ structure])
for tag,name in PLMS:
    s=r4(tag); e=site(tag)
    add(f'MIX {name} [Rep4 + site emb + Physicochemical]', np.hstack([s,e,PHYS]),'seq_plm')
    add(f'MIX {name} [Rep4 + Physicochemical + Rep5 window]', np.hstack([s,PHYS,B[f'{name} Rep5 sliding-window surprisal (i +/-2,+/-5)']]),'seq_plm')
    # predicted-structure mixed (labeled pred_struct)
    add(f'MIX {name} [Rep4 + site emb + Physicochemical + ESMFold_struct]', np.hstack([s,e,PHYS,STRUCT]),'pred_struct')
# cross-PLM mixed families: ESM-2 surprisal + ESM C embedding + physchem (mix reps ACROSS models)
add('MIX cross-PLM [ESM-2 Rep4 + ESM C site emb + Physicochemical]', np.hstack([r4('esm2_650m'),site('esmc'),PHYS]),'seq_plm')
add('MIX cross-PLM [ESM C Rep4 + ESM-2 site emb + Physicochemical]', np.hstack([r4('esmc'),site('esm2_650m'),PHYS]),'seq_plm')
add('MIX cross-PLM [ESM-2 Rep1 + ESM-1v Rep1 + ESM C Rep1 + Physicochemical + Rep5 windows]',
    np.hstack([plm('esm2_650m','rep1'),plm('esm1v','rep1'),plm('esmc','rep1'),PHYS,
               B['ESM-2 650M Rep5 sliding-window surprisal (i +/-2,+/-5)'],
               B['ESM C 600M Rep5 sliding-window surprisal (i +/-2,+/-5)']]),'seq_plm')
# GRAND mixed: all-PLM Rep4 + all-PLM site emb + physchem + physchem-window + structure
add('MIX GRAND [AllPLM Rep4 + AllPLM site emb + Physicochemical + PhyschemWindow + ESMFold_struct]',
    np.hstack([r4('esm2_650m'),r4('esm1v'),r4('esmc'),site('esm2_650m'),site('esm1v'),site('esmc'),PHYS,PW,STRUCT]),'pred_struct')

print(f'WAVE2: {len(sets)} new feature sets')
for k,(m,t) in sets.items(): print(f'  [{t:11}] {m.shape[1]:5d}  {k}')
