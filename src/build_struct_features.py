#!/usr/bin/env python
"""build_struct_features.py -- assemble the predicted-structure (ESMFold) feature blocks.

Two feature sets, both keyed by `mutant`, written to data/interim/:
  1. ESMFold_struct  (8 features)  -- WT per-position features mapped to each variant by position.
                                       Full coverage (all 4,770 pool variants). ALWAYS available.
  2. ESMFold_delta   (variable)    -- per-variant absolute + delta features from GPU folding.
                                       Only present if data/interim/esmfold_variant_features.csv
                                       exists (produced by src/colab_esmfold/09_colab_esmfold_extraction.ipynb).
                                       Full coverage only if that run covered all 4,770.
"""
import sys
from pathlib import Path
_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'.projectroot').exists())
sys.path.insert(0, str(_root)); from paths import INTERIM, PROCESSED
import numpy as np, pandas as pd

df = pd.read_parquet(PROCESSED/'tem1_firnberg_processed.parquet')
pool = df[~df.excluded_wetlab_validation].reset_index(drop=True)

# --- 1. position-mapped structural features (from WT fold) ---
d = np.load(INTERIM/'esmfold_wt_perres.npz', allow_pickle=True)
feat = d['feat']; cols = [str(c) for c in d['cols']]; positions = list(d['positions'])
pos2idx = {int(p):i for i,p in enumerate(positions)}
X = np.vstack([feat[pos2idx[int(p)]] for p in pool.position_linear]).astype(np.float32)
np.savez_compressed(INTERIM/'feat_esmfold_struct.npz', X=X, cols=np.array(cols), mutant=pool.mutant.values)
print(f'ESMFold_struct: {X.shape} (position-mapped, full coverage)')

# --- 2. per-variant delta features (Colab GPU) if present ---
colab = INTERIM/'esmfold_variant_features.csv'
if colab.exists():
    dv = pd.read_csv(colab)
    merged = pool[['mutant']].merge(dv, on='mutant', how='left')
    dcols = [c for c in dv.columns if c!='mutant']
    cov = merged[dcols[0]].notna().mean()
    Xd = merged[dcols].to_numpy(np.float32)
    np.savez_compressed(INTERIM/'feat_esmfold_delta.npz', X=Xd, cols=np.array(dcols), mutant=pool.mutant.values)
    print(f'ESMFold_delta: {Xd.shape}, coverage {cov:.1%} of pool ({dcols})')
    if cov < 0.99:
        print(f'  NOTE: partial coverage -- delta features usable as illustrative arm, not full-power benchmark cell')
else:
    print('ESMFold_delta: (no delta CSV yet -- run src/colab_esmfold/09_colab_esmfold_extraction.ipynb, '
          'drop esmfold_variant_features.csv into data/interim/)')
