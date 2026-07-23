"""
paths.py -- one source of truth for every path in the project.

Two ways to use it:
  1. simplest: paste the body of this file into a cell at the top of each notebook.
  2. keep this file and `from paths import *` -- but Python must be able to SEE it on
     the import path. That means the notebook lives inside the project; from notebooks/
     put the project root on sys.path first:
         import sys; from pathlib import Path
         root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'.projectroot').exists())
         sys.path.insert(0, str(root))
         from paths import *
Either way, no notebook contains an absolute path except (at most) the project root below.
"""
from pathlib import Path
import os
import matplotlib.pyplot as plt

def find_project_root(explicit=None, marker=".projectroot"):
    """Locate the project root, in priority order:
       1. an explicit path you pass in,
       2. the $PROJECT_ROOT environment variable,
       3. walking UP from the working directory to find the marker file.
    If none are found we RAISE instead of silently guessing the cwd -- a wrong guess
    would scatter data/ and models/ into the wrong folder."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if os.environ.get("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).expanduser().resolve()
    here = Path.cwd().resolve()
    for folder in [here, *here.parents]:
        if (folder / marker).exists():
            return folder
    raise FileNotFoundError(
        f"No project root found (looked for '{marker}' from {here} upward). "
        f"Fix with one of: run your project's setup; move this notebook into the "
        f"project's notebooks/ folder; pass find_project_root('/path/to/project'); or set $PROJECT_ROOT.")


# Leave PROJECT_ROOT = None to auto-detect (the notebook must live inside the project).
# Set an explicit path only when running from OUTSIDE the tree.
PROJECT_ROOT = None
BASE_DIR = find_project_root(PROJECT_ROOT)
DATA      = BASE_DIR / "data"
RAW       = DATA / "raw"                  # read-only, never edited by hand
INTERIM   = DATA / "interim"              # cleaned / partially processed
PROCESSED = DATA / "processed"            # model-ready
FEATURES  = DATA / "features"             # extracted feature matrices, one subfolder per method
FEATURES_PLM_MM   = FEATURES / "plm_masked_marginal"  # masked-marginal surprisal (Rep 1/3/4)
FEATURES_PLM_EMB  = FEATURES / "plm_embeddings"        # mean-pooled / site / site-delta (Rep 2)
FEATURES_ESMFOLD  = FEATURES / "esmfold_structure"     # ESMFold structural features
MODELS    = BASE_DIR / "models"           # top-level: models are first-class artifacts
RESULTS   = BASE_DIR / "results"
FIGURES   = RESULTS / "figures"
TABLES    = RESULTS / "tables"

for _d in (RAW, INTERIM, PROCESSED, FEATURES_PLM_MM, FEATURES_PLM_EMB, FEATURES_ESMFOLD, FIGURES, MODELS, TABLES):
    _d.mkdir(parents=True, exist_ok=True)
