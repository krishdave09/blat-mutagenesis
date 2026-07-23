# Beta-Lactam ML v2 — Sequence-Only Variant-Effect Benchmark

Predict **functional vs. non-functional TEM-1 β-lactamase variants from amino-acid
sequence alone** — no crystal structure, no experimental structural data, no MSA as an
input. This is the deployable clinical question: given only a resistance-gene sequence,
is the variant still functional?

The benchmark's central comparison is whether sequence-derived features that encode
biology (protein-language-model surprisal, physicochemical properties) beat a pure
residue-identity baseline (identity one-hot) — **especially on held-out protein regions**,
where a pattern-matcher should fail and a biology-aware model should generalize.

**Headline result so far (sequence-only, see `docs/05_benchmark_results.md`):** the best
sequence-only PLM recipe (ESM C 600M, per-position surprisal + substitution one-hot,
XGBoost) reaches **contiguous AUROC 0.899**, against an identity-one-hot floor of
**0.705** and a physicochemical floor of **0.726** — both statistically robust gaps.
That comparison is exactly the deployability question this repo exists to answer, and
it currently holds.

## One-line setup

```bash
git clone <repo-url> && cd blat-mutagenesis && bash setup.sh
```

`setup.sh` creates the pinned conda env `betalactam-v2` from `environment.yml` (or a pip
venv from `requirements.txt` if conda is absent), builds the directory tree, and checks
for the raw data. `make setup`, `make verify`, and `make tree` wrap the same steps.

That env is the **traditional-ML + evaluation stack only** (sklearn/xgboost/pandas/etc.).
PLM scoring needs two more conda environments this repo does not yet pin (see
**PLM scoring environments** below) — you must set those up separately before the
`03_score_plm_masked_marginal` / `04_score_plm_embeddings` / `05_esmfold_structure`
notebooks will run.

## Data you must supply

Drop the Firnberg (2014) TEM-1 DMS file into `data/raw/`:

- `BLAT_ECOLX_Firnberg_2014.csv` — 4,783 variants, standard ProteinGym schema
  (`mutant, mutated_sequence, DMS_score, DMS_score_bin`). Primary label is
  `DMS_score_bin` (median split, balance 0.501).

`data/raw/` is immutable and git-ignored; every other data file is rebuilt by code.

## PLM scoring environments (not pinned in this repo yet)

The `run_in(env, script)` helper in notebooks 03–05 dispatches to three conda
environments by name. Only `betalactam-v2` ships a committed spec here:

| env name | used for | spec |
| --- | --- | --- |
| `betalactam-v2` | traditional ML, feature derivation, PDF rendering | `environment.yml` / `requirements.txt` (this repo) |
| `esm` | ESM-2 650M + ESM-1v masked-marginal scoring, ESMFold WT fold (`fair-esm`) | not committed here — closest known-working reference is `../Beta-Lactam ML Benchmark first draft/envs/esm.yml` (`fair-esm==2.0.0`, `torch==2.13.0`) |
| `esmc` | ESM C 600M scoring via the ESM SDK (`esm.models.esmc`, `esm.sdk.api`) | not committed here — closest reference is `../Beta-Lactam ML Benchmark first draft/envs/esm-sdk.yml`, but that file names its env `esm2`, not `esmc`; verify/rename before reuse |

Before your next PLM-scoring run, either `conda env export -n esm` / `-n esmc` from
whatever environment you actually have working locally and commit the result as
`environment-esm.yml` / `environment-esmc.yml`, or adapt the two reference files above.
Until one of those exists, a fresh clone of this repo cannot reproduce the PLM-scoring
stages, only the traditional-ML baseline and any notebook that reads already-cached
`data/interim/*.npz` feature files.

## Directory layout

```
.projectroot          # marker; paths.py walks up to find this
paths.py              # single source of truth for every path
environment.yml       # pinned conda env (betalactam-v2) -- traditional ML only, see above
requirements.txt      # pinned pip fallback (same scope)
setup.sh / Makefile   # one-line reproducible setup
data/{raw,interim,processed}
notebooks/            # 00_run_all -> 08_results_document (see Pipeline status)
src/                  # feature builders, PLM scoring (score_*.py), benchmark engine
src/colab_esmfold/    # self-contained folder to upload to Google Colab for GPU
                       #   per-variant ESMFold Δ-feature extraction (not yet run)
models/                # joblib + metadata (empty -- no model is locked yet)
results/{figures,tables}
docs/                  # 01-05: literature review, feature catalog, technical guide,
                       #   selection memo, benchmark results (.md is canonical, .pdf rendered)
```

## Naming discipline

Classifiers are always named individually — **Logistic Regression, Random Forest,
XGBoost, SVM** — never "baselines". Splits are always named in full — **random split,
modulo split, contiguous (region-holdout) split** — never "our splits". The headline
generalization number is always the **contiguous split**.

## Pipeline status

- **Docs 01–04 — literature review, feature catalog, technical guide, model/feature
  selection memo** (done; see `docs/`)
- **Notebooks 00–02 — reproducible scaffolding, Firnberg load, wet-lab holdout seal
  (13 variants sealed), three splits (random/modulo/contiguous)** (done)
- **Notebooks 03–04 — ESM-2 650M / ESM-1v / ESM C 600M masked-marginal scoring and
  embeddings** (done; cached under `data/interim/*.npz`)
- **Notebook 05 — ESMFold WT-only structural features** (done; per-variant delta
  features not yet merged, see next line)
- **Notebook 06 — feature assembly + sequence-only benchmark grid** (done for the
  primary grid; the wave-2 extended grid in `docs/ALL_feature_sets.txt` — 177 feature
  sets — is defined but only partially run, see `data/interim/benchmark_partial_*.csv`)
- **Notebook 07 — significance testing + figures** (done for the primary grid)
- **Notebook 08 — results document assembly** (done; produces `docs/05_benchmark_results.md/.pdf`)
- **In progress — `src/colab_esmfold/`**: a self-contained Colab notebook to fold all
  4,770 non-sealed variants individually (not just the WT) and extract per-variant Δ
  structural features, to give the ESMFold arm a fair, adequately powered comparison
  against the PLM sequence features. Staged, not yet executed or merged back.
- **Not started — final single-mutant model lock, wet-lab-panel scoring, cross-protein
  transfer test**

