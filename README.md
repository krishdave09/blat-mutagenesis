# TEM-1 fitness project

Run notebooks in order: Part00 (project setuo) -> Part0 (EDA) -> Part1 -> Part2.
Raw data in data/raw/ is immutable; everything else is rebuilt by code.

## Notebook pipeline

| notebook | what it does | needs |
|----------|--------------|-------|
| `00_project_setup` | project scaffold | — |
| `01_EDA_traditional_ml_aa_identity` | EDA over amino-acid-identity features | `data/processed/.../modeling_dataset.parquet` |
| `02_traditional_ml_aa_identity_benchmark` | supervised AA-identity benchmark (uses raw amino acids only, no language model) | same |
| `03a_assemble_physicochemical_features` | assemble physicochemical-descriptor features (substitution matrices, distances, property deltas) | modeling_dataset.parquet |
| `03_EDA_physicochemical` | EDA over physicochemical features | physchem features |
| `04_physicochemical_benchmark` | supervised physicochemical benchmark (the D027 no-language-model control); three-way comparison vs AA-identity and zero-shot | physchem features |
| `05a_pllm_zeroshot_feature_extraction` | GPU feature extraction: ESM scores for all variants (run on Colab; committed for full reproducibility) | modeling_dataset.parquet |
| `05_EDA_pllm_zeroshot` | EDA over zero-shot ESM scores | scores parquet (below) |
| `06_pllm_zeroshot_benchmark` | zero-shot pLLM benchmark (the D027 no-training control) | scores parquet (below) |
| `07a_pllm_embedding_extraction` | GPU feature extraction: ESM embedding deltas for all variants (run on Colab; committed for full reproducibility) | modeling_dataset.parquet |
| `07_EDA_pllm_embeddings` | EDA over ESM embedding-delta features (per-block magnitude, PCA variance, single-PC association) | embeddings parquet (below) |
| `08_pllm_supervised_benchmark` | supervised pLLM benchmark — classifier on ESM embeddings (arm-3), with the 4-way comparison | embeddings parquet (below) |

## Feature extraction (GPU / Colab)

Notebooks `05`/`06` read the zero-shot ESM scores from
`data/features/plm_masked_marginal/pllm_zeroshot_scores.parquet`. That parquet is produced by the
**feature-extraction notebook `notebooks/05a_pllm_zeroshot_feature_extraction.ipynb`**, committed
in-repo for full reproducibility. It is GPU work, so a Colab-ready copy also lives in
`1.5 - Colab notebooks/` for convenience:

- **Extractor:** `notebooks/05a_pllm_zeroshot_feature_extraction.ipynb` (in-repo) or the Colab copy `1.5 - Colab notebooks/05a_pllm_zeroshot_feature_extraction_colab.ipynb`
  (self-contained — the variant list is embedded; no upload needed).
- **What it computes:** masked-marginal + wildtype-marginal substitution scores
  `log P(mut) − log P(wt)` at each mutated site, for the ESM ladder (ESM-1b, ESM-1v,
  ESM-2 150M/650M/3B, ESM-C 300M/600M).
- **GPU:** the six ≤650M models fit a free Colab T4; ESM-2 3B wants an L4 (24 GB) / A100.
- **To regenerate:** run that Colab notebook on a GPU runtime, download
  `out/pllm_zeroshot_scores.parquet`, and drop it at the path above. Then run `05` and `06`.

Scores are keyed by `seq_id` with one column per `{model}__{scheme}`
(e.g. `esm2_650m__masked_marginal`). The report at `2 - Writing/build_06_report.py` renders the
`06` results into a PDF matching the `00` format.

## Embedding extraction (GPU / Colab) — the supervised arm

Notebooks `07`/`08` read ESM **embedding deltas** (not scores) from
`data/features/plm_embeddings/pllm_embeddings.parquet`, produced by the
**feature-extraction notebook `notebooks/07a_pllm_embedding_extraction.ipynb`**, committed in-repo
for reproducibility. It is heavier GPU work than the score extractor (one forward pass per variant,
not per position), so a self-contained Colab copy lives in `1.5 - Colab notebooks/`:

- **Extractor:** `notebooks/07a_pllm_embedding_extraction.ipynb` (in-repo, defers to Colab on a
  CPU-only machine per the >20-minute rule) or the Colab copy
  `1.5 - Colab notebooks/07a_pllm_embedding_extraction_colab.ipynb` (self-contained — the variant
  list is embedded; no upload needed).
- **What it computes:** three embedding-delta blocks per model (D035/D036) — `delta_site`
  (change at the mutated residue, the primary variant-specific feature), `delta_pooled`
  (whole-sequence mean delta), `delta_local` (±7-residue window delta) — for the ESM ladder
  (ESM-1b, ESM-1v, ESM-2 150M/650M/3B, ESM-C 300M/600M). The wild-type embedding is computed once
  per model and reused for all 4,783 deltas.
- **GPU:** the six ≤650M models fit an L4 (24 GB); **ESM-2 3B wants an A100** (embeddings store
  hidden states, so it is heavier than the score run). Each model checkpoints to `out/`, so a
  disconnect never loses prior models; re-running skips models already on disk.
- **To regenerate:** run the Colab notebook on a GPU runtime, download `out/pllm_embeddings.parquet`,
  and drop it at the path above. Then run `07` and `08`.

Embeddings are keyed by `seq_id` with one column per `{model}__{block}__d{0000..}`
(e.g. `esm2_650m__delta_site__d0000`). This is a wide matrix (~10k+ columns), so `08` applies the
**D037 guardrail**: PCA fit *inside each CV fold* (train only), never on the full data, with a
leakage assertion that no identity/label column enters the feature matrix. The report at
`2 - Writing/build_08_report.py` renders the `08` results — including the 4-way comparison
(AA-identity 02 vs physicochemical 04 vs zero-shot 06 vs supervised-PLM 08) — into a PDF matching
the `00`/`06` format.

> **Status:** the `07a` extraction is a Colab GPU step; until `pllm_embeddings.parquet` is dropped
> into `data/features/plm_embeddings/`, notebooks `07`/`08` and `build_08_report.py` run in a
> reviewable PREVIEW/skeleton mode and populate automatically once the parquet is in place.
