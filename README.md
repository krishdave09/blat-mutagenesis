# TEM-1 fitness project

Run notebooks in order: Part00 (project setuo) -> Part0 (EDA) -> Part1 -> Part2.
Raw data in data/raw/ is immutable; everything else is rebuilt by code.

## Notebook pipeline

| notebook | what it does | needs |
|----------|--------------|-------|
| `00_project_setup` | project scaffold | — |
| `01_EDA_traditional_ml_aa_identity` | EDA over amino-acid-identity features | `data/processed/.../modeling_dataset.parquet` |
| `02_traditional_ml_aa_identity_benchmark` | supervised AA-identity benchmark (uses raw amino acids only, no language model) | same |
| `05a_pllm_zeroshot_feature_extraction` | GPU feature extraction: ESM scores for all variants (run on Colab; committed for full reproducibility) | modeling_dataset.parquet |
| `05_EDA_pllm_zeroshot` | EDA over zero-shot ESM scores | scores parquet (below) |
| `06_pllm_zeroshot_benchmark` | zero-shot pLLM benchmark (the D027 no-training control) | scores parquet (below) |

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
