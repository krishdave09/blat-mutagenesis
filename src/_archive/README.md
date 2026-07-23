# archived scripts

Earlier-iteration versions, kept for history -- none of these are called by any notebook or
live script. Each was superseded by a newer version during development:

| archived | superseded by |
|---|---|
| `run_baseline.py` | `run_benchmark.py` |
| `make_figs2.py` | `make_benchmark_figs.py` |
| `make_results_doc.py` | `make_benchmark_doc.py` |
| `run_significance.py` | `run_analysis.py` (significance folded in alongside zero-shot AUROC + feature weights) |
| `fold_variants.py` | `src/colab_esmfold/09_colab_esmfold_extraction.ipynb` (partial-subset CPU fold -> full 4,770-variant GPU fold) |
| `run_benchmark_distilled.py` | not superseded -- a feature-importance-guided side investigation that was never wired into the pipeline |

If you need one of these paths again, read the file directly here rather than resurrecting it
into `src/` -- confirm it still reads/writes the current table and file names before running it.
