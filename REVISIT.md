# Revisit flags

Open items to come back to. Each names the file, the provisional choice, and why it matters.

## 08 supervised pLLM benchmark — run at TRIMMED settings (provisional)

**File:** `notebooks/08_pllm_supervised_benchmark.ipynb` (config cell: `PCA_K`, `N_BOOT`)

**Current (provisional):** `PCA_K = 25`, `N_BOOT = 1000` — chosen for speed on a 16 GB laptop
(~25 min vs ~45 min at full settings). The results tables/figures/PDF currently on disk were
produced at these trimmed settings.

**Restore for the definitive run:** `PCA_K = 50`, `N_BOOT = 2000`, then re-run 08 and rebuild
`2 - Writing/build_08_report.py`.

**Why it matters (two known costs of the trim):**
1. **K=25 discards a large share of variance — not "a little".** Per the 07 `pca_variance` table,
   EVERY `delta_site` block (the primary D035 feature) needs **>100 PCs** to reach 90% variance, and
   its top-10-PC variance is only ~20–48% (ESM-C as low as ~19–20%). So at K=25 each `delta_site`
   block retains well under a quarter of its variance. Low-variance-but-predictive directions are
   especially at risk: the best class-separating direction (ESM-C 600M PC1, single-feature AUC 0.82)
   carried only ~3% of the variance. The trimmed run may **materially understate** the full-embedding
   model, especially on the contiguous split. K=50 roughly doubles retained variance but still
   under-covers — the intended range is 50–200 (D037), so treat trimmed numbers as a lower bound.
2. **1000 bootstraps widen the CI noise floor.** Close model-vs-model or arm-vs-arm gaps on the
   contiguous split may fail to reach significance (DeLong/McNemar + bootstrap CIs) that the
   2000-resample run would have resolved.

Flagged 2026-07-24.
