# Beta-Lactam ML v2 — Sequence-Only Benchmark: Setup Instructions

Paste this whole file as the first message in the new chat. It defines the goal, the
data you're starting from, the hard rules, and the exact benchmark to build.

> **Status as of this update: the benchmark described below has been run.** The
> sequence-only PLM grid (§4) is complete for the primary representations, written up
> in `docs/05_benchmark_results.md`, and the target set in §5 has been beaten: best
> sequence-only contiguous AUROC is **0.899** (ESM C 600M, Rep4 site+onehot,
> XGBoost) against the v2 goal of pushing past 0.872. The wave-2 extended grid
> (`docs/ALL_feature_sets.txt`, 177 feature sets) is defined but only partially run.
> A separate `src/colab_esmfold/` folder is staged (not yet executed) to fold all
> 4,770 non-sealed variants individually for a fair, adequately powered ESMFold arm.
> Keep this file as the historical brief for what the benchmark was designed to answer;
> `README.md` and `docs/05_benchmark_results.md` are the current source of truth for
> what has actually been measured.

---

## 1. The premise (why v2 exists)

In the clinic the input is a **gene sequence** → translated to an **amino-acid
sequence**. That is the ONLY starting data. v2 builds a variant-effect model that
predicts functional vs. non-functional TEM-1 β-lactamase mutations **from sequence
alone** — no crystal structure, no experimental structural data, as an input.

This is deliberately narrower than v1. v1 showed that adding lab-structure features
(ThermoMPNN ΔΔG, DDGun3D, B-factor) buys ~0.03–0.04 AUROC — but those features
require a solved structure we won't have for a novel resistance gene. v2 asks the
honest deployable question: **how good is a model that only ever sees a sequence?**

## 2. Hard rules (enforce these from the first cell)

1. **Sequence-in only.** Every feature must be derivable from the amino-acid
   sequence alone. Allowed: sequence PLMs (ESM2, ESM-1v, ESM C), sequence-based
   physicochemical scales, raw identity encodings.
   **Not allowed as an input:** crystal structures, experimental B-factors, SASA
   from a real structure, or any ΔΔG/structural score that needs deposited
   coordinates.
   - **Predicted structure is a SEPARATE, explicitly-labelled experimental arm**,
     never mixed into the "sequence-only" model. If you fold with ESMFold/AF2 and
     run structural PLMs on the *prediction*, that model is named e.g.
     `RandomForest_ESMFoldStructPLM` and reported as its own line — never folded
     into a "sequence" result. Keep the graceful-degradation tiers visually and
     nominally distinct.
2. **Name everything explicitly. No abstractions on graphs or tables.**
   - NOT "traditional ML" → **"Random Forest, ESM2 features"**
   - NOT "PLM" → the specific model: **"ESM2 (650M)"**, **"ESM-1v"**, **"ESM C"**
   - NOT "combo" → **"Logistic Regression, ESM2+ESM-1v concatenated embeddings"**
   - Every legend entry, axis label, bar, and table row = `<classifier>, <exact
     feature set>`. A reader must know the precise model from the label with no
     footnote.
3. **Report the honest split.** Random split is optimistic (position leakage).
   The headline number is always the **contiguous split** (whole regions held
   out). Show random/modulo/contiguous side by side so the leakage tax is visible,
   but the number you quote as "the model's accuracy" is contiguous.
4. **Lab conventions** (same as v1): `data/{raw,interim,processed}`,
   `notebooks/`, `src/`, `models/`, `results/{figures,tables}`; a single
   `paths.py` auto-detecting the `.projectroot` marker (already placed); no
   absolute paths in notebooks; Parquet for frames; joblib+meta for models;
   seed everything; assert no wet-lab holdout leaks into training.
5. **Wet-lab holdout stays sealed.** 13 variants flagged
   `excluded_wetlab_validation=True` are NEVER seen during any training,
   tuning, or model selection. Score them exactly once, at the very end,
   after the model is locked.
6. **Figures**: professional palette only (blues/purples/greens/blacks/dark
   pinks, no neon/red/orange). Load the `figure-style` skill before any
   deliverable figure. Body text of written docs = Times New Roman; figures stay
   sans-serif.

## 3. What's already in `data/raw/` (carried over from v1)

These are the reusable ground-truth files — labels and sequence only, **no
features** (v2 regenerates every feature from sequence itself):

- **`tem1_dms_labels_CARRYOVER.csv`** — 4,783 variants. Columns:
  `mutant, wt_aa, position_linear, position_ambler, mut_aa, DMS_score,
  DMS_score_bin`, five fixed-threshold label schemes
  (`label_bin_fixed_0.20…1.00`), three quantile schemes
  (`label_bin_quantile_q25/q50_median/q75`), plus `excluded_wetlab_validation,
  wetlab_primer_name, wetlab_role`.
  - Primary label = `DMS_score_bin` (median split, balance 0.501).
  - Positions: `position_linear` 24–286 (use for MSA-free work);
    `position_ambler` is the standard TEM-1 numbering for reporting.
- **`tem1_wt_mature.fasta`** — the 263-aa mature TEM-1 sequence (signal peptide
  removed), the substrate for all PLM scoring. Starts `HPETLVK…`.
- **`external_stiffler_CARRYOVER.csv`** — ⚠️ this is a per-METHOD summary from v1
  (14 rows: method, spearman, auroc), NOT variant-level. For a real external
  transfer test in v2, **re-fetch the raw Stiffler 2015 variant-level DMS** (it's
  public). Keep as reference for what v1 achieved.

## 4. The benchmark to build

A 2-axis grid, every cell named explicitly:

**Feature sets (sequence-derived only):**
- `ESM2` — masked-marginal scalar per variant
- `ESM2_embeddings` — mean-pooled embedding vector
- `ESM-1v`, `ESM C` — masked-marginal
- `SeqPLM_concat` — ESM2 + ESM-1v + ESM C scores concatenated
  (**note:** v1's "SeqPLM_all" wrongly included MSA Transformer — do NOT repeat
  that; MSA-T needs an alignment and is not sequence-only)
- `Physicochemical` — sequence-only AA scales (hydrophobicity, charge, volume…)
- `Identity_onehot` — the naive baseline (raw substitution one-hots)

**Classifiers:** Logistic Regression, Random Forest, XGBoost, SVM
(re-tuned per cell via inner CV — never hardcode one).

**Feature-representation sub-study (the v1-deferred experiment):** for ESM2
specifically, compare masked-marginal scalar vs. mean-pooled embedding vs.
per-position surprisal vector vs. sliding-window context features. This is where a
sequence-only model earns back accuracy — make it a first-class result, not a
footnote.

**Evaluation per cell:** nested CV, 3 splits (random/modulo/contiguous),
bootstrap 95% CIs, paired significance (DeLong + paired bootstrap) among the top
cells. Headline = contiguous.

## 5. Reference numbers from v1 (targets to beat / contextualize)

Contiguous AUROC, sequence-only-relevant recipes:
- Identity one-hot (naive floor): **0.720**
- ESM2 alone (LR): **0.872**  ← best pure-sequence single model in v1
- ESM C (LR): 0.836 · ESM-1v (LR): 0.833
- For context (NOT sequence-only, need structure/MSA): PLM_all 0.930,
  ALL_features 0.941

The v2 goal: with better ESM2 representations + sequence-PLM concatenation, push
the sequence-only contiguous AUROC **above 0.872** — ideally toward the 0.90 range
that v1 only reached with structural data.

## 6. First actions in the new chat

1. Confirm envs: v1 used a conda env `esm` (fair-esm 2.0.0, torch 2.12.1) and
   `esm2` (for ESM C / esm-sdk — kept separate to avoid clobbering fair-esm).
   Recreate or reuse. ESM scoring needs a GPU for reasonable speed; the local box
   is CPU-only, so wire PLM scoring for GPU dispatch.
2. Write `paths.py` (auto-detect `.projectroot`), then a `01_load_labels` notebook
   that reads the carryover CSV and asserts 4,783 rows / 13 sealed holdouts.
3. Score ESM2 masked-marginal on the 263-aa WT (v1 timing: ~192 s for all
   positions after a ~116 s model load, ~5 GB peak).
4. Build the feature-representation sub-study before the full grid — it decides
   what "ESM2 features" even means downstream.

---
*Generated from the v1 session as a clean-room restart. v1 artifacts (full
benchmark, figures, guide, benchmark document) remain in the original project for
reference.*
