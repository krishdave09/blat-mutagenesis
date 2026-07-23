# Model & Feature Selection Memo

*Decision memo for the sequence-only TEM-1 β-lactamase functionality benchmark (v2). Recommends which sequence-input models and which feature representations to carry into benchmarking, with rationale tied to the literature review (`01_literature_review.md`) and the feature catalog (`02_feature_representations.md`). This memo governs the deferred PLM phases; the traditional-ML baseline phase that runs now uses only the two PLM-free representations.*

## The decision in one paragraph

Carry the **ESM family as the primary PLM set — ESM-2 650M and ESM-1v as the two must-run models, ESM C as an efficiency candidate, ESMFold's ESM-2 trunk noted as the same feature source** — with **ProtT5/VESPA as the non-ESM comparator to beat** and **CARP/Ankh held as CPU fallbacks**. On the feature axis, carry **all eight named representations**, but treat **Representation 4 (site surprisal + one-hot substitution, 40 features)** and **Representation 6 (sequence-PLM score concatenation)** as the leading hypotheses for the best tree-model input, always concatenated with **Representation 7 (physicochemical, including charge)**, and benchmark them against the two mandatory controls, **Representation 8 (identity one-hot)** and the ESM zero-shot scalar (Representation 1). The identity one-hot control is not optional: it is the measurement that justifies the entire project.

## Model shortlist and rationale

**Tier A — must run (the core comparison).**

1. **ESM-2 650M (`esm2_t33_650M_UR50D`).** The representation workhorse. Best-quality embeddings and per-position distributions among the models we can run locally, well-validated, and the trunk shared with ESMFold so it doubles as the "structure-aware without using structure" feature source. Rationale: the review shows the ESM line is the strongest, best-validated sequence-only family, and 650M is the accuracy/compute sweet spot on CPU ([Lin 2023](https://doi.org/10.1126/science.ade2574)).
2. **ESM-1v.** The purpose-built zero-shot variant-effect scorer. It exists specifically so that its masked-marginal log-ratio predicts mutational effect, making it the reference for Representation 1 and the natural single-model baseline ([Meier 2021](https://doi.org/10.1101/2021.07.09.450648)). Running both ESM-2 and ESM-1v lets the sub-study settle whether the specialized scalar or the general embeddings win on TEM-1.

**Tier B — run if compute allows / strong comparator.**

3. **VESPA (on ProtT5).** The one published model purpose-built for exactly our task (missense variant effect from sequence), and therefore the external comparator our custom classifier must beat to justify itself ([Marquet 2021](https://doi.org/10.1007/s00439-021-02411-y)). ProtT5's independence from ESM also makes it the best partner for the cross-PLM concatenation (Representation 6) ([Elnaggar 2022](https://doi.org/10.1109/TPAMI.2021.3095381)).
4. **ESM C (ESM Cambrian).** Efficiency candidate — strong representations at lower compute, which matters on a CPU box scoring 4,783 variants ([EvolutionaryScale 2024](https://www.evolutionaryscale.ai/blog/esm-cambrian)). Flagged as vendor-reported until our benchmark confirms it; include in the concatenation set if its standalone score is competitive.

**Tier C — CPU fallbacks, only if Tier A is too slow.**

5. **CARP** (convolutional, cheaper than transformers) ([Yang 2024](https://doi.org/10.1016/j.cels.2024.01.008)) and **Ankh** (compute-optimized transformer) ([Elnaggar 2023](https://doi.org/10.1101/2023.01.16.524265)). Only reach for these if scoring the ESM models locally proves impractical; they trade some accuracy for speed.

**Reviewed but not feature sources.** AlphaFold2/3 and OmegaFold are sequence-input and analyzed in the review, but they output coordinates, and extracting features from a predicted structure leaves the direct-from-sequence regime. ESMFold is the sole "folder" whose features we use, and only via its ESM-2 trunk, which is already covered by choosing ESM-2. **Excluded entirely:** SaProt (needs Foldseek 3Di tokens from a structure), MSA Transformer and EVE (need an MSA), retrieval-Tranception (needs an MSA at inference).

## Feature shortlist and rationale

The project's second goal is to find the best *combination* of sequence-derived features to feed a decision-tree learner. The catalog defines eight; here is how to prioritize them.

**Lead hypotheses (expected best tree-model inputs):**

- **Representation 4 — site surprisal (20) + one-hot substitution (20) = 40 features.** Encodes both the position's tolerance profile and the specific mutant, which is exactly the interaction structure a Random Forest or XGBoost splits on. Cheap (263 shared masked passes). This is the front-runner for "what ESM2 features means" and the reason the project names it explicitly.
- **Representation 6 — sequence-PLM score concatenation (ESM-2 + ESM-1v + ESM C, optionally + ProtT5).** Cheap ensembling of partially-independent constraint estimates; directly serves the "best combo of PLM features" goal.
- **Representation 7 — physicochemical (charge, hydrophobicity, volume + deltas).** Always concatenated onto whichever PLM representation is used. Free to compute, and carries the catalytic/biophysical signal the review shows PLM surprisal under-weights. The user's specific request to include charge is well-founded: it is the orthogonal axis.

**Also benchmarked:**

- **Representation 2 — mean-pooled embedding (~1280-dim).** The richest single vector but the most expensive (4,783 per-variant passes) and high-dimensional relative to the dataset. Benchmark it, but it must clearly beat Rep 4 to justify its ~18× cost; if it does not, Rep 4 wins on deployability.
- **Representation 3 — per-position surprisal vector (20).** The natural intermediate; effectively Rep 4 without the substitution one-hot. Useful for isolating how much the one-hot half contributes.
- **Representation 5 — sliding-window context.** A cheap add-on to test whether local neighborhood signal helps; expected secondary.
- **Representation 1 — masked-marginal scalar.** The zero-shot control: the accuracy you get with *no trained classifier at all*. Any trained model must beat this to justify the supervision.

**Mandatory control:**

- **Representation 8 — identity one-hot.** Not a contender — the floor. Its behavior across splits is the project's central measurement (see below).

## The one comparison the project rests on

The scientific justification is a single, specific contrast, and every model/feature choice above serves it: **on the contiguous split, does a PLM-derived feature (Rep 4 or Rep 6, + Rep 7) beat identity one-hot (Rep 8), when both are given the same classifier?** Identity one-hot can only memorize which positions are intolerant, so it should score respectably on the random split and collapse on the contiguous split, where positions are unseen. A PLM feature that encodes transferable biology should hold its accuracy on the contiguous split. If that gap appears, it is direct evidence that PLMs understand biology rather than patterns — the project's reason to exist. If it does not appear, that is an equally important (negative) result telling us the PLM signal is not transferring for TEM-1.

## Honest deployability statement

Everything recommended here is computable from an amino-acid sequence with no experimental structure and no MSA. That is a deliberate accuracy sacrifice: the review notes that structure-aware (SaProt) and alignment-based (EVE, retrieval-Tranception) models report higher variant-effect accuracy, and v1 of this project found that adding real-structure features bought roughly 0.03–0.04 AUROC. We give that up on purpose, because a novel clinical resistance gene has neither a solved structure nor a reliable family alignment on the timescale that matters. The number this project will quote as "the model's accuracy" is therefore the **contiguous-split AUROC of the best sequence-only recipe**, reported alongside random and modulo so the leakage tax is visible, and validated exactly once at the end against the 13 sealed wet-lab variants. That is the honest measure of what a deployable, sequence-only tool can do.

## What runs now vs. later

- **Now (baseline phase):** only **Representation 7 (physicochemical)** and **Representation 8 (identity one-hot)** — the two representations that need no PLM — across Logistic Regression, Random Forest, XGBoost, and SVM, on all three splits. This establishes the traditional-ML, sequence-only floor and, critically, the identity-one-hot control the whole thesis is measured against.
- **Later (deferred PLM phases):** score the Tier A/B models, run the ESM-2 representation sub-study (Reps 1/2/3/5) to fix what "ESM2 features" means, then the full feature × classifier grid with the concatenations, DeLong + paired-bootstrap significance, and the sealed-holdout scoring.
