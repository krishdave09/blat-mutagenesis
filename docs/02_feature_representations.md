# Candidate Feature Representations (sequence-derived only)

*Every representation below is computable from the amino-acid sequence alone. No structure, no MSA. This document defines the menu the selection memo recommends from and pins the exact meaning of each named representation so downstream code and prose never use a vague label.*

A variant in this project is a single amino-acid substitution in TEM-1, written `wt_aa · position · mut_aa` (e.g. `M182T`). Every representation is a function of that variant plus the wild-type sequence. The representations divide into three groups: language-model-derived (require a PLM forward pass), physicochemical (require only amino-acid property tables), and identity (require nothing but the substitution itself, the naive baseline).

## Language-model-derived representations

### Representation 1 — Masked-marginal surprisal scalar

**What it computes.** For a variant at position *i*, mask position *i* in the wild-type sequence, run one PLM forward pass, and read the model's predicted log-probabilities over the 20 amino acids at that position. The score is `log P(mut_aa | context) − log P(wt_aa | context)`. One number per variant. Negative values mean the model finds the mutation less likely than wild type (predicted damaging); values near zero mean tolerated.

**Biology it carries.** Direct readout of learned evolutionary constraint at that position. High-magnitude negative scores concentrate at conserved core and functional positions.

**Shape.** 1 feature per variant (per PLM). Concatenating three PLMs gives 3 features.

**Pros.** Cheapest informative PLM feature; interpretable; the standard zero-shot method; one forward pass per position (263 passes covers all TEM-1 single mutants). **Cons.** Collapses all information at a position into one scalar, discarding which specific substitution was made beyond its own probability; weakest at catalytic positions the LM under-constrains.

**Compute.** One masked forward pass per position; ~263 passes total for TEM-1, cacheable once.

### Representation 2 — Mean-pooled embedding vector

**What it computes.** Run the mutant full-length sequence (wild type with the single substitution applied) through the PLM, take the final-layer per-residue hidden states, and average across all residues to get one fixed-length vector (1280-dim for ESM-2 650M). Optionally also embed the wild type and use the difference.

**Biology it carries.** A holistic learned representation of the whole variant sequence, including distributed effects a single-position score misses. The mutant-minus-wild-type difference vector isolates what the substitution changed globally.

**Shape.** 1280 features per variant (ESM-2 650M); varies by model (ProtT5 1024, ESM-2 3B 2560).

**Pros.** Richest single-vector representation; captures context beyond the mutated position; strong for a trained classifier. **Cons.** High-dimensional (regularization or dimensionality reduction needed with few thousand training variants); one forward pass *per variant* (4,783 passes), the most expensive option; less interpretable.

**Compute.** One full-sequence forward pass per variant; ~4,783 passes, the dominant cost of the whole feature-extraction stage.

### Representation 3 — Per-position surprisal vector (20 amino acids)

**What it computes.** At the mutated position, take the full 20-dimensional log-probability distribution the masked LM predicts (not just the wt/mut pair). One 20-vector per variant, or equivalently per position since all substitutions at a position share it.

**Biology it carries.** The complete tolerance profile of the position: which residues the model considers acceptable there, not just the one that was substituted. Encodes the "shape" of constraint (e.g. a position that tolerates only hydrophobics).

**Shape.** 20 features per variant.

**Pros.** Far richer than Representation 1 at almost no extra cost (same forward pass); lets a classifier learn position-type-specific rules; moderate dimensionality. **Cons.** Still position-centric; does not encode the mutant identity except through its own entry (pair with Representation 4).

**Compute.** Same 263 masked passes as Representation 1; the 20-vector is already computed, just retained instead of reduced to a scalar.

### Representation 4 — Site surprisal + one-hot substitution (40 features)

**What it computes.** Concatenate Representation 3 (the 20-dim position tolerance profile) with a 20-dim one-hot encoding of the mutant amino acid. 40 features per variant. This is the representation the project's naming convention calls out explicitly: "site-level surprisal across all 20 amino acids + one-hot substitution encoding, 40 features."

**Biology it carries.** Combines *where* (the position's tolerance profile) with *what* (the specific substitution), letting a tree-based model split on interactions like "this position tolerates only small residues AND the mutant is bulky."

**Shape.** 40 features per variant.

**Pros.** Explicitly encodes the substitution identity alongside constraint; well-matched to decision-tree learners; still cheap (same 263 passes). **Cons.** The one-hot half carries the same position-independent identity signal the naive baseline uses, so leakage discipline still matters.

**Compute.** Same 263 masked passes; the one-hot adds no PLM cost.

### Representation 5 — Sliding-window context features

**What it computes.** For the mutated position, aggregate surprisal (or embeddings) over a local window of ±k residues, producing summary statistics (mean, min, max surprisal in the window) that describe the local sequence neighborhood rather than the single position.

**Biology it carries.** Local structural/functional context: a destabilizing mutation often sits in a locally constrained stretch (a buried strand, an active-site loop). Window features capture that neighborhood signal.

**Shape.** Small (a handful of summary features per window per variant), tunable by window size k.

**Pros.** Cheap add-on to Representations 1/3; injects local context without full embeddings; low-dimensional. **Cons.** Ad hoc; window size is a hyperparameter; likely secondary to the direct per-position signal.

**Compute.** Derived from the already-cached per-position surprisal; negligible extra cost.

### Representation 6 — Sequence-PLM score concatenation

**What it computes.** Concatenate the masked-marginal scalars (Representation 1) from multiple sequence-input PLMs — e.g. ESM-2 + ESM-1v + ESM C — into a short vector, optionally with their per-position vectors (Representation 3).

**Biology it carries.** Different PLMs were trained on different data with different objectives, so their constraint estimates are partially independent; concatenation lets a classifier exploit agreement and disagreement between them (an ensemble signal).

**Shape.** Small when scalars only (3–6 features); larger if per-position vectors are stacked.

**Pros.** Cheap ensembling of independent evidence; robust to any single model's blind spots; directly serves the project goal of finding the best *combination* of PLM features. **Cons.** Requires running every PLM in the set (multiplies forward-pass cost); gains may be modest if the models are highly correlated.

**Compute.** Sum of the per-model costs for whichever representations are concatenated.

## Physicochemical representations

### Representation 7 — Physicochemical amino-acid scales

**What it computes.** Encode the wild-type and mutant amino acids by tabulated biophysical properties — hydrophobicity (e.g. Kyte–Doolittle), net charge, side-chain volume, polarity, and related scales — and use the values and their wild-type→mutant *deltas* (Δhydrophobicity, Δcharge, Δvolume) as features. All values come from standard property tables indexed by amino-acid identity; no sequence model needed.

**Biology it carries.** The direct biophysical consequence of the swap: a charge reversal at a salt bridge, a large-to-small change opening a cavity in the core, a hydrophilic residue introduced into a buried position. This is signal *orthogonal* to evolutionary conservation and is exactly what the review flagged as complementary to PLM surprisal at catalytic and structural positions.

**Shape.** ~10–20 features per variant depending on how many scales are included.

**Pros.** Essentially free to compute; readily available from a sequence (the user's stated priority); adds a mechanistic axis the PLMs under-weight; low-dimensional and interpretable. **Cons.** Position-agnostic on its own (a charge change matters only in the right structural context, which physicochemical features alone cannot supply); best used *with* PLM or position features, not alone.

**Compute.** Table lookup; instant.

## Identity baseline

### Representation 8 — Identity one-hot (naive floor)

**What it computes.** One-hot encode the wild-type amino acid, the mutant amino acid, and optionally the position index. No biology, no property tables — only "which residue changed to what, and where."

**Biology it carries.** None beyond the raw substitution. It is the deliberate control: any accuracy it achieves on a random split is largely position memorization, and its collapse on the contiguous split is the evidence that the PLM features carry transferable biology the one-hot does not.

**Shape.** 40 features (wt one-hot + mut one-hot), or 40 + position encoding.

**Pros.** The essential baseline for the project's central claim; trivial to compute; exposes position leakage when compared across splits. **Cons.** By design carries no generalizable signal; expected to fail on held-out regions — which is the point.

**Compute.** Instant.

## Summary table

| # | Representation | Dim (per variant) | PLM passes | Feature-source eligible | Primary role |
|---|----------------|-------------------|------------|-------------------------|--------------|
| 1 | Masked-marginal surprisal scalar | 1 / PLM | 263 (per-position) | yes | Cheap zero-shot score |
| 2 | Mean-pooled embedding | ~1280 | 4,783 (per-variant) | yes | Richest single vector |
| 3 | Per-position surprisal vector | 20 | 263 (shared) | yes | Position tolerance profile |
| 4 | Site surprisal + one-hot substitution | 40 | 263 (shared) | yes | Best tree-model input |
| 5 | Sliding-window context | ~3–9 | 0 (derived) | yes | Local-context add-on |
| 6 | Sequence-PLM score concatenation | 3–6+ | sum of models | yes | Cross-PLM ensemble |
| 7 | Physicochemical scales | ~10–20 | 0 | yes | Orthogonal biophysics |
| 8 | Identity one-hot | 40 | 0 | yes (control) | Naive leakage floor |

## How the ESM2 sub-study uses these (deferred phase)

The ESM2 feature-representation sub-study, deferred to a later session per the current plan, compares Representations 1, 2, 3, and 5 *for ESM-2 specifically* across all four classifiers and all three splits, to decide what "ESM2 features" means before the full grid runs. This document defines those representations so that sub-study, and the full PLM+ML grid after it, inherit one fixed vocabulary. The baseline phase that runs now uses only Representations 7 (physicochemical) and 8 (identity one-hot), the two that need no PLM.
