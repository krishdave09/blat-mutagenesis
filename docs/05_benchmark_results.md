# TEM-1 β-lactamase variant-effect benchmark: PLM architectures and feature weights

**Sequence-only prediction of β-lactamase functionality — a direction-finding benchmark**

---

## Summary

This benchmark tests whether protein-language-model (PLM) features predict TEM-1
β-lactamase functionality better than a model that sees only amino-acid identity, and
compares three ESM-family architectures to identify which representation to build a
final predictor on. Ground truth is the Firnberg 2014 deep-mutational-scanning (DMS)
dataset: 4770 single-missense variants with a binary functional/non-functional label
(13 wet-lab-panel variants held out and **not** scored here). All models are evaluated
by out-of-fold (OOF) AUROC under 5-fold cross-validation across three splits — random,
modulo-position, and **contiguous** (whole protein regions withheld) — with the
contiguous split as the primary, honest test of generalization to unseen regions.

**Classifier selection methodology.** Every reported cell is a (feature_set, split)
pair scored by four classifiers (LR, RF, XGBoost, SVM). The classifier reported for a
feature set is fixed by its performance on the **modulo split**, never by the contiguous
score being headlined — picking the classifier from the same number being reported would
be circular and optimistically biased. All four classifiers' contiguous-split AUROC are
also reported as a sensitivity block (§2) so the pre-specified pick is fully auditable.
This selection rule was applied retroactively to a version of this benchmark that had
picked the best-of-4 independently per split; the correction changed 3 of 21 cells by
≤0.007 AUROC, and did not change the identity of the best sequence-only cell.

**Headline result.** The best sequence-only PLM configuration (ESM C 600M, per-position
surprisal + substitution one-hot, XGBoost — the pre-specified classifier for this
feature set) reaches **contiguous AUROC 0.899**, versus the identity-only floor at
**0.705** and a physicochemical floor at **0.726** (both floors: XGBoost, also
pre-specified) — a gap of **+0.194 AUROC** that is overwhelmingly significant (DeLong
p ≈ 3.6 × 10^-131; paired-bootstrap 95% CI [0.178, 0.210]). The premise of the project —
that PLM features carry biological signal beyond position identity — holds on the
hardest split, and the gap is, if anything, slightly larger under bias-free classifier
selection than it first appeared.

*This is a proof-of-concept benchmark. No final model is locked or saved; the sealed
wet-lab holdout is untouched.*

---

## 1. Thesis: PLM features beat the identity floor

The identity one-hot baseline encodes only *which position was mutated to which residue* —
it can memorize position-level tolerance but knows no biology. Using the pre-specified
classifier (XGBoost, chosen by modulo-split performance) it reaches contiguous AUROC
**0.705**. Every PLM representation clears it by a wide margin, and the best PLM cell
does so by **+0.194 AUROC**.

![PLM features beat the identity floor by ~0.19 AUROC on held-out regions]({{artifact:art_a83b2386-68ba-454f-aa5b-a64848da0dcb}})

*Figure 1. Contiguous-split OOF AUROC (pre-specified classifier per representation) with
95% bootstrap CIs. Open markers = floors (identity, physicochemical); filled = PLM.
Marker colour = the pre-specified classifier (chosen by modulo-split AUROC, not by the
contiguous score shown). Dashed line = identity floor (0.705). The ~0.19 gap between the
floors and the PLM representations is the central result.*

The significance test on the paired OOF predictions confirms the gap is not noise. Both
classifiers below are the pre-specified (modulo-selected) choice for their feature set:

| Comparison | PLM AUROC | Floor AUROC | Δ | DeLong p | Bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- |
| ESM C Rep4 (XGBoost) vs Identity floor (XGBoost) | 0.8991 | 0.7050 | 0.1942 | 3.65e-131 | [0.1784,0.2096] |

Note the floor classifier changed from Random Forest (the earlier, contiguous-biased
pick) to XGBoost (the pre-specified pick) — the floor AUROC dropped slightly (0.7124 →
0.7050) as a result, which *widens* the reported gap rather than narrowing it.

---

## 2. Architecture comparison: ESM-2 vs ESM-1v vs ESM C

All three ESM-family architectures were scored on the same variants, splits, and
classifiers. The comparison identifies which architecture and representation to carry
forward. The classifier shown for each row is pre-specified per feature set (chosen by
modulo-split AUROC); a full four-classifier sensitivity table follows for transparency.

![Architecture comparison across representations]({{artifact:art_6b21e27b-1b47-4449-8af3-063091c3450d}})

*Figure 2. Contiguous AUROC (pre-specified classifier) by architecture and representation.
All three architectures clear the floor across every representation.*

| Architecture | Representation | Contiguous AUROC | 95% CI | Pre-specified classifier | Random | Modulo |
| --- | --- | --- | --- | --- | --- | --- |
| Floor | Physicochemical | 0.7258 | [0.712,0.740] | XGBoost | 0.7477 | 0.7287 |
| Floor | Identity one-hot | 0.7050 | [0.691,0.720] | XGBoost | 0.7382 | 0.7133 |
| ESM-2 650M | Rep4 site+onehot | 0.8892 | [0.880,0.898] | XGBoost | 0.9553 | 0.9420 |
| ESM-2 650M | Rep1 surprisal + physchem | 0.8835 | [0.874,0.894] | Logistic Regression | 0.8944 | 0.8881 |
| ESM-2 650M | Rep2b site-emb | 0.8807 | [0.871,0.890] | Logistic Regression | 0.9589 | 0.9445 |
| ESM-2 650M | Rep1 surprisal | 0.8731 | [0.862,0.884] | Logistic Regression | 0.8782 | 0.8770 |
| ESM-2 650M | Rep2c Δsite-emb | 0.8605 | [0.850,0.871] | Logistic Regression | 0.9513 | 0.9244 |
| ESM-2 650M | Rep3 surprisal-20 | 0.8458 | [0.834,0.857] | Random Forest | 0.8950 | 0.8855 |
| ESM-2 650M | Rep2a mean-emb | 0.8363 | [0.825,0.848] | Logistic Regression | 0.9529 | 0.9261 |
| ESM-1v | Rep4 site+onehot | 0.8748 | [0.865,0.885] | XGBoost | 0.9527 | 0.9333 |
| ESM-1v | Rep1 surprisal | 0.8357 | [0.824,0.847] | Logistic Regression | 0.8459 | 0.8445 |
| ESM-1v | Rep3 surprisal-20 | 0.8212 | [0.809,0.833] | Random Forest | 0.8945 | 0.8823 |
| ESM-1v | Rep2c Δsite-emb | 0.8206 | [0.808,0.833] | Logistic Regression | 0.9446 | 0.9152 |
| ESM-1v | Rep2a mean-emb | 0.7964 | [0.783,0.810] | Logistic Regression | 0.9454 | 0.9225 |
| ESM-1v | Rep2b site-emb | 0.7832 | [0.771,0.797] | Logistic Regression | 0.9567 | 0.9252 |
| ESM C 600M | Rep4 site+onehot | 0.8991 | [0.890,0.908] | XGBoost | 0.9575 | 0.9434 |
| ESM C 600M | Rep2a mean-emb | 0.8780 | [0.868,0.888] | Logistic Regression | 0.9588 | 0.9420 |
| ESM C 600M | Rep1 surprisal | 0.8733 | [0.863,0.884] | Logistic Regression | 0.8785 | 0.8777 |
| ESM C 600M | Rep2b site-emb | 0.8643 | [0.854,0.874] | Logistic Regression | 0.9596 | 0.9452 |
| ESM C 600M | Rep2c Δsite-emb | 0.8589 | [0.848,0.869] | Logistic Regression | 0.9533 | 0.9309 |
| ESM C 600M | Rep3 surprisal-20 | 0.8422 | [0.831,0.853] | Random Forest | 0.8953 | 0.8842 |

**Four-classifier sensitivity block** (contiguous AUROC for all four classifiers on each
feature set; **bold** = the pre-specified pick; `--` = not run — RF/XGBoost were gated
to ≤400-dim feature sets and SVM to ≤200-dim, so the 1,152–1,280-dim raw embedding
representations (Rep2a/b/c) are LR-only by design):

| Architecture | Representation | Pre-specified | LR | RF | XGB | SVM |
| --- | --- | --- | --- | --- | --- | --- |
| Floor | Physicochemical | **XGB** | 0.6082 | 0.7264 | 0.7258 | 0.6923 |
| Floor | Identity one-hot | **XGB** | 0.6797 | 0.7124 | 0.7050 | 0.6830 |
| ESM-2 650M | Rep4 site+onehot | **XGB** | 0.8638 | 0.8958 | 0.8892 | 0.8820 |
| ESM-2 650M | Rep1 surprisal + physchem | **LR** | 0.8835 | -- | -- | -- |
| ESM-2 650M | Rep2b site-emb | **LR** | 0.8807 | -- | -- | -- |
| ESM-2 650M | Rep1 surprisal | **LR** | 0.8731 | 0.8580 | 0.8608 | 0.8417 |
| ESM-2 650M | Rep2c Δsite-emb | **LR** | 0.8605 | -- | -- | -- |
| ESM-2 650M | Rep3 surprisal-20 | **RF** | 0.8399 | 0.8458 | 0.8338 | 0.8143 |
| ESM-2 650M | Rep2a mean-emb | **LR** | 0.8363 | -- | -- | -- |
| ESM-1v | Rep4 site+onehot | **XGB** | 0.8345 | 0.8702 | 0.8748 | 0.8508 |
| ESM-1v | Rep1 surprisal | **LR** | 0.8357 | 0.8213 | 0.8248 | 0.7973 |
| ESM-1v | Rep3 surprisal-20 | **RF** | 0.8114 | 0.8212 | 0.8077 | 0.7895 |
| ESM-1v | Rep2c Δsite-emb | **LR** | 0.8206 | -- | -- | -- |
| ESM-1v | Rep2a mean-emb | **LR** | 0.7964 | -- | -- | -- |
| ESM-1v | Rep2b site-emb | **LR** | 0.7832 | -- | -- | -- |
| ESM C 600M | Rep4 site+onehot | **XGB** | 0.8598 | 0.8964 | 0.8991 | 0.8950 |
| ESM C 600M | Rep2a mean-emb | **LR** | 0.8780 | -- | -- | -- |
| ESM C 600M | Rep1 surprisal | **LR** | 0.8733 | 0.8608 | 0.8665 | 0.8377 |
| ESM C 600M | Rep2b site-emb | **LR** | 0.8643 | -- | -- | -- |
| ESM C 600M | Rep2c Δsite-emb | **LR** | 0.8589 | -- | -- | -- |
| ESM C 600M | Rep3 surprisal-20 | **RF** | 0.8373 | 0.8422 | 0.8376 | 0.8163 |

**Reading the table.**
- **ESM-2 650M and ESM C 600M lead and are close**; ESM-1v trails on every representation,
  but by an amount that varies by representation and architecture rather than a single
  band. On surprisal and Rep4 the gap is ~0.02-0.04 against both architectures. On the raw
  embeddings the gap is larger and uneven: **ESM C's gap is consistently ~0.08** on both
  Rep2a mean-emb (0.878 vs 0.796, Δ0.082) and Rep2b site-emb (0.864 vs 0.783, Δ0.081);
  **ESM-2's gap is representation-dependent** — small on Rep2a (0.836 vs 0.796, Δ0.040) but
  similarly large on Rep2b (0.881 vs 0.783, Δ0.098). ESM C 600M holds the single best
  contiguous cell (Rep4, XGBoost, 0.899); ESM-2 650M's best cell (Rep4, XGBoost, 0.889) is
  a close second.
- **Rep4 (per-position surprisal + substitution one-hot) is the strongest representation
  on the contiguous split** for every architecture — it generalizes to unseen regions
  better than the raw embeddings.
- **Three cells changed classifier under bias-free selection** (Identity floor, Physchem
  floor, and ESM-2 Rep4 all moved from Random Forest to XGBoost — RF had won those cells
  narrowly on the contiguous split itself but not on modulo). The changes are small
  (≤0.007 AUROC) and don't affect which architecture or representation is best.
- **The random/modulo columns are much higher (0.94-0.96) than contiguous (0.86-0.90),
  especially for the embedding representations.** This is the generalization story: on
  random splits the embeddings partly exploit position information that leaks across folds;
  on the contiguous split — where whole regions are unseen — that advantage disappears and
  the compact surprisal+identity representation (Rep4) holds up best. **The contiguous
  number is the one to trust for a tool that must generalize to new regions.**

---

## 3. Is a trained classifier worth its complexity over zero-shot?

The raw masked-marginal surprisal is itself a predictor — no training required. We
compared it (as an AUROC ranker) against the best trained classifier, per architecture,
using bias-free (modulo-selected) classifiers throughout.

![Zero-shot vs best classifier]({{artifact:art_b8aff65c-908a-4c23-98f1-2b8328a9b5bc}})

*Figure 3. Zero-shot surprisal AUROC vs the best trained classifier, contiguous split.
A classifier on surprisal-only adds essentially nothing; only richer features (Rep4)
buy a small margin.*

| Architecture | Zero-shot | Classifier (surprisal only) | Classifier (richest) | Gain over zero-shot |
| --- | --- | --- | --- | --- |
| ESM-2 650M | 0.8783 | 0.8731 | 0.8892 | 0.0109 |
| ESM-1v | 0.8461 | 0.8357 | 0.8748 | 0.0287 |
| ESM C 600M | 0.8787 | 0.8733 | 0.8991 | 0.0204 |

**Finding.** Zero-shot surprisal alone already reaches **0.85-0.88** on the contiguous
split. A classifier trained on surprisal-only does **not** improve on it (within noise —
thresholding one scalar cannot beat ranking by it). The gain from a trained classifier
comes entirely from *richer features* (Rep4's substitution one-hot on top of surprisal),
and even then it is modest: **+0.011 to +0.029 AUROC** (ESM-2's gain shrank slightly
under bias-free selection, from +0.017 to +0.011, since its Rep4 cell moved from RF to
XGBoost). Practically: if simplicity and deployability matter, zero-shot surprisal
captures the large majority of the achievable signal; the trained classifier buys a
couple of points, not a transformation.

---

## 4. Feature weights: what carries the signal

To rank features without cherry-picking, a single Logistic Regression was fit on **all**
features stacked together (2,559 dimensions: surprisal scalars, 20-dim surprisal vectors,
site embeddings for ESM-2 and ESM C, physicochemical properties, ESMFold structural
features, and identity one-hot), standardized. Coefficient magnitude ranks the signal.
This analysis is unaffected by the classifier-selection fix (it fits a single LR on all
features jointly, with no per-feature-set classifier choice involved).

![Feature-weight analysis]({{artifact:art_26086228-85ba-4a49-9c33-3730a90914d1}})

*Figure 4. (a) Per-feature signal density (mean |LR coefficient| per feature) by block.
(b) Top individual features. The three masked-marginal surprisal scalars are the densest
single signals; ESM C embedding dimensions dominate the top individual features.*

| Feature block | N features | Total \|weight\| | Mean \|weight\|/feature |
| --- | --- | --- | --- |
| ESMC_Rep1_surprisal | 1 | 1.7508 | 1.7508 |
| ESM2_Rep1_surprisal | 1 | 1.4298 | 1.4298 |
| ESM1v_Rep1_surprisal | 1 | 0.9101 | 0.9101 |
| ESMC_site_emb | 1152 | 242.088 | 0.2101 |
| ESM2_site_emb | 1280 | 202.1359 | 0.1579 |
| Identity_onehot | 40 | 5.5778 | 0.1394 |
| Physicochemical | 16 | 1.901 | 0.1188 |
| ESM2_Rep4_site+onehot | 40 | 4.2146 | 0.1054 |
| ESMFold_struct | 8 | 0.747 | 0.0934 |
| ESM2_Rep3_surprisal20 | 20 | 1.2134 | 0.0607 |

**Reading the weights.**
- **The masked-marginal surprisal scalars are the densest signal in the entire matrix** —
  ESM C (1.75), ESM-2 (1.43), ESM-1v (0.91) per feature, the three highest-weighted
  individual features of anything. This is the project's hypothesis made quantitative:
  a single surprisal number per variant carries more predictive weight per feature than
  any embedding dimension, physicochemical property, or structural feature.
- **Site embeddings carry large *total* weight** (they have >1,100 dimensions each) but
  ~0.16-0.21 per feature — the signal is real but distributed across many dimensions.
- **Physicochemical and identity features contribute modestly** (~0.12-0.14/feature);
  the physicochemical charge-change term appears among the stronger individual features.
- **ESMFold structural features have the lowest total block weight** (0.747 across 8
  features; ~0.093/feature) — real but the weakest-contributing block. See §5.

This ranking justifies the feature choices for a final model — surprisal first, site
embeddings second — and is derived by giving the linear model every available feature
rather than by manual selection.

---

## 5. ESMFold predicted-structure arm

A separately-labeled arm tested predicted-structure features. These are **WT
position-mapped** structural features (pLDDT, SASA, relative SASA, burial, secondary
structure, contact number) from a single ESMFold fold of the wild-type protein, mapped to
every variant by position — i.e. *where* a mutation sits structurally, not *which*
substitution it is (per-variant delta-fold features, which would be substitution-specific,
were not computed here and remain future work). All 4,770 variants are covered, so these
AUROCs are directly comparable to the sequence-only arm. Classifier selection uses the
same bias-free (modulo-selected) rule as §2.

![ESMFold predicted-structure arm]({{artifact:art_86934d17-3612-4876-ad52-104c8b9f0908}})

*Figure 5. Contiguous AUROC for structure-alone, structure+physicochemical, and
structure+PLM combinations. Open markers = floors and structure-alone. The best cell
(rightmost) combines structure with all PLM features and physicochemical properties.*

**Findings.**
- **Structure alone reaches contiguous AUROC 0.784** (Random Forest, the pre-specified
  classifier) — above the floors (0.70-0.73) but well below every PLM surprisal
  representation (0.87-0.90). Position-level structural context carries some signal,
  consistent with the weight analysis showing it as the weakest block.
- **Structure + PLM combinations reach ~0.90-0.91.** ESMFold + ESM C Rep4 reaches 0.9072;
  the grand ESMFold + all-PLM Rep4 + physicochemical reaches **0.9107 [0.902,0.919]**
  (XGBoost) — nominally above the best sequence-only cell (0.8991 [0.890,0.908]), but
  **the two 95% confidence intervals now overlap** under bias-free classifier selection
  (grand-combo lower bound 0.902 vs sequence-only upper bound 0.908). An earlier version
  of this analysis reported these as non-overlapping (+0.018 gain); that used
  contiguous-biased classifier picks for both cells (Random Forest for the grand combo)
  and overstated the separation. **The honest read: adding structure to the best PLM
  features may help a little, but this benchmark cannot distinguish that gain from noise
  — it is a plausible small effect, not a demonstrated one.**
- **Interpretation.** Predicted structure is a weak feature on its own, and its value in
  combination with the strongest PLM features is not established at this sample size —
  it is not ruled out either. Because it is a single WT fold reused across all variants
  (folded once, not per-variant), the cost of including it in a deployable pipeline would
  be negligible regardless of whether it turns out to help. Because it is WT
  position-mapped, it cannot distinguish substitutions at the same position, and it is
  blind to catalytic effects that do not perturb the fold; its value for
  substitution-specific or epistatic questions is not tested here. **The final
  single-mutant model (next session) should run an ablation pass (sequence-only →
  +physicochemical → +structure → +all) to determine, with adequate power, whether
  structure contributes anything once the strongest PLM features are already included —
  rather than assuming either that it does or that it doesn't.**

---

## 6. Methods (brief)

- **Data.** Firnberg 2014 DMS for TEM-1 (BLAT_ECOLX), 4770 single-missense variants after
  sealing 13 wet-lab-panel variants; binary label `DMS_score_bin` (balance 0.50).
- **Splits.** 5-fold, seed 42. Random (variant KFold), modulo (position mod 5), contiguous
  (five linear-position blocks 24-76/77-129/130-182/183-234/235-286). Contiguous is primary.
- **Features.** ESM-2 650M, ESM-1v, ESM C 600M masked-marginal surprisal (Rep1 scalar, Rep3
  20-dim vector, Rep4 + substitution one-hot) and mean/site/site-delta embeddings (Rep2a/b/c);
  physicochemical (16); identity one-hot (40); ESMFold WT position-mapped structural (8).
- **Classifiers.** Logistic Regression, Random Forest, XGBoost, SVM (RBF), each with inner
  3-fold grid search. SVM gated to ≤200-dim sets; tree models gated to ≤400-dim sets on the
  wide embedding stacks (LR is both faster and stronger there) — a direction-finding scope,
  not an exhaustive certification.
- **Classifier selection (bias correction).** The classifier reported for a (feature_set)
  is the one with the highest **modulo-split** AUROC, applied identically across random,
  modulo, and contiguous columns. This is a leakage-free choice with respect to the
  contiguous headline: contiguous scores play no role in picking which classifier's number
  gets reported on the contiguous split. All four classifiers' contiguous AUROC are
  additionally reported as a sensitivity block (§2) for full auditability.
- **Metric.** Out-of-fold AUROC; 95% CIs by 2,000-fold bootstrap; significance by DeLong and
  5,000-fold paired bootstrap on paired OOF predictions, using the pre-specified classifier
  on both arms of the comparison.
- **Not done (by design).** No model is locked or saved; the 13 sealed wet-lab variants are
  not scored; per-variant ESMFold delta features were not computed; RF/XGBoost on the wide
  embedding stacks and distilled-feature model selection are deferred.

---

## 7. Limitations

- **Single dataset, single protein.** One DMS study (Firnberg, TEM-1). External validation
  on a second β-lactamase or DMS dataset is required before the finding generalizes.
- **Single mutants only.** This benchmark establishes signal for single missense variants;
  it does not test prediction of unseen double/triple mutants.
- **Direction-finding scope.** The classifier grid was gated for speed (LR-only on wide
  embeddings, trees deferred there); a final model-selection study would run the full grid
  on the shortlisted feature sets.
- **Predicted structure is WT-only,** and its incremental value on top of the best PLM
  features is not resolved by this benchmark (§5) — the point estimate is positive but the
  confidence intervals overlap, so a dedicated, adequately powered comparison is needed
  before treating structure as either helpful or not.

*Tables (CSV) and figures (PNG) accompanying this document are in `results/`.*
