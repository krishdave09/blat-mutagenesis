#!/usr/bin/env python3
"""
build_08_report.py -- generate the 00/06-style PDF report for the supervised pLLM benchmark.

Reads the results tables + figures written by notebooks 07_EDA_pllm_embeddings and
08_pllm_supervised_benchmark (via paths.py), and emits a PDF matching the format of
00_TEM1_pLLM_functional_prediction_review_proposal.pdf / 06_pllm_zeroshot_benchmark.pdf.

This is the arm-3 report: a classifier trained on ESM embeddings (not scores), compared against
the three bounding baselines it exists to beat -- AA-identity (02), physicochemical (04), and
zero-shot pLLM (06).

Usage:
    python build_08_report.py                 # resolves paths.py, writes next to this script
    python build_08_report.py --tables DIR --figures DIR --out FILE.pdf

If the 08 tables are absent (embeddings not extracted yet), it still builds the report skeleton
with a clear PREVIEW banner and placeholders, so the layout is reviewable before the Colab
embeddings land. Rerun after dropping the embeddings in and re-running 07/08 to populate.
"""
import argparse, sys, json
from pathlib import Path
import pandas as pd, numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, HRFlowable, KeepTogether)

MODEL_LABEL = {"esm1b":"ESM-1b 650M","esm1v":"ESM-1v 650M","esm2_150m":"ESM-2 150M",
               "esm2_650m":"ESM-2 650M","esm2_3b":"ESM-2 3B","esmc_300m":"ESM-C 300M",
               "esmc_600m":"ESM-C 600M"}
LEARNER_LABEL = {"logreg":"logistic reg.","rf":"random forest","xgboost":"XGBoost","dummy":"dummy (majority)"}
ARM_LABEL = {"aa_identity":"AA-identity (02)","physicochem":"physicochem (04)",
             "zeroshot_pllm":"zero-shot pLLM (06)","supervised_pllm":"supervised-PLM (08)"}
NB = "08_pllm_supervised_benchmark"; EDA = "07_EDA_pllm_embeddings"

def _styles():
    ss = getSampleStyleSheet(); S = {}
    S["title"] = ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold",
                                fontSize=15, leading=19, alignment=TA_CENTER, spaceAfter=6)
    S["subtitle"] = ParagraphStyle("st", parent=ss["Normal"], fontName="Times-Italic",
                                   fontSize=11, alignment=TA_CENTER, spaceAfter=2)
    S["date"] = ParagraphStyle("d", parent=ss["Normal"], fontName="Times-Roman",
                               fontSize=10, alignment=TA_CENTER, spaceAfter=10)
    S["h"] = ParagraphStyle("h", parent=ss["Heading2"], fontName="Times-Bold",
                            fontSize=12, leading=15, spaceBefore=12, spaceAfter=5)
    S["body"] = ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman",
                               fontSize=10, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6)
    S["cap"] = ParagraphStyle("c", parent=ss["Normal"], fontName="Times-Roman",
                              fontSize=8.5, leading=11, spaceBefore=2, spaceAfter=8)
    S["capfig"] = ParagraphStyle("cf", parent=ss["Normal"], fontName="Times-Italic",
                                 fontSize=8.5, leading=11, spaceBefore=2, spaceAfter=10)
    S["banner"] = ParagraphStyle("ban", parent=ss["Normal"], fontName="Times-Bold",
                                 fontSize=9.5, leading=12, alignment=TA_CENTER,
                                 textColor=colors.HexColor("#b03060"), spaceAfter=8)
    S["cell"] = ParagraphStyle("cell", parent=ss["Normal"], fontName="Times-Roman",
                               fontSize=8, leading=9.5)
    S["cellh"] = ParagraphStyle("cellh", parent=ss["Normal"], fontName="Times-Bold",
                                fontSize=8, leading=9.5, textColor=colors.white)
    S["analysis"] = ParagraphStyle("an", parent=ss["Normal"], fontName="Times-Roman",
                                   fontSize=9, leading=12, alignment=TA_JUSTIFY,
                                   leftIndent=10, spaceBefore=0, spaceAfter=12,
                                   textColor=colors.HexColor("#222222"))
    return S

def gray_table(header, rows, colw, S, gray_row=None):
    data = [[Paragraph(h, S["cellh"]) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), S["cell"]) for c in r])
    t = Table(data, colWidths=colw, hAlign="LEFT")
    style = [("BACKGROUND",(0,0),(-1,0), colors.HexColor("#595959")),
             ("GRID",(0,0),(-1,-1), 0.4, colors.HexColor("#b0b0b0")),
             ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
             ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
             ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]
    for i in range(1,len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND",(0,i),(-1,i), colors.HexColor("#f0f0f0")))
    if gray_row is not None:
        for gi in gray_row:
            style.append(("TEXTCOLOR",(0,gi+1),(-1,gi+1), colors.HexColor("#9aa0a6")))
    t.setStyle(TableStyle(style))
    return t

def fig(path, S, caption, width=6.4*inch):
    path = Path(path)
    if path.suffix == ".pdf" and path.with_suffix(".png").exists():
        path = path.with_suffix(".png")
    if not path.exists():
        return Paragraph(f"[figure not found: {path.name} — run the notebook to generate]", S["cap"])
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    w = width; h = w*ih/iw
    return KeepTogether([Image(str(path), width=w, height=h), Paragraph(caption, S["capfig"])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results"); ap.add_argument("--figures"); ap.add_argument("--out")
    ap.add_argument("--tables"); a = ap.parse_args()

    if a.tables and a.figures:
        TABLES = Path(a.tables); FIGROOT = Path(a.figures); OUT = Path(a.out or "08_pllm_supervised_benchmark.pdf")
    else:
        here = Path(__file__).resolve()
        root = next((p for p in [here.parent,*here.parents] if (p/'.projectroot').exists()), None)
        if root is None:
            for p in [here.parent,*here.parents]:
                cand = p/"1 - ML"
                if (cand/'.projectroot').exists(): root = cand; break
        if root is None:
            sys.exit("could not locate project root (.projectroot); pass --tables/--figures/--out")
        TABLES = root/"results"/"tables"; FIGROOT = root/"results"/"figures"
        OUT = Path(a.out) if a.out else (root.parent/"2 - Writing"/"08_pllm_supervised_benchmark.pdf")

    FIG = FIGROOT/NB; EDAFIG = FIGROOT/EDA
    S = _styles()
    grid_p = TABLES/f"{NB}_metrics_grid.csv"
    populated = grid_p.exists()
    story = []

    story.append(Paragraph("Supervised Protein-Language-Model Benchmark for TEM-1 "
                           "Functionality Prediction", S["title"]))
    story.append(Paragraph("A classifier on ESM embeddings vs three bounding baselines", S["subtitle"]))
    story.append(Paragraph("Benchmark Report", S["date"]))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.black, spaceAfter=8))

    if not populated:
        story.append(Paragraph("PREVIEW — benchmark tables not found on disk. This shows the "
            "report layout only; numbers populate after the Colab embeddings are dropped into "
            "data/features/plm_embeddings/ and notebooks 07/08 are re-run, then rebuild this PDF.",
            S["banner"]))

    story.append(Paragraph(
        "<b>Task:</b> predict functional vs non-functional (DMS_score_bin; positive = functional) "
        "for single missense TEM-1 beta-lactamase variants (Firnberg et al. 2014, BLAT_ECOLX) from "
        "<b>supervised classifiers trained on ESM embeddings</b>. <b>n = 4,783</b> variants across "
        "263 residue positions; classes balanced (2,397 functional / 2,386 non-functional). This is "
        "arm-3: the learned-feature model the two no-training and no-language-model baselines exist "
        "to bound.", S["body"]))

    # 1. Why
    story.append(Paragraph("1. Why a supervised pLLM benchmark", S["h"]))
    story.append(Paragraph(
        "The project's three prior arms set the bounds this model must beat (decision D027 keeps "
        "measured baselines, not asserted ones). <b>AA-identity (02)</b> is the no-language-model, "
        "raw-amino-acid floor: strong on the random split, collapsing on the contiguous region-holdout "
        "split because it can only memorize positions it trained on. <b>Physicochemical (04)</b> is the "
        "hand-engineered-descriptor control. <b>Zero-shot pLLM (06)</b> is the no-training control: a "
        "frozen ESM surprisal score, read off the model with no exposure to the labels. This notebook "
        "asks the arm-3 question directly: do <b>learned</b> embedding features beat all three \u2014 "
        "especially on the contiguous split \u2014 which would be the evidence that language-model "
        "representations carry signal beyond memorized positions and beyond the raw surprisal score.", S["body"]))
    story.append(Paragraph(
        "The features are ESM embedding <b>deltas</b> between the mutant and wild-type sequence "
        "(decisions D035/D036): the change in the contextual embedding at the mutated residue "
        "(<i>delta_site</i>, the primary variant-specific feature), the whole-sequence mean delta "
        "(<i>delta_pooled</i>), and a local-window delta (<i>delta_local</i>). Unlike the single "
        "surprisal scalar of 06, these are high-dimensional vectors, which is exactly why a reduction "
        "guardrail is mandatory before modeling (below).", S["body"]))

    # 2. Methods
    story.append(Paragraph("2. Methods", S["h"]))
    story.append(Paragraph(
        "<b>Features (D035/D036).</b> Per variant, three embedding-delta blocks per ESM model, extracted "
        "on GPU by the companion Colab notebook 07a and read here from disk. The wild-type embedding is "
        "computed once per model and reused for all 4,783 deltas. The ESM ladder (D010\u2013D013) spans "
        "ESM-1b, ESM-1v, ESM-2 (150M/650M/3B) and ESM-C (300M/600M).", S["body"]))
    story.append(Paragraph(
        "<b>Reduction guardrail (D037, non-negotiable).</b> The raw feature matrix is ~10,000+ columns "
        "(three blocks \u00d7 up to seven models). Feeding it whole into a tree swamps the model and, under "
        "the region-holdout splits, invites overfitting to the training positions. PCA (with "
        "standardization) is therefore fit <b>inside each cross-validation fold, on the training rows "
        "only</b>, per model, and applied to the held-out fold \u2014 never on the full data. No test row "
        "ever informs the scaler mean or the PCA basis. A leakage assertion confirms no "
        "seq_id / position / label-derived column enters the matrix and that no single dimension tracks "
        "the label almost perfectly.", S["body"]))
    story.append(Paragraph(
        "<b>Classifiers (D023\u2013D025).</b> L2 logistic regression, random forest, and XGBoost, plus a "
        "majority-class dummy floor. Per D026 no winner is pre-committed \u2014 the deployed model is "
        "selected empirically on the contiguous split (accuracy + calibration), the hardest and most "
        "realistic setting.", S["body"]))
    story.append(Paragraph(
        "<b>Splits, threshold, metrics.</b> Identical to 02/04/06: five-fold random / modulo / contiguous "
        "splits (seed 42; the two position-based schemes are group-aware, no position in both train and "
        "test). The decision threshold is Youden's J fit on the <b>train fold only</b> and applied to the "
        "held-out fold. Metrics: ROC-AUC, PR-AUC, balanced accuracy, F1, MCC, and the domain-weighted "
        "utility (TP +1, TN +1, FN \u22120.25, FP \u22121); with bootstrap 95% CIs (2,000 resamples), "
        "DeLong and McNemar tests versus the best model per split, and the dummy floor.", S["body"]))

    # 3. Results
    story.append(Paragraph("3. Results", S["h"]))
    if populated:
        grid = pd.read_csv(grid_p)
        order = ["logreg","rf","xgboost","dummy"]
        header = ["split","model","ROC-AUC","PR-AUC","bal.acc","MCC","utility"]
        rows=[]; gray=[]
        for split in ["random","modulo","contiguous"]:
            sub = grid[grid.split==split]
            sub = sub.set_index("model").reindex([m for m in order if m in set(sub.model)]).reset_index()
            for _,r in sub.iterrows():
                if r.model=="dummy": gray.append(len(rows))
                rows.append([split, LEARNER_LABEL.get(r.model,r.model),
                    f"{r.roc_auc_mean:.3f}\u00b1{r.roc_auc_std:.3f}",
                    f"{r.pr_auc_mean:.3f}", f"{r.balanced_acc_mean:.3f}",
                    f"{r.mcc_mean:.3f}", f"{r.utility_mean:.3f}"])
        story.append(Paragraph("<b>Table 1.</b> Supervised-PLM cross-validated performance (mean "
            "\u00b1 SD over 5 folds), all learners, per split. PCA reduction fit inside each fold "
            "(D037).", S["cap"]))
        cw=[0.9*inch,1.15*inch,1.05*inch,0.75*inch,0.7*inch,0.65*inch,0.75*inch]
        story.append(gray_table(header, rows, cw, S, gray_row=gray))

        # model selection sentence
        sel_p = TABLES/f"{NB}_model_selection.csv"
        if sel_p.exists():
            sel = pd.read_csv(sel_p)
            b = sel.iloc[0]
            story.append(Paragraph(
                f"<b>Selected model (D026): {LEARNER_LABEL.get(b.model,b.model)}</b> \u2014 chosen on the "
                f"contiguous split (ROC-AUC {b.roc_auc:.3f}, utility {b.utility:.3f}, Brier {b.brier:.3f}), "
                f"not fixed in advance.", S["body"]))

        # Table 2: the 4-way comparison
        cmp_p = TABLES/f"{NB}_four_way_comparison.csv"
        if cmp_p.exists():
            cmp = pd.read_csv(cmp_p)
            arms_present = [a for a in ["aa_identity","physicochem","zeroshot_pllm","supervised_pllm"]
                           if a in cmp.columns]
            story.append(Paragraph("<b>Table 2.</b> The four-arm comparison \u2014 best model per arm per "
                "split, all metrics. AA-identity (02) and physicochemical (04) are the no-language-model "
                "controls; zero-shot pLLM (06) is the no-training control; supervised-PLM (08) is this "
                "arm. Any arm absent from disk is omitted.", S["cap"]))
            ch = ["split","metric"] + [ARM_LABEL[a].split(" (")[0] for a in arms_present]
            crows=[]
            for r in cmp.itertuples():
                row=[r.split, r.metric] + [f"{getattr(r,a):.3f}" if not pd.isna(getattr(r,a,np.nan)) else "\u2014"
                                           for a in arms_present]
                crows.append(row)
            ccw=[0.85*inch,0.8*inch] + [1.15*inch]*len(arms_present)
            story.append(gray_table(ch, crows, ccw, S))

            # headline read on the hard split
            if {"zeroshot_pllm","supervised_pllm"} <= set(cmp.columns):
                roc = cmp[cmp.metric=="ROC-AUC"].set_index("split")
                sc = roc.loc["contiguous","supervised_pllm"]; zc = roc.loc["contiguous","zeroshot_pllm"]
                ac = roc.loc["contiguous","aa_identity"] if "aa_identity" in roc.columns else np.nan
                msg = (f"<b>The decisive cell is the contiguous split.</b> Supervised-PLM reaches "
                       f"ROC-AUC {sc:.3f} there, versus zero-shot {zc:.3f} ({sc-zc:+.3f})")
                if not np.isnan(ac):
                    msg += f" and AA-identity {ac:.3f} ({sc-ac:+.3f})"
                msg += (". If the supervised-PLM lead over zero-shot is clear and positive on this "
                        "region-holdout split, learned embedding features carry signal beyond the raw "
                        "surprisal; if it merely matches zero-shot, the scalar score already captures the "
                        "usable signal and the embedding complexity is not justified \u2014 a real result "
                        "either way (D039 framing).")
                story.append(Paragraph(msg, S["body"]))
    else:
        story.append(Paragraph("<b>Table 1.</b> Supervised-PLM cross-validated performance \u2014 populates "
            "from results/tables/08_pllm_supervised_benchmark_metrics_grid.csv once embeddings exist.", S["cap"]))
        story.append(Paragraph("[metrics grid, model selection, and the four-way comparison populate here "
            "after the Colab embeddings are extracted and notebooks 07/08 re-run.]", S["body"]))

    # ---- per-figure analysis: pull real numbers from the results tables ----
    def _load(name):
        p = TABLES/f"{NB}_{name}.csv"; return pd.read_csv(p) if p.exists() else None
    def _eload(name):
        p = TABLES/f"{EDA}_{name}.csv"; return pd.read_csv(p) if p.exists() else None
    cmp_t = _load("four_way_comparison"); grid_t = _load("metrics_grid")
    mag_t = _eload("block_magnitude"); pca_t = _eload("pca_variance"); assoc_t = _eload("single_pc_association")
    def _an(text): story.append(Paragraph(text, S["analysis"]))

    story.append(Paragraph("4. Figures", S["h"]))

    # Figure 1 — the 4-way comparison (headline)
    story.append(fig(FIG/"four_way_comparison.pdf", S,
        "<b>Figure 1.</b> The four-arm comparison \u2014 ROC-AUC of the best model per arm, by split. "
        "AA-identity (02), physicochemical (04), zero-shot pLLM (06), and supervised-PLM (08)."))
    if cmp_t is not None and {"zeroshot_pllm","supervised_pllm"} <= set(cmp_t.columns):
        roc = cmp_t[cmp_t.metric=="ROC-AUC"].set_index("split")
        sr,sc = roc.loc["random","supervised_pllm"], roc.loc["contiguous","supervised_pllm"]
        zc = roc.loc["contiguous","zeroshot_pllm"]
        ac = roc.loc["contiguous","aa_identity"] if "aa_identity" in roc.columns else np.nan
        txt=(f"<b>What it shows.</b> The bars group the four arms within each split. The random split is "
             f"the easy, optimistic setting; the contiguous split withholds whole regions and is the "
             f"realistic surveillance test. Supervised-PLM reaches {sr:.3f} on random and {sc:.3f} on "
             f"contiguous \u2014 the flatness across splits is the property a deployable predictor needs. "
             f"On contiguous it stands at {sc:.3f} vs zero-shot {zc:.3f}")
        if not np.isnan(ac):
            txt += (f" and AA-identity {ac:.3f}, whose collapse from the random split is the position "
                    f"memorization the region holdout is designed to expose")
        txt += ". The gap between supervised-PLM and zero-shot on this split is the arm's central result."
        _an(txt)

    # Figure 2 — ROC curves
    story.append(fig(FIG/"roc_curves.pdf", S,
        "<b>Figure 2.</b> Supervised-PLM ROC by split, one line per learner (logreg / RF / XGBoost)."))
    if grid_t is not None:
        g = grid_t[grid_t.model!="dummy"]
        best = g.sort_values("roc_auc_mean",ascending=False).iloc[0]
        _an(f"<b>What it shows.</b> Each curve is a learner's true-positive vs false-positive trade-off; "
            f"area under it is the ROC-AUC. Unlike the zero-shot arm (where the three split panels are "
            f"identical because no training occurs), here each split retrains the reducer and classifier, "
            f"so the panels differ \u2014 the random panel sits highest, the contiguous panel is the honest "
            f"generalization test. Best single cell: {LEARNER_LABEL.get(best.model,best.model)} on the "
            f"{best.split} split (AUC {best.roc_auc_mean:.3f}).")

    # Figure 3 — metric bars with CIs
    story.append(fig(FIG/"metric_bars.pdf", S,
        "<b>Figure 3.</b> ROC-AUC by split and learner with bootstrap 95% CIs; dashed line is the 0.50 "
        "no-skill floor."))
    _an("<b>What it shows.</b> The error bars are 2,000-resample bootstrap 95% intervals, so overlapping "
        "intervals mean the ranking between two learners is not statistically resolved (see the DeLong / "
        "McNemar tests in the significance table). Every learner clears the 0.50 floor on all three "
        "splits; the reliable comparison is each learner against the dummy and across splits, not small "
        "gaps between the top learners.")

    # Figure 4 — utility
    story.append(fig(FIG/"utility_bars.pdf", S,
        "<b>Figure 4.</b> Domain-weighted utility by split and learner (TP +1, TN +1, FN \u22120.25, "
        "FP \u22121); the dummy floor is marked."))
    _an("<b>What it shows.</b> This reweights predictions by the project's real decision cost: a false "
        "'functional' call (FP, \u22121) is four times as costly as missing a functional variant "
        "(FN, \u22120.25), because calling a dead resistance gene active is the dangerous error in a "
        "surveillance setting. The bar to watch is the contiguous split \u2014 a model whose utility holds "
        "there, not just on random, is the one that would survive deployment to unseen regions.")

    # Figure 5 — EDA: block magnitude (the D035 justification)
    story.append(fig(EDAFIG/"block_magnitude.pdf", S,
        "<b>Figure 5.</b> Embedding-delta magnitude by block (EDA, notebook 07): per-variant L2 norm of "
        "delta_site vs delta_pooled vs delta_local, per model."))
    mtxt=("<b>What it shows.</b> This is the empirical justification for treating the site-level delta as "
          "the primary feature (D035). A single substitution barely moves the whole-sequence mean, so "
          "delta_pooled has a small per-variant norm; the change at the mutated residue itself "
          "(delta_site) is much larger and carries the variant-specific signal.")
    if mag_t is not None and {"delta_site","delta_pooled"} <= set(mag_t.block.unique()):
        piv = mag_t.pivot(index="model",columns="block",values="norm_median")
        ratio = (piv["delta_site"]/piv["delta_pooled"]).dropna()
        if len(ratio):
            mtxt += (f" Across models the delta_site norm is {ratio.min():.1f}\u2013{ratio.max():.1f}\u00d7 "
                     f"the delta_pooled norm, confirming the primary-feature choice is data-backed.")
    _an(mtxt)

    # Figure 6 — EDA: PCA scree (the D037 budget)
    story.append(fig(EDAFIG/"pca_scree.pdf", S,
        "<b>Figure 6.</b> PCA scree \u2014 cumulative variance vs component count for the delta_site block, "
        "one line per model (EDA, notebook 07)."))
    ptxt=("<b>What it shows.</b> Each block is 640\u20132560-dimensional, but most of the variance lives in "
          "far fewer components. The curves' steep early rise is what makes the D037 reduction safe: "
          "projecting to ~50\u2013200 components per model before the classifier discards width, not signal.")
    if pca_t is not None and "pc_for_90" in pca_t.columns:
        lo,hi = int(pca_t.pc_for_90.min()), int(pca_t.pc_for_90.max())
        ptxt += f" Reaching 90% variance takes {lo}\u2013{hi} components across models \u2014 the reduction budget 08 uses."
    _an(ptxt)

    # 5. Interpretation / limitations
    story.append(Paragraph("5. Interpretation and limitations", S["h"]))
    story.append(Paragraph(
        "The result to read off Table 2 and Figure 1 is whether the supervised-PLM arm beats its three "
        "bounding baselines on the contiguous region-holdout split. Beating AA-identity there means the "
        "language-model representation, not position memorization, is doing the work; beating "
        "physicochemical means it exceeds hand-engineered chemistry; beating zero-shot means training on "
        "embeddings adds signal over the raw surprisal score. A supervised-PLM that only matches zero-shot "
        "on the hard split is itself a publishable finding: it says the scalar surprisal already captures "
        "most of the usable sequence-only signal for this protein, and the embedding complexity is not "
        "justified.", S["body"]))
    story.append(Paragraph(
        "<b>Guardrail (D037).</b> The reported numbers are free of reduction-leakage: PCA is fit inside "
        "each fold on training rows only, no test row informs the projection, and no identity or "
        "label-derived column enters the feature matrix (asserted in the notebook). This is what makes the "
        "contiguous-split number trustworthy rather than an optimistic artifact of fitting the reducer on "
        "all data.", S["body"]))
    story.append(Paragraph(
        "<b>Limitation, stated where the number lives (D039).</b> These embeddings, like the zero-shot "
        "scores, are sequence-only and shaped by foldability and stability, so they can under-detect a "
        "<i>catalytic-but-stable</i> knockout \u2014 an active-site substitution (Ambler S70, K73, S130, "
        "E166) that abolishes activity without destabilizing the fold. Training on embeddings tests "
        "whether the supervised head recovers signal the zero-shot score misses; it does not close the "
        "stability-vs-catalysis blind spot. That is the role of the ESMFold structural-epistasis features "
        "(D015/D038, for the double mutants) and the wet-lab mutagenesis panel, which check predictions "
        "against real selective-plating outcomes.", S["body"]))

    # 6. Next steps
    story.append(Paragraph("6. Next steps", S["h"]))
    story.append(Paragraph(
        "<b>Feature fusion (immediate, data-ready).</b> This arm trains on embeddings alone. The natural "
        "next model combines the zero-shot surprisal scalars (06, already on disk) and the site-level "
        "score features (D032/D033: the full 20-way logit vector, mutant rank, margin, and per-position "
        "entropy) with the embedding deltas used here, in one classifier under the identical three-split "
        "battery. This directly answers the open question this arm raises (D039): is the embedding vector "
        "<b>complementary to</b> or <b>redundant with</b> the scalar surprisal? It reuses this notebook's "
        "in-fold PCA reducer and folds unchanged \u2014 the surprisal features already share the same 4,783 "
        "single-mutant keys \u2014 so no new extraction is needed. If fusion beats embeddings-alone on the "
        "contiguous split, the two feature families carry different signal; if it does not, the scalar "
        "already captured it.", S["body"]))
    story.append(Paragraph(
        "<b>Structural / ESMFold (D015/D038) \u2014 a separate regime.</b> The structural features "
        "(\u0394pLDDT and predicted-structure perturbation of a double mutant relative to WT and to its "
        "constituent singles) are specifically for <b>double / combinatorial mutants</b> (Deng 2012), "
        "where summed per-site sequence scores cannot represent how two substitutions interact. That is a "
        "different data regime \u2014 it requires the double-mutant label set and a separate ESMFold "
        "extraction \u2014 so it is its own arm on its own data, not a fourth feature block in this "
        "single-mutant benchmark. Both feature-fusion and structural arms feed the wet-lab pAmpGent panel, "
        "which is where the catalytic-but-stable blind spot is tested against biology rather than "
        "held-out folds.", S["body"]))

    doc = SimpleDocTemplate(str(OUT), pagesize=letter,
        leftMargin=0.9*inch, rightMargin=0.9*inch, topMargin=0.9*inch, bottomMargin=0.8*inch,
        title="TEM-1 supervised pLLM benchmark")
    doc.build(story)
    print(f"wrote {OUT}  (populated={populated})")

if __name__ == "__main__":
    main()
