#!/usr/bin/env python
"""build_architecture_comparison.py -- curated architecture x representation comparison.

Builds results/tables/architecture_comparison.csv from plm_benchmark_sequence_only.csv.

Classifier selection methodology (bias-corrected):
  For each (architecture, representation, feature_set) triple and EACH split, the
  reported classifier is picked using the MODULO-split AUROC -- never the split being
  reported. Concretely: for a given feature_set, find the classifier with the highest
  modulo-split AUROC; that SAME classifier's score is then reported for random, modulo,
  and contiguous splits alike. This avoids selecting the classifier from the same score
  being headlined (picking by contiguous AUROC and then reporting that contiguous AUROC
  is circular and optimistically biased).

  All four classifiers' contiguous-split AUROCs are also included as a sensitivity block
  (columns clf_LR/RF/XGB/SVM_contiguous) so the headline choice is auditable and the
  reader can see how much the pre-specified pick differs from the best-of-4 alternative.

  This corrects an earlier version of this script that picked argmax(AUROC) independently
  per split -- i.e. selected the classifier using the contiguous score itself for the
  contiguous column. The fix changes only 3 of 21 cells by a small margin (<=0.0074
  AUROC); the headline PLM-vs-floor cell (ESM C 600M Rep4, XGBoost) is unaffected because
  XGBoost already wins that feature set on both the modulo and the contiguous split.

  Note on Rep2a/b/c (raw embedding) rows: these show NaN for RF/XGB/SVM because the
  benchmark gated RF/XGBoost to <=400-dim feature sets and SVM to <=200-dim (embeddings
  are 1152-1280 dim) -- this is a genuine design gate, verified clean across 8 of 9
  Rep2a/b/c sets. One exception was found and removed from the source CSV: 'ESM-2 650M
  Rep2a mean-pooled emb' carried a single stray Random Forest row (random split only, not
  modulo/contiguous) left over from an interrupted benchmark run, not the design gate.
  This selection logic already required a classifier present on BOTH modulo and
  contiguous, so the stray row could not have been selected regardless -- but it has been
  deleted from plm_benchmark_sequence_only.csv so it cannot mislead a future reader
  scanning the raw table.

The curated triples are a deliberate editorial subset (not every feature_set in the
benchmark is shown, and ESM-2 650M gets one representation -- Rep1 surprisal + physchem --
that ESM-1v/ESM C 600M don't) -- stated explicitly below rather than derived, so a rerun
can't silently change which representations get surfaced.

    python src/build_architecture_comparison.py --root .
"""
import argparse
from pathlib import Path
import pandas as pd


def find_root(root):
    p = Path(root).resolve()
    while p != p.parent and not (p / ".projectroot").exists():
        p = p.parent
    return p


# (architecture, representation label, feature_set) -- matches the existing file row order exactly
CURATED = [
    ("Floor", "Physicochemical", "Physicochemical"),
    ("Floor", "Identity one-hot", "Identity one-hot"),
    ("ESM-2 650M", "Rep4 site+onehot", "ESM-2 650M Rep4 site+onehot"),
    ("ESM-2 650M", "Rep1 surprisal + physchem", "ESM-2 650M Rep1 masked-marginal scalar + Physicochemical"),
    ("ESM-2 650M", "Rep2b site-emb", "ESM-2 650M Rep2b site emb"),
    ("ESM-2 650M", "Rep1 surprisal", "ESM-2 650M Rep1 masked-marginal scalar"),
    ("ESM-2 650M", "Rep2c \u0394site-emb", "ESM-2 650M Rep2c site-delta emb"),
    ("ESM-2 650M", "Rep3 surprisal-20", "ESM-2 650M Rep3 surprisal vector"),
    ("ESM-2 650M", "Rep2a mean-emb", "ESM-2 650M Rep2a mean-pooled emb"),
    ("ESM-1v", "Rep4 site+onehot", "ESM-1v Rep4 site+onehot"),
    ("ESM-1v", "Rep1 surprisal", "ESM-1v Rep1 masked-marginal scalar"),
    ("ESM-1v", "Rep3 surprisal-20", "ESM-1v Rep3 surprisal vector"),
    ("ESM-1v", "Rep2c \u0394site-emb", "ESM-1v Rep2c site-delta emb"),
    ("ESM-1v", "Rep2a mean-emb", "ESM-1v Rep2a mean-pooled emb"),
    ("ESM-1v", "Rep2b site-emb", "ESM-1v Rep2b site emb"),
    ("ESM C 600M", "Rep4 site+onehot", "ESM C 600M Rep4 site+onehot"),
    ("ESM C 600M", "Rep2a mean-emb", "ESM C 600M Rep2a mean-pooled emb"),
    ("ESM C 600M", "Rep1 surprisal", "ESM C 600M Rep1 masked-marginal scalar"),
    ("ESM C 600M", "Rep2b site-emb", "ESM C 600M Rep2b site emb"),
    ("ESM C 600M", "Rep2c \u0394site-emb", "ESM C 600M Rep2c site-delta emb"),
    ("ESM C 600M", "Rep3 surprisal-20", "ESM C 600M Rep3 surprisal vector"),
]

SPLITS = ["random", "modulo", "contiguous"]
CLF_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "SVM"]
CLF_SHORT = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB", "SVM": "SVM"}


def pre_specified_row(bench, feature_set):
    """Select the classifier on modulo-split AUROC (bias-free), then report that SAME
    classifier's score on every split. Also attach all 4 classifiers' contiguous AUROC
    as a sensitivity block."""
    sub = bench[bench["feature_set"] == feature_set]
    assert len(sub), f"feature_set not found in plm_benchmark_sequence_only.csv: {feature_set!r}"
    mod = sub[sub["split"] == "modulo"]
    con = sub[sub["split"] == "contiguous"]
    assert len(mod) and len(con), f"missing modulo/contiguous rows for {feature_set!r}"

    # classifiers available on BOTH modulo and contiguous (guards against a partial grid)
    common = set(mod["classifier"]) & set(con["classifier"])
    assert common, f"no classifier present on both modulo and contiguous for {feature_set!r}"
    mod_common = mod[mod["classifier"].isin(common)]
    pre_clf = mod_common.loc[mod_common["auroc"].idxmax(), "classifier"]

    out = {"selected_classifier": pre_clf}
    for split in SPLITS:
        s = sub[(sub["split"] == split) & (sub["classifier"] == pre_clf)]
        if not len(s):
            # classifier absent for this split (rare partial grid) -- fall back to that
            # split's own best-of-available, and flag it explicitly rather than silently
            # substituting a different criterion.
            s_any = sub[sub["split"] == split]
            row = s_any.loc[s_any["auroc"].idxmax()]
            out[f"{split}_auroc"] = round(float(row["auroc"]), 4)
            out[f"{split}_ci"] = f"[{row['ci_lo']:.3f},{row['ci_hi']:.3f}]"
            out[f"{split}_clf"] = row["classifier"] + " (fallback: pre-specified clf missing)"
        else:
            row = s.iloc[0]
            out[f"{split}_auroc"] = round(float(row["auroc"]), 4)
            out[f"{split}_ci"] = f"[{row['ci_lo']:.3f},{row['ci_hi']:.3f}]"
            out[f"{split}_clf"] = pre_clf

    # sensitivity block: all 4 classifiers' contiguous AUROC (± CI), whatever is available
    for clf in CLF_ORDER:
        r = con[con["classifier"] == clf]
        tag = CLF_SHORT[clf]
        if len(r):
            out[f"contiguous_{tag}"] = round(float(r["auroc"].iloc[0]), 4)
        else:
            out[f"contiguous_{tag}"] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = find_root(args.root)
    tables = root / "results" / "tables"

    bench = pd.read_csv(tables / "plm_benchmark_sequence_only.csv")
    rows = []
    for architecture, representation, feature_set in CURATED:
        row = {"architecture": architecture, "representation": representation, "feature_set": feature_set}
        row.update(pre_specified_row(bench, feature_set))
        rows.append(row)

    cols = ["architecture", "representation", "feature_set", "selected_classifier",
            "random_auroc", "random_ci", "random_clf", "modulo_auroc", "modulo_ci", "modulo_clf",
            "contiguous_auroc", "contiguous_ci", "contiguous_clf",
            "contiguous_LR", "contiguous_RF", "contiguous_XGB", "contiguous_SVM"]
    out = pd.DataFrame(rows, columns=cols)
    out_path = tables / "architecture_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows)")

    # report cells where the pre-specified classifier differs from the naive best-of-4-on-contiguous
    naive_best = out[["contiguous_LR", "contiguous_RF", "contiguous_XGB", "contiguous_SVM"]].idxmax(axis=1)
    changed = out[out.apply(lambda r: CLF_SHORT[r["selected_classifier"]] != naive_best[r.name].replace("contiguous_", ""), axis=1)]
    if len(changed):
        print(f"\n{len(changed)} cell(s) where pre-specified classifier != naive best-of-4-on-contiguous:")
        print(changed[["feature_set", "selected_classifier", "contiguous_auroc"]].to_string(index=False))
    else:
        print("\nPre-specified classifier matches naive best-of-4-on-contiguous in every row.")


if __name__ == "__main__":
    main()
