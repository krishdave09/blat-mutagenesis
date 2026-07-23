#!/usr/bin/env python
"""build_esmfold_summary.py -- bias-corrected classifier selection for the ESMFold-arm
document table/figure (docs/05_benchmark_results.md Section 5, Figure 5).

Same methodology as build_architecture_comparison.py: classifier is selected per
feature_set using the MODULO-split AUROC (not contiguous), then that fixed classifier's
scores are reported across splits. All 4 classifiers' contiguous AUROC are attached as a
sensitivity block.

    python src/build_esmfold_summary.py --root .
"""
import argparse
from pathlib import Path
import pandas as pd


def find_root(root):
    p = Path(root).resolve()
    while p != p.parent and not (p / ".projectroot").exists():
        p = p.parent
    return p


ESMFOLD_SELECTED = [
    "Identity one-hot", "Physicochemical", "ESMFold_struct",
    "ESMFold_struct + Physicochemical",
    "ESMFold_struct + ESM-2 650M Rep1 masked-marginal scalar",
    "ESMFold_struct + ESM-2 650M Rep4 site+onehot",
    "ESMFold_struct + ESM C 600M Rep4 site+onehot",
    "ESMFold_struct + AllPLM Rep4 + Physicochemical",
]
SPLITS = ["random", "modulo", "contiguous"]
CLF_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "SVM"]
CLF_SHORT = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB", "SVM": "SVM"}


def pre_specified_row(bench, feature_set):
    sub = bench[bench["feature_set"] == feature_set]
    assert len(sub), f"feature_set not found: {feature_set!r}"
    mod = sub[sub["split"] == "modulo"]; con = sub[sub["split"] == "contiguous"]
    assert len(mod) and len(con), f"missing modulo/contiguous rows for {feature_set!r}"
    common = set(mod["classifier"]) & set(con["classifier"])
    assert common, f"no shared classifier for {feature_set!r}"
    mod_common = mod[mod["classifier"].isin(common)]
    pre_clf = mod_common.loc[mod_common["auroc"].idxmax(), "classifier"]
    out = {"feature_set": feature_set, "selected_classifier": pre_clf}
    for split in SPLITS:
        s = sub[(sub["split"] == split) & (sub["classifier"] == pre_clf)]
        row = s.iloc[0]
        out[f"{split}_auroc"] = round(float(row["auroc"]), 4)
        out[f"{split}_ci"] = f"[{row['ci_lo']:.3f},{row['ci_hi']:.3f}]"
    for clf in CLF_ORDER:
        r = con[con["classifier"] == clf]
        out[f"contiguous_{CLF_SHORT[clf]}"] = round(float(r["auroc"].iloc[0]), 4) if len(r) else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = find_root(args.root)
    tables = root / "results" / "tables"
    bench = pd.read_csv(tables / "plm_benchmark_esmfold.csv")
    rows = [pre_specified_row(bench, fs) for fs in ESMFOLD_SELECTED]
    cols = ["feature_set", "selected_classifier", "random_auroc", "random_ci",
            "modulo_auroc", "modulo_ci", "contiguous_auroc", "contiguous_ci",
            "contiguous_LR", "contiguous_RF", "contiguous_XGB", "contiguous_SVM"]
    out = pd.DataFrame(rows, columns=cols)
    out_path = tables / "esmfold_arm_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows)")
    print(out[["feature_set", "selected_classifier", "contiguous_auroc"]].to_string(index=False))


if __name__ == "__main__":
    main()
