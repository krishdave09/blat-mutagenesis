#!/usr/bin/env bash
# One-line reproducible setup for Beta-Lactam ML v2 (sequence-only benchmark).
#
#   git clone <repo> && cd blat-mutagenesis && bash setup.sh
#
# Creates the conda env `betalactam-v2` from environment.yml (falls back to a
# pip venv if conda is absent), ensures the directory tree exists, and prints a
# next-step pointer. Idempotent: safe to re-run.
set -euo pipefail

ENV_NAME="betalactam-v2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "==> Beta-Lactam ML v2 setup"
echo "    project root: $HERE"

# --- 1. Environment -------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
        echo "==> conda env '${ENV_NAME}' already exists — updating from environment.yml"
        conda env update -n "${ENV_NAME}" -f environment.yml --prune
    else
        echo "==> creating conda env '${ENV_NAME}' from environment.yml"
        conda env create -f environment.yml
    fi
    echo "==> activate with:  conda activate ${ENV_NAME}"
else
    echo "==> conda not found; creating a pip venv in .venv from requirements.txt"
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "==> activate with:  source .venv/bin/activate"
fi

# --- 2. Directory tree (paths.py also does this, but make it explicit) -----
echo "==> ensuring directory tree"
mkdir -p data/raw data/interim data/processed notebooks src models results/figures results/tables
for d in data/raw data/interim data/processed models results/figures results/tables; do
    [ -e "$d/.gitkeep" ] || touch "$d/.gitkeep"
done

# --- 3. Data check --------------------------------------------------------
if ls data/raw/*Firnberg*.csv >/dev/null 2>&1 || ls data/raw/*firnberg*.csv >/dev/null 2>&1; then
    echo "==> Firnberg DMS data found in data/raw/"
else
    echo "==> NOTE: no Firnberg DMS CSV in data/raw/ yet."
    echo "    Drop BLAT_ECOLX_Firnberg_2014.csv into data/raw/ before running the pipeline."
fi

echo ""
echo "==> setup complete. Next: run notebooks/01_project_setup.ipynb top to bottom"
echo "    to verify the environment with a printed PASS/FAIL checklist."
