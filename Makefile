# Beta-Lactam ML v2 — sequence-only benchmark
# One-line setup:  make setup
ENV_NAME := betalactam-v2

.PHONY: help setup verify clean-interim tree

help:
	@echo "targets:"
	@echo "  make setup   - create/update the conda env and directory tree (runs setup.sh)"
	@echo "  make verify  - run the setup verification notebook headless (PASS/FAIL checklist)"
	@echo "  make tree    - print the project directory tree"
	@echo "  make clean-interim - remove regenerable interim/processed data (keeps data/raw)"

setup:
	bash setup.sh

verify:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_project_setup.ipynb
	@echo "verification notebook executed; open it to read the PASS/FAIL checklist output"

tree:
	@find . -type d -not -path '*/.*' | sort | sed 's|[^/]*/|  |g'

clean-interim:
	rm -f data/interim/* data/processed/*
	@echo "cleared data/interim and data/processed (data/raw untouched)"
