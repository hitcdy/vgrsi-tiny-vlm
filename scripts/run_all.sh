#!/usr/bin/env bash
# One-command reproduction of every table and figure in the paper.
#   bash scripts/run_all.sh          # real backbone (needs weights)
#   bash scripts/run_all.sh mock     # simulated backbone, no weights needed
set -euo pipefail
BACKEND="${1:-real}"
SEED="${2:-42}"

python -m src.verifier_audit                                   # Table 4 (model-free)
for G in 4 8; do
  python -m src.experiments --backend "$BACKEND" --arm all --grid "$G" --seed "$SEED"
done
python -m src.tables  --backend "$BACKEND" --grid 4            # Table 1 / Table 2 inputs
python -m src.figures --backend "$BACKEND" --grid 4            # Fig. 2-6
echo "[run_all] done. Tables -> paper/table1_main.md, Figures -> paper/figs/"