#!/usr/bin/env bash
# Regenerates every dataset from fixed seeds, then checks them.
#   data/idea1_testfns, data/idea1_solvers   (committed; Idea 1)
#   data/diffusion, data/helmholtz           (gitignored, ~120 MB; Idea 2)
#   results/baselines_*.csv, results/*.png   (Idea 2 baseline solver sweeps; slow, ~1 h)
# Requires: pip install numpy scipy h5py pyamg matplotlib
set -euo pipefail
cd "$(dirname "$0")/python"
python3 idea1.py ../data/idea1_testfns
python3 idea1_solvers.py ../data/idea1_solvers
python3 diffusion.py ../data/diffusion
python3 helmholtz.py ../data/helmholtz
python3 audit.py
if [[ "${1:-}" == "--baselines" ]]; then
    python3 baselines.py diffusion helmholtz
    python3 figures.py
fi
