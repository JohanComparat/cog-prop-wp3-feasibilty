#!/usr/bin/env bash
# On the laptop (jaxgpu env, the local GPU): a second Gauss-Newton chain of script 26, in its own
# directory (WP3FEAS_MAP_TAG), step -> search -> jac for each iterate; stops when a search stops it.
#
#     nohup local/map_chain.sh _s0 _d10 10 1 4 > ~/data/cog-prop-wp3-feasibility/logs/map_s0_d10.log 2>&1 &
#
# arguments: layout, tag, damping, first iterate, number of iterates
set -euo pipefail
layout="${1-_s0}"; tag="${2:-_local}"; damp="${3:-10}"; start="${4:-1}"; n="${5:-3}"
cd "$(dirname "$0")/.."
export WP3FEAS_MAP_TAG="$tag" PYTHONPATH="$HOME/software/ggah_sens_study_wt_v170${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/software/miniforge3/envs/jaxgpu/bin/python
for k in $(seq "$start" $((start + n - 1))); do
    echo "== $(date -Is) iterate $k"
    $PY -u scripts/26_map_fit.py step --it "$k" --layout "$layout" --damp "$damp" --tag "$tag"
    $PY -u scripts/26_map_fit.py search --it "$k" --layout "$layout" --tag "$tag"
    if grep -q '"stop"' "$HOME/data/cog-prop-wp3-feasibility/26_map${layout}${tag}/it$k/theta.json"; then
        echo "chain stopped at iterate $k"; exit 0
    fi
    $PY -u scripts/26_map_fit.py jac --it "$k" --layout "$layout"
done
echo "== $(date -Is) done"
