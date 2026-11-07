#!/usr/bin/env bash
# From the laptop: submit the Gauss-Newton chain of script 26 at CC-IN2P3 for one layout,
# each step waiting for the previous one (afterok):
#     jac --it 1 (GPU) -> step --it 2 (CPU) -> jac --it 2 -> ... -> step --it N+1
# Iterate 1 is the step already taken on the laptop (its theta.json sent by cc/deploy.sh).
#
#     cc/submit_map.sh _s0 4          # slice 0 alone, four Jacobians
#     cc/submit_map.sh "" 3           # the three shells
set -euo pipefail
layout="${1-_s0}"; n="${2:-3}"; damp="${DAMP:-10}"
R=/sps/lsst/users/jcomparat01/ggah/wp3feas/software/cog-prop-wp3-feasibilty
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-cc -o ControlPersist=8h -o BatchMode=yes"
tag="${layout:-_s3}"
$SSH cc "cd $R && bash -s" <<REMOTE
set -euo pipefail
J=../../data/logs/jobs.txt
dep=""
for k in \$(seq 1 $n); do
    id=\$(sbatch --parsable --job-name=wp3-jac$tag-it\$k \${dep:+--dependency=afterok:\$dep} cc/gpu.sbatch \
        scripts/26_map_fit.py jac --it \$k --layout "$layout")
    echo "\$id wp3-jac$tag-it\$k" | tee -a \$J
    nxt=\$((k + 1))
    dep=\$(sbatch --parsable --job-name=wp3-step$tag-it\$nxt --dependency=afterok:\$id cc/cpu.sbatch \
        scripts/26_map_fit.py step --it \$nxt --layout "$layout" --damp $damp)
    echo "\$dep wp3-step$tag-it\$nxt" | tee -a \$J
done
REMOTE
