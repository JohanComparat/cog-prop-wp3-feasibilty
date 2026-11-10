#!/usr/bin/env bash
# From the laptop: submit the Gauss-Newton chain of script 26 at CC-IN2P3 for one layout,
# each job waiting for the previous one (afterok), all on one GPU:
#     step --it k -> search --it k (model alone at 1, 1/2, 1/4, 1/8 of the step) -> jac --it k -> step --it k+1 ...
# The first step (START, default 1) uses the Jacobian of iterate START-1 (iterate 0: the
# defaults, sent by cc/deploy.sh); AFTER makes it wait for a running job (that Jacobian).
# A search where no fraction improves stops the chain: the later jobs then exit at once.
#
#     cc/submit_map.sh _s0 4                       # slice 0 alone, four iterations
#     START=5 AFTER=63097310 cc/submit_map.sh _s0 3
set -euo pipefail
layout="${1-_s0}"; n="${2:-3}"; damp="${DAMP:-1}"; start="${START:-1}"; after="${AFTER:-}"
R=/sps/lsst/users/jcomparat01/ggah/wp3feas/software/cog-prop-wp3-feasibilty
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-cc -o ControlPersist=8h -o BatchMode=yes"
tag="${layout:-_s3}"
$SSH cc "cd $R && bash -s" <<REMOTE
set -euo pipefail
J=../../data/logs/jobs.txt
dep="$after"
for k in \$(seq $start \$(($start + $n - 1))); do
    for cmd in step search jac; do
        extra=""; [ \$cmd = step ] && extra="--damp $damp"
        id=\$(sbatch --parsable --job-name=wp3-\${cmd}$tag-it\$k \${dep:+--dependency=afterok:\$dep} cc/gpu.sbatch \
            scripts/26_map_fit.py \$cmd --it \$k --layout "$layout" \$extra)
        echo "\$id wp3-\${cmd}$tag-it\$k" | tee -a \$J
        dep=\$id
    done
done
REMOTE
