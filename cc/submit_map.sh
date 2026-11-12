#!/usr/bin/env bash
# From the laptop: submit the Gauss-Newton chain of script 26 at CC-IN2P3 for one layout,
# each job waiting for the previous ones (afterok):
#     step --it k -> search --it k (model alone at 1, 1/2, 1/4, 1/8 of the step) -> jac --it k -> step --it k+1 ...
# With SHARDS=n (default 1) the search and the Jacobian of each iterate run as n jobs in
# parallel (--shard i/n, every n-th block), and a search without --shard combines the blocks.
# The first step (START, default 1) uses the Jacobian of iterate START-1 (iterate 0: the
# defaults, sent by cc/deploy.sh); AFTER (job ids joined by ':') makes it wait for running jobs.
# A search where no fraction improves stops the chain: the later jobs then exit at once.
# The gpu QOS takes at most 100 submitted jobs per user; beyond it sbatch refuses silently
# (an iterate costs 2 + 2 n jobs with SHARDS=n): count with `squeue -u $USER -h | wc -l`.
#
#     cc/submit_map.sh _s0 4                       # slice 0 alone, four iterations
#     START=5 AFTER=63097310 cc/submit_map.sh _s0 3
#     MAPTAG=_d10 DAMP=10 SHARDS=4 cc/submit_map.sh _s0 4      # a second chain in 26_map_s0_d10
# VECTOR (a path at CC) fits another vector than v0, MASK a row mask of scripts/25_scale_cuts.py.
set -euo pipefail
layout="${1-_s0}"; n="${2:-3}"; damp="${DAMP:-1}"; start="${START:-1}"; after="${AFTER:-}"
mtag="${MAPTAG:-}"; shards="${SHARDS:-1}"; vec="${VECTOR:-}"; mask="${MASK:-}"
R=/sps/lsst/users/jcomparat01/ggah/wp3feas/software/cog-prop-wp3-feasibilty
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-cc -o ControlPersist=8h -o BatchMode=yes"
tag="${layout:-_s3}"
$SSH cc "cd $R && bash -s" <<REMOTE
set -euo pipefail
J=../../data/logs/jobs.txt
dep="$after"
sub() {  # sub <name> <dependency> <script args...>: prints the job id
    local name=\$1 d=\$2; shift 2
    local id
    id=\$(sbatch --parsable --job-name=\$name --export=ALL,WP3FEAS_MAP_TAG=$mtag${vec:+,WP3FEAS_VECTOR=$vec}${mask:+,WP3FEAS_MASK=$mask} \${d:+--dependency=afterok:\$d} \
        cc/gpu.sbatch scripts/26_map_fit.py "\$@" --layout "$layout" --tag "$mtag" 2>/dev/null)
    echo "\$id \$name" >> \$J
    echo "\$id \$name" >&2
    echo \$id
}
for k in \$(seq $start \$(($start + $n - 1))); do
    s=\$(sub wp3-step$tag$mtag-it\$k "\$dep" step --it \$k --damp $damp)
    if [ $shards -gt 1 ]; then
        ids=""
        for i in \$(seq 0 \$(($shards - 1))); do
            ids=\$ids:\$(sub wp3-search$tag$mtag-it\$k-\$i "\$s" search --it \$k --shard \$i/$shards)
        done
        c=\$(sub wp3-search$tag$mtag-it\$k "\${ids#:}" search --it \$k)
        ids=""
        for i in \$(seq 0 \$(($shards - 1))); do
            ids=\$ids:\$(sub wp3-jac$tag$mtag-it\$k-\$i "\$c" jac --it \$k --shard \$i/$shards)
        done
        dep=\${ids#:}
    else
        c=\$(sub wp3-search$tag$mtag-it\$k "\$s" search --it \$k)
        dep=\$(sub wp3-jac$tag$mtag-it\$k "\$c" jac --it \$k)
    fi
done
REMOTE
