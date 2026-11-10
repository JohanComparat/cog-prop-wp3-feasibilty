#!/usr/bin/env bash
# Waits for the local chain (local/map_chain.sh, pid $1); if it ends without "done" or
# "chain stopped" in its log ($2), sends its iterates to CC and continues the chain there
# (cc/submit_map.sh with the same tag and damping) from the first iterate without a Jacobian.
#
#     nohup local/fallback_cc.sh <pid> <log> _s0 _d10 10 4 > <log>.fallback 2>&1 &
set -uo pipefail
pid="$1"; log="$2"; layout="$3"; tag="$4"; damp="$5"; last="$6"
cd "$(dirname "$0")/.."
while kill -0 "$pid" 2>/dev/null; do sleep 60; done
if grep -qE "^== .* done|chain stopped" "$log"; then
    echo "$(date -Is) local chain finished"; exit 0
fi
D=$HOME/data/cog-prop-wp3-feasibility/26_map${layout}${tag}
k=1
while [ -f "$D/it$k/theta.json" ] && [ "$(ls "$D/it$k"/*.npz 2>/dev/null | wc -l)" -gt 100 ]; do k=$((k + 1)); done
echo "$(date -Is) local chain failed; continuing at CC from iterate $k"
rm -rf "${D:?}/it$k"
SSH="ssh -o ControlPath=$HOME/.ssh/cm-cc -o BatchMode=yes"
rsync -a -e "$SSH" "$D/" "cc:/sps/lsst/users/jcomparat01/ggah/wp3feas/data/26_map${layout}${tag}/"
START=$k MAPTAG="$tag" DAMP="$damp" cc/submit_map.sh "$layout" $((last - k + 1))
