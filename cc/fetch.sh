#!/usr/bin/env bash
# From the laptop: bring back from CC-IN2P3 the result JSONs (newer than the local ones),
# the iterates of script 26 and the logs.
#
#     cc/fetch.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W=/sps/lsst/users/jcomparat01/ggah/wp3feas
D="${WP3FEAS_DATA:-$HOME/data/cog-prop-wp3-feasibility}"
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-cc -o ControlPersist=8h -o BatchMode=yes"
rsync -a --update -e "$SSH" "cc:$W/software/cog-prop-wp3-feasibilty/results/" "$ROOT/results/"
rsync -a --update -e "$SSH" --include="26_map*/***" --exclude="*" "cc:$W/data/" "$D/cc/"
mkdir -p "$D/logs/cc"
rsync -a -e "$SSH" "cc:$W/data/logs/" "$D/logs/cc/"
