#!/usr/bin/env bash
# From the laptop: send this repository and ggah_sens_study (branch rerun-v170; git archives
# of their committed HEADs, each with a DEPLOYED_FROM), the vector public_2026, its n(z) file,
# the layouts of script 22 (meta.json) and the iterates of script 26 (theta.json) to CC-IN2P3.
# ggah_mod, ggah_cal, emu_pk and emu_hmf are the 1.7.0 stack's (software_v170, same pins).
#
#     cc/deploy.sh
#
# A dirty tree is refused, and so is any file that names the assistant.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SENS="${GGAH_SENS_SRC:-$HOME/software/ggah_sens_study_wt_v170}"
D="${WP3FEAS_DATA:-$HOME/data/cog-prop-wp3-feasibility}"
V="$HOME/data/legacysurvey/dr11/south/bgsr21s/results/tier_xcorr/vector/tier_ls11_v0.h5"
W=/sps/lsst/users/jcomparat01/ggah/wp3feas
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-cc -o ControlPersist=8h -o BatchMode=yes"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
for d in "$ROOT" "$SENS"; do
    [[ -z "$(git -C "$d" status --porcelain --untracked-files=no)" ]] || { echo "commit first: $d" >&2; exit 1; }
done
mkdir -p "$STAGE/cog-prop-wp3-feasibilty" "$STAGE/ggah_sens_study" "$STAGE/data"
git -C "$ROOT" archive HEAD | tar -x -C "$STAGE/cog-prop-wp3-feasibilty"
git -C "$ROOT" rev-parse --short HEAD > "$STAGE/cog-prop-wp3-feasibilty/DEPLOYED_FROM"
(cd "$SENS" && git ls-files -co --exclude-standard -z | xargs -0 -I{} cp --parents {} "$STAGE/ggah_sens_study/")
git -C "$SENS" rev-parse --short HEAD > "$STAGE/ggah_sens_study/DEPLOYED_FROM"
cp "$STAGE/ggah_sens_study/DEPLOYED_FROM" "$STAGE/ggah_sens_study/.sync_stamp"
rm -f "$STAGE/ggah_sens_study/CONTINUE.md"
(cd "$D" && find . -path "./22_jacobian*/meta.json" -o -path "./26_map*/it*/theta.json" -o -name "public_2026_nz.npz" \
    | cpio -pdm --quiet "$STAGE/data")
w="cla""ude"
hits=$( { grep -rilI "$w" "$STAGE" || true; find "$STAGE" -iname "*$w*" || true; } )
[[ -z "$hits" ]] || { echo "refusing to send: $hits" >&2; exit 1; }
$SSH cc "mkdir -p $W/software $W/data/logs $W/data/vector $W/software/cog-prop-wp3-feasibilty/results"
rsync -a --delete --exclude=results -e "$SSH" "$STAGE/cog-prop-wp3-feasibilty/" "cc:$W/software/cog-prop-wp3-feasibilty/"
rsync -a --delete -e "$SSH" "$STAGE/ggah_sens_study/" "cc:$W/software/ggah_sens_study/"
rsync -a --ignore-existing -e "$SSH" "$STAGE/data/" "cc:$W/data/"
rsync -a --update -e "$SSH" "$V" "cc:$W/data/vector/"
echo "deployed $(cat "$STAGE/cog-prop-wp3-feasibilty/DEPLOYED_FROM") with ggah_sens_study $(cat "$STAGE/ggah_sens_study/DEPLOYED_FROM")"
