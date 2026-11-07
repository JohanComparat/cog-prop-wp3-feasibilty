#!/usr/bin/env bash
# One resumable GPU step of this study at CC-IN2P3, one V100: the command is rerun, a few
# blocks per process (a fresh process frees the GPU), until it prints "done in".
#
#     sbatch cc/gpu.sbatch scripts/26_map_fit.py jac --it 1 --layout _s0
#SBATCH --job-name=wp3feas-gpu
#SBATCH --partition=gpu_v100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --licenses=sps
#SBATCH --output=/sps/lsst/users/jcomparat01/ggah/wp3feas/data/logs/%x-%j.log
set -euo pipefail
source /sps/lsst/users/jcomparat01/ggah/wp3feas/software/cog-prop-wp3-feasibilty/cc/env.sh
export JAX_COMPILATION_CACHE_DIR="${TMPDIR:-/tmp}/jax_cache_${SLURM_JOB_ID:-$$}"
cd "$WP3FEAS_REPO"
nvidia-smi -L || true
echo "start $(date -Is) on $(hostname), repository $(cat DEPLOYED_FROM), ggah_sens_study $(cat ../ggah_sens_study/DEPLOYED_FROM): $*"
L="${TMPDIR:-/tmp}/run_${SLURM_JOB_ID:-$$}.log"
: > "$L"
for i in $(seq 1 200); do
    "$SENS_PY" -u "$@" --max-blocks 6 2>&1 | tee -a "$L" | grep -v "Warning\|check_calibration" || true
    grep -q "^done in" "$L" && break
    grep -q "^Traceback" "$L" && { echo "failed" >&2; exit 1; }
done
grep -q "^done in" "$L" || { echo "not finished" >&2; exit 1; }
echo "end $(date -Is)"
