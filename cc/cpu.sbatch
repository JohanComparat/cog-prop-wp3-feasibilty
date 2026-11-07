#!/usr/bin/env bash
# One CPU script of this study at CC-IN2P3, its arguments passed on:
#
#     sbatch cc/cpu.sbatch scripts/26_map_fit.py step --it 2 --layout _s0
#SBATCH --job-name=wp3feas-cpu
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --licenses=sps
#SBATCH --output=/sps/lsst/users/jcomparat01/ggah/wp3feas/data/logs/%x-%j.log
set -euo pipefail
source /sps/lsst/users/jcomparat01/ggah/wp3feas/software/cog-prop-wp3-feasibilty/cc/env.sh
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$WP3FEAS_REPO"
echo "start $(date -Is) on $(hostname), repository $(cat DEPLOYED_FROM): $*"
"$SENS_PY" -u "$@"
echo "end $(date -Is)"
