#!/usr/bin/env bash
# Script 29 at CC-IN2P3, in the sum_stat venv (Corrfunc), its arguments passed on:
#
#     sbatch cc/wpspec.sbatch map                       # spectroscopic fraction per pixel, first
#     sbatch --array=0-11 cc/wpspec.sbatch run          # one slice-0 sample per task
#     sbatch cc/wpspec.sbatch collect
#SBATCH --job-name=wp3-wpspec
#SBATCH --partition=htc
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --licenses=sps
#SBATCH --output=/sps/lsst/users/jcomparat01/ggah/wp3feas/data/logs/%x-%A_%a.log
set -euo pipefail
SPS=/sps/lsst/users/jcomparat01
export LS11_ROOT=$SPS/ls11_rmagGAL/bgsr21s
export WP3FEAS_DATA=$SPS/ggah/wp3feas/data
export PYTHONPATH=$SPS/tier_ls11/code
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd $SPS/ggah/wp3feas/software/cog-prop-wp3-feasibilty
echo "start $(date -Is) on $(hostname), repository $(cat DEPLOYED_FROM): $*"
$SPS/envs/sumstat/bin/python -u scripts/29_wp_spec.py "$@"
echo "end $(date -Is)"
