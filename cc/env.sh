# The environment of this study at CC-IN2P3: the ggah_mod 1.7.0 stack (ggah_sens_study
# local/cc_stack_v170.sh, eps_dpl closure) with this study's own copy of ggah_sens_study
# (branch rerun-v170, with JointConfig.nz_file and beams_off) first on the path, and the
# paths of wp3feas.  Sourced by the jobs, or in a login shell:
#
#     source /sps/lsst/users/jcomparat01/ggah/wp3feas/software/cog-prop-wp3-feasibilty/cc/env.sh
_WG=/sps/lsst/users/jcomparat01/ggah
_WW=$_WG/wp3feas
source $_WG/software_v170/ggah_sens_study/local/cc_stack_v170.sh
export PYTHONPATH="$_WW/software/ggah_sens_study:$PYTHONPATH"
export WP3FEAS_DATA=$_WW/data
export WP3FEAS_VECTOR=$_WW/data/vector/tier_ls11_v0.h5
export WP3FEAS_REPO=$_WW/software/cog-prop-wp3-feasibilty
unset _WG _WW
