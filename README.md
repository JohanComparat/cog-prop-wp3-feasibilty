# cog-prop-wp3-feasibilty

This repository tests whether the full data vector of five redshift shells can be measured, modelled and fitted as WP3 of the ERC Consolidator proposal UNCUT plans.
It takes the forecast runs of `ggah_sens_study` (reference `S5_public_2031`: public data, ACT DR6 $y$ map) and a public data vector built from the Legacy Surveys DR11, eRASS1 and ACT DR6, and produces:
- the multipole each map and the shear reach in each shell;
- the tolerance and cost of catastrophic photometric-redshift outliers;
- a compression of the vector and the validation of its covariance on sub-blocks, with the number of light-cones it needs;
- the cost of one likelihood gradient and of a Hamiltonian Monte Carlo chain;
- one summary file, `results/wp3_feasibility_summary.json`, read by the proposal.

## Reading order

1. `PLAN.md`: the questions, the method, the schedule and the Tracker.
2. `CONTINUE.md`: what was done in each session and what comes next.
3. `docs/`: the results, one page per part of the plan, as they are written.

## Environment

Everything runs in the shared conda env `dev` of `~/software/dev_env`; NaMaster runs in `dev-full`, GPU timing in `jaxgpu`.
The repository has no environment of its own, and `wp3feas/` is not installed: run the scripts from the repository root and the tests with `python -m pytest`.
Outputs larger than a JSON go to `~/data/cog-prop-wp3-feasibility/`.
