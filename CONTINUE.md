# Where this stands, and what happens next

## 2026-11-07 — Part D: Fisher, default-model projection, first MAP step

**Done.**
- `scripts/22_fisher_public.py`: the Jacobian of the 1.7.0 model (152 parameters, 2 247 pairs, 175 blocks) under `WP3FEAS_DATA/22_jacobian/`; the Fisher in the reported basis (σ8, S8) with the study's priors.
  σ(S8) is 0.0062, 0.0074 and 0.0100 in the slices z 0.1–0.2, 0.2–0.3 and 0.3–0.6, and 0.0054 jointly (`results/22_fisher_public.json`).
  The covariance is block-diagonal jackknife, so these are lower limits.
- `scripts/23_project_default.py`: χ² = 208 869 over 17 748 rows in 1 220 blocks at the defaults.
  The shear C_ℓ pass in 462 of 552 blocks; the galaxy C_ℓ do not (blue 10–10.5 at z 0.1–0.2 is 2–3 times above the model); y×y fails (noise not subtracted); AGN × X-ray at z 0.3–0.6 is 5–20 times above the model.
- `scripts/26_map_fit.py`: a linear Gauss–Newton step with cosmology fixed predicts χ² 127 549 (λ = 10), with steps up to 8 prior widths.

**Next.**
1. Score the full model at the step (`26_map_fit.py score --it 1`); iterate, or move to a gradient-based MAP on per-block vector–Jacobian products.
2. Script 24 (null tests) and 25 (scale cuts: y×y out, AGN × X-ray investigated).
3. Script 27 (NUTS) on the masked vector; the analytic covariance (Part E).

## 2026-11-03 — repository set up

**Done.**
- Layout copied from `cog-prop-wp1-feasibilty`: `wp3feas/` (provenance, store, timing, style), tests, hooks, plan.

**Next.**
1. Part A: `scripts/01_inputs.py` on `shells_2031_summary.json` (`S5_public_2031`).
2. Part B: the $\ell_{\max}$ table.
3. Part D: the vector `public_2026` comes from `~/software/sum_stat` (HDF5), hoped for Fri 6 Nov; its fit the week of 9 Nov. Write the adapter once the first files exist.
