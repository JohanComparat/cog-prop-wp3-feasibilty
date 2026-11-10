# Where this stands, and what happens next

## 2026-11-10 — Part D: MAP chains at CC-IN2P3, line search, audit of slice 0 planned

**Done.**
- Three shells: iterate 1 (trust 0.125) accepted, χ² + prior 205 386 → 159 277 on 17 733 rows (linear prediction 160 796).
- Slice 0 alone: iterates 1–3 (trust 1, 0.5, 0.25) rejected, χ² + prior 1.4 × 10³³, 5 × 10¹⁸⁰ and 133 547 against 132 427 at the defaults; the X-ray predictions diverge at the larger steps.
  About 18 GPU hours of Jacobians went to these rejected points.
- `scripts/26_map_fit.py search`: the model alone at 1, 1/2, 1/4 and 1/8 of each step, each block compiled once; the best fraction that lowers χ² + prior becomes the iterate, else the chain stops.
  `cc/submit_map.sh` now chains step → search → jac, with `START` and `AFTER` to continue a running chain.
- Resubmitted at CC: slice 0 iterations 5–7 (jobs 63274484–92, after jac it4 63097310), three shells iterations 3–5 (63274494–502, after jac it2 63097319).
- PLAN.md: audit D0 of slice 0 (`scripts/20_vector_audit.py`), four groups of tests.
- `scripts/20_vector_audit.py data` (1a–1c) and `abundance` (4a), run at CC: no pair of slice-0 samples has r > 1 beyond 2σ (91 pairs); the bias ratios from the autos agree with the median of 26 crosses (galaxies, AGN, y, 13 shear bins) within 1.3σ; C(ALL) equals the RED/BLUE combination to 0.1 per cent.
  Data over default-model number density: BLUE 1.51, 1.93, 2.47 (M* 10–10.5, 10.5–11, 11–11.5), RED 0.65, 0.99, 1.12, 0.91, AGN 0.09–0.13.
- The BLUE names of sum_stat carry a count truncated by one digit (e.g. `N_130142` for 1 301 429); nothing reads it.

- `scripts/20_vector_audit.py wp`: linear bias from w_p (10–30 Mpc), from the data C_ℓ and from the default model's C_ℓ; the model C_ℓ of the galaxy autos is 0.39–0.55 of the data.
  The default-model amplitudes of script 23 (slice 0): galaxy × galaxy 2.18, AGN × galaxy 1.99, galaxy × shear 1.01, galaxy × X-ray 0.99 (medians): the discrete × discrete C_ℓ of the model is suspect, not the data.
- PLAN.md: D1, the validation of the C_ℓ and w(θ) path against w_p and the data (V1–V5).
- The pending MAP jobs at CC are held (`scontrol hold`; release with `scontrol release <ids>`).

- D1 V1 and V2 (`scripts/28_cl_validation.py`, CPU at CC): the model's 3D bias equals the data w_p bias (RED 10–10.5: 1.31 against 1.31) and its w_p matches the data to 0.83–1.19 on 1–30 Mpc; the build's C_ℓ equals an independent Limber integral of its own P_gg and kernels to 0.1 per cent.
  The data C_ℓ autos are 1.8–2.6 times the model (RED, flat from ℓ = 30 to 2000); a top-hat n(z) on the slice, the narrowest possible, removes only a factor 1.2–1.4.
  Slice-0 × slice-1 ALL crosses: r = 0.05–0.14 on 30 ≤ ℓ ≤ 200.

- D1 V3: data w(θ) over model w(θ) on 12–120 arcmin is 1.80–2.54, as in C_ℓ.
- D1 xslice: the slice-0 × slice-1 RED crosses agree with the model (r 0.02–0.11 against 0.05–0.11, within 1σ); BLUE data r 0.22–0.25 against model 0.18–0.20.
- D1 V1b: with the photo-z scatter of the members (f_spec 0.20–0.44, σ_z 0.02), the model w_p falls to 0.58–0.77 of the data on 1–30 Mpc: the data w_p would need a 3D bias higher by about 1.25, which brings the model C_ℓ up by about 1.55.

- The LS10 w_p of ggah_cal were corrected for photo-z dilution; the tier w_p of sum_stat (BEST_Z) are not.
  The photo-z members of slice 0 have BEST_Z_ERR 0.017 at the median (0.009–0.027, 16th–84th percentile); the spectroscopic fraction exceeds 0.8 in nside-64 pixels holding 6 per cent of the members.
- `scripts/29_wp_spec.py` (V1c, sum_stat venv at CC, `cc/wpspec.sbatch`): w_p of all members (BEST_Z) against w_p of the spectroscopic members (weight 1/f_spec per pixel) in those pixels, eleven slice-0 samples; jobs 63294646 (map), 63294647 (array 0–10), 63294648 (collect).
- Galaxy × shear amplitude per sample (script 23, slice 0, inverse-variance over 13 bins) against √(galaxy auto amplitude): RED 1.02, 0.90, 0.85, 0.88 (± 0.02–0.04) against 1.54, 1.39, 1.31, 1.54; BLUE 1.54, 1.22, 1.00 against 1.93, 1.52, 1.45.
  Lensing does not allow the bias to rise by 1.3–1.5 for RED: the auto excess has a correlation coefficient with matter of about 0.6–0.7 relative to the model, or the galaxy–lensing prediction is high.

- Script 29 result (`results/29_wp_spec.json`): in the spectroscopic region (f_spec 0.84–0.95, 6 per cent of the members), w_p of all members over w_p of the spectroscopic members is 0.95–1.01 on 1–10 Mpc (± 0.01–0.03) for the samples below 11.5.
  The region's w_p over the full-sky tier w_p (f_spec 0.20–0.36) is 0.92–0.95 for RED, 0.98–1.08 for ALL and 1.15–1.19 for BLUE on 1–30 Mpc.
  The full-sky RED w_p is not lower than the nearly spectroscopic one: its photo-z dilution is at most 10 per cent, not the 35 per cent of V1b (the V1b Gaussian model overstates it).
  The tier w_p stands, the default bias is right, and explanation 1 (bias raised by dilution, lensing low) is ruled out for RED.
  The excess sits in the discrete × discrete angular spectra (C_ℓ and w(θ)) alone.

- `scripts/28_cl_validation.py proj` (model-free): the data w_p projected with the data n(z) gives a w(θ) below the data w(θ) by 1.14–1.55 on 12–120 arcmin (ALL and RED, π_max = 100 Mpc, so a lower limit), while the model w(θ) is 2 times below: the model's angular projection is at fault.
- `kern`: **bug found.** In the one-shell layout the build names its tracers without the `@0` suffix (`gb10p`), the n(z) file keys carry it (`gb10p@0`), so `selection()` found no measured n(z) and every slice-0 photometric galaxy and AGN fell back to the top-hat smoothed with σ_z0 = 0.03(1 + z): ∫W² dχ = 0.00160 /Mpc against 0.00192–0.00322 /Mpc for the measured n(z), an auto C_ℓ low by 1.2–2.0.
  Galaxy × shear and galaxy × X-ray barely depend on the kernel width, which is why they passed.
  Fix in `wp3feas.model.public_config`: a one-shell layout reads `public_2026_nz_s0.npz`, the slice-0 tables without the suffix; after it the kernels match the n(z) to 1 per cent (job 63296994).
  The three-shell build has matching names and is not affected.
- The slice-0 Jacobian at the defaults is void: moved to `22_jacobian_s0_tophat` at CC, recomputed with the fix (job 63296995, gpu_h100); the slice-0 MAP iterates built on it are void too.

**Next.**
- When job 63296995 ends: script 23 `_s0`, `28 v3` and `xslice` again; the galaxy autos should come down from about 2 to the 1.1–1.5 left by `proj`, part of which is the π_max truncation.
- D1 (superseded by the bug): find the factor 2 of the discrete × discrete angular spectra: the model kernel of photometric samples (normalisation of S = n(z) vol/dV, the units of n̄), the data δ maps (weights, n̄ over the footprint), and w(θ) of a spectroscopic subsample against its w_p projected with its own n(z).
- D1 (superseded): collect script 29; then decide between a bias raised by the dilution with low lensing, and an excess in the discrete × discrete spectra alone (galaxy × galaxy 2.18, AGN × galaxy 1.99 against galaxy × shear 1.01 and galaxy × X-ray 0.99).
- D1 (earlier list): data w_p of the spectroscopic members alone (photo-z dilution at π_max = 100 Mpc); model w(θ) against data w(θ); model C_ℓ of the slice-0 × slice-1 crosses (`scripts/28_cl_validation.py`, CPU at CC).
- Model side: the eRASS1 flux limit in the AGN selection; the quenched fraction against the observed g − z split; then 1d, 2, 3 of D0.

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
