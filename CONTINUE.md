# Where this stands, and what happens next

## 2026-11-12 — Part D: slice-0 chains, χ² by block at the best point

**Done.**
- χ² + prior, slice 0 (10 976 rows): λ = 3 it 5 65 115 (1/4), it 6 62 820 (full); λ = 30 it 3 66 463 (1/2), it 4 61 027 (full); λ = 10 it 3 81 405, stopped.
  Three shells: it 4 117 949 (1/2).
- λ = 30's Jacobian of it 4 had been cancelled by the faulty `scancel` of 10 Nov: resubmitted in four shards, iterations 5–8 sharded after it; λ = 3 at iterations 7–8.
- `23_project_default.py --model-dir 26_map_s0_d3/it6` (`results/23_project_default_d3it6.json`): χ² 113 410 → 66 374; by block type (χ², rows, median amplitude), defaults → it 6:
  galaxy × galaxy 49 893 → 14 844 (420 rows, amplitude 1.47 → 1.08); galaxy × X-ray 11 918 → 4 865; AGN × galaxy 1 844 → 1 087; galaxy × y 1 162 → 385 (amplitude 0.55 → 1.10).
  Unchanged: AGN × X-ray 17 573 (675 rows, amplitude 3.2), X-ray × X-ray 11 874 (1 800 rows), shear × shear 6 232 (994 rows, χ²/n 6.3), y × y 3 559 (15 rows, noise not subtracted).
- Those four hold 39 000 of the 66 374: they need a scale cut or a systematics model (script 25), not more iterations.

**Next.**
- Script 25 on these four: y × y out; AGN × X-ray (the eRASS1 flux limit of the AGN selection, data/model abundance 0.09–0.13); X-ray autos (photon shot noise, the PSF); shear × shear (covariance of the 13 bins, the source n(z)).
- Galaxy × galaxy at χ²/n 35 with amplitude 1.08: the shape, scale by scale (BLUE 10–10.5 at high ℓ).

## 2026-11-11 — Part D: slice-0 MAP chains with the measured n(z)

**Done.**
- Slice 0 (10 976 rows), χ² + prior after each accepted line search (fraction of the step kept):

  | λ | it 1 | it 2 | it 3 | it 4 |
  |---|---|---|---|---|
  | 1 | 100 684 (1/8) | 80 820 (1/2) | 75 780 (1/4) | – |
  | 3 | 100 739 (1/8) | 80 812 (1/2) | 75 779 (1/4) | 66 453 (1) |
  | 10 | 100 237 (1/4) | 96 091 (1/8) | running | |
  | 30 | 78 816 (1/2) | 72 749 (1/4) | running | |

  The full steps at it 1–3 send the model to χ² 10⁵–10¹¹ (the X-ray and AGN blocks), so the line search is what makes progress; λ = 1 and λ = 3 follow the same path.
- Three shells: iterate 3 accepted at the full step, χ² + prior 141 928 → 126 699 on 17 733 rows.
- Sharded Jacobian of slice 0: four jobs of 51–57 min against 3 h 32 min for one job (both on H100).
- λ = 1 stopped (redundant with λ = 3); λ = 3 continued sharded, iterations 5–8.

**Done (later).**
- Script 25 (`results/25_scale_cuts_mask*.json`): ℓ ≥ 30, ℓ ≤ 1 h/Mpc χ(z), y × y out keeps 45 785 of 85 606 C_ℓ rows; at λ = 3 it 6, χ² 62 816 → 43 136 on 9 251 rows; without the X-ray autos of v0, 31 262 on 7 451 rows (4.2 per row). Left: galaxy × galaxy 7 946 / 280, AGN × X-ray 8 769 / 450, shear × shear 6 232 / 994.
- X-ray re-measurement (sum_stat `tier_xray_background.py`, `tier_xray_fields.py --mask`, `tier_xray_footprint_cl.py`): background map from eRASS1 Main; footprints b30_p25 … b20_p70 of 2 347–7 794 deg²; masks: Magellanic Clouds, Virgo, Fornax, 96 789 Galactic eRASS1 sources, 63 422 Gaia stars G < 8; AGN and clusters kept.
  Broad band 0.5–2.0 keV with exposure weights: effective f_sky 0.014–0.036 (the deep ecliptic-pole fields dominate), power 4–27 times Lau et al. (2025) Fig. 2, rising with ℓ: the unmasked point sources (Lau mask every EXT_LIKE = 0 source). The decision rule picks b30_p25 (successive steps differ by 7–22 per cent).
  Comparison with Lau et al. (2025) on b30_p70, binary mask (f_sky 0.151): our ℓ(ℓ+1)C_ℓ/2π over their Fig. 2 is 2.3–9.7 (ℓ = 100–2000) with the AGN kept; with every eRASS1 point source masked (903 521, EXT_LIKE = 0, 2 × APE_RADIUS_1; 5 631 deg², f_sky 0.130) it is 0.74, 0.81, 1.16, 1.24, 1.37 at ℓ = 100, 200, 500, 1000, 2000, within the 20 per cent of reading their figure plus their absorption correction and cluster mask; shot noise over signal 0.21 at ℓ = 500.
  So the map is sound, and the excess with the AGN kept is the AGN.
  Binary weights (f_sky 0.055, 0.088, 0.120, 0.151, 0.182 for b30_p25 … b20_p70): C_ℓ over the step before 0.96 (χ²/band 3.9), 1.02 (1.0), 1.06 (12.2), 0.96 (8.6).
  The strict rule (χ²/band < 2) stops at b30_p25; we take **b30_p55** (5 132 deg²): p25 → p55 changes C_ℓ by 2 per cent in total and p40, p55 agree; the first clear excess (+6 per cent) comes with the higher-background fields of p70.
  Full C_ℓ at CC with the b30_p55 X-ray fields (`tier_ls11_measure.py --tag _xrb30p55 cl --xray --xray-footprint b30_p55`, job 63560954: jackknife 94/100 after 4 h 40 min); the strict footprint b30_p25 in `cl_nside1024_xrb30p25.h5` (job 63618595): X-ray autos and crosses with galaxies, AGN and y; the shear crosses then through `tier_shear_cl.py`.
- `docs/model_status.md`: the model per work package; the defaults of ggah_mod stay until the masked fit converges.

**Next.**
- Compare λ = 3, 10, 30 after iterations 3–5; keep one chain, shard it, stop the others.
- The residual blocks at the best point: where the remaining χ² of 66 453 sits (BLUE autos, galaxy × y, AGN × X-ray, y × y).

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

- The suffix fix also went into ggah_sens_study (branch rerun-v170, 3533e61): `joint.survey.nz_tables` answers `name` with the table `name@0` in a one-shell layout. sum_stat never writes build names and needs no change.
- `proj` with BLUE (n(z) files matched by stem): data w(θ) over projected data w_p 1.75 and 1.52 (BLUE 10–10.5, 10.5–11), 1.14–1.28 (RED), 1.17–1.55 (ALL); the π_max truncation is 1 per cent in linear theory, so the excess left after the kernel fix is real, largest for BLUE 10–10.5.
- Queued at CC after the slice-0 Jacobian (63296995): script 23 `_s0` (63307317), 28 v2 (63307318), v3 (63307319), slice-0 MAP iterations 1–3 (63307321–29; the void iterates moved to `26_map_s0_tophat`).
  Three shells: Jacobian of iterate 2 completed (63307330), iterations 3–5 (63307331–39).

- With the measured n(z) (slice-0 Jacobian 63296995, 2 h 10 min on H100): χ² at the defaults 135 910 → 113 410 over 10 991 rows; median amplitudes galaxy × galaxy 2.18 → 1.47, AGN × galaxy 1.99 → 1.50, galaxy × X-ray 0.99 → 0.85 (`results/23_project_default_s0_nz.json`).
  w(θ) data/model on 12–120 arcmin: RED 1.13–1.39, BLUE 1.52–1.86, as `proj` predicted; v2 build = independent Limber to 0.1 per cent.
- Slice-0 MAP iterate 1: linear χ² 109 927 → 67 947, 39 of 79 parameters at the trust limit; line search running.
- Three shells: step of iterate 3 stopped on `bcut@2` outside its range at the start; `26_map_fit.py step` now clips such parameters, chain resubmitted (63396878–86).

- Parallel chains on slice 0 at CC, λ = 1, 3, 10, 30 (`MAPTAG`, iterates in `26_map_s0{,_d3,_d10,_d30}`); the laptop GPU could not run the step (one XLA compile took 30 min), so λ = 10 went to CC through `local/fallback_cc.sh`.
- `--shard i/n` for the search and the Jacobian (scripts 22, 26), `SHARDS=n` in `cc/submit_map.sh`: λ = 1 and λ = 3 run sharded (4 GPUs per search and per Jacobian) from their first Jacobian on; λ = 10, λ = 30 and the three shells unsharded.
  The gpu QOS caps a user at 100 submitted jobs and sbatch then refuses silently: shard the other chains once jobs have drained.

**Next.**
- The BLUE excess left (w(θ) 1.5–1.9 at fixed w_p) and galaxy × y at 0.55, AGN × X-ray at 2.9 (slice 0).
- Done: when job 63296995 ends: script 23 `_s0`, `28 v3` and `xslice` again; the galaxy autos should come down from about 2 to the 1.1–1.5 left by `proj`, part of which is the π_max truncation.
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
