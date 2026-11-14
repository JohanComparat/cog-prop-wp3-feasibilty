# WP3 feasibility: the full data vector in five redshift shells

*Plan written on 3 November 2026. The work runs from 3 to 27 November 2026, inside the November plan of the proposal (`~/software/erc-cog-prop/review/plan_2026-11.md`, Gates W1–W3).*

## Tracker

*Status: `open`, `running`, `done (date, where)` or `dropped (reason)`. Last update: 2026-11-14 (chain `_shot` with the AGN Poisson term: χ² + prior 35 866 at it 11).*

| Item | What | Proposal item | Status |
|---|---|---|---|
| S | repository set up (layout, provenance, tests, plan) | – | done (2026-11-03) |
| A | inputs: layout, row and spectra counts of `S5_public_2031`; what it assumes public for LSST Y3 and Euclid DR2 | I4 | open |
| B | $\ell_{\max}$ per map and shell at $k = 1, 3, 10$ h/Mpc against the ACT beam, the eROSITA PSF, shear shape noise and shot noise | N8, I11 | open |
| C | catastrophic photometric-redshift outliers: tolerance on $f_{\rm out}$ at $0.3\sigma$ on $S_8$, and cost when free | Ref. 1 pt 5, Table 3 | open |
| D | the public data vector `public_2026` (`tier_ls11_v0.h5`, LS DR11 × eRASS1 × ACT DR6 × DES/KiDS/HSC shear) fitted with `ggah_mod` 1.7.0: reader, Fisher per redshift slice, projection of the default model, null tests, scale cuts, MAP, NUTS | I19, I24 | running: reader (21), Fisher per slice (22: σ(S8) = 0.0054 joint, block-diagonal jackknife) and default-model projection (23: χ² = 208 869 over 17 748 rows) done 2026-11-07 in `results/`; MAP (26) running; null tests, scale cuts and NUTS open |
| D0 | audit of the slice 0.1–0.2 of `public_2026` against the data themselves, external measurements and the default model (`scripts/20_vector_audit.py`, at CC-IN2P3) | I19 | running: tests 1a–1c and 4a done 2026-11-10 (`results/20_vector_audit_{data,abundance}.json`): the data are self-consistent; the model's abundances are not (BLUE ×1.5–2.5 too few, RED 10–10.5 ×1.5 too many, AGN ×8–11 too many); AGN 2026-11-13 (`results/25_scale_cuts_agnx.json`): ×1.3–1.8 too many once the counts use the AGN footprint and the model the catalogue band 0.2–2.3 keV and the eRASS1 completeness |
| D1 | validation of the model's $C_\ell$ and $w(\theta)$ against its $w_p$ and the data, steps V1–V5 (`scripts/28_cl_validation.py`, at CC-IN2P3); MAP chains held until it passes | I19 | running: V1 and V2 passed 2026-11-10 (`results/28_cl_validation_v{1,2}.json`): the model's w_p fits the data w_p (0.83–1.19 on 1–30 Mpc) and its C_ℓ is the Limber integral of its P_gg to 0.1 per cent; the data C_ℓ of the galaxy autos is 1.8–2.6 times the model's, flat in ℓ for RED |
| E | covariance: light-cone count, compression, sub-block validation, one-shell C3 and its five-shell scaling | N11, I12, I27 | open |
| F | timing of one likelihood value and gradient, and of a NUTS chain for 212 parameters | N10, I13 | open |
| G | synthesis: `results/wp3_feasibility_summary.json`, the B2 figures, N1, N6, N7 read from `ggah_sens_study` | N1, N6, N7 | open |
| P | proposal edits: B2 WP3, Table 3, R3–R4, Gantt (PD4 from M16, T1–T2 at M40–M44, a science result per PhD), B1 lines | I17–I21 | open |

## The questions

The mock reviews (`erc-cog-prop/review/step2_evaluation_2026-10.md` §1, §3.2–3.3, §4.2) leave these WP3 points open:
1. To which multipole does each map contribute in each shell (Referee 1, pt 3)?
2. How are catastrophic outliers of the photometric shells modelled, and what do they cost (Referee 1, pt 5)?
3. How is a covariance of about 80 000 rows validated, on how many light-cones (Referee 1, pt 4)?
4. What do one likelihood gradient and a sampling chain cost (interview question 11)?
5. What part of the vector can already be measured on public data, and does the model describe it?

The intrinsic alignments (N6), the matched calibration pair (N1) and the shell edges (N7) are computed in `ggah_sens_study`; this study reads them.

## Method

**Part A.** `scripts/01_inputs.py` reads `shells_2031_summary.json` and the `S5_public_2031` configuration, and records the shells, samples, bands and counts that the other parts use.

**Part B.** `scripts/11_ellmax.py` converts $k$ to $\ell = k\,\chi(z_i)$ per shell and compares it with the scale at which each map loses its signal; it reports the fraction of the Fisher information on $S_8$ above each map's $\ell_{\max}$.

**Part C.** `scripts/12_photoz_outliers.py` adds an outlier fraction to the $n(z)$ of the photometric samples, computes the induced cross-shell spectra in the Limber approximation with the `ggah_mod` power spectra, and propagates them to $S_8$ with the existing Jacobians.

**Part D.** The vector `public_2026` is `tier_ls11_v0.h5` (written on 5 November 2026).
It holds 4 484 blocks: 2 702 C_ℓ, 1 020 shear C_ℓ, 496 w(θ), 182 w_p and 84 ΔΣ, with 100 jackknife realisations.
The redshift slices are z = 0.10–0.20 and 0.20–0.30, and 0.30–0.60 for the AGN with $L_X \geq 10^{43.5}$ erg/s.
The model is the `ggah_mod` 1.7.0 default (`agn_energy="eps_dpl"`, the sector defaults, the `StudyFiducial` cosmology of `ggah_sens_study`).
A jackknife covariance from 100 regions has rank at most 99, so it serves blocks of fewer than about 50 rows.
The joint fits use the analytic Gaussian covariance of `ggah_mod`, validated against the jackknife on the diagonals and on sub-blocks (Part E).
- `scripts/20_vector_audit.py` (D0) looks for the cause of the misfit of the slice 0.1–0.2, which the default model misses by χ² = 135 910 over 10 991 rows, and which three Gauss–Newton steps of 1, 0.5 and 0.25 prior widths did not lower.
  The tests come in four groups, ordered from those that need no model to those that need one; each states which outcome points to the data and which to the model.
  1. *Internal consistency, no model.*
     (a) The cross-correlation coefficient $r_{ab}(\ell) = C_{ab}/\sqrt{C_{aa}C_{bb}}$ of every pair of slice-0 fields, with its jackknife error: $r > 1$ beyond $2\sigma$ means an auto-spectrum with too much shot noise subtracted or a cross with too little.
     (b) The linear-bias triangle: on $30 \le \ell \le 200$, $b_a/b_b$ from $C_{aa}/C_{bb}$, from $C_{ac}/C_{bc}$ for every third field $c$ (galaxies, y, each shear bin) must agree.
     (c) The nested samples: the spectra of ALL must equal the number-weighted sum of the RED and BLUE autos and their cross.
     (d) The shot noise: the subtracted level against $1/\bar n$ from `tier_ls11_counts.py`, and the high-ℓ plateau of the autos before subtraction.
     (e) $C_\ell$ against $w(\theta)$, already passed by the galaxy autos in sum_stat (median ratio 1.004–1.029), extended to the AGN, the y and the photon crosses.
     (f) Each lens × shear spectrum from DES Y3, KiDS-1000 and HSC Y3 at the same lens, rescaled by its source $n(z)$, and the shear EE against the published spectra.
  2. *Normalisation of the fields.*
     The photon fields: their unit (counts per pixel, per exposure, per band width and effective area) against the one the band emissivity of `ggah_mod` predicts, and the band autos against the published eRASS1 diffuse-background power spectrum.
     The y map: y × y minus the simulated noise (`tier_y_noise.py data`) against the ACT DR6 tSZ power spectrum, and the stacked y against Das et al. (2023), already in `compare_das2023.py`.
  3. *Redshift distributions.*
     The large-scale bias from $w_p$ on 10–30 Mpc, which does not depend on $n(z)$, against the bias from $C_\ell$ on $30 \le \ell \le 200$ with the sum_stat $n(z)$: a ratio away from 1 points to the width or the tails of $n(z)$.
     The same with a top-hat $n(z)$ on the slice, and with the closure-test shift and stretch at $\pm1\sigma$.
  4. *Against the default model.*
     The number density of each sample (the discrete rows) against the model's: a factor of 3 in the low-mass BLUE autos with the right abundance points to the occupation; with the wrong abundance, to the stellar-mass definition (LS11 masses against the model's SHMR).
     A figure per block family of slice 0, the data against the default and against the best accepted iterate of script 26, with the ratio and the whitened residual per ℓ, under `WP3FEAS_DATA/20_vector_audit/`.
  The script writes `results/20_vector_audit.json` with one verdict per test (pass, data, model, undecided); the tests that fail on the data go back to sum_stat as issues on the vector.
- `scripts/28_cl_validation.py` (D1, at CC-IN2P3) validates the angular path of the model, $C_\ell$ and $w(\theta)$, against the projected path $w_p$, which `ggah_cal` validated on the LS10 data.
  It was opened on 2026-11-10 because, at the defaults in slice 0, the model needs a factor of 2.18 (median of 28 blocks) on galaxy × galaxy and 1.99 on AGN × galaxy, but 1.01 on galaxy × shear and 0.99 on galaxy × X-ray, whereas the data give the same bias from the autos and from the crosses (D0, test 1b).
  The steps go from the 3D spectrum outwards, each with a tolerance, and the next step starts only when the previous one passes.
  V1. *One 3D spectrum.* For one sample (RED 10.5–11, z 0.1–0.2), $P_{gg}(k, z)$ and $P_{gm}(k, z)$ from `ggah_mod` at the parameters with which $w_p$ (`ggah_mod.observables.real_space.wp`) fits the LS11 $w_p$, against the spectrum that `ggah_sens_study.joint.model.build` holds for the same pair on its Limber nodes: same $k$ (h/Mpc), $z$ and units, to 0.1 per cent.
  V2. *The projection.* From that $P_{gg}$ and the sample's $n(z)$, $C_\ell$ by an independent Limber integral (numpy, as in script 20), by `ggah_mod.observables.limber.c_ell` with `kernels.number_counts`, and by the joint build (zones summed as in script 22's `_assemble`); the three agree to 0.5 per cent on $30 \le \ell \le 2048$.
  The same for galaxy × shear and galaxy × y, where the model already matches the data, and for AGN × galaxy.
  A disagreement here is a code defect, fixed where it lives (`ggah_mod`, or `ggah_sens_study` on branch `rerun-v170`), with a unit test there.
  V3. *Real space.* $w(\theta)$ from the model $C_\ell$ (`transforms.cl_to_wtheta`, with the bandpower and pixel conventions of sum_stat) against `ggah_mod.observables.WTheta`, to 1 per cent; then the data $w(\theta)$ against the data $C_\ell$ transformed the same way, for every slice-0 galaxy and AGN block (sum_stat checked the galaxy autos: median ratio 1.004–1.029 at $\theta \geq 12$ arcmin).
  V4. *Approximations.* Limber against the exact projection at $\ell < 100$ for slices of $\Delta z = 0.1$, redshift-space distortions of the photometric samples, and the $n(z)$ shift and stretch of the closure test: each a correction on $C_\ell$ with its size, kept if above a third of the jackknife error.
  V5. *Data and model together.* The occupation of each slice-0 sample fitted on $w_p$ alone ($r_p$ = 0.1–30 Mpc), then $C_\ell$ and $w(\theta)$ predicted at that point, with the $n(z)$ shift and stretch free within their priors: per block, PTE > 0.01 and a best-fit amplitude within $2\sigma$ of 1.
  The step passes when $C_\ell$, $w(\theta)$ and $w_p$ agree in all slice-0 galaxy samples; the AGN follow, once the eRASS1 flux limit is in their selection (D0).
  Until D1 passes, the MAP chains of script 26 are held at CC-IN2P3 (the Jacobians at the defaults are kept; a code fix in V2 makes them stale and they are recomputed).
- `scripts/21_read_vector.py` reads the vector (`wp3feas/vector.py`), counts rows and spectra per statistic and slice, and maps them onto the `S5_public_2031` layout.
- `scripts/22_fisher_public.py` computes the Jacobian at the default (`ggah_sens_study.fisher.jacobian`) and the Fisher matrix per slice and joint, with and without the Planck prior, and compares it with `S5_public_2031` on the same shells.
- `scripts/23_project_default.py` evaluates the default model (and the 1.6.0 closure, `agn_energy="soltan"`) on every block and reports χ², PTE and whitened residuals per block and per ℓ.
- `scripts/24_null_tests.py` compares $y$ with CIB-deprojected $y$, the bands with and without the point-source mask, and the shear of the three surveys, and reads the jackknife `null` blocks.
- `scripts/25_scale_cuts.py` sets the mask before any fit of $S_8$: a block or ℓ range is removed if it fails a null test (PTE < 0.01), lies beyond $\ell = k_{\max}\chi(z)$ with $k_{\max} = 1$ h/Mpc (3 h/Mpc as a variant), beyond the beam or PSF scale, or below ℓ = 30, or if its residual survives a ±20 per cent freedom of the baryon and AGN sectors; it reports the shift of $S_8$, in Fisher σ, when each removed block is put back.
- `scripts/26_map_fit.py` finds the MAP (L-BFGS from several starts in the prior box) on the masked vector and reports χ², degrees of freedom, PTE and the Laplace covariance.
- `scripts/27_nuts.py` (env `jaxgpu`) runs four `numpyro` NUTS chains from the MAP with a dense mass matrix, and reports $\hat R$, effective sample size, the posterior widths against the Fisher widths, and the wall time (Part F); chains go to `WP3FEAS_DATA`.

**Part E.** `scripts/31_lightcones.py` counts the light-cones and the realisations each route needs; `32_compression.py` compresses the vector with the Jacobians and checks the loss on $\sigma(S_8)$; `33_subblock_validation.py` compares measured and analytic covariances on sub-blocks.

**Part F.** `scripts/41_timing.py` (env `jaxgpu`) times the five-shell likelihood and a short NUTS run, and scales them to 212 parameters.

**Part G.** `scripts/91_summary.py` and `92_figures.py`.

## Schedule

- 3–6 Nov: S, A, B; F started (Gate W1).
- 9–13 Nov: C; D reader, Fisher and projection of the default model (21–23); E light-cones and compression.
- 16–20 Nov: D null tests, scale cuts, MAP and NUTS (24–27); E sub-blocks; G; proposal edits P (Gate W3).
- 23–27 Nov: N1, N6, N7 read in; final numbers; B1 lines.

## Verification

- `python -m pytest`.
- The row and spectra counts of Part A equal `\nRowsFive` and `\nSpectraFive` in `erc-cog-prop/shared/numbers.tex`.
- An injection test (vector = default model plus jackknife noise) recovers the input parameters within 1σ with the MAP, and NUTS widths agree with the Fisher widths within 10 per cent.
- The compressed Fisher $\sigma(S_8)$ lies within 1 per cent of the full `S5_public_2031` value.
- In `erc-cog-prop`: `scripts/make_proposal_figures.py` writes the WP3 macros and `./build.sh check` passes.
