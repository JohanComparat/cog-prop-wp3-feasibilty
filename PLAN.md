# WP3 feasibility: the full data vector in five redshift shells

*Plan written on 3 November 2026. The work runs from 3 to 27 November 2026, inside the November plan of the proposal (`~/software/erc-cog-prop/review/plan_2026-11.md`, Gates W1–W3).*

## Tracker

*Status: `open`, `running`, `done (date, where)` or `dropped (reason)`. Last update: 2026-11-07 (Part D steps 1–3 run, the first MAP step taken).*

| Item | What | Proposal item | Status |
|---|---|---|---|
| S | repository set up (layout, provenance, tests, plan) | – | done (2026-11-03) |
| A | inputs: layout, row and spectra counts of `S5_public_2031`; what it assumes public for LSST Y3 and Euclid DR2 | I4 | open |
| B | $\ell_{\max}$ per map and shell at $k = 1, 3, 10$ h/Mpc against the ACT beam, the eROSITA PSF, shear shape noise and shot noise | N8, I11 | open |
| C | catastrophic photometric-redshift outliers: tolerance on $f_{\rm out}$ at $0.3\sigma$ on $S_8$, and cost when free | Ref. 1 pt 5, Table 3 | open |
| D | the public data vector `public_2026` (`tier_ls11_v0.h5`, LS DR11 × eRASS1 × ACT DR6 × DES/KiDS/HSC shear) fitted with `ggah_mod` 1.7.0: reader, Fisher per redshift slice, projection of the default model, null tests, scale cuts, MAP, NUTS | I19, I24 | running: reader (21), Fisher per slice (22: σ(S8) = 0.0054 joint, block-diagonal jackknife) and default-model projection (23: χ² = 208 869 over 17 748 rows) done 2026-11-07 in `results/`; MAP (26) running; null tests, scale cuts and NUTS open |
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
