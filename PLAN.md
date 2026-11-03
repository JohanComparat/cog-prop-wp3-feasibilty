# WP3 feasibility: the full data vector in five redshift shells

*Plan written on 3 November 2026. The work runs from 3 to 27 November 2026, inside the November plan of the proposal (`~/software/erc-cog-prop/review/plan_2026-11.md`, Gates W1–W3).*

## Tracker

*Status: `open`, `running`, `done (date, where)` or `dropped (reason)`. Last update: 2026-11-03 (repository set up).*

| Item | What | Proposal item | Status |
|---|---|---|---|
| S | repository set up (layout, provenance, tests, plan) | – | done (2026-11-03) |
| A | inputs: layout, row and spectra counts of `S5_public_2031`; what it assumes public for LSST Y3 and Euclid DR2 | I4 | open |
| B | $\ell_{\max}$ per map and shell at $k = 1, 3, 10$ h/Mpc against the ACT beam, the eROSITA PSF, shear shape noise and shot noise | N8, I11 | open |
| C | catastrophic photometric-redshift outliers: tolerance on $f_{\rm out}$ at $0.3\sigma$ on $S_8$, and cost when free | Ref. 1 pt 5, Table 3 | open |
| D | the public data vector (LS DR11 × eRASS1 × ACT DR6): adapter, coverage of the final vector, null tests, fit import | I19, I24 | open (waits for the vector's path and format) |
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

**Part D.** `scripts/21_read_vector.py` maps the public vector onto the `S5_public_2031` layout; `24_null_tests.py` runs the $y$ against CIB-deprojected $y$ and the map nulls; `25_fit_import.py` carries the fit into the summary.

**Part E.** `scripts/31_lightcones.py` counts the light-cones and the realisations each route needs; `32_compression.py` compresses the vector with the Jacobians and checks the loss on $\sigma(S_8)$; `33_subblock_validation.py` compares measured and analytic covariances on sub-blocks.

**Part F.** `scripts/41_timing.py` (env `jaxgpu`) times the five-shell likelihood and a short NUTS run, and scales them to 212 parameters.

**Part G.** `scripts/91_summary.py` and `92_figures.py`.

## Schedule

- 3–6 Nov: S, A, B; F started (Gate W1).
- 9–13 Nov: C; D adapter; E light-cones and compression.
- 16–20 Nov: D nulls; E sub-blocks; G; proposal edits P (Gate W3).
- 23–27 Nov: N1, N6, N7 read in; final numbers; B1 lines.

## Verification

- `python -m pytest`.
- The row and spectra counts of Part A equal `\nRowsFive` and `\nSpectraFive` in `erc-cog-prop/shared/numbers.tex`.
- The compressed Fisher $\sigma(S_8)$ lies within 1 per cent of the full `S5_public_2031` value.
- In `erc-cog-prop`: `scripts/make_proposal_figures.py` writes the WP3 macros and `./build.sh check` passes.
