# The state of the model per work package (12 November 2026)

This page says, for each work package of the proposal UNCUT, which parts of the model it owns, what is validated today and with which number, and what remains before its deliverables.
The model is `ggah_mod` 1.7.0 with the `eps_dpl` AGN closure; the joint forecasts and fits go through `ggah_sens_study` (branch rerun-v170).
The numbers come from the result files named in brackets, in the feasibility repositories of WP1, WP2 and WP3.

## WP1, the halo layer (D1.1–D1.6)

WP1 owns the halo mass function, the halo bias and the concentration–mass relation that every observable of the model integrates over.

What works today:
- The suite is costed and its first boxes ran: 1.62 × 10⁷ core-h and 3 480 H100-h, 14 per cent of a Kraken year (`wp1_feasibility_summary.json`).
- In the two Planck boxes, dn/dM is 0.97–0.98 of Tinker08 at z ≤ 0.5 and 1.007 ± 0.003 of Comparat17 at z = 0 (`63_planck_literature.json`).
- The concentration is 0.948 ± 0.006 of Seppi21 and within 1.2 per cent of Klypin16 at 200c.

What remains:
- The bias: three estimators give 0.94–1.03 of Tinker10 at z = 0, a spread of 8 per cent, so the error of Tinker10 is not yet measured (D1.1).
- The 5–13 per cent difference of the two Planck boxes' dn/dM below 1 000 particles, needed before any 1 per cent mass function.
- The move of `ggah_sens_study` and the proposal generator to 1.7.0.

## WP2, the feedback model (D2.1–D2.9)

WP2 owns the gas, the AGN energetics and the closure that ties the energy of supernovae and AGN to the missing baryons.

What works today:
- The coupling the observations require rises with mass: 0.020, 0.19 and 0.61 at 10¹³, 10¹⁴ and 10¹⁵ Msun, zero below 10^12.75 Msun; a double power law fits it, a constant does not (WP2 Q8).
- `ggah_mod` 1.7.0 makes that fit the default (`agn_energy = "eps_dpl"`, `eps_sn = 0.0286`), and the closure at the defaults returns the census to 4 × 10⁻¹⁶.
- The suppression of the matter power reaches 0.36 per cent of BAHAMAS hiAGN and 0.29 per cent of lowAGN (Q7).

What remains:
- Illustris stays at 5.5 per cent: one shape of the closure does not fit every suite (D2.4).
- The gas fractions within R500c give χ² = 31.2 for 10 bins at the posterior, 19 of it from Mantz et al. (2016): the mass scales of the X-ray samples differ (D2.3).
- The refit of the gas with the closure in the loop.

## WP3, the full data vector in five redshift shells (D3.1–D3.7)

WP3 owns the projection of the model onto the observables: the kernels of each tracer, the C_ℓ, w(θ) and w_p, the covariance, and the fit to the data.
We test it on the public vector `public_2026` (LS DR11 × eRASS1 × ACT DR6 × DES, KiDS and HSC shear), first on the slice z = 0.1–0.2.

What works today:
- The build's C_ℓ equals an independent Limber integral of its own kernels and P(k) to 0.1 per cent (`28_cl_validation_v2.json`).
- The model's w_p matches the data to 0.83–1.19 on 1–30 Mpc, and its 3D bias equals the bias from the data w_p (RED 10–10.5: 1.31 against 1.31; `28_cl_validation_v1.json`).
- The tier w_p is not diluted by the photo-z members by more than 10 per cent: the spectroscopic members alone give the same w_p, 0.95–1.01 on 1–10 Mpc (`29_wp_spec.json`).
- The photometric kernels equal the measured n(z) to 1 per cent, since the fix of the one-shell names (`28_cl_validation_kern.json`).
- At the defaults, slice 0 gives χ² = 113 410 over 10 991 rows; six Gauss–Newton iterates bring it to 66 374, the galaxy auto amplitude from 1.47 to 1.08, galaxy × y from 0.55 to 1.10 (`23_project_default_s0_nz.json`, `23_project_default_d3it6.json`).
- A sharded Jacobian of slice 0 takes four H100 jobs of 51–57 min, against 3 h 32 min for one.

What remains:
- AGN × X-ray (χ² 17 573 on 675 rows, amplitude 3.2) and the AGN density (data over model 0.09–0.13): the eRASS1 flux limit is not in the model's AGN selection (D3.5).
- X-ray × X-ray (χ² 11 874 on 1 800 rows): being re-measured on a clean footprint (the background, |b| > 20° or 30°, stars, the Magellanic Clouds, Virgo and Fornax masked), to be compared with Lau et al. (2025).
- Shear × shear (χ²/n = 6.3): the covariance of the 13 bins and the source n(z) (D3.1, D3.4).
- y × y: the noise of the y map is not subtracted; the block is cut (D3.6).
- BLUE 10–10.5: w(θ) is 1.5–1.9 times the model at a w_p that matches; the shape of its C_ℓ at high ℓ.
- The covariance: block-diagonal jackknife today, so the χ² and the Fisher errors are lower limits (D3.4).
- The model has no trap for a tracer whose n(z) table is missing: it falls back to a smoothed top-hat silently; it should refuse by name.

## WP4, integration (D4.1–D4.4)

WP4 owns the joint fit and the releases.
Today the fit machinery runs end to end on slice 0: the vector loader, the projection of the model, the jackknife covariance with the Hartlap factor, the Gauss–Newton step with a line search, and parallel chains at CC-IN2P3.
What remains: the scale cuts (script 25), the three shells together, NUTS (script 27), and the analytic covariance.

## On updating the defaults of `ggah_mod`

We do not update the defaults from this fit yet.
The fit is not converged (χ²/n ≈ 6), and the four block types it cannot move pull on the gas and AGN parameters that moved most (`gamma_n`, `eta_ej`, `alpha_out_n`, `log10_eps_agn_p`) and on the intrinsic-alignment amplitude (5.74 → 2.66).
The occupation shifts (`mu_c` 1.53 → 1.02, `gamma` 2.18 → 2.69, `lg_mh_qs` 12.84 → 13.31, `lg_m0star` 10.24 → 9.98, `bsat` 11.28 → 10.55) are candidates once the masked fit converges and they also fit the tier w_p; they would then go into `ggah_mod` as a named preset of this data set, not as a change of the defaults.
