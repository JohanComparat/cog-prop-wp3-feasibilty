# Scale cuts per block type of slice 0

This page proposes one scale cut per block type of the slice-0 data vector `public_2026` (v1, clean X-ray footprint), chosen so that the model gives a reasonable description of each block.
The cut is a maximum wavenumber k_max: a row is kept when ℓ ≤ k_max χ(z̄), with χ(z̄) the comoving distance to the mean redshift of the discrete tracers (galaxies, AGN) of the pair.
At slice 0 (0.10 < z < 0.20) χ(z̄) ≈ 440 Mpc/h, so k_max = 1, 0.5, 0.3, 0.2, 0.15, 0.1 h/Mpc give ℓ_max ≈ 440, 220, 130, 90, 66, 44.
The other cuts of the mask `_v1xsnx` stay: ℓ ≥ 30, y × y out, X-ray × X-ray out (PLAN D3), shear × shear across surveys out.
Pairs with no discrete tracer (shear, X-ray, y) carry no k_max cut.

## χ² against k_max

We evaluated the χ² of each block type at the iterate 16 of the chain `_v1final`, which was fitted with k_max = 1 h/Mpc, on the masks `_v1xsnx_k<k>` (`scripts/25_scale_cuts.py mask --drop-cross-shear --drop-xray-auto --kmax <k>`, then `chi2`).
Sources: `results/25_scale_cuts_chi2_v1xsnx{,_k05,_k03,_k02,_k015,_k01}_v1final_it16.json`.
Entries are χ² / rows, with χ²/n in brackets.

| block type | k_max 1 | 0.5 | 0.3 | 0.2 | 0.15 | 0.1 |
|---|---|---|---|---|---|---|
| galaxy × galaxy | 1 519 / 280 (5.43) | 567 / 196 (2.89) | 192 / 140 (1.37) | 148 / 112 (1.32) | 74 / 84 (0.88) | 65 / 56 (1.16) |
| galaxy × X-ray | 1 697 / 1 120 (1.52) | 1 098 / 784 (1.40) | 599 / 560 (1.07) | 352 / 448 (0.79) | 206 / 336 (0.61) | 73 / 224 (0.33) |
| galaxy × shear | 1 103 / 819 (1.35) | 849 / 546 (1.55) | 633 / 364 (1.74) | 445 / 273 (1.63) | 248 / 182 (1.36) | 181 / 91 (1.99) |
| galaxy × y | 69 / 70 (0.99) | 48 / 49 (0.98) | 30 / 35 (0.86) | 24 / 28 (0.86) | 15 / 21 (0.71) | 10 / 14 (0.71) |
| AGN × galaxy | 241 / 210 (1.15) | 168 / 147 (1.14) | 148 / 105 (1.41) | 89 / 84 (1.06) | 65 / 63 (1.03) | 54 / 42 (1.29) |
| AGN × AGN | 142 / 60 (2.37) | 92 / 42 (2.19) | 40 / 30 (1.33) | 37 / 24 (1.54) | 36 / 18 (2.00) | 12 / 12 (1.00) |
| AGN × X-ray | 554 / 480 (1.15) | 281 / 336 (0.84) | 191 / 240 (0.80) | 120 / 192 (0.63) | 86 / 144 (0.60) | 53 / 96 (0.55) |
| AGN × y | 32 / 30 (1.07) | 24 / 21 (1.14) | 18 / 15 (1.20) | 15 / 12 (1.25) | 13 / 9 (1.44) | 5 / 6 (0.83) |
| AGN × shear | 336 / 351 (0.96) | 201 / 234 (0.86) | 144 / 156 (0.92) | 106 / 117 (0.91) | 74 / 78 (0.95) | 42 / 39 (1.08) |
| shear × shear | 560 / 490 (1.14) | no k_max cut | | | | |
| shear × X-ray | 3 731 / 2 912 (1.28) | no k_max cut | | | | |
| shear × y | 192 / 182 (1.05) | no k_max cut | | | | |
| X-ray × y | 300 / 240 (1.25) | no k_max cut | | | | |

## Proposed selection

For each block type we take the largest k_max at which χ²/n is close to 1 and stays there at smaller k_max.

| block type | k_max [h/Mpc] | ℓ_max | χ² / rows | χ²/n | note |
|---|---|---|---|---|---|
| galaxy × galaxy | 0.15 | 66 | 74 / 84 | 0.88 | the misfit lies at 1–3 Mpc/h (one-halo to two-halo transition) |
| galaxy × X-ray | 0.3 | 130 | 599 / 560 | 1.07 | |
| galaxy × shear | 0.15 | 66 | 248 / 182 | 1.36 | no k_max brings χ²/n to 1: an amplitude (0.78) issue, open |
| galaxy × y | 1 | 440 | 69 / 70 | 0.99 | |
| AGN × galaxy | 0.2 | 90 | 89 / 84 | 1.06 | |
| AGN × AGN | 0.3 | 130 | 40 / 30 | 1.33 | 30 rows |
| AGN × X-ray | 0.5 | 220 | 281 / 336 | 0.84 | |
| AGN × y | 1 | 440 | 32 / 30 | 1.07 | |
| AGN × shear | 1 | 440 | 336 / 351 | 0.96 | |
| shear × shear | none | | 560 / 490 | 1.14 | same-survey pairs only |
| shear × X-ray | none | | 3 731 / 2 912 | 1.28 | the largest χ²; its ℓ dependence not yet examined |
| shear × y | none | | 192 / 182 | 1.05 | |
| X-ray × y | none | | 300 / 240 | 1.25 | |
| X-ray × X-ray | out | | | | PLAN D3 |
| y × y | out | | | | noise not subtracted |

Summed over the block types, the selection holds χ² = 6 551 on 5 551 rows (χ²/n 1.18), against 10 477 on 7 244 rows (χ²/n 1.45) with k_max = 1 h/Mpc for all.

## Caveats

- The cuts were chosen by looking at the χ² of the fitted model, so they are not blind; the S8 analysis must fix its cuts before any S8 fit (PLAN.md).
- The χ² are evaluated at a point fitted with k_max = 1 h/Mpc; a refit with the proposed selection will lower them and may move the occupation parameters (`mu_s`, `mu_c`), which the k_max = 1 fit shifted by −10.6σ and −8.1σ.
- The sum over block types ignores the covariance between them.
- At k_max = 0.15 h/Mpc galaxy × galaxy keeps only 30 ≤ ℓ ≤ 66, so slice 0 loses most of its galaxy-clustering information; the higher-redshift slices keep larger ℓ ranges at the same k_max.
- χ²/n well below 1 at small k_max (galaxy × X-ray 0.33, AGN × X-ray 0.55 at 0.1 h/Mpc) points to jackknife errors that are too large at large scales.
