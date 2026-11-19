r"""The Laplace posterior of the fitted slice 0.1-0.2 (script 26, one chain and iterate) with the
cosmology free: S8, Omega_m and their correlation with S(k) = P_mm / P_mm^DMO.

    sbatch --export=ALL,<the chain's WP3FEAS_*> cc/cpu.sbatch scripts/95_laplace_cosmo.py --it 21   # at CC

Script 26 fits with the cosmology held fixed, but the Jacobian of every iterate holds all the
columns, the five cosmological ones included.
Here :math:`A = \sum_b J_b^{\rm T} C_b^{-1} J_b` (the block-diagonal jackknife of script 26) is
taken in the reported basis of script 22 (:math:`\sigma_8` for :math:`\ln10^{10}A_s`, its gradient
at the defaults), with the priors of script 26 on the astrophysical parameters and the Planck/BBN
priors on h, n_s and Omega_b; Omega_m and sigma8 have no prior.
The posterior covariance is :math:`(A + P)^{-1}` on the free parameters (those of script 26 and
the five cosmological ones).
One linearised step from the iterate, :math:`\Delta = (A + P)^{-1} (J^{\rm T} C^{-1} r - P(x - x_0))`,
moves the cosmology from its fixed value; S8 and S(k) are quoted at that step.
S(k) at z = 0 and 0.15 comes from script 91's Jacobian (``WP3FEAS_DATA/91_sk_band<tag>.npz``).

Writes ``results/95_laplace_cosmo<layout><tag>_it<it>.json`` and the full covariance (reported and
raw basis) to the iterate's ``laplace_cov_cosmo.npz`` for the draws of other scripts.
The posterior is first order around the point and uses the block-diagonal jackknife covariance.
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import ggah_sens_study  # noqa: E402,F401  (x64 first)
from wp3feas.provenance import DATA, Record  # noqa: E402

S22 = importlib.import_module("22_fisher_public")
S26 = importlib.import_module("26_map_fit")
COSMO = ("Omega_m", "sigma8", "h", "n_s", "Omega_b")
K_QUOTE = (0.3, 1.0, 3.0, 10.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--it", type=int, required=True)
    ap.add_argument("--layout", default="_s0", choices=list(S22.LAYOUTS))
    a = ap.parse_args(argv)
    S22.set_layout(a.layout)
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import decompose as DC
    from ggah_sens_study.priors import planck_prior_fisher

    rec = Record(__file__)
    tag = f"{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', '')}_it{a.it}"
    meta, Jd, r, blocks, x = S26.point(a.it, None)
    names = list(meta["names"])
    th0 = np.asarray(meta["theta0"])
    n = len(names)
    A = np.zeros((n, n))
    b = np.zeros(n)
    for sel, iC in blocks:
        A += Jd[sel].T @ iC @ Jd[sel]
        b += Jd[sel].T @ iC @ r[sel]
    chi2 = float(sum(r[s] @ iC @ r[s] for s, iC in blocks))
    n_rows = int(sum(s.size for s, _ in blocks))

    # the reported basis: d theta_raw = T d theta_rep
    fid = load_fiducial()
    T, rep, centre, value = DC.reported_basis({"names": names, "theta0": th0}, fid)
    Ti = np.linalg.inv(T)
    Ar, br = T.T @ A @ T, T.T @ b
    sp = S26.prior_sigma(names)                      # inf for the cosmology (FIXED of script 26)
    Pr = np.diag(np.where(np.isfinite(sp), 1.0 / sp ** 2, 0.0)) + planck_prior_fisher(rep, fid.raw["cosmology"])
    dx_rep = Ti @ (x - th0)                          # the iterate's offset from the defaults, reported basis
    fixed_free = np.load(S26.it_dir(a.it) / "laplace_cov.npz")["free"]
    ic = [rep.index(c) for c in COSMO]
    f = np.union1d(fixed_free, ic)
    H = Ar[np.ix_(f, f)] + Pr[np.ix_(f, f)]
    sc = np.sqrt(np.clip(np.diag(H), 1e-300, None))
    C = np.linalg.inv(H / np.outer(sc, sc)) / np.outer(sc, sc)
    step = C @ (br[f] - Pr[np.ix_(f, f)] @ dx_rep[f])

    # S8 = sigma8 sqrt(Omega_m / 0.3) at the defaults (the fit holds the cosmology there)
    om, s8 = centre["Omega_m"], centre["sigma8"]
    S8_0 = s8 * np.sqrt(om / 0.3)
    g = np.zeros(f.size)
    g[list(f).index(rep.index("sigma8"))] = np.sqrt(om / 0.3)
    g[list(f).index(rep.index("Omega_m"))] = 0.5 * s8 / np.sqrt(0.3 * om)
    sig_S8 = float(np.sqrt(g @ C @ g))
    io = list(f).index(rep.index("Omega_m"))
    cosmo = {}
    for c in COSMO:
        j = list(f).index(rep.index(c))
        cosmo[c] = {"value_fixed": float(value[c]), "step": float(step[j]), "value": float(value[c] + step[j]),
                    "error": float(np.sqrt(C[j, j]))}
    S8 = {"value_fixed": float(S8_0), "value": float(S8_0 + g @ step), "error": sig_S8,
          "error_rel": sig_S8 / S8_0,
          "corr_Omega_m": float((g @ C[:, io]) / (sig_S8 * np.sqrt(C[io, io])))}

    # the astrophysical errors with and without the cosmology free
    L = np.load(S26.it_dir(a.it) / "laplace_cov.npz")
    err_fixed = dict(zip([names[i] for i in L["free"]], np.sqrt(np.clip(np.diag(L["C"]), 0, None))))
    astro = {}
    for j, i in enumerate(f):
        if rep[i] in COSMO or names[i] not in err_fixed:
            continue
        e = float(np.sqrt(C[j, j]))
        astro[names[i]] = {"error_fixed_cosmology": float(err_fixed[names[i]]), "error_free_cosmology": e,
                           "ratio": e / float(err_fixed[names[i]]) if err_fixed[names[i]] > 0 else None}
    ratios = np.array([v["ratio"] for v in astro.values() if v["ratio"]])

    # S(k): script 91's Jacobian G (raw basis) -> reported basis, on the free parameters
    sk = None
    gpath = DATA / f"91_sk_band{tag}.npz"
    if gpath.is_file():
        Gz = np.load(gpath)
        assert list(Gz["names"]) == names and np.allclose(Gz["theta"], x)
        k = Gz["k"]
        sk = {}
        for iz, z in enumerate(Gz["z"]):
            G = (Gz["G"][iz] @ T)[:, f]
            row = {}
            for kk in K_QUOTE:
                i = int(np.argmin(np.abs(np.log(k / kk))))
                gk = G[i]
                s_ = float(np.sqrt(gk @ C @ gk))
                # the error with the cosmology fixed: the same G on the astrophysical block
                astro_j = [j for j, i_ in enumerate(f) if rep[i_] not in COSMO]
                Ca = np.linalg.inv(H[np.ix_(astro_j, astro_j)])
                row[f"{kk:g}"] = {"k": float(k[i]), "S_point": float(Gz["S"][iz][i]),
                                  "S_step": float(Gz["S"][iz][i] + gk @ step),
                                  "sigma_free_cosmology": s_,
                                  "sigma_fixed_cosmology": float(np.sqrt(gk[astro_j] @ Ca @ gk[astro_j])),
                                  "corr_S8": float((g @ C @ gk) / (sig_S8 * s_)) if s_ > 0 else None}
            sk[f"{float(z):g}"] = row
        rec.inputs.append(str(gpath))

    out = S26.it_dir(a.it) / "laplace_cov_cosmo.npz"
    np.savez(out, names=np.array(names), names_rep=np.array(rep), T=T, theta=x, theta0=th0, free=f, C=C,
             step=step, chi2=chi2, n_rows=n_rows, S8_fixed=S8_0)
    print(f"chi2 {chi2:.1f} on {n_rows} rows; {f.size} free parameters ({len(COSMO)} cosmological)")
    print(f"S8 = {S8['value']:.4f} +- {sig_S8:.4f} (fixed {S8_0:.4f}); corr(S8, Omega_m) {S8['corr_Omega_m']:.2f}")
    for c, v in cosmo.items():
        print(f"  {c:8s} {v['value']:.4f} +- {v['error']:.4f} (fixed {v['value_fixed']:.4f})")
    print(f"astrophysical errors, free / fixed cosmology: median {np.median(ratios):.3f}, max {ratios.max():.3f}")
    if sk:
        for z, row in sk.items():
            for kk, v in row.items():
                print(f"  z {z} k {kk}: S {v['S_step']:.4f} +- {v['sigma_free_cosmology']:.4f} "
                      f"(fixed cosmology +- {v['sigma_fixed_cosmology']:.4f}), corr(S8) {v['corr_S8']:.2f}")
    rec.data.update(it=a.it, layout=a.layout, map_tag=os.environ.get("WP3FEAS_MAP_TAG", ""),
                    env={k: v for k, v in os.environ.items() if k.startswith("WP3FEAS_")},
                    chi2=chi2, n_rows=n_rows, n_free=int(f.size), cov_file=str(out),
                    priors="script 26 on the astrophysical parameters; Planck/BBN on h, n_s, Omega_b; "
                           "none on Omega_m and sigma8",
                    covariance="jackknife per block, Hartlap, block-diagonal",
                    S8=S8, cosmology=cosmo, sk=sk,
                    astro_error_ratio={"median": float(np.median(ratios)), "p84": float(np.percentile(ratios, 84)),
                                       "max": float(ratios.max()),
                                       "largest": sorted(astro, key=lambda n_: -(astro[n_]["ratio"] or 0))[:8]},
                    astro=astro)
    print(rec.write(f"95_laplace_cosmo{tag}"))


if __name__ == "__main__":
    main()
