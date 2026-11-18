r"""Part E: score (MOPED) compression of the joint vector and its loss on :math:`\sigma(S_8)`.

    python scripts/32_compression.py [--k-max 1 3] [--tier T5] [--trials 200]

The run is the one-shell joint forecast of ``ggah_sens_study``,
``joint_defaults-v1.6.0_stage4_closure`` (18 255 rows before the cuts, 68 parameters, C1
covariance), read and never written; the five-shell ``S5_public_2031`` keeps no Jacobian on disk.
The spectra are compressed to one number per parameter,
:math:`t = W^{T}(d - \mu)` with :math:`W = C^{-1}J` (the score; MOPED is the same up to a
Gram–Schmidt rotation, which leaves the Fisher matrix unchanged), and the number densities are
kept as they are.
The Fisher matrix of :math:`t` is :math:`F_t = (W^TJ)^T(W^TCW)^{-1}(W^TJ)`.
We report :math:`\sigma(S_8)/S_8` (Planck prior, prior boxes as in ``scripts/72_joint_grid.py``)
for the full vector and for:

1. ``exact``: :math:`W` from the true :math:`J` and :math:`C` (lossless by construction: a check);
2. ``jac_today``: :math:`W` from the Jacobian of ``joint_defaults-v1.6.0_today_closure`` (another
   point of the model), applied to the stage-4 data;
3. ``diag_cov``: :math:`W = D^{-1}J`, the diagonal of :math:`C` only;
4. ``sim_cov``: :math:`W` exact, but the 68 × 68 covariance of :math:`t` estimated from
   :math:`n_s` Gaussian realisations, with the Hartlap factor; median and 16th–84th percentiles of
   :math:`\sigma(S_8)` over ``--trials`` draws, for each :math:`n_s` (the route of script 31).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import SENS, Record  # noqa: E402

RUN = SENS / "joint_defaults-v1.6.0_stage4_closure"
RUN_TODAY = SENS / "joint_defaults-v1.6.0_today_closure"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-max", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--tier", default="T5")
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 150, 250, 500, 1000, 3000])
    ap.add_argument("--trials", type=int, default=200)
    a = ap.parse_args(argv)

    from ggah_sens_study.basis import ad_sigma8_gradient
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import analysis as AN
    from ggah_sens_study.joint import covariance as CV
    from ggah_sens_study.joint import model as M
    from ggah_sens_study.joint.survey import JointConfig, Noise

    rec = Record(__file__)
    for f in (RUN / "jacobian.npz", RUN / "info.npz", RUN / "config.json", RUN_TODAY / "jacobian.npz"):
        rec.add_input(f)
    cfg = JointConfig.from_dict(json.loads((RUN / "config.json").read_text()))
    v0, J, info = AN.load_run(RUN)
    zt = np.load(RUN_TODAY / "jacobian.npz", allow_pickle=True)
    assert [str(n) for n in zt["names"]] == list(info["names"]) and zt["J"].shape == J.shape
    J_today = zt["J"]
    fid = load_fiducial()
    cosmo = fid.cosmology
    s8, ds8 = ad_sigma8_gradient(fid)
    names = list(info["names"])
    i0 = len(M.GAL)
    Mc = np.eye(len(M.COSMO))
    Mc[M.COSMO.index("ln10A_s")] = [ds8[n] for n in M.COSMO]
    T = np.eye(len(names))
    T[i0:i0 + len(M.COSMO), i0:i0 + len(M.COSMO)] = np.linalg.inv(Mc)
    rep = list(names)
    rep[rep.index("ln10A_s")] = "sigma8"
    th = dict(zip(rep, np.asarray(info["theta0"], float)))
    centre = {"Omega_m": float(th["Omega_m"]), "sigma8": s8}
    noise = Noise()
    rng = np.random.default_rng(32)

    def s8rel(F):
        res, _ = AN.constraints_joint(F, rep, centre, cfg.sources, cosmo_prior="planck",
                                      cosmo_raw=fid.raw["cosmology"])
        return res["S8_rel"]

    def fisher_t(W, Jr, C):
        G = W.T @ Jr
        S = W.T @ C @ W
        return G.T @ np.linalg.solve(S, G)

    out = {}
    for k in a.k_max:
        nbar, Cb = CV.binned(v0, info)
        Jn, JC = CV.binned_jacobian(J, info)
        _, JCt = CV.binned_jacobian(J_today, info)
        N = CV.noise_power(cfg, noise, info, nbar, cosmo)
        masks = CV.ell_cut_rows(info, AN.ell_cuts(info, k))
        keep = AN.tier_pairs(info["pairs"], a.tier)
        C, rows = CV.covariance(Cb, N, nbar, cfg, cosmo, keep, masks)
        Jr = np.vstack([JC[pr][L] for _, pr, L in rows]) @ T
        Jt = np.vstack([JCt[pr][L] for _, pr, L in rows]) @ T
        # densities: independent of the spectra, added to every Fisher matrix
        disc = AN.tier_discrete(info["discrete"], a.tier)
        var = CV.nbar_variance(nbar, cfg, cosmo, info["discrete"])
        Fn = np.zeros((len(names),) * 2)
        for d in disc:
            g = Jn[info["discrete"].index(d)] @ T
            Fn += np.outer(g, g) / var[info["discrete"].index(d)]
        # whiten by the standard deviations; the correlation matrix on its eigenvectors
        sd = np.sqrt(np.diag(C))
        R = C / np.outer(sd, sd)
        w, V = np.linalg.eigh(R)
        ok = w > AN.EIG_FLOOR * w.max()
        Rinv = (V[:, ok] / w[ok]) @ V[:, ok].T
        Jw, Jtw = Jr / sd[:, None], Jt / sd[:, None]
        F_full = Jw.T @ Rinv @ Jw + Fn
        base = s8rel(F_full)
        cases = {}
        W = Rinv @ Jw
        F_ex = fisher_t(W, Jw, R) + Fn
        cases["exact"] = s8rel(F_ex)
        cases["jac_today"] = s8rel(fisher_t(Rinv @ Jtw, Jw, R) + Fn)
        cases["diag_cov"] = s8rel(fisher_t(Jw, Jw, R) + Fn)  # D^{-1}J in whitened units
        p = len(names)
        Ft = F_ex - Fn  # covariance of t (= its Fisher matrix)
        Lt = np.linalg.cholesky(Ft + 1e-12 * np.trace(Ft) / p * np.eye(p))
        sims = {}
        for ns in a.ns:
            vals = []
            for _ in range(a.trials):
                x = rng.standard_normal((ns, p)) @ Lt.T
                S = np.cov(x, rowvar=False)
                h = (ns - p - 2.0) / (ns - 1.0)
                vals.append(s8rel(Ft @ (h * np.linalg.solve(S, Ft)) + Fn))
            q = np.percentile(vals, [16, 50, 84]) / base
            sims[str(ns)] = {"ratio_p16": q[0], "ratio_median": q[1], "ratio_p84": q[2]}
            print(f"k_max {k:g} n_s {ns:5d}: sigma(S8) / full = {q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]",
                  flush=True)
        loss = {c: v / base - 1.0 for c, v in cases.items()}
        out[f"{k:g}"] = {"rows": len(rows), "n_dropped": int((~ok).sum()), "n_params": p,
                         "S8_rel_full": base, "S8_rel": cases, "loss": loss, "sim_cov": sims}
        print(f"k_max {k:g}: {len(rows)} rows -> {p}; S8_rel full {base:.4e}; "
              + ", ".join(f"{c} {v:+.2%}" for c, v in loss.items()), flush=True)
    rec.data.update(run=RUN.name, tier=a.tier, level="C1", trials=a.trials, results=out,
                    note="loss = sigma(S8)_compressed / sigma(S8)_full - 1; densities uncompressed")
    print(rec.write())


if __name__ == "__main__":
    main()
