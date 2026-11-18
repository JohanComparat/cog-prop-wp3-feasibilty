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

With ``--shells`` (at CC, where ``shells_2031/S5_public_2031/jacobian.npz`` exists: 381 003 rows,
212 parameters) the same test runs on the five-shell run, through the ``Run`` class and the
block structure of ``ggah_sens_study`` ``scripts/77_shells_covariance.py``: C1 has no covariance
between bands, so each band block is factorised alone and the densities are compressed with the
spectra; the case ``jac_today`` has no five-shell counterpart and is left out.
The full-vector Fisher matrix is checked against ``shells_fisher.npz`` (key ``T5|k|C1``).

    python scripts/32_compression.py --shells --k-max 0.3 1 3 10     (cc/cpu.sbatch)
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
    ap.add_argument("--shells", action="store_true", help="the five-shell S5_public_2031 (at CC)")
    a = ap.parse_args(argv)
    if a.shells:
        return main_shells(a)

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


def main_shells(a):
    import importlib.util

    from scipy.linalg import cholesky, solve_triangular

    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import analysis as AN
    from ggah_sens_study.joint.survey import Noise

    run_dir = SENS / "shells_2031" / "S5_public_2031"
    import ggah_sens_study
    s77 = Path(ggah_sens_study.__file__).resolve().parents[1] / "scripts" / "77_shells_covariance.py"
    spec = importlib.util.spec_from_file_location("s77", s77)
    m77 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m77)
    m77.shells_root = lambda: run_dir.parent
    rec = Record(__file__)
    for f in (run_dir / "jacobian.npz", run_dir / "info.npz", run_dir / "config.json",
              run_dir / "shells_fisher.npz", s77):
        rec.add_input(f)
    fid = load_fiducial()
    run = m77.Run("S5_public_2031", fid)
    scheme = run.schemes[0]
    ref = np.load(run_dir / "shells_fisher.npz", allow_pickle=True)
    ref_keys = [str(k) for k in ref["keys"]]
    rng = np.random.default_rng(32)

    def s8rel(F):
        return float(run.constraints(F, scheme)["S8_rel"])

    out = {}
    for k in a.k_max:
        blocks, Jf, disc, rows = AN._cell_blocks(run.v0, run.J, run.info, run.cfg, Noise(),
                                                 fid.cosmology, tier=a.tier, k_max=k, T=run.T,
                                                 y_noise=run.y_noise)
        p = Jf.shape[1]
        Fx = np.zeros((p, p))   # J^T C^-1 J
        G = np.zeros((p, p))    # diagonal weights: Jw^T Jw
        S = np.zeros((p, p))    # Jw^T R Jw
        nd = 0
        for idx, blk in blocks:
            d = np.sqrt(np.diag(blk))
            R = blk / np.outer(d, d)
            Jw = Jf[idx] / d[:, None]
            try:
                U = cholesky(R, lower=False, check_finite=False)
                X = solve_triangular(U, Jw, trans="T", lower=False, check_finite=False)
                Fx += X.T @ X
            except np.linalg.LinAlgError:
                w, V = np.linalg.eigh(R)
                ok = w > AN.EIG_FLOOR * w.max()
                nd += int((~ok).sum())
                P = V[:, ok].T @ Jw
                Fx += (P.T / w[ok]) @ P
            G += Jw.T @ Jw
            S += Jw.T @ R @ Jw
        base = s8rel(Fx)
        key = f"{a.tier}|{k:g}|C1"
        check = None
        if key in ref_keys and ref["F"].shape[1] == p:
            Fr = ref["F"][ref_keys.index(key)]
            check = {"S8_rel_shells_fisher": s8rel(Fr),
                     "max_rel_diag": float(np.max(np.abs(np.diag(Fx) - np.diag(Fr))
                                                  / np.maximum(np.abs(np.diag(Fr)), 1e-300)))}
        cases = {"exact": base, "diag_cov": s8rel(G @ np.linalg.solve(S, G))}
        Lt = np.linalg.cholesky(Fx + 1e-12 * np.trace(Fx) / p * np.eye(p))
        sims = {}
        for ns in a.ns:
            if ns <= p + 3:
                continue
            vals = []
            for _ in range(a.trials):
                x = rng.standard_normal((ns, p)) @ Lt.T
                Sh = np.cov(x, rowvar=False)
                h = (ns - p - 2.0) / (ns - 1.0)
                vals.append(s8rel(Fx @ (h * np.linalg.solve(Sh, Fx))))
            q = np.percentile(vals, [16, 50, 84]) / base
            sims[str(ns)] = {"ratio_p16": q[0], "ratio_median": q[1], "ratio_p84": q[2]}
            print(f"k_max {k:g} n_s {ns:5d}: sigma(S8) / full = {q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]",
                  flush=True)
        loss = {c: v / base - 1.0 for c, v in cases.items()}
        out[f"{k:g}"] = {"rows": int(Jf.shape[0]), "n_dropped": nd, "n_params": p,
                         "S8_rel_full": base, "S8_rel": cases, "loss": loss, "sim_cov": sims,
                         "check": check}
        print(f"k_max {k:g}: {Jf.shape[0]} rows -> {p}; S8_rel full {base:.4e}; "
              + ", ".join(f"{c} {v:+.2%}" for c, v in loss.items()) + f"; check {check}", flush=True)
    rec.data.update(run="S5_public_2031", tier=a.tier, level="C1", scheme=scheme,
                    trials=a.trials, results=out,
                    note="loss = sigma(S8)_compressed / sigma(S8)_full - 1; densities compressed "
                         "with the spectra; n_s <= p + 3 skipped")
    print(rec.write("32_compression_S5_public_2031"))


if __name__ == "__main__":
    main()
