r"""A candidate for the non-galaxy defaults: the posterior of today's slice-0 fit, conditional on
the galaxy parameters at their defaults, times the gas fit to the cluster and group scaling
relations of ggah_cal (the fit behind the gas defaults).

    python scripts/93_combined_point.py --it 21      (WP3FEAS_MAP_TAG=_cuts)

Both are Gaussians (Laplace approximations).
Today's, :math:`N(\mu, C)` on the free parameters of script 26 (``laplace_cov.npz``), is
conditioned on the galaxy parameters ``g`` at their defaults ``g_0``:
:math:`\mu_{n|g} = \mu_n + C_{ng} C_{gg}^{-1} (g_0 - \mu_g)`, precision :math:`(C^{-1})_{nn}`.
The gas fit's covariance is marginalised onto the parameters the two share, and the product
has precision :math:`\Lambda = \Lambda_t + \Lambda_g` and mean
:math:`\Lambda^{-1}(\Lambda_t\mu_t + \Lambda_g\mu_g)`.
Each holds its own prior, so the priors count twice: the point leans to the box centres a
little more than it should.
The photometric nuisances (``dzp@``, ``szp@``) stay at their defaults in the candidate.

Writes ``results/93_combined_point<layout><tag>_it<it>.json``: per parameter the default,
today's conditional value, the gas fit's value and the combined value, with their errors.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import DATA, Record  # noqa: E402

GAS_MAP = Path.home() / "data/ggah_cal_results_v1.5.0/gas_r6v/gpu/P3/merged/map_result.json"
GAL = ("lg_m1h", "lg_m0star", "beta", "delta", "gamma", "sigma_lnmstar", "eta", "fc", "bsat",
       "beta_sat", "bcut", "beta_cut", "alpha_sat", "lg_mh_qc", "mu_c", "lg_mh_qs", "mu_s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--it", type=int, default=21)
    ap.add_argument("--layout", default="_s0")
    a = ap.parse_args(argv)
    rec = Record(__file__)
    tag = os.environ.get("WP3FEAS_MAP_TAG", "")
    lp = DATA / "cc" / f"26_map{a.layout}{tag}" / f"it{a.it}" / "laplace_cov.npz"
    L = np.load(lp)
    names = [str(n) for n in L["names"]]
    x, f, C = L["theta"], L["free"], L["C"]
    summ = json.loads((ROOT / "results" / f"26_map_summary{a.layout}{tag}_it{a.it}.json").read_text())["parameters"]
    fn = [names[i] for i in f]
    mu = x[f]
    d0 = np.array([summ[n]["default"] for n in fn])
    g = np.array([n in GAL for n in fn])
    n_ = ~g
    # conditional on the galaxy parameters at their defaults
    Cgg, Cng = C[np.ix_(g, g)], C[np.ix_(n_, g)]
    mu_t = mu[n_] + Cng @ np.linalg.solve(Cgg, d0[g] - mu[g])
    Lam_t = np.linalg.inv(C)[np.ix_(n_, n_)]
    tn = [n for n, k in zip(fn, n_) if k]
    # the gas fit, marginalised onto the shared parameters
    G = json.loads(GAS_MAP.read_text())
    gn = list(G["names"])
    shared = [n for n in tn if n in gn]
    ig = [gn.index(n) for n in shared]
    Cg = np.asarray(G["laplace_cov"])[np.ix_(ig, ig)]
    mu_g = np.asarray(G["x"])[ig]
    Lam_g = np.zeros_like(Lam_t)
    b_g = np.zeros(len(tn))
    it_ = [tn.index(n) for n in shared]
    Pg = np.linalg.inv(Cg)
    Lam_g[np.ix_(it_, it_)] = Pg
    b_g[it_] = Pg @ mu_g
    Lam = Lam_t + Lam_g
    Cc = np.linalg.inv(Lam)
    mu_c = Cc @ (Lam_t @ mu_t + b_g)
    rows = {}
    for j, n in enumerate(tn):
        r = {"default": float(d0[n_][j]), "fit": float(mu[n_][j]), "today_given_gal_default": float(mu_t[j]),
             "today_sigma": float(np.sqrt(np.linalg.inv(Lam_t)[j, j])), "combined": float(mu_c[j]),
             "combined_sigma": float(np.sqrt(Cc[j, j])), "nuisance": n.startswith(("dzp@", "szp@"))}
        if n in shared:
            k = shared.index(n)
            r.update(gas_fit=float(mu_g[k]), gas_sigma=float(np.sqrt(Cg[k, k])))
        rows[n] = r
    cand = {n: r["combined"] for n, r in rows.items() if not r["nuisance"]}
    print(f"{len(tn)} non-galaxy parameters, {len(shared)} shared with the gas fit")
    print(f"{'parameter':22s} {'default':>8s} {'fit':>8s} {'t|g0':>8s} {'gas':>8s} {'comb':>8s} {'sig':>7s}")
    for n, r in sorted(rows.items(), key=lambda kv: -abs(kv[1]["combined"] - kv[1]["default"]) / kv[1]["combined_sigma"]):
        print(f"{n:22s} {r['default']:8.3f} {r['fit']:8.3f} {r['today_given_gal_default']:8.3f} "
              f"{r.get('gas_fit', np.nan):8.3f} {r['combined']:8.3f} {r['combined_sigma']:7.3f}")
    rec.inputs += [str(lp), str(GAS_MAP)]
    rec.data.update(it=a.it, layout=a.layout, map_tag=tag, shared_with_gas_fit=shared,
                    galaxy_fixed_at_defaults=list(GAL), parameters=rows, candidate=cand)
    print(rec.write(f"93_combined_point{a.layout}{tag}_it{a.it}"))


if __name__ == "__main__":
    main()
