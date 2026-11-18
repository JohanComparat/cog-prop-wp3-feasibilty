r"""Part E: how many light-cones each covariance route of ``S5_public_2031`` needs.

    python scripts/31_lightcones.py [--target 0.05 0.10]

A covariance estimated from :math:`n_s` independent realisations of a vector of :math:`p` rows,
fitted for :math:`n_\theta` parameters, biases the inverse (Hartlap et al. 2007) and inflates the
parameter variance (Dodelson & Schneider 2013; Percival et al. 2014).
With :math:`A = 2/[(n_s-p-1)(n_s-p-4)]` and :math:`B = (n_s-p-2)/[(n_s-p-1)(n_s-p-4)]`:

- Hartlap factor :math:`h = (n_s-p-2)/(n_s-1)` (the inverse is unbiased after multiplying by it);
- Dodelson–Schneider variance factor :math:`1 + B(p-n_\theta)` (the scatter of the best fit);
- Percival :math:`m_1 = [1 + B(p-n_\theta)]/[1 + A + B(n_\theta+1)]`, the factor applied to the
  posterior variance so that it covers the extra scatter.

For each route we report the smallest :math:`n_s` that meets both criteria at a target :math:`t`:
:math:`\sqrt{m_1}-1 \le t` (the inflation of :math:`\sigma(S_8)`), and the noise of the estimated
error itself, :math:`\sigma[\hat\sigma]/\sigma = \tfrac12\sqrt{2/(n_s-p-4)} \le t` (Taylor, Joachimi &
Kitching 2013), i.e. :math:`n_s \ge p + 4 + 1/(2t^2)`; and the number of full-sky light-cones that gives,
when each light-cone holds :math:`\lfloor 41\,253/A\rfloor` independent footprints of area
:math:`A` (the largest map footprint of the run).

Routes:
1. the full vector estimated from simulations, :math:`p = 71\,342` (T5, k_max = 10 h/Mpc);
2. the score-compressed vector, :math:`p = n_\theta = 212` (one number per parameter, script 32);
3. the analytic covariance (Gaussian + SSC, C3 for one shell) validated on sub-blocks: the
   simulations needed to detect a fractional amplitude error :math:`\delta` of a sub-block of
   :math:`p_b` rows at 3σ, :math:`n_s = 18/(p_b\delta^2)` (the mean whitened :math:`\chi^2/p_b`
   over :math:`n_s` draws has standard deviation :math:`\sqrt{2/(p_b n_s)}`), with :math:`n_s > p_b`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import SENS, Record  # noqa: E402

RUN = SENS / "shells_2031" / "S5_public_2031"
SUMMARY = SENS / "shells_2031" / "shells_2031_summary.json"
FULL_SKY = 41252.96


def factors(ns, p, nt):
    d = (ns - p - 1.0) * (ns - p - 4.0)
    A, B = 2.0 / d, (ns - p - 2.0) / d
    ds = 1.0 + B * (p - nt)
    return {"hartlap": (ns - p - 2.0) / (ns - 1.0), "dodelson_schneider": ds,
            "percival_m1": ds / (1.0 + A + B * (nt + 1.0))}


def n_needed(p, nt, target, key="percival_m1"):
    """Smallest n_s with sqrt(factor) - 1 <= target (bisection over integers)."""
    lo, hi = p + 5, p + 5
    while np.sqrt(factors(hi, p, nt)[key]) - 1.0 > target:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if np.sqrt(factors(mid, p, nt)[key]) - 1.0 > target:
            lo = mid
        else:
            hi = mid
    return int(hi)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, nargs="+", default=[0.05, 0.10],
                    help="tolerated inflation of sigma(S8)")
    ap.add_argument("--delta", type=float, nargs="+", default=[0.05, 0.10],
                    help="route 3: amplitude error of a sub-block to detect at 3 sigma")
    ap.add_argument("--pb", type=int, nargs="+", default=[50, 200, 1000],
                    help="route 3: rows of the validated sub-block")
    a = ap.parse_args(argv)
    rec = Record(__file__)
    for f in (RUN / "config.json", RUN / "info.npz", SUMMARY):
        rec.add_input(f)
    cfg = json.loads((RUN / "config.json").read_text())
    nt = len(np.load(RUN / "info.npz")["names"])
    p_full = json.loads(SUMMARY.read_text())["runs"]["S5_public_2031"]["checks"]["min_corr_eig_T5_k10"]["rows"]
    area = max(v for k, v in cfg["areas"].items() if k != "x_arch")
    per_cone = int(FULL_SKY // area)

    routes = {}
    for name, p in (("full", p_full), ("compressed", nt)):
        r = {"p": p, "n_theta": nt, "targets": {}}
        for t in a.target:
            ns_m1 = n_needed(p, nt, t)
            ns_tjk = int(np.ceil(p + 4 + 1.0 / (2.0 * t * t)))
            ns = max(ns_m1, ns_tjk)
            r["targets"][f"{t:g}"] = {"n_s": ns, "n_s_percival": ns_m1, "n_s_error_on_error": ns_tjk, "light_cones": int(np.ceil(ns / per_cone)),
                                      "hartlap": factors(ns, p, nt)["hartlap"],
                                      "dodelson_schneider": factors(ns, p, nt)["dodelson_schneider"],
                                      "percival_m1": factors(ns, p, nt)["percival_m1"]}
            print(f"{name:10s} p={p:6d} sigma inflation <= {t:.2f}: n_s = {ns:7d}, "
                  f"light-cones = {r['targets'][f'{t:g}']['light_cones']}", flush=True)
        routes[name] = r
    val = {}
    for pb in a.pb:
        for d in a.delta:
            ns = max(int(np.ceil(18.0 / (pb * d * d))), pb + 5)
            val[f"pb{pb}_d{d:g}"] = {"p_b": pb, "delta": d, "n_s": ns,
                                     "light_cones": int(np.ceil(ns / per_cone))}
            print(f"validation p_b={pb:5d} delta={d:.2f}: n_s = {ns}, light-cones = {val[f'pb{pb}_d{d:g}']['light_cones']}")
    routes["analytic_validated"] = val
    rec.data.update(run="S5_public_2031", rows_full=p_full, n_theta=nt, footprint_deg2=area,
                    footprints_per_light_cone=per_cone, routes=routes,
                    note="sigma inflation = sqrt(Percival m1) - 1; light-cones are full-sky, "
                         "cut into floor(41253/area) independent footprints of the largest map")
    print(rec.write())


if __name__ == "__main__":
    main()
