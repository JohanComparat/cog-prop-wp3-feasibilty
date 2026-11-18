r"""Part C: catastrophic photometric-redshift outliers in the five shells of ``S5_public_2031``.

    python scripts/12_photoz_outliers.py --kmax 3        (at CC: sbatch cc/cpu.sbatch ...)

A fraction :math:`f` of each photometric sample of shell :math:`i` lies in another shell
:math:`j`: :math:`n_a \to (1-f)\,n_a + f\sum_j w_j n_{a@j}`, where :math:`a@j` is the same
sample (colour, stellar-mass bin) in shell :math:`j`, with its clustering there.
To first order in :math:`f` each leg :math:`L` of a spectrum changes by

.. math::

    \partial C_{Lb}/\partial f = -C_{Lb} + \sum_j w_j\,C_{L@j,\,b},

summed over both legs; :math:`C_{L@j,b}` is read from the vector of the run, and is zero
where the run holds no such pair (no overlap of the n(z) in the Limber approximation).
This template :math:`t` holds the dilution of every spectrum and the cross-shell signal
the outliers induce.
Rows that the run does not hold (pairs of shells two or more apart) are not added: the
tolerance is that of the vector as specified.
The densities do not change (the counts are measured; only the n(z) moves).

Destinations: ``uniform`` (every other shell where the sample is photometric, equal
weights) and ``adjacent`` (the shells i +- 1); one global :math:`f`, and one :math:`f`
per shell (``uniform``).

As in ``ggah_sens_study`` ``scripts/77`` ``step_sys`` (Jacobian, covariance, priors and
calibration case of the run, read only): the first-order bias per unit :math:`f`,
:math:`\Delta S_8 = [(F+P)^{-1} J^T C^{-1} t]_{S_8}`, the tolerance
:math:`f_{\max} = 0.3\,\sigma(S_8)/|\Delta S_8/f|`, and the cost when :math:`f` is free:
:math:`\sigma(S_8)` with the template columns marginalised under a Gaussian prior of
width ``--prior`` (Schur complement of the extended Fisher matrix).

Writes ``results/12_photoz_outliers_k<kmax>.json``.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import Record  # noqa: E402

RUN = "S5_public_2031"
#: The sens-study results this study reads (v1.6.0).
SENS_ROOT_CC = "/sps/lsst/users/jcomparat01/ggah/results/ggah_sens_study_results_v1.6.0"


def _s90():
    import ggah_sens_study

    p = Path(ggah_sens_study.__file__).resolve().parents[1] / "scripts" / "90_shells_2031.py"
    spec = importlib.util.spec_from_file_location("s90", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s90"] = mod
    spec.loader.exec_module(mod)
    return mod


def split(name):
    base, sh = name.rsplit("@", 1)
    return base, int(sh)


def templates(v0, info, nsh):
    """``name -> t`` on the node vector of the run."""
    disc, P = list(info["discrete"]), [tuple(p) for p in info["pairs"]]
    nd, ne = len(disc), len(info["ell"])
    photo = set(info["photo"]) & ({a for a, _ in P} | {b for _, b in P})
    idx = {}
    for k, (a, b) in enumerate(P):
        idx[(a, b)] = k
        idx[(b, a)] = k

    def blk(k):
        return v0[nd + k * ne: nd + (k + 1) * ne]

    def dests(L, mode):
        base, i = split(L)
        js = [j for j in range(nsh) if j != i and f"{base}@{j}" in photo
              and (mode == "uniform" or abs(j - i) == 1)]
        return [(f"{base}@{j}", 1.0 / len(js)) for j in js] if js else []

    def build(mode, shells):
        t = np.zeros_like(v0)
        for k, (a, b) in enumerate(P):
            for L, other in ((a, b), (b, a)):
                if L not in photo or split(L)[1] not in shells:
                    continue
                ds = dests(L, mode)
                if not ds:
                    continue
                d = -blk(k).copy()
                for Lj, w in ds:
                    if (Lj, other) in idx:
                        d += w * blk(idx[(Lj, other)])
                t[nd + k * ne: nd + (k + 1) * ne] += d
        return t

    allsh = set(range(nsh))
    out = {"f_uniform": build("uniform", allsh), "f_adjacent": build("adjacent", allsh)}
    for i in range(nsh):
        out[f"f_uniform@{i}"] = build("uniform", {i})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=float, default=3.0)
    ap.add_argument("--prior", type=float, nargs="+", default=[0.1, 0.02],
                    help="Gaussian prior widths on f when free")
    ap.add_argument("--dry", action="store_true", help="templates only, no Fisher matrix")
    a = ap.parse_args(argv)
    if "GGAH_SENS_RESULTS" not in os.environ and Path(SENS_ROOT_CC).is_dir():
        os.environ["GGAH_SENS_RESULTS"] = SENS_ROOT_CC

    S90 = _s90()
    S90._setup()
    S77 = S90.S77
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import analysis as AN
    from ggah_sens_study.joint.survey import Noise

    rec = Record(__file__)
    rdir = S90.root() / RUN
    for f in ("jacobian.npz", "info.npz", "config.json", "ssc.npz"):
        rec.add_input(rdir / f)
    fid = load_fiducial()
    run = S90.S78.Run(RUN, fid, drop=S90.dropped(RUN, S90._info(RUN)["discrete"]))
    v0 = np.asarray(run.v0, float)
    tm = templates(v0, run.info, run.nsh)
    names_t = list(tm)
    norms = {k: float(np.abs(t).sum()) for k, t in tm.items()}
    print("templates", {k: f"{v:.3e}" for k, v in norms.items()}, flush=True)
    if a.dry:
        return
    Jx = np.hstack([run.J, np.stack([tm[n] for n in names_t], axis=1)])
    n = run.J.shape[1]
    Tx = np.eye(n + len(names_t))
    Tx[:n, :n] = run.T
    F, st = AN.cell_fisher(v0, Jx, run.info, run.cfg, Noise(), run.fid.cosmology, tier="T5",
                           k_max=a.kmax, ssc=run.ssc(), area_scales=(1.0,), T=Tx,
                           y_noise=run.y_noise)
    S8 = run.centre["sigma8"] * np.sqrt(run.centre["Omega_m"] / 0.3)
    groups = {"f_uniform": ["f_uniform"], "f_adjacent": ["f_adjacent"],
              "f_uniform_per_shell": [f"f_uniform@{i}" for i in range(run.nsh)]}
    res = []
    for lev in S77.LEVELS:
        Fx = F[lev][0]
        Fp, B, Ftt = Fx[:n, :n], Fx[:n, n:], Fx[n:, n:]
        for sch in run.schemes:
            r = run.constraints(Fp, sch, b=B)
            sig = float(r["sigma"]["S8"])
            bias = {}
            for j, key in enumerate(names_t):
                d8 = float(r["dS8"][j])
                bias[key] = {"dS8_per_unit_f": d8,
                             "f_max_own": 0.3 * sig / abs(d8) if d8 else None,
                             "f_max_target": 0.3 * S77.TARGET * S8 / abs(d8) if d8 else None}
            free = {}
            for g, members in groups.items():
                jj = [names_t.index(m) for m in members]
                for pw in a.prior:
                    Fg = Ftt[np.ix_(jj, jj)] + np.eye(len(jj)) / pw ** 2
                    Bg = B[:, jj]
                    Fm = Fp - Bg @ np.linalg.solve(Fg, Bg.T)
                    s = float(run.constraints(Fm, sch)["sigma"]["S8"])
                    free[f"{g}|prior{pw:g}"] = {"sigma_S8": s, "ratio": s / sig}
            res.append({"level": lev, "scheme": sch, "sigma_S8_fixed": sig,
                        "S8_rel": float(r["S8_rel"]), "bias": bias, "free": free})
            print(f"{lev} {sch}: sigma(S8) {sig:.4g}; f_max(uniform) "
                  f"{bias['f_uniform']['f_max_own']}; free ratio(uniform, 0.1) "
                  f"{free['f_uniform|prior0.1']['ratio']:.3f}", flush=True)
    rec.data.update(run=RUN, sens_root=str(S90.root()), tier="T5", k_max=a.kmax,
                    rows=st.get("rows"), S8=S8, calib=S77.CALIB_MAIN, dropped=run.drop,
                    templates=names_t, template_l1=norms, priors=a.prior,
                    rule="f_max = 0.3 sigma(S8) / |dS8/df|; first-order bias (F+P)^-1 J^T C^-1 t; "
                         "free: template columns marginalised with a Gaussian prior",
                    results=res)
    print(rec.write(f"12_photoz_outliers_k{a.kmax:g}"))
    print("done in", f"{rec.data.get('wall', 0)}")


if __name__ == "__main__":
    main()
