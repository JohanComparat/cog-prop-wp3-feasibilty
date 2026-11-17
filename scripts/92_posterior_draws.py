r"""Parameter draws for the figures of ggah_mod's technical paper (Paper II) with a band.

    source ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh
    PYTHONPATH=~/software/ggah_sens_study_wt_v170:$PYTHONPATH python scripts/92_posterior_draws.py today --it 21
    PYTHONPATH=... python scripts/92_posterior_draws.py forecast

``today`` draws from the Laplace posterior of the slice-0 fit (script 26, chain ``WP3FEAS_MAP_TAG``,
``laplace_cov.npz`` of iterate ``--it``): a Gaussian at the best fit with the covariance
:math:`(J^{\rm T}C^{-1}J + P)^{-1}` on the free parameters, the cosmology fixed.

``forecast`` draws from the Fisher forecast of the five shells of 2030 (ggah_sens_study
``shells_2030/<run>/shells_fisher.npz``, key ``T5|<kmax>|C1+SSC``: the full vector, Gaussian
plus super-sample covariance), with the priors and the evolution scheme of its script 78
(``Run.constraints``), centred on the defaults; each draw is evaluated at ``--z`` (a global
value plus its slope ``p/dz`` times ``z - pivot``).

Both write ``WP3FEAS_DATA/92_draws_<case>.npz`` (``names``, ``centre``, ``draws``) and a JSON
under ``results/``.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import DATA, Record  # noqa: E402

SENS = Path(os.environ.get("SENS_WT", Path.home() / "software/ggah_sens_study_wt_v170"))


def cmd_today(a):
    rec = Record(__file__)
    tag = os.environ.get("WP3FEAS_MAP_TAG", "")
    p = DATA / "cc" / f"26_map{a.layout}{tag}" / f"it{a.it}" / "laplace_cov.npz"
    L = np.load(p)
    names, x, f, C = list(L["names"]), L["theta"], L["free"], L["C"]
    rng = np.random.default_rng(a.seed)
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    Ls = V * np.sqrt(np.clip(w, 0, None))
    d = np.tile(x, (a.n, 1))
    d[:, f] += rng.standard_normal((a.n, f.size)) @ Ls.T
    case = f"today{a.layout}{tag}_it{a.it}"
    out = DATA / f"92_draws_{case}.npz"
    np.savez(out, names=np.array(names), centre=x, draws=d, free=f)
    rec.inputs.append(str(p))
    rec.data.update(case=case, n=a.n, seed=a.seed, n_free=int(f.size), chi2=float(L["chi2"]),
                    n_rows=int(L["n_rows"]), file=str(out),
                    sigma={names[i]: float(np.sqrt(C[j, j])) for j, i in enumerate(f)})
    print(rec.write(f"92_posterior_draws_{case}"))


def cmd_forecast(a):
    rec = Record(__file__)
    sys.path.insert(0, str(SENS / "scripts"))
    sp = importlib.util.spec_from_file_location("s78", SENS / "scripts/78_shells_2030.py")
    S = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(S)
    S._setup()
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import analysis as AN

    run = S.Run(a.run, load_fiducial())
    fis = np.load(run.dir / "shells_fisher.npz", allow_pickle=True)
    key = f"T5|{a.kmax:g}|C1+SSC"
    F = dict(zip([str(k) for k in fis["keys"]], fis["F"]))[key]
    r = run.constraints(F, "E1", widths=run.widths(calib=S.S77.CALIB_MAIN))
    red, cov = list(r["reduced"]), np.asarray(r["cov"])
    centre = dict(zip([str(n) for n in run.rep], np.asarray(run.info["theta0"], float)))
    centre.update({n: float(v) for n, v in run.centre.items()})
    rng = np.random.default_rng(a.seed)
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    dr = rng.standard_normal((a.n, len(red))) @ (V * np.sqrt(np.clip(w, 0, None))).T
    names = [n for n in red if "/" not in n and "[" not in n]
    x0 = np.array([centre.get(n, np.nan) for n in names])
    d = np.tile(x0, (a.n, 1))
    for j, n in enumerate(names):
        d[:, j] += dr[:, red.index(n)]
        if f"{n}/dz" in red:
            d[:, j] += (a.z - AN.PIVOT_Z) * dr[:, red.index(f"{n}/dz")]
        if f"{n}[0]" in red and a.z < 0.2:
            d[:, j] += dr[:, red.index(f"{n}[0]")]
    ok = np.isfinite(x0)
    case = f"forecast_{a.run}_k{a.kmax:g}"
    out = DATA / f"92_draws_{case}.npz"
    np.savez(out, names=np.array(names)[ok], centre=x0[ok], draws=d[:, ok])
    rec.inputs.append(str(run.dir / "shells_fisher.npz"))
    rec.data.update(case=case, run=a.run, key=key, scheme="E1", calib=S.S77.CALIB_MAIN, z=a.z,
                    pivot=float(AN.PIVOT_Z), n=a.n, seed=a.seed, sigma_S8=float(r["sigma"]["S8"]),
                    file=str(out), dropped_no_centre=[n for n, k in zip(names, ok) if not k],
                    sigma={n: float(np.std(d[:, j])) for j, n in enumerate(names) if ok[j]})
    print(f"sigma(S8) = {r['sigma']['S8']:.5f}; {int(ok.sum())} parameters")
    print(rec.write(f"92_posterior_draws_{case}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("today", "forecast"))
    ap.add_argument("--it", type=int, default=21)
    ap.add_argument("--layout", default="_s0")
    ap.add_argument("--run", default="S5_lsst_2030")
    ap.add_argument("--kmax", type=float, default=10.0, help="h/Mpc")
    ap.add_argument("--z", type=float, default=0.15)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    {"today": cmd_today, "forecast": cmd_forecast}[a.cmd](a)


if __name__ == "__main__":
    main()
