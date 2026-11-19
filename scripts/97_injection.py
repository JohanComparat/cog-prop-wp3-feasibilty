r"""Injection test of the slice-0 MAP fit (PLAN, Verification): a synthetic vector with a known
truth, fitted by the Gauss-Newton chain of script 26 from a point away from it.

    # at CC, CPU: the synthetic vector (WP3FEAS_VECTOR = v1, the vector of the truth's chain)
    WP3FEAS_MAP_TAG=_cuts python scripts/97_injection.py make --truth-it 21 --seed 1
    # the chain's start: the defaults (script 22's model and Jacobian), as iterate 2 of _inj
    python scripts/97_injection.py seed
    # then cc/submit_map.sh with VECTOR = the synthetic vector, MAPTAG=_inj, START=3
    # after iterate k: 26 summary (MAPTAG=_inj), then
    WP3FEAS_MAP_TAG=_inj WP3FEAS_VECTOR=<synthetic> python scripts/97_injection.py summary --it k

The truth is the MAP of the `_cuts` chain at iterate 21, not the defaults: its model D was
computed by the chain itself (band 0.2--2.3 keV of the AGN, v1 fields), so the fit sees the
same model function as the one that made the data. The fit starts from the defaults, 3--9
Laplace sigma away from the truth in the occupation parameters.
The synthetic data are ``P D_truth`` plus one draw of the jackknife scatter of the vector:
``sqrt((n-1)/n) z^T (jk - mean)`` with ``z ~ N(0, 1)`` per region, whose covariance is the
jackknife covariance of every row, so the noise has the correlations the fit assumes per block
and those across blocks it neglects. The Poisson term of the AGN x X-ray blocks
(``WP3FEAS_SHOT``) is left out of the data and of the fit alike.

``summary`` writes ``results/97_injection_s0_it<k>.json`` (and ``97_injection_s0.json``):
per free parameter ``(theta_hat - theta_true) / sigma`` with the Laplace sigma of
``26 summary``, and chi2 at the truth and at the iterate on the same rows.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from wp3feas.provenance import DATA, RESULTS, Record  # noqa: E402
from wp3feas.vector import VECTOR  # noqa: E402

S22 = importlib.import_module("22_fisher_public")
OUT = DATA / "97_injection"
CHAIN = "_inj"


def vector_path(seed):
    return OUT / f"tier_ls11_v1_inj_s{seed}.h5"


def cmd_make(a):
    import h5py

    S26 = importlib.import_module("26_map_fit")
    from wp3feas.model import projection
    from wp3feas.vector import load_vector

    rec = Record(__file__)
    S22.set_layout("_s0")
    truth_dir = S26.it_dir(a.truth_it)
    rec.add_input(VECTOR)
    rec.add_input(truth_dir / "theta.json")
    meta, D, _ = S22._assemble(truth_dir, jac=False)
    info = {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]], "ell": meta["ell"]}
    vec = load_vector()
    P, rows = projection(vec, info)
    m = P @ D
    jk = vec.jk
    fin = np.isfinite(jk)
    dev = np.where(fin, jk - np.nanmean(np.where(fin, jk, np.nan), axis=0), 0.0)
    rng = np.random.default_rng(a.seed)
    n = jk.shape[0]
    noise = np.sqrt((n - 1) / n) * (rng.standard_normal(n) @ dev)
    data = np.array(vec.data, copy=True)
    ok = np.isfinite(m) & np.isfinite(data[rows])
    data[rows[ok]] = m[ok] + noise[rows[ok]]
    OUT.mkdir(parents=True, exist_ok=True)
    out = vector_path(a.seed)
    shutil.copyfile(VECTOR, out)
    with h5py.File(out, "r+") as f:
        f["data"][...] = data
        f.attrs["injection_seed"] = a.seed
        f.attrs["injection_truth"] = str(truth_dir)
        f.attrs["injection_source"] = str(VECTOR)
    th = json.loads((truth_dir / "theta.json").read_text())
    (OUT / f"truth_s{a.seed}.json").write_text(json.dumps(
        {"names": th["names"], "theta": th["theta"], "truth_dir": str(truth_dir), "seed": a.seed,
         "vector": str(out), "source_vector": str(VECTOR), "rows_replaced": int(ok.sum())}))
    print(f"{out}: {ok.sum()} rows replaced by the model of {truth_dir} plus jackknife noise (seed {a.seed})")
    rec.data.update(seed=a.seed, truth_dir=str(truth_dir), vector=str(out), rows_replaced=int(ok.sum()))
    print(rec.write(f"97_injection_make_s{a.seed}"))
    print("done in make")


def cmd_seed(a):
    """The defaults (script 22's blocks, D and J at theta0) as iterate 2 of the chain, with a
    placeholder objective at iterate 1, so that the step of iterate 3 starts from them and
    nothing is written into script 22's directory."""
    S22.set_layout("_s0")
    meta = json.loads((S22.OUT / "meta.json").read_text())
    chain = DATA / f"26_map_s0{CHAIN}"
    it2, it1 = chain / "it2", chain / "it1"
    if it2.exists():
        sys.exit(f"{it2} exists")
    it2.mkdir(parents=True)
    for name in meta["blocks"]:
        shutil.copy(S22.OUT / f"{name}.npz", it2 / f"{name}.npz")
    (it2 / "theta.json").write_text(json.dumps({"names": meta["names"], "theta": meta["theta0"], "trust": 1.0,
                                                "base": 1, "slices": None, "lambda": 1.0}))
    it1.mkdir(exist_ok=True)
    (it1 / "objective.json").write_text(json.dumps({"chi2": 1e30, "prior": 0.0, "n_rows": 0, "placeholder": True,
                                                    "note": "placeholder: the chain starts at it2 = the defaults"}))
    print(f"seeded {it2} from {S22.OUT} (the defaults)")


def cmd_summary(a):
    S26 = importlib.import_module("26_map_fit")
    rec = Record(__file__)
    S22.set_layout("_s0")
    seed = a.seed
    truth = json.loads((OUT / f"truth_s{seed}.json").read_text())
    tag = os.environ.get("WP3FEAS_MAP_TAG", CHAIN)
    sfile = RESULTS / f"26_map_summary_s0{tag}_it{a.it}.json"
    rec.add_input(sfile)
    summ = json.loads(sfile.read_text())
    tv = dict(zip(truth["names"], truth["theta"]))
    pars = {}
    for n, p in summ["parameters"].items():
        pull = (p["value"] - tv[n]) / p["error"] if p["error"] else None
        pars[n] = {"truth": tv[n], "value": p["value"], "error": p["error"], "start": p["default"], "pull": pull,
                   "start_pull": (p["default"] - tv[n]) / p["error"] if p["error"] else None}
    pulls = np.array([v["pull"] for v in pars.values() if v["pull"] is not None])
    # chi2 at the truth on the same rows: the truth's model against the synthetic data
    meta, D, _ = S22._assemble(Path(truth["truth_dir"]), jac=False)
    _, r, blocks = S26.system(D)
    chi2_truth = float(sum(r[s] @ iC @ r[s] for s, iC in blocks))
    n_rows = int(sum(s.size for s, _ in blocks))
    o = summ["objective"]
    res = {"it": a.it, "seed": seed, "n_free": len(pars), "chi2_truth": chi2_truth, "n_rows_truth": n_rows,
           "chi2_map": o["chi2"], "prior_map": o["prior"], "n_rows_map": o["n_rows"],
           "pull_rms": float(np.sqrt(np.mean(pulls ** 2))), "pull_max_abs": float(np.max(np.abs(pulls))),
           "n_beyond_2sigma": int(np.sum(np.abs(pulls) > 2)), "n_beyond_3sigma": int(np.sum(np.abs(pulls) > 3)),
           "start_pull_rms": float(np.sqrt(np.mean([v["start_pull"] ** 2 for v in pars.values()
                                                    if v["start_pull"] is not None]))),
           "parameters": pars}
    print(f"it {a.it}: chi2 truth {chi2_truth:.1f} on {n_rows}, MAP {o['chi2']:.1f} on {o['n_rows']}; "
          f"pull rms {res['pull_rms']:.2f}, max {res['pull_max_abs']:.2f}, >3 sigma {res['n_beyond_3sigma']}"
          f" (start: rms {res['start_pull_rms']:.2f})")
    rec.data.update(res)
    print(rec.write(f"97_injection_s0_it{a.it}"))
    print(rec.write("97_injection_s0"))
    print("done in summary")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("make", "seed", "summary"))
    ap.add_argument("--truth-it", type=int, default=21)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--it", type=int, default=0)
    ap.add_argument("--max-blocks", type=int, default=0, help="ignored (cc/gpu.sbatch)")
    a = ap.parse_args(argv)
    {"make": cmd_make, "seed": cmd_seed, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
