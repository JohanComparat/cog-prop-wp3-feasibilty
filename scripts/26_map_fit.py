"""Part D, step 6: the MAP of public_2026 under the ggah_mod 1.7.0 model, by Gauss-Newton steps.

    source ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh gpu
    PYTHONPATH=~/software/ggah_sens_study_wt_v170:$PYTHONPATH python scripts/26_map_fit.py step [--it 1]
    PYTHONPATH=... python scripts/26_map_fit.py eval --it 1 [--max-blocks 20]
    PYTHONPATH=... python scripts/26_map_fit.py score --it 1

``step`` takes the model and Jacobian at iterate ``it - 1`` (iterate 0: script 22's, at the
defaults) and solves the damped, prior-regularised normal equations for the step,
cosmology held fixed (the first pass fits the occupation, AGN, energetics and nuisances).
``eval`` evaluates the full model at the new point block by block (resumable).
``score`` gives chi^2 at the new point and the per-block table of script 23.
The covariance is script 22's (jackknife per block, Hartlap); y x y is left out
(its instrument noise is not subtracted).
"""

import argparse
import gc
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import ggah_sens_study  # noqa: E402,F401
from wp3feas.model import NZ_FILE, build_public, projection, public_config  # noqa: E402
from wp3feas.provenance import DATA, Record  # noqa: E402
from wp3feas.vector import VECTOR, hartlap, jk_cov, load_vector  # noqa: E402

S22 = importlib.import_module("22_fisher_public")
FIXED = ("Omega_m", "ln10A_s", "h", "n_s", "Omega_b")
EXCLUDE = ("cl|y|y|",)
DAMP = (0.0, 0.1, 1.0, 10.0)


def it_dir(it):
    return S22.OUT if it == 0 else DATA / f"26_map/it{it}"


def theta_of(it):
    meta = json.loads((S22.OUT / "meta.json").read_text())
    if it == 0:
        return np.asarray(meta["theta0"])
    return np.asarray(json.loads((it_dir(it) / "theta.json").read_text())["theta"])


def system(D, slices=None):
    """``(r, iC blocks, rows)``: the residuals and per-block inverse covariances on the used rows."""
    meta = json.loads((S22.OUT / "meta.json").read_text())
    info = {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]], "ell": meta["ell"]}
    vec = load_vector()
    P, rows = projection(vec, info)
    m = P @ D
    d = vec.data[rows]
    ell = vec.index["x"][rows]
    fin = (np.isfinite(d) & np.isfinite(m) & np.all(np.isfinite(vec.jk[:, rows]), axis=0)
           & (ell >= S22.SCALE_LMIN))
    key = np.array(["|".join(vec.index[k][r].decode() for k in ("stat", "a", "b", "survey")) for r in rows])
    tag = np.array(["los" if not t else "+".join(map(str, t)) for t in S22._slice_of(vec, rows)])
    if slices:
        fin &= np.isin(tag, slices)
    out = []
    for k in np.unique(key[fin]):
        if k in EXCLUDE:
            continue
        sel = np.flatnonzero((key == k) & fin)
        C = jk_cov(vec.jk[:, rows[sel]])
        h = hartlap(vec.n_jk, sel.size)
        if h > 0 and np.all(np.isfinite(C)):
            out.append((sel, h * np.linalg.pinv(C)))
    return P, d - m, out


def prior_sigma(names):
    from ggah_sens_study.joint import analysis as AN

    cfg = public_config()
    w = AN.photo_widths(names, cfg, [0.5 * (lo + hi) for lo, hi in cfg.shells])
    return np.array([np.inf if n in FIXED else (w[n] if n in w else (AN.box_sigma(n, cfg.sources) or np.inf))
                     for n in names])


def cmd_step(a):
    rec = Record(__file__)
    meta, D, J = S22._assemble(it_dir(a.it - 1)) if a.it == 1 else (None, None, None)
    if a.it != 1:
        raise SystemExit("iterations past 1 need the Jacobian at the new point: rerun 22 jac there")
    names = meta["names"]
    th0 = np.asarray(meta["theta0"])
    P, r, blocks = system(D, a.slices)
    Jd = P @ J
    used = np.flatnonzero(np.abs(Jd[np.concatenate([s for s, _ in blocks])]).sum(0) > 0)
    free = np.array([n not in FIXED for n in names]) & np.isin(np.arange(len(names)), used)
    sp = prior_sigma(names)
    A = np.zeros((len(names),) * 2)
    g = np.zeros(len(names))
    chi2 = 0.0
    for sel, iC in blocks:
        Jb = Jd[sel]
        A += Jb.T @ iC @ Jb
        g += Jb.T @ iC @ r[sel]
        chi2 += float(r[sel] @ iC @ r[sel])
    Pr = np.where(np.isfinite(sp), 1.0 / sp ** 2, 0.0)
    f = np.flatnonzero(free)
    sc = np.where(np.isfinite(sp[f]), sp[f], 1.0)
    out = []
    for lam in DAMP:
        Af = (A[np.ix_(f, f)] + np.diag(Pr[f])) * np.outer(sc, sc)
        Af += lam * np.eye(f.size)
        dx = np.linalg.solve(Af, g[f] * sc) * sc  # step from the defaults (prior centred there)
        pred = chi2 - 2 * g[f] @ dx + dx @ A[np.ix_(f, f)] @ dx
        out.append({"lambda": lam, "chi2_pred": float(pred), "prior": float(np.sum(Pr[f] * dx ** 2)),
                    "max_pull_prior": float(np.max(np.abs(dx / sc)))})
        print(f"lambda={lam:5.1f} chi2 {chi2:.0f} -> {pred:.0f} (linear), prior {out[-1]['prior']:.1f}, "
              f"max step {out[-1]['max_pull_prior']:.2f} prior sigma", flush=True)
        if lam == a.damp:
            th1 = th0.copy()
            th1[f] += dx
    d = it_dir(a.it)
    d.mkdir(parents=True, exist_ok=True)
    if not a.dry:
        (d / "theta.json").write_text(json.dumps({"names": names, "theta": th1.tolist(), "lambda": a.damp}))
    steps = {n: float(th1[i] - th0[i]) for i, n in enumerate(names) if free[i]}
    big = sorted(steps, key=lambda n: -abs(steps[n] / sp[names.index(n)]))[:20]
    print("largest steps (prior sigma):", {n: round(steps[n] / sp[names.index(n)], 2) for n in big})
    rec.data.update(it=a.it, slices=a.slices, chi2_start=chi2, n_rows=int(sum(s.size for s, _ in blocks)),
                    n_free=int(f.size), fixed=FIXED, exclude=EXCLUDE, damping=out, lambda_used=a.damp,
                    theta=dict(zip(names, th1.tolist())), step_over_prior={n: steps[n] / sp[names.index(n)] for n in steps})
    print(rec.write(f"26_map_step{a.it}{a.tag}"))


def cmd_eval(a):
    import jax
    import jax.numpy as jnp

    d = it_dir(a.it)
    th = jnp.asarray(theta_of(a.it))
    t0 = time.time()
    f, info, _, _ = build_public(public_config(), jit=False, block_pairs=a.block_pairs,
                                 block_pairs_los=a.block_pairs_los)
    meta = json.loads((S22.OUT / "meta.json").read_text())
    assert [b["name"] for b in info["blocks"]] == meta["blocks"], "block layout differs from script 22"
    done = 0
    for b in info["blocks"]:
        path = d / f"{b['name']}.npz"
        if path.is_file():
            continue
        if a.max_blocks and done >= a.max_blocks:
            print("stopping: --max-blocks reached", flush=True)
            return
        done += 1
        t1 = time.time()
        y = np.asarray(jax.jit(b["f"])(th))
        np.savez(path, rows=np.asarray(b["rows"]), D=y)
        print(f"  {b['name']}: {time.time() - t1:.1f} s", flush=True)
        jax.clear_caches()
        gc.collect()
    print(f"done in {time.time() - t0:.1f} s")


def cmd_score(a):
    rec = Record(__file__)
    for p in (VECTOR, NZ_FILE):
        rec.add_input(p)
    _, D0, _ = S22._assemble(it_dir(a.it - 1), jac=False)
    _, D1, _ = S22._assemble(it_dir(a.it), jac=False)
    res = {}
    for tag, D in (("start", D0), ("new", D1)):
        _, r, blocks = system(D)
        res[tag] = float(sum(r[s] @ iC @ r[s] for s, iC in blocks))
    n = int(sum(s.size for s, _ in blocks))
    print(f"chi2 {res['start']:.0f} -> {res['new']:.0f} over {n} rows")
    rec.data.update(it=a.it, slices=a.slices, chi2=res, n_rows=n)
    print(rec.write(f"26_map_score{a.it}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("step", "eval", "score"))
    ap.add_argument("--it", type=int, default=1)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--block-pairs", type=int, default=40)
    ap.add_argument("--block-pairs-los", type=int, default=10)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--slices", nargs="*", default=None, help='slice tags kept ("0", "1", "0+1", "los")')
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry", action="store_true", help="step: do not write theta")
    a = ap.parse_args(argv)
    {"step": cmd_step, "eval": cmd_eval, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
