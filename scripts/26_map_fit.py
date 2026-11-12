"""Part D, step 6: the MAP of public_2026 under the ggah_mod 1.7.0 model, by Gauss-Newton steps.

    source ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh gpu
    PYTHONPATH=~/software/ggah_sens_study_wt_v170:$PYTHONPATH python scripts/26_map_fit.py step [--it 1]
    PYTHONPATH=... python scripts/26_map_fit.py eval --it 1 [--max-blocks 20]
    PYTHONPATH=... python scripts/26_map_fit.py score --it 1

``search --it k`` evaluates the model alone at fractions 1, 1/2, 1/4, 1/8 of step k and keeps
the best one that lowers chi^2 + prior (else the chain stops), so no Jacobian is spent on a
rejected point.
``jac --it k`` writes the model and its Jacobian at iterate k (script 22's jac) for the next step.
``step`` takes the model and Jacobian at iterate ``it - 1`` (iterate 0: script 22's, at the
defaults) and solves the damped, prior-regularised normal equations, inside the parameter
ranges and a trust region (an iterate that loses rows or worsens the objective is rejected),
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
import os
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


def it_dir(it):
    # WP3FEAS_MAP_TAG keeps a second chain (another damping, another machine) in its own directory
    return S22.OUT if it == 0 else DATA / f"26_map{S22.LAYOUT}{os.environ.get('WP3FEAS_MAP_TAG', '')}/it{it}"


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
    if os.environ.get("WP3FEAS_MASK"):   # a row mask of scripts/25_scale_cuts.py (keep over the vector rows)
        fin &= np.load(os.environ["WP3FEAS_MASK"])["keep"][rows]
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

    cfg = public_config(**S22.LAYOUTS[S22.LAYOUT])
    w = AN.photo_widths(names, cfg, [0.5 * (lo + hi) for lo, hi in cfg.shells])
    return np.array([np.inf if n in FIXED else (w[n] if n in w else (AN.box_sigma(n, cfg.sources) or np.inf))
                     for n in names])


def bounds(names, th0, sp):
    """``(lo, hi)``: each parameter's range: its ``PARAM_BOX`` (1 per cent of the width inside
    the edges, always holding the default), else five prior widths about the default."""
    from ggah_cal.campaign import PARAM_BOX
    from ggah_sens_study.priors import base_name

    lo, hi = np.full(len(names), -np.inf), np.full(len(names), np.inf)
    for i, n in enumerate(names):
        b = base_name(n)
        if b in PARAM_BOX and "@" not in n:
            l, h = PARAM_BOX[b]
            m = 0.01 * (h - l)
            lo[i], hi[i] = min(l + m, th0[i]), max(h - m, th0[i])
        elif np.isfinite(sp[i]) and b not in PARAM_BOX:
            lo[i], hi[i] = th0[i] - 5 * sp[i], th0[i] + 5 * sp[i]
    return lo, hi


def shell_limits(names, x, lo, hi):
    """``(lo, hi)`` with each shell offset ``p@s`` limited so that ``p + p@s`` stays in p's range."""
    lo, hi = lo.copy(), hi.copy()
    for i, n in enumerate(names):
        if "@" in n and n.split("@")[0] in names:
            g = names.index(n.split("@")[0])
            lo[i], hi[i] = lo[g] - x[g], hi[g] - x[g]
    return lo, hi


def point(k, slices):
    """The model, Jacobian, residuals and blocks of iterate ``k``."""
    meta, D, J = S22._assemble(it_dir(k))
    P, r, blocks = system(D, slices)
    return meta, P @ J, r, blocks, theta_of(k)


def objective(k, r, blocks, x, th0, sp):
    pr = np.where(np.isfinite(sp), 1.0 / sp ** 2, 0.0)
    c = float(sum(r[s] @ iC @ r[s] for s, iC in blocks))
    o = {"chi2": c, "prior": float(np.sum(pr * (x - th0) ** 2)), "n_rows": int(sum(s.size for s, _ in blocks))}
    (it_dir(k) / "objective.json").write_text(json.dumps(o))
    return o


def _with(x, i, v):
    y = x.copy()
    y[i] = v
    return y


def probe():
    """``ok(theta)``: whether the first block of every shell and of the line of sight is finite."""
    import jax
    import jax.numpy as jnp

    _, info, _, _ = build_public(public_config(**S22.LAYOUTS[S22.LAYOUT]), jit=False,
                                 block_pairs=40, block_pairs_los=10)
    first = {}
    for b in info["blocks"]:
        first.setdefault(b["name"].split(".")[0], b)
    fns = [jax.jit(b["f"]) for b in first.values()]
    print("probe blocks:", [b["name"] for b in first.values()], flush=True)
    return lambda th: all(bool(np.all(np.isfinite(np.asarray(g(jnp.asarray(th)))))) for g in fns)


def cmd_step(a):
    """Iterate ``it``: a bounded, trust-limited Gauss-Newton step, from iterate ``it - 1`` if it
    improved the objective on all the rows, else from its base with half the trust region."""
    from scipy.optimize import lsq_linear

    rec = Record(__file__)
    k = a.it - 1
    if k >= 1 and json.loads((it_dir(k) / "theta.json").read_text()).get("stop"):
        _stop_forward(k, a.it)
        print("done in step")
        return
    meta, Jd, r, blocks, x = point(k, a.slices)
    names = meta["names"]
    th0 = np.asarray(meta["theta0"])
    sp = prior_sigma(names)
    o = objective(k, r, blocks, x, th0, sp)
    trust, status, base = a.trust, "start", k
    if k >= 1:
        h = json.loads((it_dir(k) / "theta.json").read_text())
        ob = json.loads((it_dir(h["base"]) / "objective.json").read_text())
        ok = (o["n_rows"] == ob["n_rows"] and o["chi2"] + o["prior"] < ob["chi2"] + ob["prior"])
        status = "accepted" if ok else "rejected"
        print(f"iterate {k}: {status}: chi2 + prior {o['chi2'] + o['prior']:.1f} on {o['n_rows']} rows, "
              f"base {h['base']}: {ob['chi2'] + ob['prior']:.1f} on {ob['n_rows']} rows", flush=True)
        trust = min(h["trust"] * 1.5, a.trust) if ok else h["trust"] / 2
        if not ok:
            base = h["base"]
            meta, Jd, r, blocks, x = point(base, a.slices)
            o = ob
    used_rows = np.concatenate([s for s, _ in blocks])
    if not (np.all(np.isfinite(Jd[used_rows])) and np.all(np.isfinite(r[used_rows]))):
        raise RuntimeError(f"iterate {base}: non-finite model or Jacobian on the used rows")
    used = np.flatnonzero(np.abs(Jd[used_rows]).sum(0) > 0)
    free = np.array([n not in FIXED for n in names]) & np.isin(np.arange(len(names)), used)
    A = np.zeros((len(names),) * 2)
    g = np.zeros(len(names))
    for sel, iC in blocks:
        Jb = Jd[sel]
        A += Jb.T @ iC @ Jb
        g += Jb.T @ iC @ r[sel]
    Pr = np.where(np.isfinite(sp), 1.0 / sp ** 2, 0.0)
    f = np.flatnonzero(free)
    sc = np.where(np.isfinite(sp[f]), sp[f], 1.0)
    gf = (g[f] - Pr[f] * (x[f] - th0[f])) * sc  # minus half the gradient at x, in prior widths
    H = (A[np.ix_(f, f)] + np.diag(Pr[f])) * np.outer(sc, sc) + a.damp * np.eye(f.size)
    w, V = np.linalg.eigh(H)
    keep = w > w.max() * 1e-14
    R, y = (V[:, keep] * np.sqrt(w[keep])) @ V[:, keep].T, (V[:, keep] / np.sqrt(w[keep])) @ (V[:, keep].T @ gf)
    lo, hi = shell_limits(names, x, *bounds(names, th0, sp))

    def solve(frozen, tr):
        lb = np.minimum(np.maximum((lo[f] - x[f]) / sc, -tr), 0.0)
        ub = np.maximum(np.minimum((hi[f] - x[f]) / sc, tr), 0.0)
        for j, i in enumerate(f):
            if i in frozen:
                lb[j], ub[j] = -1e-12, 1e-12
        sol = lsq_linear(R, y, bounds=(lb, ub), method="bvls", max_iter=2000)
        th = x.copy()
        th[f] += sol.x * sc
        return sol, lb, ub, th

    # the ranges are not enough: the gas profiles' slopes depend on mass, and their finite
    # region couples parameters, so the candidate is tested on the compiled probe blocks
    ok_at = probe()
    if not ok_at(x):
        raise RuntimeError(f"iterate {base}: the probe blocks are not finite at the starting point")
    frozen = set()
    for _ in range(10):
        sol, lb, ub, th1 = solve(frozen, trust)
        if ok_at(th1):
            break
        new = [i for i in f if i not in frozen and th1[i] != x[i] and not ok_at(_with(x, i, th1[i]))]
        print("not finite: frozen", [names[i] for i in new] or f"none alone, trust {trust / 2:.3g}", flush=True)
        if new:
            frozen |= set(new)
        else:
            trust /= 2
    else:
        raise RuntimeError("no finite step found")
    dx = sol.x * sc
    Af = A[np.ix_(f, f)]
    pred = o["chi2"] - 2 * g[f] @ dx + dx @ Af @ dx
    pri = float(np.sum(Pr * (th1 - th0) ** 2))
    bad = [i for i in f if not lo[i] - 1e-9 <= th1[i] <= hi[i] + 1e-9]
    if bad:
        # a parameter already outside its range at the start (shell_limits depend on x) is put
        # back on the nearest edge; the line search scores the clipped point with the full model
        print("clipped into their ranges:", {names[i]: (float(th1[i]), float(lo[i]), float(hi[i])) for i in bad},
              flush=True)
        th1[bad] = np.clip(th1[bad], lo[bad], hi[bad])
        if not ok_at(th1):
            raise RuntimeError(f"step leaves the ranges of {[names[i] for i in bad]} and the clipped point is not finite")
    frozen_names = [names[i] for i in sorted(frozen)]
    n_lim = int(np.sum(np.isclose(sol.x, lb) | np.isclose(sol.x, ub)))
    print(f"from iterate {base}: chi2 {o['chi2']:.0f} -> {pred:.0f} (linear) on {o['n_rows']} rows, prior "
          f"{o['prior']:.1f} -> {pri:.1f}, trust {trust:.3g}, {n_lim} of {f.size} parameters at a limit", flush=True)
    d = it_dir(a.it)
    d.mkdir(parents=True, exist_ok=True)
    if not a.dry:
        (d / "theta.json").write_text(json.dumps({"names": names, "theta": th1.tolist(), "lambda": a.damp,
                                                  "trust": float(trust), "base": int(base),
                                                  "slices": a.slices}))
    off = {n: float((th1[i] - th0[i]) / sp[i]) for i, n in enumerate(names) if free[i]}
    big = sorted(off, key=lambda n: -abs(off[n]))[:20]
    print("largest offsets from the defaults (prior sigma):", {n: round(off[n], 2) for n in big})
    rec.data.update(it=a.it, layout=S22.LAYOUT, slices=a.slices, previous_status=status, base=int(base),
                    chi2_base=o["chi2"], prior_base=o["prior"], n_rows=o["n_rows"], n_free=int(f.size),
                    chi2_pred=float(pred), prior_new=pri, trust=float(trust), n_at_limit=n_lim, frozen_not_finite=frozen_names, fixed=FIXED,
                    exclude=EXCLUDE, lambda_used=a.damp, theta=dict(zip(names, th1.tolist())), offset_over_prior=off)
    print(rec.write(f"26_map_step{a.it}{S22.LAYOUT}{a.tag}"))
    print("done in step")  # cc/gpu.sbatch reruns a command until it prints this


def _stop_forward(k, it):
    """Carry the stop of iterate ``k`` (no fraction of its step improved) to iterate ``it``."""
    d = it_dir(it)
    d.mkdir(parents=True, exist_ok=True)
    h = json.loads((it_dir(k) / "theta.json").read_text())
    (d / "theta.json").write_text(json.dumps({**h, "stop": h["stop"]}))
    print(f"iterate {k} stopped the chain ({h['stop']}): nothing to do", flush=True)


FRACTIONS = (1.0, 0.5, 0.25, 0.125)


def cmd_search(a):
    """A line search along the step of iterate ``it``: the model alone (no Jacobian) at the
    fractions ``FRACTIONS`` of the step, each block compiled once; the best fraction that
    lowers chi^2 + prior on the base's rows becomes iterate ``it``, else the chain stops."""
    import jax
    import jax.numpy as jnp

    d = it_dir(a.it)
    h = json.loads((d / "theta.json").read_text())
    if h.get("stop"):
        print(f"iterate {a.it} already stopped ({h['stop']})")
        print("done in search")
        return
    if "search" in h:
        print(f"iterate {a.it}: line search already done")
        print("done in search")
        return
    sd = d / "search"
    sd.mkdir(exist_ok=True)
    xb, x1 = theta_of(h["base"]), np.asarray(h["theta"])
    pts = [xb + t * (x1 - xb) for t in FRACTIONS]
    t0 = time.time()
    _, info, _, _ = build_public(public_config(**S22.LAYOUTS[S22.LAYOUT]), jit=False, block_pairs=a.block_pairs,
                                 block_pairs_los=a.block_pairs_los)
    meta = json.loads((S22.OUT / "meta.json").read_text())
    assert [b["name"] for b in info["blocks"]] == meta["blocks"], "block layout differs from script 22"
    done = 0
    si, sn = S22.shard_of(a)
    for ib, b in enumerate(info["blocks"]):
        path = sd / f"{b['name']}.npz"
        if path.is_file() or ib % sn != si:
            continue
        if a.max_blocks and done >= a.max_blocks:
            print("stopping: --max-blocks reached", flush=True)
            return
        done += 1
        t1 = time.time()
        g = jax.jit(b["f"])
        D = np.stack([np.asarray(g(jnp.asarray(p))) for p in pts])
        np.savez(path, rows=np.asarray(b["rows"]), D=D)
        print(f"  {b['name']}: {time.time() - t1:.1f} s", flush=True)
        del g
        jax.clear_caches()
        gc.collect()
    if sn > 1:
        print(f"shard {si}/{sn} done; the search without --shard combines the blocks")
        print("done in search")
        return
    missing = [b["name"] for b in info["blocks"] if not (sd / f"{b['name']}.npz").is_file()]
    if missing:
        raise RuntimeError(f"search {a.it}: blocks missing {missing[:5]}...")
    # all blocks done: the objective at each fraction
    rec = Record(__file__)
    n_rows = len(meta["discrete"]) + len(meta["pairs"]) * len(meta["ell"])
    D = np.zeros((len(FRACTIONS), n_rows))
    for name in meta["blocks"]:
        with np.load(sd / f"{name}.npz") as z:
            D[:, z["rows"]] += z["D"]
    th0 = np.asarray(meta["theta0"])
    sp = prior_sigma(meta["names"])
    pr = np.where(np.isfinite(sp), 1.0 / sp ** 2, 0.0)
    ob = json.loads((it_dir(h["base"]) / "objective.json").read_text())
    res = []
    for t, p, Dt in zip(FRACTIONS, pts, D):
        _, r, blocks = system(Dt, h.get("slices"))
        n = int(sum(s.size for s, _ in blocks))
        c = float(sum(r[s] @ iC @ r[s] for s, iC in blocks))
        res.append({"fraction": t, "chi2": c, "prior": float(np.sum(pr * (p - th0) ** 2)), "n_rows": n})
        print(f"fraction {t:5.3f}: chi2 {c:.1f} + prior {res[-1]['prior']:.1f} on {n} rows "
              f"(base {h['base']}: {ob['chi2'] + ob['prior']:.1f} on {ob['n_rows']})", flush=True)
    good = [q for q in res if q["n_rows"] == ob["n_rows"] and np.isfinite(q["chi2"])
            and q["chi2"] + q["prior"] < ob["chi2"] + ob["prior"]]
    if good:
        q = min(good, key=lambda q: q["chi2"] + q["prior"])
        i = FRACTIONS.index(q["fraction"])
        h.update(theta=pts[i].tolist(), trust=h["trust"] * q["fraction"], search=res, fraction=q["fraction"])
        print(f"iterate {a.it}: fraction {q['fraction']} kept, chi2 + prior {q['chi2'] + q['prior']:.1f}")
    else:
        h.update(stop=f"no fraction of the step from iterate {h['base']} improves chi2 + prior", search=res)
        print(f"iterate {a.it}: {h['stop']}")
    (d / "theta.json").write_text(json.dumps(h))
    rec.data.update(it=a.it, layout=S22.LAYOUT, base=h["base"], base_objective=ob, search=res,
                    fraction=h.get("fraction"), stop=h.get("stop"))
    print(rec.write(f"26_map_search{a.it}{S22.LAYOUT}{a.tag}"))
    print(f"done in {time.time() - t0:.1f} s")


def cmd_jac(a):
    """The model and its Jacobian at iterate ``it`` (script 22's jac), into that iterate's directory."""
    h = json.loads((it_dir(a.it) / "theta.json").read_text())
    if h.get("stop"):
        print(f"iterate {a.it} stopped the chain ({h['stop']}): no Jacobian")
        print("done in jac")
        return
    a.at, a.out = theta_of(a.it), it_dir(a.it)
    S22.cmd_jac(a)


def cmd_eval(a):
    import jax
    import jax.numpy as jnp

    d = it_dir(a.it)
    th = jnp.asarray(theta_of(a.it))
    t0 = time.time()
    f, info, _, _ = build_public(public_config(**S22.LAYOUTS[S22.LAYOUT]), jit=False, block_pairs=a.block_pairs,
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
    print(rec.write(f"26_map_score{a.it}{S22.LAYOUT}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("step", "search", "jac", "eval", "score"))
    ap.add_argument("--it", type=int, default=1)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--trust", type=float, default=1.0, help="largest change of a parameter per step, in prior widths")
    ap.add_argument("--block-pairs", type=int, default=40)
    ap.add_argument("--block-pairs-los", type=int, default=10)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--slices", nargs="*", default=None, help='slice tags kept ("0", "1", "0+1", "los")')
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry", action="store_true", help="step: do not write theta")
    ap.add_argument("--layout", default="", choices=list(S22.LAYOUTS))
    ap.add_argument("--shard", default=None, help="search, jac: i/n, every n-th block from block i")
    a = ap.parse_args(argv)
    S22.set_layout(a.layout)
    {"step": cmd_step, "search": cmd_search, "jac": cmd_jac, "eval": cmd_eval, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
