"""A systematic error per block type, raised until each block type fits, and what it costs S(k).

    # at CC, in the environment of the chain (WP3FEAS_MAP_TAG, _VECTOR, _MASK, _SHOT, _AGN_BAND):
    sbatch --mem=120G --export=ALL,... cc/cpu.sbatch scripts/28_sys_floor.py fit --it 18 --layout _s0
    # on the laptop, after fetching 28_sys_floor<tag>_it<k>.npz to WP3FEAS_DATA/cc/:
    python scripts/28_sys_floor.py band --it 18 --layout _s0 --tag _v1final

The jackknife covariance of a block type that the model misses (chi^2/n more than two standard
deviations above one, 1 + 2 sqrt(2/n)) is not the error of the comparison: the model has an
error of its own.  ``fit`` adds to every block of such a type a fraction f of the model in
quadrature, C_b -> C_b / h + diag(f m)^2 (h the Hartlap factor), with f set so that the type
reaches chi^2 = n, and refits; m is the model at iterate ``it``, so that the floor does not
move with the fit.  The refit is the Gauss-Newton solution of the problem
linearised at iterate ``it`` (its model and Jacobian), every non-cosmological parameter free
under the priors of ``26_map_fit.py``; f and the fit are iterated until f moves by less than
1e-4.  It writes the point, the Laplace covariance (J^T C^-1 J + P)^-1 without and with the
floors, and with the floors and each block type left out in turn, to
``WP3FEAS_DATA/28_sys_floor<tag>_it<k>.npz`` and the table to
``results/28_sys_floor<tag>_it<k>.json``, and the floor of every row of the vector,
f m at iterate ``it``, to ``28_sys_floor<tag>_it<k>_sigma.npz``: ``26_map_fit.py`` adds it to
the covariance when ``WP3FEAS_SYS`` names that file, so that the chain refits with the floors
held fixed (the linearised refit here is only a guide: it leaves the range of the Jacobian).

``band`` propagates the three to S(k, z) with the Jacobian of ``91_sk_band.py jac`` at the same
iterate (first order, cosmology fixed): the band without and with the floors, and for each block
type the factor by which the band at k = 0.3, 1, 3 and 10 h/Mpc grows without it, the parts of
the vector that carry S(k).
"""

import argparse
import collections
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from wp3feas.provenance import DATA, Record  # noqa: E402

K_REPORT = (0.3, 1.0, 3.0, 10.0)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tag(a):
    return f"{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', a.tag)}_it{a.it}"


def blocks_of(s26, D):
    """Per block: rows (into the used rows), the jackknife covariance over Hartlap, the model, the
    block type; with the data and the projection.  As ``26_map_fit.system``, keeping C."""
    from wp3feas.model import projection
    from wp3feas.vector import hartlap, jk_cov, load_vector

    s25 = _load("s25", "25_scale_cuts.py")
    meta = json.loads((s26.S22.OUT / "meta.json").read_text())
    info = {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]], "ell": meta["ell"]}
    vec = load_vector()
    P, rows = projection(vec, info)
    m = P @ D + s26.shot_rows(vec, rows)
    d = vec.data[rows]
    ell = vec.index["x"][rows]
    fin = (np.isfinite(d) & np.isfinite(m) & np.all(np.isfinite(vec.jk[:, rows]), axis=0)
           & (ell >= s26.S22.SCALE_LMIN))
    key = np.array(["|".join(vec.index[k][r].decode() for k in ("stat", "a", "b", "survey")) for r in rows])
    if os.environ.get("WP3FEAS_MASK"):
        fin &= np.load(os.environ["WP3FEAS_MASK"])["keep"][rows]

    def typ(i):
        A, B = vec.index["a"][i].decode(), vec.index["b"][i].decode() or vec.index["a"][i].decode()
        return vec.index["stat"][i].decode() + ":" + "x".join(sorted((s25.kind(A) or "?", s25.kind(B) or "?")))

    out = []
    for k in np.unique(key[fin]):
        if k in s26.EXCLUDE:
            continue
        sel = np.flatnonzero((key == k) & fin)
        C = jk_cov(vec.jk[:, rows[sel]])
        h = hartlap(vec.n_jk, sel.size)
        if h > 0 and np.all(np.isfinite(C)):
            out.append({"sel": sel, "C": C / h, "type": typ(rows[sel[0]])})
    return P, d, m, out, rows


def weights(blocks, m, f):
    """The inverse covariance of each block with the floor f[type] times the model."""
    return [np.linalg.pinv(b["C"] + np.diag((f.get(b["type"], 0.0) * m[b["sel"]]) ** 2)) for b in blocks]


def chi2_by_type(blocks, W, r):
    t = collections.defaultdict(lambda: [0.0, 0])
    for b, w in zip(blocks, W):
        t[b["type"]][0] += float(r[b["sel"]] @ w @ r[b["sel"]])
        t[b["type"]][1] += b["sel"].size
    return dict(t)


def normal(blocks, W, J, r, skip=()):
    A = np.zeros((J.shape[1],) * 2)
    g = np.zeros(J.shape[1])
    for b, w in zip(blocks, W):
        if b["type"] in skip:
            continue
        Js = J[b["sel"]]
        A += Js.T @ w @ Js
        g += Js.T @ w @ r[b["sel"]]
    return A, g


def solve_f(blocks, m, r, t, n):
    """The floor of type t that gives chi^2 = n on the residual r (bisection in f)."""
    bt = [b for b in blocks if b["type"] == t]

    def c2(f):
        return sum(float(r[b["sel"]] @ np.linalg.pinv(b["C"] + np.diag((f * m[b["sel"]]) ** 2)) @ r[b["sel"]])
                   for b in bt)

    lo, hi = 0.0, 0.05
    while c2(hi) > n and hi < 100:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if c2(mid) > n else (lo, mid)
    return hi


def cmd_fit(a):
    s26 = _load("s26", "26_map_fit.py")
    s26.S22.set_layout(a.layout)
    rec = Record(__file__)
    it = s26.it_dir(a.it)
    meta, D, Jfull = s26.S22._assemble(it)
    P, d, m0, blocks, rows = blocks_of(s26, D)
    J = P @ Jfull
    names = meta["names"]
    th0 = np.asarray(meta["theta0"])
    x0 = s26.theta_of(a.it)
    sp = s26.prior_sigma(names)
    Pr = np.where(np.isfinite(sp), 1.0 / sp ** 2, 0.0)
    W0 = weights(blocks, m0, {})
    A0, _ = normal(blocks, W0, J, d - m0)
    free = np.flatnonzero(np.array([n not in s26.FIXED for n in names]) & (np.diag(A0) + Pr > 0))

    def fit(f):
        """The minimum of the linearised chi^2 + prior with the floors f (f scales the model at x0)."""
        W = weights(blocks, m0, f)
        A, g = normal(blocks, W, J, d - m0)
        Af = A[np.ix_(free, free)] + np.diag(Pr[free])
        bf = g[free] - Pr[free] * (x0[free] - th0[free])
        dx = np.zeros(len(names))
        dx[free] = np.linalg.solve(Af, bf)
        return dx, W

    def lap(W, skip=()):
        A, _ = normal(blocks, W, J, d - m0, skip)
        return np.linalg.pinv(A[np.ix_(free, free)] + np.diag(Pr[free]))

    types = sorted({b["type"] for b in blocks})
    c0 = chi2_by_type(blocks, W0, d - m0)
    f = {}
    for k in range(a.max_iter):
        dx, W = fit(f)
        m = m0 + J @ dx
        r = d - m
        c = chi2_by_type(blocks, weights(blocks, m0, f), r)
        new = dict(f)
        for t in types:
            x2, n = c[t]
            if x2 / n > 1 + 2 * np.sqrt(2 / n) or (f.get(t, 0) > 0 and abs(x2 / n - 1) > 1e-3):
                new[t] = solve_f(blocks, m0, r, t, n)
        move = max([abs(new.get(t, 0) - f.get(t, 0)) for t in types])
        print(f"pass {k}: " + ", ".join(f"{t} {v:.4f}" for t, v in sorted(new.items())) + f"; largest move {move:.2e}",
              flush=True)
        f = new
        if move < 1e-4:
            break
    dx, W = fit(f)
    m = m0 + J @ dx
    r = d - m
    Wf = weights(blocks, m0, f)
    c1 = chi2_by_type(blocks, Wf, r)
    C_base = lap(W0)
    C_sys = lap(Wf)
    C_drop = np.stack([lap(Wf, skip=(t,)) for t in types])
    # the floor per row of the vector, f m at this iterate, for 26_map_fit.py (WP3FEAS_SYS)
    from wp3feas.vector import load_vector
    sig = np.zeros(load_vector().data.size)
    for b in blocks:
        sig[rows[b["sel"]]] = f.get(b["type"], 0.0) * np.abs(m0[b["sel"]])
    np.savez(DATA / f"28_sys_floor{tag(a)}_sigma.npz", sigma=sig, floors=json.dumps(f))
    np.savez(DATA / f"28_sys_floor{tag(a)}.npz", names=np.array(names), theta=x0 + dx, theta_it=x0,
             theta0=th0, free=free, C_base=C_base, C_sys=C_sys, C_drop=C_drop, types=np.array(types))
    big = sorted(free, key=lambda i: -abs(dx[i] / np.sqrt(C_sys[list(free).index(i)][list(free).index(i)])))[:12]
    rec.data.update(
        it=a.it, map_tag=os.environ.get("WP3FEAS_MAP_TAG", ""), layout=a.layout, floor_model="f * model",
        accept="chi2/n > 1 + 2 sqrt(2/n)", floors=f,
        chi2={t: {"n": c0[t][1], "chi2_it": c0[t][0], "chi2_sys": c1[t][0]} for t in types},
        chi2_total_it=sum(v[0] for v in c0.values()), chi2_total_sys=sum(v[0] for v in c1.values()),
        n_rows=sum(v[1] for v in c0.values()),
        shifts={names[i]: {"from": float(x0[i]), "to": float(x0[i] + dx[i]),
                           "error_sys": float(np.sqrt(C_sys[list(free).index(i)][list(free).index(i)])),
                           "error_base": float(np.sqrt(C_base[list(free).index(i)][list(free).index(i)]))}
                for i in big})
    for t in types:
        print(f"{t:22s} n {c0[t][1]:5d}  chi2/n {c0[t][0] / c0[t][1]:6.2f} -> {c1[t][0] / c1[t][1]:5.2f}"
              f"  floor {f.get(t, 0):.4f}")
    print(rec.write(f"28_sys_floor{tag(a)}"))


def cmd_band(a):
    rec = Record(__file__)
    jpath = DATA / f"91_sk_band{a.layout}{a.tag}_it{a.it}.npz"
    fpath = DATA / "cc" / f"28_sys_floor{a.layout}{a.tag}_it{a.it}.npz"
    J, F = np.load(jpath), np.load(fpath)
    assert list(J["names"]) == list(F["names"]) and np.allclose(J["theta"], F["theta_it"])
    free, types = F["free"], [str(t) for t in F["types"]]
    k = J["k"]
    idx = {kk: int(np.argmin(np.abs(np.log(k / kk)))) for kk in K_REPORT}
    out = {}
    for iz, z in enumerate(J["z"]):
        G = J["G"][iz][:, free]
        dS = G @ (F["theta"] - F["theta_it"])[free]

        def sig(C):
            return np.sqrt(np.clip(np.einsum("ki,ij,kj->k", G, C, G), 0, None))
        sb, ss = sig(F["C_base"]), sig(F["C_sys"])
        drop = {t: sig(F["C_drop"][j]) for j, t in enumerate(types)}
        row = {"S_it": J["S"][iz].tolist(), "S_sys": (J["S"][iz] + dS).tolist(), "S_default": J["S_default"][iz].tolist(),
               "sigma_base": sb.tolist(), "sigma_sys": ss.tolist(), "k": k.tolist(), "at": {}}
        for kk, i in idx.items():
            row["at"][f"{kk:g}"] = {
                "S_it": float(J["S"][iz][i]), "sigma_base": float(sb[i]), "S_sys": float(J["S"][iz][i] + dS[i]),
                "sigma_sys": float(ss[i]), "S_default": float(J["S_default"][iz][i]),
                "growth_without": {t: float(drop[t][i] / ss[i]) for t in types}}
            g = row["at"][f"{kk:g}"]["growth_without"]
            top = sorted(g, key=lambda t: -g[t])[:4]
            print(f"z {z:g} k {k[i]:5.2f}: S {J['S'][iz][i]:.4f} +- {sb[i]:.4f} -> {J['S'][iz][i] + dS[i]:.4f} +- "
                  f"{ss[i]:.4f} (default {J['S_default'][iz][i]:.4f}); without: "
                  + ", ".join(f"{t} x{g[t]:.2f}" for t in top))
        out[f"{z:g}"] = row
    rec.inputs += [str(jpath), str(fpath)]
    rec.data.update(it=a.it, layout=a.layout, map_tag=a.tag, cosmology="fixed", band=out, types=types)
    print(rec.write(f"28_sys_floor_band{a.layout}{a.tag}_it{a.it}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "band"))
    ap.add_argument("--it", type=int, required=True)
    ap.add_argument("--layout", default="_s0")
    ap.add_argument("--tag", default="", help="band: the chain tag (fit reads WP3FEAS_MAP_TAG)")
    ap.add_argument("--max-iter", type=int, default=30)
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "band": cmd_band}[a.cmd](a)


if __name__ == "__main__":
    main()
