"""Part D: the scale cuts and exclusions of public_2026, decided before any fit of S8.

    python scripts/25_scale_cuts.py mask                 # the row mask and its reasons
    python scripts/25_scale_cuts.py chi2 --model-dir DIR --layout _s0   # chi2 by block type, masked and not

``mask`` removes a row of the vector when
  (a) l < L_MIN (mask mode coupling);
  (b) l > k_max chi(z) for any discrete tracer of the pair (k_max = 1 h/Mpc; 3 with --kmax 3),
      chi at the mean redshift of its slice;
  (c) the pair is y x y (the noise of the y map is not subtracted in sum_stat);
  (d) the pair involves an X-ray field of v0 (to be replaced by the re-measurement on a clean
      footprint, sum_stat tier_xray_background.py) when --drop-xray-auto is given: X-ray x X-ray only.
The mask goes to WP3FEAS_DATA/25_mask.npz (keep, bool over the vector rows), which
scripts/26_map_fit.py reads through WP3FEAS_MASK.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import DATA, Record  # noqa: E402
from wp3feas.vector import VECTOR, load_vector, parse_tracer  # noqa: E402

L_MIN = 30.0
H = 0.6774


def chi_mpch(z):
    from astropy.cosmology import FlatLambdaCDM
    return FlatLambdaCDM(H0=100 * H, Om0=0.3089).comoving_distance(z).value * H


def kind(n):
    return parse_tracer(n)["kind"] if n else None


def cmd_mask(a):
    rec = Record(__file__)
    rec.add_input(VECTOR)
    vec = load_vector()
    ix = vec.index
    ell = ix["x"].astype(float)
    a_ = np.char.decode(ix["a"]) if ix["a"].dtype.kind == "S" else ix["a"]
    b_ = np.char.decode(ix["b"]) if ix["b"].dtype.kind == "S" else ix["b"]
    st = np.char.decode(ix["stat"]) if ix["stat"].dtype.kind == "S" else ix["stat"]
    is_cl = np.char.startswith(st.astype(str), "cl")
    keep = np.ones(ell.size, bool)
    why = collections.Counter()
    lmax_of = {}
    for n in set(a_) | set(b_):
        t = parse_tracer(n) if n else {"z": None}
        if t.get("z") is not None and t["kind"] in ("gal", "agn"):
            lmax_of[n] = a.kmax * chi_mpch(0.5 * sum(t["z"]))
    for i in np.flatnonzero(is_cl):
        A, B = a_[i], b_[i] or a_[i]
        r = None
        if ell[i] < L_MIN:
            r = "l < 30"
        elif min(lmax_of.get(A, np.inf), lmax_of.get(B, np.inf)) < ell[i]:
            r = f"l > {a.kmax:g} h/Mpc chi(z)"
        elif kind(A) == "y" and kind(B) == "y":
            r = "y x y (noise not subtracted)"
        elif a.drop_xray_auto and kind(A) == "xray" and kind(B) == "xray":
            r = "X-ray x X-ray of v0 (re-measured on a clean footprint)"
        if r:
            keep[i] = False
            why[r] += 1
    out = DATA / f"25_mask{a.tag}.npz"
    np.savez(out, keep=keep)
    n_cl = int(is_cl.sum())
    print(f"{n_cl} C_l rows, {int(keep[is_cl].sum())} kept; removed: {dict(why)} -> {out}")
    rec.data.update(kmax_h_mpc=a.kmax, l_min=L_MIN, drop_xray_auto=a.drop_xray_auto, n_rows=int(ell.size),
                    n_cl_rows=n_cl, n_cl_kept=int(keep[is_cl].sum()), removed=dict(why), mask_file=str(out),
                    lmax_by_tracer={k: float(v) for k, v in sorted(lmax_of.items())})
    print(rec.write(f"25_scale_cuts_mask{a.tag}"))


def cmd_chi2(a):
    """chi2 by block type at the model of --model-dir, on all rows and on the kept rows."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s26", ROOT / "scripts/26_map_fit.py")
    s26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s26)
    s26.S22.set_layout(a.layout)
    rec = Record(__file__)
    keep = np.load(DATA / f"25_mask{a.tag}.npz")["keep"]
    _, D, _ = s26.S22._assemble(Path(a.model_dir), jac=False)
    vec = load_vector()
    P, r, blocks = s26.system(D)
    meta = json.loads((s26.S22.OUT / "meta.json").read_text())
    from wp3feas.model import projection
    _, rows = projection(vec, {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]],
                               "ell": meta["ell"]})

    def typ(i):
        A, B = vec.index["a"][i].decode(), vec.index["b"][i].decode() or vec.index["a"][i].decode()
        return vec.index["stat"][i].decode() + ":" + "x".join(sorted((kind(A) or "?", kind(B) or "?")))

    tot = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])
    for sel, iC in blocks:
        t = typ(rows[sel[0]])
        tot[t][0] += float(r[sel] @ iC @ r[sel])
        tot[t][1] += sel.size
        k = keep[rows[sel]]
        if k.any():
            s2 = sel[k]
            C = s26.jk_cov(vec.jk[:, rows[s2]])
            h = s26.hartlap(vec.n_jk, s2.size)
            tot[t][2] += float(r[s2] @ (h * np.linalg.pinv(C)) @ r[s2])
            tot[t][3] += s2.size
    rows_out = {t: {"chi2_all": v[0], "n_all": v[1], "chi2_kept": v[2], "n_kept": v[3]} for t, v in tot.items()}
    for t, v in sorted(rows_out.items(), key=lambda kv: -kv[1]["chi2_all"]):
        print(f"{t:18s} all {v['chi2_all']:9.0f} / {v['n_all']:5d}   kept {v['chi2_kept']:9.0f} / {v['n_kept']:5d}")
    s_all = sum(v["chi2_all"] for v in rows_out.values())
    s_k = sum(v["chi2_kept"] for v in rows_out.values())
    n_k = sum(v["n_kept"] for v in rows_out.values())
    print(f"total: all {s_all:.0f}, kept {s_k:.0f} on {n_k} rows")
    rec.data.update(model_dir=a.model_dir, layout=a.layout, by_type=rows_out, chi2_all=s_all,
                    chi2_kept=s_k, n_kept=n_k)
    print(rec.write(f"25_scale_cuts_chi2{a.tag}{a.out_tag}"))


def cmd_xshot(a):
    """X-ray x X-ray blocks: chi2 of the model of --model-dir as it is, with a free amplitude,
    and with a free amplitude plus a constant C_l (a shot noise of the resolved sources),
    each fitted by generalised least squares on the block's kept rows."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s26", ROOT / "scripts/26_map_fit.py")
    s26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s26)
    s26.S22.set_layout(a.layout)
    rec = Record(__file__)
    keep = np.load(DATA / f"25_mask{a.tag}.npz")["keep"]
    _, D, _ = s26.S22._assemble(Path(a.model_dir), jac=False)
    vec = load_vector()
    meta = json.loads((s26.S22.OUT / "meta.json").read_text())
    from wp3feas.model import projection
    P, rows = projection(vec, {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]],
                               "ell": meta["ell"]})
    m = P @ D
    ix = vec.index
    A_, B_ = ix["a"][rows].astype(str), ix["b"][rows].astype(str)
    out, tot = {}, np.zeros(3)
    for name in sorted({f"{x}__{y}" for x, y in zip(A_, B_) if x.startswith("EVT") and y.startswith("EVT")}):
        x, y = name.split("__")
        sel = np.flatnonzero((A_ == x) & (B_ == y) & keep[rows])
        if sel.size < 4:
            continue
        d = vec.data[rows[sel]]
        iC = s26.hartlap(vec.n_jk, sel.size) * np.linalg.pinv(s26.jk_cov(vec.jk[:, rows[sel]]))
        res = []
        for X in (None, m[sel][:, None], np.c_[m[sel], np.ones(sel.size)]):
            if X is None:
                r = d - m[sel]
                coef = []
            else:
                coef = np.linalg.solve(X.T @ iC @ X, X.T @ iC @ d)
                r = d - X @ coef
            res.append((float(r @ iC @ r), [float(c) for c in coef]))
        out[name] = {"n": int(sel.size), "chi2_model": res[0][0], "chi2_amp": res[1][0], "amp": res[1][1][0],
                     "chi2_amp_shot": res[2][0], "amp_with_shot": res[2][1][0], "shot": res[2][1][1],
                     "shot_over_model_l1000": res[2][1][1] / float(np.interp(1000, ix["x"][rows[sel]], m[sel]))}
        tot += [res[0][0], res[1][0], res[2][0]]
    n = sum(v["n"] for v in out.values())
    print(f"{len(out)} X-ray x X-ray blocks, {n} rows: chi2 model {tot[0]:.0f}, + amplitude {tot[1]:.0f}, "
          f"+ amplitude and shot noise {tot[2]:.0f}")
    au = {k: v for k, v in out.items() if k.split("__")[0] == k.split("__")[1]}
    for k, v in list(au.items())[:30]:
        print(f"{k:30s} n {v['n']:3d}  chi2 {v['chi2_model']:8.0f} {v['chi2_amp']:8.0f} {v['chi2_amp_shot']:8.0f}  "
              f"A {v['amp_with_shot']:.2f}  N/model(l=1000) {v['shot_over_model_l1000']:.2f}")
    rec.data.update(model_dir=a.model_dir, n_blocks=len(out), n_rows=n, chi2_model=tot[0], chi2_amp=tot[1],
                    chi2_amp_shot=tot[2], blocks=out)
    print(rec.write(f"25_scale_cuts_xshot{a.tag}{a.out_tag}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("mask", "chi2", "xshot"))
    ap.add_argument("--kmax", type=float, default=1.0, help="h/Mpc")
    ap.add_argument("--drop-xray-auto", action="store_true")
    ap.add_argument("--tag", default="", help="mask variant (file 25_mask<tag>.npz)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--layout", default="_s0")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args(argv)
    {"mask": cmd_mask, "chi2": cmd_chi2, "xshot": cmd_xshot}[a.cmd](a)


if __name__ == "__main__":
    main()
