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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("mask", "chi2"))
    ap.add_argument("--kmax", type=float, default=1.0, help="h/Mpc")
    ap.add_argument("--drop-xray-auto", action="store_true")
    ap.add_argument("--tag", default="", help="mask variant (file 25_mask<tag>.npz)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--layout", default="_s0")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args(argv)
    {"mask": cmd_mask, "chi2": cmd_chi2}[a.cmd](a)


if __name__ == "__main__":
    main()
