r"""Part E: the Gaussian (Knox) covariance against the jackknife covariance of public_2026, on sub-blocks.

    python scripts/33_subblock_validation.py inputs            (laptop, light: reads the sum_stat fields)
    python scripts/33_subblock_validation.py run --model-dir DIR --tag _cuts --out-tag _cuts_it21   (CC)

``inputs`` packs what CC does not hold into ``WP3FEAS_DATA/33_inputs.npz``: the noise N_l of every
harmonic pair (``nl`` of sum_stat ``cl_nside1024_xray.h5``, the shared-object terms of the crosses
included; y x y carries its noise in C_l, N = 0) and the mask weight of every field, averaged on
HEALPix nside 128 (w and w^2 per coarse pixel; the y field from the ACT mask).

``run`` takes every harmonic block of the vector (``cl``; ``cl_sh`` but shear x shear, which
scripts/30_cov_shear.py covers) on the rows of the mask ``25_mask<tag>.npz`` with l >= 30, cut into
sub-blocks of at most 50 consecutive rows, and compares the jackknife covariance (100 regions) with
the Gaussian one, diagonal in l,

    sigma^2_ab(l) = [(S_aa + N_aa)(S_bb + N_bb) + (S_ab + N_ab)^2] / ((2l + 1) Delta l f_sky,ab),

S the model of --model-dir (an iterate of scripts/26_map_fit.py) where it is finite, else the
measurement clipped at zero (as in script 30); N_shear the white shape noise of shear_noise.json;
f_sky,ab = <w_a w_b>^2 / <w_a^2 w_b^2> from the coarse masks (for a shear field, the area of the bin,
the smaller of the two).
Per sub-block: the jackknife over the Gaussian sigma per row; the mean chi^2/p of the jackknife
realisations under the Gaussian covariance (1 if the Gaussian holds, the diagonal ratio squared on
average); the mean absolute correlation of the jackknife covariance off the diagonal (0 for the
Gaussian); and chi^2/p of data - model under each covariance (jackknife with the Hartlap factor).
Summaries by block type go to ``results/33_subblock_validation<out_tag>.json``.
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import DATA, Record  # noqa: E402
from wp3feas.vector import VECTOR, hartlap, jk_cov, load_vector, parse_tracer  # noqa: E402

TIER = Path(os.environ.get("WP3FEAS_TIER", Path.home() / "data/legacysurvey/dr11/south/bgsr21s/results/tier_xcorr"))
INPUTS = DATA / "33_inputs.npz"
NSIDE_C = 128
FULL_SKY_DEG2 = 41252.96
P_MAX = 50
LMIN = 30.0


def cmd_inputs(a):
    import h5py
    import healpy as hp
    rec = Record(__file__)
    cl = TIER / "measurements/cl/cl_nside1024_xray.h5"
    rec.add_input(cl)
    out = {}
    with h5py.File(cl, "r") as h:
        edges = h["ell_edges"][()]
        ell = h["ell_eff"][()]
        pairs = [k for k in h if "__" in k]
        out["nl_pairs"] = np.array(pairs)
        out["nl"] = np.array([h[k]["nl"][()] for k in pairs])
    out["ell_edges"], out["ell_eff"] = edges, ell
    names, w1, w2 = [], [], []
    for f in sorted((TIER / "inputs/cl_fields").glob("*_nside1024.npz")):
        m = np.load(f)["mask"].astype(np.float64)
        names.append(f.name.replace("_nside1024.npz", ""))
        w1.append(hp.ud_grade(m, NSIDE_C, order_in="RING"))
        w2.append(hp.ud_grade(m ** 2, NSIDE_C, order_in="RING"))
    act = TIER / "inputs/act_mask_nside1024.fits"
    m = hp.read_map(act).astype(np.float64)
    for n in ("y", "y_std"):
        names.append(n)
        w1.append(hp.ud_grade(m, NSIDE_C))
        w2.append(hp.ud_grade(m ** 2, NSIDE_C))
    rec.add_input(act)
    out["fields"] = np.array(names)
    out["w1"] = np.array(w1, dtype=np.float32)
    out["w2"] = np.array(w2, dtype=np.float32)
    np.savez_compressed(INPUTS, **out)
    rec.data.update(n_pairs=len(pairs), n_fields=len(names), nside_coarse=NSIDE_C, file=str(INPUTS))
    print(rec.write("33_subblock_inputs"))


def band_widths(x, edges):
    """Delta l of each bandpower centre x, from the sum_stat edges (the bin holding x)."""
    i = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    return np.diff(edges)[i]


def btype(stat, A, B):
    ka, kb = parse_tracer(A)["kind"] or "?", parse_tracer(B)["kind"] or "?"
    ka, kb = ("x-ray" if k in ("xray", "xsrc") else k for k in (ka, kb))
    return "x".join(sorted((ka, kb)))


def cmd_run(a):
    spec_path = ROOT / "scripts/30_cov_shear.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("s30", spec_path)
    s30 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s30)
    rec = Record(__file__)
    rec.add_input(VECTOR)
    rec.add_input(INPUTS)
    rec.add_input(a.noise)
    inp = np.load(INPUTS)
    edges = inp["ell_edges"]
    nl = {str(p): v for p, v in zip(inp["nl_pairs"], inp["nl"])}
    ell_nl = inp["ell_eff"]
    fidx = {str(n): i for i, n in enumerate(inp["fields"])}
    w1, w2 = inp["w1"].astype(np.float64), inp["w2"].astype(np.float64)
    shear = json.loads(Path(a.noise).read_text())
    keep = np.load(DATA / f"25_mask{a.tag}.npz")["keep"]
    vec = load_vector()
    ix = vec.index
    st = ix["stat"].astype(str)
    A_, B_ = ix["a"].astype(str), ix["b"].astype(str)
    model = s30.model_rows(a.model_dir, a.layout, vec) if a.model_dir else None

    def is_shear(n):
        return parse_tracer(n)["kind"] == "shear"

    # signal rows by ordered pair, both statistics
    rows_of = collections.defaultdict(list)
    for k in np.flatnonzero(np.char.startswith(st, "cl")):
        rows_of[(A_[k], B_[k])].append(k)
    rows_of = {p: np.array(v)[np.argsort(ix["x"][v])] for p, v in rows_of.items()}

    def signal(x, y, ell):
        k = rows_of.get((x, y))
        if k is None:
            k = rows_of.get((y, x))
        if k is None:
            return None
        c = model[k] if model is not None and np.all(np.isfinite(model[k])) else vec.data[k]
        return np.interp(ell, ix["x"][k], c)

    def noise(x, y, ell):
        if is_shear(x) or is_shear(y):
            return s30.white_noise(shear, x) * np.ones_like(ell) if x == y else np.zeros_like(ell)
        v = nl.get(f"{x}__{y}", nl.get(f"{y}__{x}"))
        return np.zeros_like(ell) if v is None else np.interp(ell, ell_nl, v)

    def fsky(x, y):
        if is_shear(x) or is_shear(y):
            ar = [shear[n]["area"] for n in (x, y) if is_shear(n) and n in shear]
            ar += [FULL_SKY_DEG2 * np.mean(w1[fidx[n]] ** 2) ** 2 / np.mean(w2[fidx[n]] ** 2)
                   for n in (x, y) if not is_shear(n) and n in fidx]
            return min(ar) / FULL_SKY_DEG2 if ar else np.nan
        if x not in fidx or y not in fidx:
            return np.nan
        i, j = fidx[x], fidx[y]
        return float(np.mean(w1[i] * w1[j]) ** 2 / np.mean(w2[i] * w2[j]))

    sub, summ = [], collections.defaultdict(lambda: collections.defaultdict(list))
    skipped = collections.Counter()
    for (x, y), k in sorted(rows_of.items()):
        stat = st[k[0]]
        if stat == "cl_sh" and is_shear(x) and is_shear(y):
            continue  # shear x shear: scripts/30_cov_shear.py
        k = k[keep[k] & (ix["x"][k] >= LMIN)]
        if k.size < 2:
            skipped["fewer than 2 kept rows"] += 1
            continue
        ell = ix["x"][k].astype(float)
        sxx, syy = signal(x, x, ell), signal(y, y, ell)
        if sxx is None or syy is None:
            skipped["auto-spectrum missing"] += 1
            continue
        fs = fsky(x, y)
        if not np.isfinite(fs) or fs <= 0:
            skipped["no mask"] += 1
            continue
        sxy = signal(x, y, ell)
        tot_xx = np.clip(sxx, 0, None) + noise(x, x, ell)
        tot_yy = np.clip(syy, 0, None) + noise(y, y, ell)
        tot_xy = sxy + noise(x, y, ell)
        var_g = (tot_xx * tot_yy + tot_xy ** 2) / ((2 * ell + 1) * band_widths(ell, edges) * fs)
        t = btype(stat, x, y)
        for c0 in range(0, k.size, P_MAX):
            s = slice(c0, min(c0 + P_MAX, k.size))
            kk, vg = k[s], var_g[s]
            p = kk.size
            if p < 2 or vec.n_jk - p - 2 <= 0:
                continue
            C = jk_cov(vec.jk[:, kk])
            if not np.all(np.isfinite(C)) or np.any(vg <= 0):
                skipped["non-finite covariance"] += 1
                continue
            sd = np.sqrt(np.diag(C))
            ratio = sd / np.sqrt(vg)
            corr = C / np.outer(sd, sd)
            offd = np.abs(corr[~np.eye(p, dtype=bool)]).mean()
            d = vec.jk[:, kk] - np.nanmean(vec.jk[:, kk], axis=0)
            d = d * np.sqrt(vec.n_jk - 1)  # mean_k d_k^T V^-1 d_k = tr(V^-1 C_jk) with jk_cov's (n-1)/n
            chi2_real = float(np.nanmean(np.sum(d ** 2 / vg, axis=1)) / p)
            row = dict(type=t, a=x, b=y, stat=stat, p=int(p), fsky=fs, l_min=float(ell[s][0]), l_max=float(ell[s][-1]),
                       ratio_median=float(np.median(ratio)), chi2_real_per_p=chi2_real, offdiag_abs_corr=float(offd))
            if model is not None and np.all(np.isfinite(model[kk])):
                r = vec.data[kk] - model[kk]
                h = hartlap(vec.n_jk, p)
                row.update(chi2_gauss_per_p=float(np.sum(r ** 2 / vg) / p),
                           chi2_jk_per_p=float(h * r @ np.linalg.solve(C, r) / p))
            sub.append(row)
            summ[t]["ratio"].extend(ratio.tolist())
            for key in ("chi2_real_per_p", "offdiag_abs_corr", "chi2_gauss_per_p", "chi2_jk_per_p"):
                if key in row:
                    summ[t][key].append((row[key], p))
    by_type = {}
    print(f"{'type':16s} {'sub':>4s} {'rows':>5s} {'jk/G p16':>8s} {'p50':>6s} {'p84':>6s} {'real chi2/p':>11s} "
          f"{'|corr|':>6s} {'res G':>6s} {'res jk':>6s}")
    for t, v in sorted(summ.items()):
        q = np.percentile(v["ratio"], [16, 50, 84])

        def wmean(key):
            if not v[key]:
                return None
            x, w = np.array(v[key]).T
            return float(np.sum(x * w) / np.sum(w))
        o = {"n_subblocks": len(v["chi2_real_per_p"]), "n_rows": len(v["ratio"]),
             "sigma_jk_over_gauss_p16_p50_p84": q.tolist()}
        for key in ("chi2_real_per_p", "offdiag_abs_corr", "chi2_gauss_per_p", "chi2_jk_per_p"):
            o[key] = wmean(key)
        by_type[t] = o
        f = lambda z: f"{z:6.2f}" if z is not None else "     -"  # noqa: E731
        print(f"{t:16s} {o['n_subblocks']:4d} {o['n_rows']:5d} {q[0]:8.2f} {q[1]:6.2f} {q[2]:6.2f} "
              f"{f(o['chi2_real_per_p']):>11s} {f(o['offdiag_abs_corr'])} {f(o['chi2_gauss_per_p'])} {f(o['chi2_jk_per_p'])}")
    s30_json = ROOT / "results/30_cov_shear_shot_it11.json"
    shear_shear = json.loads(s30_json.read_text())["by_pair"] if s30_json.exists() else None
    rec.data.update(model_dir=a.model_dir, mask_tag=a.tag, lmin=LMIN, p_max=P_MAX, n_jk=int(vec.n_jk),
                    by_type=by_type, skipped=dict(skipped), shear_x_shear_from_script_30=shear_shear,
                    subblocks=sub)
    print(rec.write(f"33_subblock_validation{a.out_tag}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("inputs", "run"))
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--layout", default="_s0")
    ap.add_argument("--tag", default="_cuts", help="mask 25_mask<tag>.npz")
    ap.add_argument("--noise", default=str(DATA / "shear_noise.json"))
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args(argv)
    {"inputs": cmd_inputs, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
