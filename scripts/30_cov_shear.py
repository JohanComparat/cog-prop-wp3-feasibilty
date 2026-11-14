"""Part E: the covariance of the shear x shear blocks of public_2026, jackknife against Gaussian.

    python scripts/30_cov_shear.py [--model-dir DIR --layout _s0]

Per block (survey pair, bin pair) of cl_sh shear x shear:
  (a) the B-mode null (``null``, ``null_jk`` of the vector file) against zero with the jackknife
      covariance (Hartlap): chi2/n and PTE, which tests the jackknife and the B-mode systematics;
  (b) the jackknife sigma of each bandpower over the Gaussian (Knox) one,
      sigma^2 = [(C_ii + N_i)(C_jj + N_j) + (C_ij + N_ij)^2] / ((2l + 1) Delta l f_sky),
      N_i the white shape noise of bin i on the full sky (sum_stat tier_shear_cl.py stores the
      coupled bias N <w^2> as ``noise`` of the shear map file, with the mask weight w;
      WP3FEAS_DATA/shear_noise.json), N_ij = 0 for i != j, f_sky the area of the bin
      (the smaller of the two for a cross-survey pair), C the model of --model-dir when given,
      else the measured C clipped at zero;
  (c) with --model-dir, chi2 of data - model with the jackknife and with the Gaussian diagonal.
Summaries by survey pair go to results/30_cov_shear<out_tag>.json.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import DATA, Record  # noqa: E402
from wp3feas.vector import VECTOR, hartlap, jk_cov, load_vector  # noqa: E402

FULL_SKY_DEG2 = 41252.96


def band_widths(x):
    """Delta l of bandpowers centred on x, edges at the geometric midpoints."""
    lx = np.log(x)
    e = np.exp(np.r_[lx[0] - (lx[1] - lx[0]) / 2, (lx[1:] + lx[:-1]) / 2, lx[-1] + (lx[-1] - lx[-2]) / 2])
    return np.diff(e)


def white_noise(noise, b):
    """N of bin b on the full sky: sum_stat stores the noise bias of the coupled pseudo-C_l
    (``noise`` = N <w^2>, w the mask weight), so N = noise / <w^2>."""
    d = noise.get(b, {})
    return d["noise"] / d["w2_mean"] if d.get("w2_mean") else 0.0


def model_rows(model_dir, layout, vec):
    """The model of an iterate of scripts/26_map_fit.py on the vector rows (NaN where not modelled)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s26", ROOT / "scripts/26_map_fit.py")
    s26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s26)
    s26.S22.set_layout(layout)
    _, D, _ = s26.S22._assemble(Path(model_dir), jac=False)
    meta = json.loads((s26.S22.OUT / "meta.json").read_text())
    from wp3feas.model import projection
    P, rows = projection(vec, {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]],
                               "ell": meta["ell"]})
    m = np.full(vec.data.size, np.nan)
    m[rows] = P @ D
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--layout", default="_s0")
    ap.add_argument("--noise", default=str(DATA / "shear_noise.json"))
    ap.add_argument("--lmin", type=float, default=30.0)
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args(argv)
    rec = Record(__file__)
    rec.add_input(VECTOR)
    rec.add_input(a.noise)
    noise = json.loads(Path(a.noise).read_text())
    vec = load_vector()
    ix = vec.index
    stat = ix["stat"].astype(str)
    A_, B_ = ix["a"].astype(str), ix["b"].astype(str)
    sv = ix["survey"].astype(str)
    model = model_rows(a.model_dir, a.layout, vec) if a.model_dir else None
    h5 = h5py.File(VECTOR, "r")

    def auto(name, ell, idx_of):
        """C_ii on ell: the model if given, else the measurement, clipped at zero."""
        k = idx_of.get((name, name))
        if k is None:
            return np.zeros_like(ell)
        c = model[k] if model is not None and np.all(np.isfinite(model[k])) else vec.data[k]
        return np.clip(np.interp(ell, ix["x"][k], c), 0, None)

    sel_sh = np.flatnonzero((stat == "cl_sh") & (np.char.find(A_, "_bin") >= 0) & (np.char.find(B_, "_bin") >= 0))
    idx_of = {}
    for k in sel_sh:
        idx_of.setdefault((A_[k], B_[k]), []).append(k)
    idx_of = {p: np.array(v)[np.argsort(ix["x"][v])] for p, v in idx_of.items()}
    blocks, summ = {}, collections.defaultdict(lambda: collections.defaultdict(list))
    for (x, y), k in sorted(idx_of.items()):
        ell = ix["x"][k]
        use = ell >= a.lmin
        k = k[use]
        ell = ell[use]
        n = k.size
        h = hartlap(vec.n_jk, n)
        C = jk_cov(vec.jk[:, k])
        iC = h * np.linalg.pinv(C)
        pair = f"{x.split('_')[0]}x{y.split('_')[0]}"
        row = {"n": int(n), "pair": pair}
        # (a) the B-mode null
        g = h5[f"blocks/cl_sh/{x}__{y}__{sv[k[0]]}"]
        if "null" in g:
            xb = g["x"][()] if "x" in g else ell
            nb = np.interp(ell, xb, g["null"][()])
            sel = np.searchsorted(xb, ell)
            jb = g["null_jk"][()][:, np.clip(sel, 0, xb.size - 1)]
            iB = hartlap(jb.shape[0], n) * np.linalg.pinv(jk_cov(jb))
            c2 = float(nb @ iB @ nb)
            row.update(null_chi2=c2, null_pte=float(stats.chi2.sf(c2, n)))
        # (b) jackknife sigma over the Gaussian one
        ni, nj = white_noise(noise, x), white_noise(noise, y)
        area = min(noise.get(x, {}).get("area", np.nan), noise.get(y, {}).get("area", np.nan))
        fsky = area / FULL_SKY_DEG2
        cii, cjj = auto(x, ell, idx_of), auto(y, ell, idx_of)
        cij = (model[k] if model is not None and np.all(np.isfinite(model[k])) else vec.data[k])
        nij = ni if x == y else 0.0
        var_g = ((cii + ni) * (cjj + nj) + (cij + nij) ** 2) / ((2 * ell + 1) * band_widths(ix["x"][idx_of[(x, y)]])[use] * fsky)
        ratio = np.sqrt(np.diag(C)) / np.sqrt(var_g)
        row.update(sigma_jk_over_gauss_median=float(np.median(ratio)), sigma_ratio=ratio.tolist(), fsky=fsky)
        # (c) chi2 of the fit with each covariance
        if model is not None and np.all(np.isfinite(model[k])):
            r = vec.data[k] - model[k]
            row.update(chi2_jk=float(r @ iC @ r), chi2_gauss_diag=float(np.sum(r ** 2 / var_g)),
                       chi2_jk_diag=float(np.sum(r ** 2 / np.diag(C))))
        blocks[f"{x}__{y}"] = row
        for key in ("null_chi2", "chi2_jk", "chi2_gauss_diag", "chi2_jk_diag", "sigma_jk_over_gauss_median", "n"):
            if key in row:
                summ[pair][key].append(row[key])
    out = {}
    print(f"{'pair':12s} {'blocks':>6s} {'rows':>5s} {'B null chi2/n':>13s} {'sig jk/G':>8s} "
          f"{'chi2/n jk':>9s} {'jk diag':>8s} {'G diag':>8s}")
    for pair, v in sorted(summ.items()):
        nn = sum(v["n"])
        o = {"n_blocks": len(v["n"]), "n_rows": nn,
             "null_chi2_per_row": sum(v["null_chi2"]) / nn if v["null_chi2"] else None,
             "sigma_jk_over_gauss_median": float(np.median(v["sigma_jk_over_gauss_median"]))}
        for key in ("chi2_jk", "chi2_jk_diag", "chi2_gauss_diag"):
            if v[key]:
                o[f"{key}_per_row"] = sum(v[key]) / nn
        out[pair] = o
        f = lambda z: f"{z:8.2f}" if z is not None else "       -"  # noqa: E731
        print(f"{pair:12s} {o['n_blocks']:6d} {nn:5d} {f(o['null_chi2_per_row']):>13s} {f(o['sigma_jk_over_gauss_median'])} "
              f"{f(o.get('chi2_jk_per_row')):>9s} {f(o.get('chi2_jk_diag_per_row'))} {f(o.get('chi2_gauss_diag_per_row'))}")
    rec.data.update(model_dir=a.model_dir, lmin=a.lmin, by_pair=out, blocks=blocks)
    print(rec.write(f"30_cov_shear{a.out_tag}"))


if __name__ == "__main__":
    main()
