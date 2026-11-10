"""Part D0: an audit of the slice z 0.1-0.2 of public_2026, against the data themselves and the default model.

    python scripts/20_vector_audit.py data          # tests 1a-1c of PLAN.md, no model
    python scripts/20_vector_audit.py abundance     # test 4a: number densities, data and default model

``data`` reads only the vector:
1a, the cross-correlation coefficient r_ab = C_ab / sqrt(C_aa C_bb) of every pair of slice-0
    galaxy and AGN samples (r > 1 points to an auto-spectrum with too much shot noise removed,
    or a cross with too little);
1b, the linear-bias ratio of each sample to a reference sample, from the autos and from the
    crosses with every third field (galaxies, AGN, y, each shear bin): an auto that disagrees
    with all its crosses points to the auto (shot noise, systematics), not to the model;
1c, the nested samples: C(ALL) against the number-weighted sum of C(RED), C(BLUE) and
    C(RED, BLUE), all shot-noise subtracted.
Ratios are averaged over 30 <= ell <= 200 (``LIN``); their errors are jackknife errors of the
ratio of averages, replicate by replicate.

``abundance`` compares the number density of each slice-0 sample with the model's at the
defaults (the discrete rows of script 22's iterate 0, one-shell layout).
"""

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import DATA, Record  # noqa: E402
from wp3feas.vector import VECTOR, load_vector, parse_tracer  # noqa: E402

LIN = (30.0, 200.0)       # ell range of the linear-bias ratios
REF = "LS11_ALL_10.50_Mstar_11.00_0.10_z_0.20"
AREA_DEG2 = 16515.8       # LS11 bgsr21s deep/mbins footprint (sum_stat tier_xcorr/counts/galaxies.txt)
H0, OM = 67.74, 0.3089    # cosmology of the measurements (sum_stat docs, tier_ls11_data_vector)


class Spectra:
    """The C_ell blocks of the vector by unordered pair, restricted to ``LIN``."""

    def __init__(self, vec):
        self.v, self.by = vec, {}
        for (stat, a, b, survey, part), idx in vec.blocks():
            if stat not in ("cl", "cl_sh"):
                continue
            ell = vec.index["x"][idx]
            idx = idx[(ell >= LIN[0]) & (ell <= LIN[1])]
            if idx.size:
                self.by[frozenset((a, b or a))] = idx

    def get(self, a, b):
        """``(mean, jk means)`` of C_ab over ``LIN``, or None."""
        idx = self.by.get(frozenset((a, b)))
        if idx is None:
            return None
        d, jk = self.v.data[idx], self.v.jk[:, idx]
        if not (np.all(np.isfinite(d)) and np.all(np.isfinite(jk))):
            return None
        return d.mean(), jk.mean(1)


def jk_sigma(x):
    n = x.size
    return float(np.sqrt((n - 1) / n * np.sum((x - x.mean()) ** 2)))


def short(name):
    return re.sub(r"_N_\d+$", "", name)


def slice0(vec):
    names = {n.decode() for k in ("a", "b") for n in vec.index[k]} - {""}
    gal = sorted(n for n in names if parse_tracer(n)["slice"] == 0 and parse_tracer(n)["kind"] == "gal")
    agn = sorted(n for n in names if parse_tracer(n)["slice"] == 0 and parse_tracer(n)["kind"] == "agn")
    shear = sorted(n for n in names if parse_tracer(n)["kind"] == "shear")
    return gal, agn, shear


def cmd_data(a):
    rec = Record(__file__)
    rec.add_input(VECTOR)
    vec = load_vector()
    S = Spectra(vec)
    gal, agn, shear = slice0(vec)
    disc = gal + agn
    # 1a: cross-correlation coefficients
    r_out = {}
    for x, y in combinations(disc, 2):
        A, B, C = S.get(x, x), S.get(y, y), S.get(x, y)
        if None in (A, B, C) or A[0] <= 0 or B[0] <= 0:
            continue
        with np.errstate(invalid="ignore"):
            rj = C[1] / np.sqrt(A[1] * B[1])
        r, s = C[0] / np.sqrt(A[0] * B[0]), jk_sigma(rj)
        r_out[f"{short(x)} x {short(y)}"] = {"r": float(r), "sigma": s, "pull_above_1": float((r - 1) / s)}
    bad_r = {k: v for k, v in r_out.items() if v["pull_above_1"] > 2}
    print(f"1a: {len(r_out)} pairs, {len(bad_r)} with r > 1 beyond 2 sigma")
    for k, v in sorted(bad_r.items(), key=lambda kv: -kv[1]["pull_above_1"])[:15]:
        print(f"   r = {v['r']:.2f} +- {v['sigma']:.2f}  {k}")
    # 1b: bias ratio to the reference sample, auto against crosses
    ref = next(n for n in gal if n.startswith(REF))
    others = disc + ["y"] + shear
    b_out = {}
    Rr = S.get(ref, ref)
    for x in disc:
        if x == ref:
            continue
        Ax = S.get(x, x)
        if Ax is None or Ax[0] <= 0:
            continue
        auto, auto_j = np.sqrt(Ax[0] / Rr[0]), np.sqrt(Ax[1] / Rr[1])
        crosses = {}
        for c in others:
            if c in (x, ref):
                continue
            Xc, Rc = S.get(x, c), S.get(ref, c)
            if Xc is None or Rc is None:
                continue
            same_part = len(Xc[1]) == len(auto_j) and not c.startswith(("DES", "KIDS", "HSC"))
            rj = Xc[1] / Rc[1]
            val = Xc[0] / Rc[0]
            s = jk_sigma(rj - auto_j) if same_part else float(np.hypot(jk_sigma(rj), jk_sigma(auto_j)))
            crosses[short(c)] = {"ratio": float(val), "sigma_diff": s, "pull": float((auto - val) / s)}
        pulls = np.array([v["pull"] for v in crosses.values()])
        ratios = np.array([v["ratio"] for v in crosses.values()])
        b_out[short(x)] = {"auto": float(auto), "auto_sigma": jk_sigma(auto_j),
                           "cross_median": float(np.median(ratios)) if ratios.size else None,
                           "median_pull": float(np.median(pulls)) if pulls.size else None,
                           "n_cross": int(pulls.size), "crosses": crosses}
    print(f"1b: bias ratio to {REF}, from the auto and the median of the crosses (30 <= ell <= 200)")
    for k, v in b_out.items():
        if v["n_cross"]:
            print(f"   auto {v['auto']:.2f} +- {v['auto_sigma']:.2f}, crosses {v['cross_median']:.2f} "
                  f"(median pull {v['median_pull']:+.1f}, {v['n_cross']})  {k}")
    # 1c: nested samples, ALL = RED + BLUE
    n_out = {}
    for x in gal:
        p = parse_tracer(x)
        if p["colour"] != "all":
            continue
        stem = x.split("_N_")[0]
        red = next((n for n in gal if n.startswith(stem.replace("ALL", "RED"))), None)
        blue = next((n for n in gal if n.startswith(stem.replace("ALL", "BLUE"))), None)
        if not (red and blue):
            continue
        n_all, n_red = int(x.split("_N_")[1]), int(red.split("_N_")[1])
        fr = n_red / n_all
        fb = 1 - fr  # the BLUE names carry a truncated count; RED + BLUE = ALL
        parts = [S.get(red, red), S.get(blue, blue), S.get(red, blue), S.get(x, x)]
        if None in parts:
            continue
        pred = [fr ** 2 * parts[0][i] + fb ** 2 * parts[1][i] + 2 * fr * fb * parts[2][i] for i in (0, 1)]
        q, qj = parts[3][0] / pred[0], parts[3][1] / pred[1]
        n_out[short(x)] = {"f_red": fr, "all_over_sum": float(q), "sigma": jk_sigma(qj)}
        print(f"1c: C(ALL) / (f_r^2 C_RR + f_b^2 C_BB + 2 f_r f_b C_RB) = {q:.3f} +- {jk_sigma(qj):.3f}, "
              f"f_red {fr:.3f}  {short(x)}")
    rec.data.update(slice="0.10-0.20", ell_range=list(LIN), reference=REF, r_ab=r_out, bias_ratio=b_out,
                    nested=n_out, n_r_above_1=len(bad_r))
    print(rec.write("20_vector_audit_data"))


def comoving_volume(z0, z1, area_deg2):
    """Comoving volume [(Mpc/h)^3] of a flat LCDM shell over ``area_deg2``."""
    z = np.linspace(0, z1, 4001)
    ez = np.sqrt(OM * (1 + z) ** 3 + 1 - OM)
    chi = 2997.92458 * np.concatenate([[0], np.cumsum(0.5 * (1 / ez[1:] + 1 / ez[:-1]) * np.diff(z))])
    c0 = np.interp(z0, z, chi)
    return area_deg2 * (np.pi / 180) ** 2 / 3 * (chi[-1] ** 3 - c0 ** 3)


def cmd_abundance(a):
    rec = Record(__file__)
    d22 = DATA / "22_jacobian_s0"
    rec.add_input(d22 / "meta.json")
    meta = json.loads((d22 / "meta.json").read_text())
    n_disc = len(meta["discrete"])
    model = np.zeros(n_disc)
    for b in meta["blocks"]:
        with np.load(d22 / f"{b}.npz") as z:
            sel = z["rows"] < n_disc
            model[z["rows"][sel]] += z["D"][sel]
    model = dict(zip(meta["discrete"], model))
    vec = load_vector()
    gal, agn, _ = slice0(vec)
    V = comoving_volume(0.1, 0.2, AREA_DEG2)
    out = {}
    for x in gal + agn:
        p = parse_tracer(x)
        n = int(x.split("_N_")[1])
        if p["kind"] == "gal":
            if p["colour"] == "all":
                continue
            if p["colour"] == "blue":  # truncated count in the name
                al = next(g for g in gal if g.startswith(x.split("_N_")[0].replace("BLUE", "ALL")))
                rd = next(g for g in gal if g.startswith(x.split("_N_")[0].replace("BLUE", "RED")))
                n = int(al.split("_N_")[1]) - int(rd.split("_N_")[1])
            key = f"g{p['colour'][0]}{p['mstar'][0]:g}p"
        else:
            key = f"a{p['lx_min']:g}"
        if key not in model:
            continue
        out[short(x)] = {"model_key": key, "N": n, "nbar_data": n / V, "nbar_model": float(model[key]),
                         "data_over_model": float(n / V / model[key])}
        print(f"{short(x):45s} {key:8s} data {n / V:.3e}  model {model[key]:.3e}  ratio {n / V / model[key]:.2f}")
    rec.data.update(volume_mpch3=V, area_deg2=AREA_DEG2, units="(Mpc/h)^-3", abundance=out)
    print(rec.write("20_vector_audit_abundance"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("data", "abundance"))
    a = ap.parse_args(argv)
    {"data": cmd_data, "abundance": cmd_abundance}[a.cmd](a)


if __name__ == "__main__":
    main()
