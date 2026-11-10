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


class Linear:
    """Linear theory in h units: Eisenstein & Hu (1998) no-wiggle P(k), sigma8-normalised,
    growth by the integral of Heath (1977); xi, w_p and Limber C_ell of a tracer with b = 1.
    sigma8 cancels in every ratio this script reports."""

    def __init__(self, h=H0 / 100, om=OM, ob=0.0486, ns=0.9667, s8=0.81):
        self.h, self.om, self.ob, self.ns = h, om, ob, ns
        self.k = np.logspace(-4, 2, 3000)
        p = self.k ** ns * self._t(self.k) ** 2
        x = self.k * 8.0
        w = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
        s2 = np.trapezoid(p * w ** 2 * self.k ** 3, np.log(self.k)) / (2 * np.pi ** 2)
        self.p0 = p * s8 ** 2 / s2
        self.zg = np.linspace(0, 1.5, 1501)
        ez = self.E(self.zg)
        a = 1 / (1 + self.zg)
        integ = np.array([np.trapezoid(1 / (np.linspace(1e-4, ai, 400) * self.E(1 / np.linspace(1e-4, ai, 400) - 1)) ** 3,
                                       np.linspace(1e-4, ai, 400)) for ai in a])
        d = 2.5 * om * ez * integ
        self.D = d / d[0]
        self.chi = 2997.92458 * np.concatenate([[0], np.cumsum(0.5 * (1 / ez[1:] + 1 / ez[:-1]) * np.diff(self.zg))])

    def E(self, z):
        return np.sqrt(self.om * (1 + z) ** 3 + 1 - self.om)

    def _t(self, k):
        om, ob, h = self.om, self.ob, self.h
        omh2, f = om * h * h, ob / om
        s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * (ob * h * h) ** 0.75)
        al = 1 - 0.328 * np.log(431 * omh2) * f + 0.38 * np.log(22.3 * omh2) * f ** 2
        g = om * h * (al + (1 - al) / (1 + (0.43 * k * h * s) ** 4))
        q = k * (2.7255 / 2.7) ** 2 / g
        l0, c0 = np.log(2 * np.e + 1.8 * q), 14.2 + 731 / (1 + 62.5 * q)
        return l0 / (l0 + c0 * q * q)

    def pk(self, k, z):
        return np.interp(np.log(k), np.log(self.k), self.p0) * np.interp(z, self.zg, self.D) ** 2

    def wp(self, rp, z, pi_max):
        """w_p [Mpc/h] at rp [Mpc/h] for b = 1, from xi by a damped spherical Bessel transform."""
        r = np.logspace(-1, np.log10(np.hypot(rp.max(), pi_max)) + 0.05, 600)
        pkz = self.pk(self.k, z) * np.exp(-(self.k / 10) ** 2)
        xi = np.array([np.trapezoid(self.k ** 3 * pkz * np.sinc(self.k * ri / np.pi), np.log(self.k))
                       for ri in r]) / (2 * np.pi ** 2)
        pi = np.linspace(0, pi_max, 800)
        return np.array([2 * np.trapezoid(np.interp(np.hypot(x, pi), r, xi), pi) for x in rp])

    def cl(self, ell, za, na, zb=None, nb=None):
        """Limber C_ell of two tracers with b = 1 and normalised n(z)."""
        z = self.zg[1:]
        n1 = np.interp(z, za, na, left=0, right=0)
        n2 = n1 if zb is None else np.interp(z, zb, nb, left=0, right=0)
        chi = self.chi[1:]
        hz = self.E(z) / 2997.92458
        out = []
        for l in np.atleast_1d(ell):
            out.append(np.trapezoid(hz * n1 * n2 / chi ** 2 * self.pk((l + 0.5) / chi, z), z))
        return np.array(out)


RP_LIN = (10.0, 30.0)     # Mpc (the vector's unit, h = 0.6774) for the w_p bias
PI_MAX = 100.0            # Mpc, sum_stat's w_p


def cmd_wp(a):
    """Linear bias of every slice-0 galaxy and AGN sample from w_p (10-30 Mpc), from the data
    C_ell (30 <= ell <= 200, Limber with the n(z) the model uses) and from the default model's
    C_ell; the AGN from their crosses with the reference galaxy sample as well."""
    from wp3feas.model import NZ_FILE, _in_build
    from wp3feas.nz import build_name

    rec = Record(__file__)
    d22 = DATA / "22_jacobian_s0"
    for p in (VECTOR, NZ_FILE, d22 / "meta.json"):
        rec.add_input(p)
    vec = load_vector()
    L = Linear()
    h = L.h
    zeff = 0.15
    gal, agn, _ = slice0(vec)
    blocks = {(k[0], frozenset((k[1], k[2] or k[1]))): i for k, i in vec.blocks()}
    nz = np.load(NZ_FILE)
    meta = json.loads((d22 / "meta.json").read_text())
    n_disc, ell_m = len(meta["discrete"]), np.asarray(meta["ell"])
    pair_of = {}
    for i, (p, q) in enumerate(meta["pairs"]):
        pair_of[(p, q)] = pair_of[(q, p)] = i
    D = np.zeros(n_disc + len(meta["pairs"]) * ell_m.size)
    for b in meta["blocks"]:
        with np.load(d22 / f"{b}.npz") as z:
            D[z["rows"]] += z["D"]

    def b2_wp(x, y):
        idx = blocks.get(("wp", frozenset((x, y))))
        if idx is None:
            return None
        rp = vec.index["x"][idx]
        s = idx[(rp >= RP_LIN[0]) & (rp <= RP_LIN[1])]
        t = L.wp(vec.index["x"][s] * h, zeff, PI_MAX * h) / h  # back to Mpc
        num = lambda v: float(np.sum(v * t) / np.sum(t * t))
        return num(vec.data[s]), np.array([num(j) for j in vec.jk[:, s]])

    def b2_cl(x, y):
        idx = blocks.get(("cl", frozenset((x, y))))
        bx, by = build_name(x), build_name(y)
        if idx is None or bx is None or by is None:
            return None
        ell = vec.index["x"][idx]
        s = idx[(ell >= LIN[0]) & (ell <= LIN[1])]
        ls = vec.index["x"][s]
        t = L.cl(ls, nz[f"{bx}.z"], nz[f"{bx}.nz"], nz[f"{by}.z"], nz[f"{by}.nz"])
        num = lambda v: float(np.sum(v * t) / np.sum(t * t))
        A, B = _in_build(bx, pair_of), _in_build(by, pair_of)
        mod = None
        if (A, B) in pair_of:
            j = n_disc + pair_of[(A, B)] * ell_m.size
            cm = np.interp(np.log(ls), np.log(ell_m), D[j:j + ell_m.size])
            mod = num(cm)
        return num(vec.data[s]), np.array([num(j) for j in vec.jk[:, s]]), mod

    def sq(v, j):
        with np.errstate(invalid="ignore"):
            return float(np.sign(v) * np.sqrt(abs(v))), jk_sigma(np.sign(j) * np.sqrt(np.abs(j)))

    ref = next(n for n in gal if n.startswith(REF))
    out = {}
    print(f"{'sample':45s} {'b_wp':>12s} {'b_cl data':>12s} {'b_cl model':>10s}  cl/wp  model/data")
    for x in gal + agn:
        row = {}
        w, c = b2_wp(x, x), b2_cl(x, x)
        if w:
            row["b_wp"], row["b_wp_sigma"] = sq(*w)
        if c:
            row["b_cl"], row["b_cl_sigma"] = sq(c[0], c[1])
            if c[2] is not None:
                row["b_cl_model"] = sq(c[2], np.array([c[2]]))[0]
        if x in agn:  # through the cross with the reference galaxies: b_A = b_gA^2 / b_g
            wx, wr, cx, cr = b2_wp(x, ref), b2_wp(ref, ref), b2_cl(x, ref), b2_cl(ref, ref)
            if wx and wr:
                row["b_wp_cross"] = wx[0] / np.sqrt(wr[0])
                row["b_wp_cross_sigma"] = jk_sigma(wx[1] / np.sqrt(np.abs(wr[1])))
            if cx and cr:
                row["b_cl_cross"] = cx[0] / np.sqrt(cr[0])
                row["b_cl_cross_sigma"] = jk_sigma(cx[1] / np.sqrt(np.abs(cr[1])))
                if cx[2] is not None and cr[2] is not None:
                    row["b_cl_cross_model"] = cx[2] / np.sqrt(cr[2])
        out[short(x)] = row
        f = lambda k: f"{row[k]:6.2f}+-{row[k + '_sigma']:4.2f}" if k in row else " " * 12
        g = row.get("b_cl_model")
        print(f"{short(x):45s} {f('b_wp')} {f('b_cl')} {g if g is None else round(g, 2)!s:>10s}  "
              f"{row['b_cl'] / row['b_wp'] if 'b_cl' in row and 'b_wp' in row else float('nan'):5.2f}  "
              f"{(g / row['b_cl']) ** 2 if g and 'b_cl' in row else float('nan'):5.2f}")
        if x in agn:
            print(f"{'   via the cross with ' + short(ref)[5:]:45s} {f('b_wp_cross')} {f('b_cl_cross')} "
                  f"{row.get('b_cl_cross_model', float('nan')):10.2f}")
    rec.data.update(z_eff=zeff, rp_range_mpc=list(RP_LIN), pi_max_mpc=PI_MAX, ell_range=list(LIN),
                    pk="Eisenstein-Hu no-wiggle, sigma8 0.81 (cancels in the ratios)", bias=out)
    print(rec.write("20_vector_audit_wp"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("data", "abundance", "wp"))
    a = ap.parse_args(argv)
    {"data": cmd_data, "abundance": cmd_abundance, "wp": cmd_wp}[a.cmd](a)


if __name__ == "__main__":
    main()
