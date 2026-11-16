"""PLAN D4: the systematics of the galaxy x galaxy blocks of slice 0, beyond the photo-z width.

    python scripts/27_gal_systematics.py imaging [--nside 256]     # at CC (env ls11: healpy, fitsio)
    python scripts/27_gal_systematics.py rsd                        # laptop, env dev-full (pyccl)

imaging, per slice-0 galaxy sample of the tier (and its slice-1 counterpart of the same colour and
mass where one exists), on HEALPix maps of nside 256 (RING) over the pixels whose random count is at
least half the median:
  (c) imaging: the overdensity with and without WEIGHT_SYS; its linear regression on the 15
      sys_mapping templates (Gaia stars, E(B-V), galaxy depth, number of passes and PSF size in
      g r i z, W1 depth), standardised on the footprint; the pseudo-C_l / f_sky of the weighted
      field, of the unweighted one and of the weighted one cleaned of the regression, and the
      contamination C_sys = c^T (C^tt)^-1 c from the template cross-spectra c and template
      spectra C^tt, each with the shot noise subtracted;
  (f) shot noise: the analytic noise Omega_pix <sum w^2 / (alpha r)^2> against the spectrum of
      the half-difference of two random halves of the galaxies;
  (d) leakage between slices: the cross-correlation coefficient
      r_l = C_01 / sqrt(C_00 C_11) of the slice-0 and slice-1 samples (noise-subtracted autos),
      to be compared with the n(z) overlap of ``rsd``.
rsd, per slice-0 sample (pyccl, Planck 2018, linear bias b = 1 and 2):
  (e) C_l with redshift-space distortions and the exact (non-Limber) projection over the Limber
      C_l without them, with the measured n(z) and with the closure-corrected one;
  (d) the Limber cross-correlation coefficient of the slice-0 and slice-1 samples from their
      n(z) alone (the same P(k) and bias in both), to set against the measured r_l.
Outputs: results/27_gal_systematics_<cmd>.json.
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import Record  # noqa: E402

LS11 = Path("/sps/lsst/users/jcomparat01/ls11_rmagGAL/bgsr21s")
NZ_DIR = Path.home() / "data/legacysurvey/dr11/south/bgsr21s/results/tier_xcorr/nz"
EDGES = np.unique(np.geomspace(30, 700, 12).astype(int))
PAT = re.compile(r"LS11_(ALL|RED|BLUE)_(\d+\.\d+)_Mstar_(\d+\.\d+)_(\d\.\d\d)_z_(\d\.\d\d)_N_(\d+)")


def samples(root, z0="0.10"):
    out = {}
    for p in sorted(glob.glob(str(root / f"LS11_*_{z0}_z_*_DATA.fits"))):
        m = PAT.search(p)
        if m and int(m.group(6)) >= 1000:
            out[(m.group(1), m.group(2), m.group(3))] = p[: -len("_DATA.fits")]
    return out


def bin_cl(cl):
    ell = np.arange(cl.size)
    return np.array([cl[(ell >= lo) & (ell < hi)].mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def cmd_imaging(a):
    import fitsio
    import healpy as hp

    rec = Record(__file__)
    ns = a.nside
    npix = hp.nside2npix(ns)
    tdir = LS11 / "final/systematics" / f"{ns:04d}"
    tfiles = sorted(tdir.glob("*.fits"))
    T = np.array([hp.read_map(str(f)) for f in tfiles])
    tnames = [f.stem for f in tfiles]
    root = LS11 / "results/tier_xcorr/samples/gal"
    s0, s1 = samples(root, "0.10"), samples(root, "0.20")
    lmax = 3 * ns - 1
    ellc = np.sqrt(EDGES[:-1] * EDGES[1:])
    out = {}

    def maps(base, seed=0):
        d = fitsio.read(base + "_DATA.fits", columns=["RA", "DEC", "WEIGHT_SYS"])
        r = fitsio.read(base + "_RAND.fits", columns=["RA", "DEC"])
        pd = hp.ang2pix(ns, d["RA"], d["DEC"], lonlat=True)
        pr = hp.ang2pix(ns, r["RA"], r["DEC"], lonlat=True)
        w = d["WEIGHT_SYS"].astype(float)
        half = np.random.default_rng(seed).random(w.size) < 0.5
        return {"r": np.bincount(pr, minlength=npix).astype(float),
                "w": np.bincount(pd, w, minlength=npix), "w2": np.bincount(pd, w ** 2, minlength=npix),
                "u": np.bincount(pd, minlength=npix).astype(float),
                "wA": np.bincount(pd[half], w[half], minlength=npix),
                "wB": np.bincount(pd[~half], w[~half], minlength=npix)}

    def spectra(fields, mask, fsky):
        al = {k: hp.map2alm(v * mask, lmax=lmax) for k, v in fields.items()}

        def cl(x, y):
            return bin_cl(hp.alm2cl(al[x], al[y])) / fsky
        return cl

    for key, base in s0.items():
        name = "_".join(key)
        M = maps(base)
        med = np.median(M["r"][M["r"] > 0])
        foot = (M["r"] >= 0.5 * med) & np.all(np.isfinite(T) & (T > -1e20), axis=0)
        fsky = foot.mean()
        alw = M["w"][foot].sum() / M["r"][foot].sum()
        alu = M["u"][foot].sum() / M["r"][foot].sum()
        dw = np.where(foot, M["w"] / np.maximum(alw * M["r"], 1e-30) - 1, 0.0)
        du = np.where(foot, M["u"] / np.maximum(alu * M["r"], 1e-30) - 1, 0.0)
        dd = np.where(foot, 0.5 * (M["wA"] - M["wB"]) / np.maximum(alw * M["r"], 1e-30), 0.0)
        Ts = np.where(foot, (T - T[:, foot].mean(1)[:, None]) / T[:, foot].std(1)[:, None], 0.0)
        # (c) pixel regression, inverse-variance weighted by the expected count
        var = np.where(foot, M["w2"] / np.maximum((alw * M["r"]) ** 2, 1e-30), np.inf)
        var = np.where(foot & (M["w2"] == 0), 1.0 / np.maximum(alw * M["r"], 1e-30), var)
        iv = np.where(foot, 1.0 / var, 0.0)
        X = Ts[:, foot].T
        A = X.T @ (X * iv[foot, None])
        coef = np.linalg.solve(A, X.T @ (iv[foot] * dw[foot]))
        dc = dw.copy()
        dc[foot] -= X @ coef
        chi2_drop = float(coef @ A @ coef)
        # (f) analytic shot noise of the weighted field, on the pseudo-C_l normalised by f_sky
        pix = hp.nside2pixarea(ns)
        N_w = pix * float(np.mean(M["w2"][foot] / (alw * M["r"][foot]) ** 2))
        N_u = pix * float(np.mean(M["u"][foot] / (alu * M["r"][foot]) ** 2))
        fields = {"w": dw, "u": du, "c": dc, "d": dd, **{f"t{i}": Ts[i] for i in range(len(tnames))}}
        cl = spectra(fields, foot.astype(float), fsky)
        Cw, Cu, Cc, Cd = cl("w", "w"), cl("u", "u"), cl("c", "c"), cl("d", "d")
        ct = np.array([cl("w", f"t{i}") for i in range(len(tnames))])
        Ctt = np.array([[cl(f"t{i}", f"t{j}") for j in range(len(tnames))] for i in range(len(tnames))])
        Csys = np.array([ct[:, b] @ np.linalg.pinv(Ctt[:, :, b]) @ ct[:, b] for b in range(ellc.size)])
        Sw = Cw - N_w
        row = {"n_gal": int(M["u"].sum()), "fsky": float(fsky), "ell": ellc.tolist(),
               "cl_w_minus_noise": Sw.tolist(), "cl_u_minus_noise": (Cu - N_u).tolist(),
               "cl_clean_minus_noise": (Cc - N_w).tolist(), "cl_sys": Csys.tolist(),
               "ratio_unweighted_over_weighted": ((Cu - N_u) / Sw).tolist(),
               "ratio_clean_over_weighted": ((Cc - N_w) / Sw).tolist(),
               "sys_fraction": (Csys / Sw).tolist(),
               "noise_analytic": N_w, "noise_split": (4 * Cd).tolist(),  # the half-difference holds N/4
               "noise_split_over_analytic": (4 * Cd / N_w).tolist(),
               "noise_over_signal": (N_w / Sw).tolist(),
               "regression": {"coef": dict(zip(tnames, coef.tolist())), "chi2_drop": chi2_drop,
                              "n_pix": int(foot.sum())}}
        # (d) the slice-1 counterpart
        if key in s1:
            M1 = maps(s1[key], seed=1)
            f1 = foot & (M1["r"] >= 0.5 * np.median(M1["r"][M1["r"] > 0]))
            a1 = M1["w"][f1].sum() / M1["r"][f1].sum()
            a0 = M["w"][f1].sum() / M["r"][f1].sum()
            d0 = np.where(f1, M["w"] / np.maximum(a0 * M["r"], 1e-30) - 1, 0.0)
            d1 = np.where(f1, M1["w"] / np.maximum(a1 * M1["r"], 1e-30) - 1, 0.0)
            N0 = pix * float(np.mean(M["w2"][f1] / (a0 * M["r"][f1]) ** 2))
            N1 = pix * float(np.mean(M1["w2"][f1] / (a1 * M1["r"][f1]) ** 2))
            c2 = spectra({"0": d0, "1": d1}, f1.astype(float), f1.mean())
            C00, C11, C01 = c2("0", "0") - N0, c2("1", "1") - N1, c2("0", "1")
            row["cross_slice1"] = {"r_l": (C01 / np.sqrt(np.clip(C00 * C11, 1e-30, None))).tolist(),
                                   "c01": C01.tolist(), "c00": C00.tolist(), "c11": C11.tolist(),
                                   "sample1": Path(s1[key]).name}
        out[name] = row
        print(f"{name:24s} fsky {fsky:.3f}  unw/w {np.median(row['ratio_unweighted_over_weighted']):.3f}  "
              f"clean/w {np.median(row['ratio_clean_over_weighted']):.3f}  sys/C {np.median(row['sys_fraction']):.3f}  "
              f"Nsplit/N {np.median(row['noise_split_over_analytic']):.3f}  N/S {np.median(row['noise_over_signal']):.2f}"
              + (f"  r01 {np.median(row['cross_slice1']['r_l']):.3f}" if "cross_slice1" in row else ""), flush=True)
    rec.data.update(nside=ns, edges=EDGES.tolist(), templates=tnames, samples=out)
    print(rec.write("27_gal_systematics_imaging"))


def cmd_rsd(a):
    import pyccl as ccl

    from wp3feas.nz import closure_corrected

    rec = Record(__file__)
    cosmo = ccl.Cosmology(Omega_c=0.2607, Omega_b=0.0490, h=0.6766, sigma8=0.8102, n_s=0.9665)
    ell = np.unique(np.geomspace(30, 2000, 20).astype(int)).astype(float)
    out = {}

    def tracer(z, n, b, rsd):
        return ccl.NumberCountsTracer(cosmo, has_rsd=rsd, dndz=(z, n), bias=(z, np.full_like(z, b)))

    files = {Path(p).name: p for p in glob.glob(str(NZ_DIR / "LS11_*_nz.npz"))}
    for f, p in sorted(files.items()):
        m = PAT.search(f)
        if not m or m.group(4) != "0.10" or int(m.group(6)) < 1000:
            continue
        d = np.load(p)
        z = np.asarray(d["z"], float)
        sel = z > 1e-3
        row = {}
        for tag, n in (("measured", np.asarray(d["nz"], float)), ("closure", closure_corrected(d)[1])):
            for b in (1.0, 2.0):
                lim = ccl.angular_cl(cosmo, tracer(z[sel], n[sel], b, False), tracer(z[sel], n[sel], b, False), ell)
                ex = ccl.angular_cl(cosmo, tracer(z[sel], n[sel], b, True), tracer(z[sel], n[sel], b, True), ell,
                                    l_limber=-1, non_limber_integration_method="FKEM")
                row[f"{tag}_b{b:g}"] = (ex / lim).tolist()
        # (d) the slice-1 counterpart of the same colour and mass
        k1 = [g for g in files if g.startswith(f"LS11_{m.group(1)}_{m.group(2)}_Mstar_{m.group(3)}_0.20_z_0.30")]
        if k1:
            d1 = np.load(files[k1[0]])
            z1, n1 = np.asarray(d1["z"], float), np.asarray(d1["nz"], float)
            t0, t1 = tracer(z[sel], np.asarray(d["nz"], float)[sel], 1.0, False), tracer(z1[z1 > 1e-3], n1[z1 > 1e-3], 1.0, False)
            c01, c00, c11 = (ccl.angular_cl(cosmo, x, y, ell) for x, y in ((t0, t1), (t0, t0), (t1, t1)))
            row["r_l_nz"] = (c01 / np.sqrt(c00 * c11)).tolist()
            row["sample1"] = k1[0]
        name = "_".join(m.group(1, 2, 3))
        out[name] = row
        print(f"{name:24s} exact+RSD / Limber at l = {ell[0]:.0f}, {ell[3]:.0f}, {ell[8]:.0f}: "
              + ", ".join(f"{row['measured_b1'][i]:.3f}" for i in (0, 3, 8))
              + (f"   r01(n(z)) {np.median(row['r_l_nz']):.3f}" if "r_l_nz" in row else ""), flush=True)
    rec.data.update(ell=ell.tolist(), cosmology="Planck 2018 (Omega_c 0.2607, Omega_b 0.049, h 0.6766, sigma8 0.8102, n_s 0.9665)",
                    samples=out)
    print(rec.write("27_gal_systematics_rsd"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("imaging", "rsd"))
    ap.add_argument("--nside", type=int, default=256)
    a = ap.parse_args(argv)
    {"imaging": cmd_imaging, "rsd": cmd_rsd}[a.cmd](a)


if __name__ == "__main__":
    main()
