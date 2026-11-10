"""D1 V1c: w_p of the spectroscopic members alone against w_p of all members (BEST_Z).

The tier w_p of sum_stat uses BEST_Z (spectroscopic where one exists, the photo-z otherwise) and
applies no photo-z dilution correction; the ggah_cal defaults were fitted on LS10 w_p that were
corrected. This script measures the dilution directly, in the region where the spectroscopic
completeness is high (nside-64 pixels whose slice-0 members are more than F_MIN spectroscopic):

    A   all members, BEST_Z, randoms of the tier sample (z drawn from BEST_Z)
    B   spectroscopic members, weight WEIGHT_SYS / f_spec(pixel), randoms with z drawn from them

A / B is the dilution of the tier w_p; A over the tier's full-sky w_p checks that the region is
representative. Also: the photo-z errors (BEST_Z_ERR) of the photo-z members.

Runs in the sum_stat venv at CC (Corrfunc), one sample per array task:

    sbatch --array=0-11 cc/wpspec.sbatch
    python scripts/29_wp_spec.py --list
    python scripts/29_wp_spec.py run --task 3
    python scripts/29_wp_spec.py collect
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wp3feas.provenance import DATA, Record  # noqa: E402

LS11 = Path(os.environ.get("LS11_ROOT", Path.home() / "data/legacysurvey/dr11/south/bgsr21s"))
TIER = LS11 / "results/tier_xcorr"
OUT = DATA / "29_wp_spec"
COSMO = FlatLambdaCDM(H0=67.74, Om0=0.3089)
RP_EDGES = np.logspace(np.log10(0.01), np.log10(60.0), 31)
RP_EDGES = RP_EDGES[RP_EDGES <= 30.0 * 1.0001]          # Mpc, the tier bins up to 30 Mpc
PI_MAX = 100.0
Z0, Z1 = 0.10, 0.20
MASS = [(10.0, 10.5), (10.5, 11.0), (11.0, 11.5), (11.5, 11.75)]
CLASSES = ("ALL", "RED", "BLUE")
NSIDE_F, F_MIN, NSIDE_JK = 64, 0.8, 16
RAND_RATIO = 5.0


def read(path, cols):
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        return {c: np.asarray(d[c]).astype(np.asarray(d[c]).dtype.newbyteorder("=")) for c in cols}


def mbins():
    out = {}
    for f in (LS11 / "deep/mbins").glob("*_DATA.fits"):
        sid = f.name.replace("_DATA.fits", "")
        p = sid.split("_")
        out[(float(p[3]), float(p[5]))] = sid
    return out


def parents(m):
    """Slice-0 members of the mass cell m, all columns needed, with their colour class."""
    from sum_stat.selection import two_class
    bins = mbins()
    keep = [k for k in bins if k[0] >= m[0] - 1e-6 and k[1] <= m[1] + 1e-6]
    parts = []
    for k in sorted(keep):
        sid = bins[k]
        d = read(LS11 / "deep/mbins" / f"{sid}_DATA.fits", ["RA", "DEC", "BEST_Z", "BEST_Z_ERR", "Z_SOURCE"])
        c = read(LS11 / "deep/mbins" / f"{sid}_COLOUR.fits", ["G_MAG", "Z_MAG"])
        w = read(LS11 / "results/sysmap_deep_mbins" / f"{sid}_NSIDE0064_WEIGHTS.fits", ["WEIGHT_SYS"])
        s = (d["BEST_Z"] >= Z0) & (d["BEST_Z"] < Z1)
        t = {k2: v[s] for k2, v in d.items()}
        t["W"] = w["WEIGHT_SYS"][s].astype(float)
        t["GZ"] = (c["G_MAG"] - c["Z_MAG"])[s]
        parts.append(t)
    t = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    t["CLASS"] = two_class(t["GZ"], t["BEST_Z"])
    return t


def spec_map():
    """Spectroscopic fraction of all slice-0 members per nside-64 pixel (cached)."""
    f = OUT / "spec_fraction.npz"
    if f.exists():
        return np.load(f)["frac"]
    n = np.zeros(hp.nside2npix(NSIDE_F))
    ns = np.zeros_like(n)
    for m in MASS:
        t = parents(m)
        p = hp.ang2pix(NSIDE_F, t["RA"], t["DEC"], lonlat=True)
        n += np.bincount(p, minlength=n.size)
        ns += np.bincount(p[t["Z_SOURCE"] == 1], minlength=n.size)
    frac = np.where(n > 20, ns / np.maximum(n, 1), 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(f, frac=frac, n=n, ns=ns)
    return frac


def tasks():
    return [(c, m) for m in MASS for c in CLASSES]


def tier_name(c, m, kind):
    pat = f"LS11_{c}_{m[0]:.2f}_Mstar_{m[1]:.2f}_{Z0:.2f}_z_{Z1:.2f}_N_*_{kind}.fits"
    hits = sorted((TIER / "samples/gal").glob(pat))
    return hits[0] if hits else None


def cmd_list(a):
    for i, (c, m) in enumerate(tasks()):
        print(i, c, m, tier_name(c, m, "RAND"))


def cmd_run(a):
    from sum_stat.catalogue import GalaxyCatalogue
    from sum_stat.twopcf.projected import wp_patches
    c, m = tasks()[a.task]
    rng = np.random.default_rng(a.task)
    frac = spec_map()
    good = frac > F_MIN
    t = parents(m)
    if c != "ALL":
        t = {k: v[t["CLASS"] == c] for k, v in t.items()}
    pf = hp.ang2pix(NSIDE_F, t["RA"], t["DEC"], lonlat=True)
    spec = t["Z_SOURCE"] == 1
    err = t["BEST_Z_ERR"][~spec]
    info = {"class": c, "mass": m, "n_all": int(spec.size), "f_spec": float(spec.mean()),
            "photo_err_pct_16_50_84": np.percentile(err[np.isfinite(err)], [16, 50, 84]).tolist()}
    inr = good[pf]
    t = {k: v[inr] for k, v in t.items()}
    spec = spec[inr]
    pf = pf[inr]
    r = read(tier_name(c, m, "RAND"), ["RA", "DEC", "Z"])
    rin = good[hp.ang2pix(NSIDE_F, r["RA"], r["DEC"], lonlat=True)]
    r = {k: v[rin] for k, v in r.items()}
    nr = min(r["RA"].size, int(RAND_RATIO * t["RA"].size))
    j = rng.choice(r["RA"].size, nr, replace=False)
    r = {k: v[j] for k, v in r.items()}
    # jackknife: nside-16 pixels of the region, relabelled 0..n-1
    pix_d = hp.ang2pix(NSIDE_JK, t["RA"], t["DEC"], lonlat=True)
    pix_r = hp.ang2pix(NSIDE_JK, r["RA"], r["DEC"], lonlat=True)
    u = np.unique(np.concatenate([pix_d, pix_r]))
    lab_d, lab_r = np.searchsorted(u, pix_d), np.searchsorted(u, pix_r)
    info.update(n_region=int(spec.size), n_spec_region=int(spec.sum()),
                f_spec_region=float(spec.mean()), n_rand=int(nr), n_patch=int(u.size),
                area_deg2=float(good.sum() * hp.nside2pixarea(NSIDE_F, degrees=True)))
    log = lambda s: print(f"{time.strftime('%H:%M:%S')} {c} {m}: {s}", flush=True)  # noqa: E731
    out = {}
    # A: all members, BEST_Z
    g = GalaxyCatalogue(ra=t["RA"].astype(float), dec=t["DEC"].astype(float),
                        redshift=t["BEST_Z"].astype(float), weight=t["W"])
    rc = GalaxyCatalogue(ra=r["RA"].astype(float), dec=r["DEC"].astype(float),
                         redshift=r["Z"].astype(float), weight=np.ones(nr))
    res = wp_patches(g, rc, RP_EDGES, PI_MAX, COSMO, patch_gal=lab_d, patch_rand=lab_r, n_threads=a.threads)
    out["A"] = (res.wp(), res.jackknife_subsamples())
    log("A done")
    # B: spectroscopic members, weight / f_spec(pixel), randoms with z from them
    ws = t["W"][spec] / frac[pf[spec]]
    zs = t["BEST_Z"][spec].astype(float)
    g = GalaxyCatalogue(ra=t["RA"][spec].astype(float), dec=t["DEC"][spec].astype(float), redshift=zs, weight=ws)
    rz = rng.choice(zs, nr, p=ws / ws.sum())
    rc = GalaxyCatalogue(ra=r["RA"].astype(float), dec=r["DEC"].astype(float), redshift=rz, weight=np.ones(nr))
    res = wp_patches(g, rc, RP_EDGES, PI_MAX, COSMO, patch_gal=lab_d[spec], patch_rand=lab_r, n_threads=a.threads)
    out["B"] = (res.wp(), res.jackknife_subsamples())
    log("B done")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"task{a.task:02d}.npz", rp_edges=RP_EDGES, wp_A=out["A"][0], jk_A=out["A"][1],
             wp_B=out["B"][0], jk_B=out["B"][1], info=json.dumps(info))


def tier_wp(c, m):
    """The tier's full-sky w_p of this sample (auto), if measured."""
    d = TIER / "measurements/wp"
    s = tier_name(c, m, "DATA")
    if s is None:
        return None
    s = s.name.replace("_DATA.fits", "")
    f = d / f"{s}__{s}_wp.h5"
    if not f.exists():
        return None
    with h5py.File(f) as h:
        return h["rp_edges"][:], h["wp"][:]


def cmd_collect(a):
    rec = Record(__file__)
    rec.add_input(LS11 / "deep/mbins")
    rec.add_input(TIER / "samples/gal")
    rows = {}
    for i, (c, m) in enumerate(tasks()):
        f = OUT / f"task{i:02d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        info = json.loads(str(z["info"]))
        rp = np.sqrt(z["rp_edges"][1:] * z["rp_edges"][:-1])
        jA, jB = z["jk_A"], z["jk_B"]
        n = jA.shape[0]
        ratio = z["wp_A"] / z["wp_B"]
        rjk = jA / jB
        sig = np.sqrt((n - 1) / n * ((rjk - rjk.mean(0)) ** 2).sum(0))
        row = dict(info, rp_mpc=rp.tolist(), wp_A=z["wp_A"].tolist(), wp_B=z["wp_B"].tolist(),
                   ratio_A_over_B=ratio.tolist(), ratio_sigma=sig.tolist())
        for lo, hi in ((1, 10), (10, 30)):
            s = (rp >= lo) & (rp <= hi)
            # mean ratio in the range, with its jackknife error
            mr = (jA[:, s].sum(1) / jB[:, s].sum(1))
            row[f"ratio_{lo}_{hi}"] = float(z["wp_A"][s].sum() / z["wp_B"][s].sum())
            row[f"ratio_{lo}_{hi}_sigma"] = float(np.sqrt((n - 1) / n * ((mr - mr.mean()) ** 2).sum()))
        tw = tier_wp(c, m)
        if tw is not None and np.allclose(tw[0][:len(z["rp_edges"])], z["rp_edges"]):
            k = len(rp)
            full = tw[1][:k]
            s = (rp >= 1) & (rp <= 30)
            row["region_over_fullsky_1_30"] = float(z["wp_A"][s].sum() / full[s].sum())
        rows[f"{c}_{m[0]:.2f}_{m[1]:.2f}"] = row
        print(f"{c:4s} {m}: f_spec {info['f_spec']:.2f} (region {info['f_spec_region']:.2f}), "
              f"A/B 1-10 {row['ratio_1_10']:.3f}+-{row['ratio_1_10_sigma']:.3f}, "
              f"10-30 {row['ratio_10_30']:.3f}+-{row['ratio_10_30_sigma']:.3f}, "
              f"region/full {row.get('region_over_fullsky_1_30', float('nan')):.3f}, "
              f"photo err {info['photo_err_pct_16_50_84'][1]:.4f}")
    rec.data = {"f_min": F_MIN, "nside_f": NSIDE_F, "nside_jk": NSIDE_JK, "pi_max_mpc": PI_MAX,
                "samples": rows}
    print(rec.write())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["list", "map", "run", "collect"])
    p.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = p.parse_args()
    {"list": cmd_list, "map": lambda a: print(spec_map().__gt__(F_MIN).sum(), "pixels"),
     "run": cmd_run, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
