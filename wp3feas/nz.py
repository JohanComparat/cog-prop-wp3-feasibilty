"""The measured n(z) of public_2026 as the nz_file of ggah_sens_study's JointConfig.

Each tier tracer takes its build name (``gr10.5p@1``, ``a43@0``, ``ks{i}``) and its table
``<name>.z``, ``<name>.nz`` (normalised to unit integral over z).
The source bins are numbered DES 0-3, KiDS 4-8, HSC 9-12 (:data:`SHEAR_ORDER`).
"""

from pathlib import Path

import numpy as np

from .vector import parse_tracer

TIER = Path.home() / "data/legacysurvey/dr11/south/bgsr21s/results/tier_xcorr"
DATA = Path.home() / "data"
SHEAR_ORDER = [("DES", i) for i in range(4)] + [("KIDS", i) for i in range(5)] + [("HSC", i) for i in range(4)]
KIDS_NZ = ("K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2"
           "_SOMcols_Fid_blindC_TOMO{}_Nz.asc")


def build_name(name):
    """The ggah_sens_study tracer a tier tracer is modelled by, or None (no model counterpart)."""
    t = parse_tracer(name)
    if t["kind"] == "gal" and t["colour"] != "all" and t["slice"] is not None:
        return f"g{t['colour'][0]}{t['mstar'][0]:g}p@{t['slice']}"
    if t["kind"] == "agn" and t["slice"] is not None:
        return f"a{t['lx_min']:g}@{t['slice']}"
    if t["kind"] == "xray" and not t["psmask"] and 500 <= t["band_ev"][0] < 2000:
        return f"x{t['band_ev'][0]:04d}"
    if t["kind"] == "y" and t["cib_deproj"]:
        return "y"
    if t["kind"] == "shear":
        return f"ks{SHEAR_ORDER.index((t['survey'], t['zbin']))}"
    return None


def _norm(z, n):
    n = np.clip(np.asarray(n, float), 0.0, None)
    return np.asarray(z, float), n / np.trapezoid(n, z)


def shear_tables():
    """``{ks{i}: (z, n)}`` of the 13 source bins, from the surveys' released n(z)."""
    from astropy.table import Table

    des = Table.read(DATA / "DES/des_y3.hdf5", path="redshift")
    hsc = Table.read(DATA / "HSC/nz.fits")
    kz = np.genfromtxt(DATA / "KIDS/DR4" / KIDS_NZ.format(1))[:, 0] + 0.025
    out = {}
    for i, (s, b) in enumerate(SHEAR_ORDER):
        if s == "DES":
            z, n = des["z"], des["n"][:, b]
        elif s == "HSC":
            z, n = hsc["Z_MID"], hsc[f"BIN{b + 1}"]
        else:
            z, n = kz, np.genfromtxt(DATA / "KIDS/DR4" / KIDS_NZ.format(b + 1))[:, 1]
        out[f"ks{i}"] = _norm(z, n)
    return out


def closure_corrected(d, n_min=1000):
    """The n(z) of a galaxy table with its photo-z part shifted and stretched by the closure test
    of sum_stat ``tier_nz.py``: the estimated n(z) of the spectroscopic analogues of the photo-z
    members has mean ``closure_mean`` and width ``closure_width``, their true redshifts
    ``true_mean`` and ``true_width``; the photo-z part moves by the difference of the means and
    scales about its own mean by the ratio of the widths.  Tables without a closure test, or
    with fewer than ``n_min`` analogues, are returned as they are.  The true redshifts are
    truncated at the parents' edges, so the stretch is an upper bound on the correction."""
    z, nz = np.asarray(d["z"], float), np.asarray(d["nz"], float)
    if "closure_width" not in d.files or int(d["n_closure"]) < n_min:
        return z, nz
    ph = np.asarray(d["nz_phot"], float)
    w = np.trapezoid(ph, z)
    if w <= 0:
        return z, nz
    m = np.trapezoid(z * ph, z) / w
    s = float(d["true_width"]) / float(d["closure_width"])
    dz = float(d["true_mean"]) - float(d["closure_mean"])
    ph2 = np.interp((z - m - dz) / s + m, z, ph, left=0.0, right=0.0) / s
    # kept on the support of the measured table, so that the build's blocks stay the same
    ph2 = np.where(nz > 0, np.maximum(ph2, 1e-9 * ph2.max()), 0.0)
    ph2 *= w / np.trapezoid(ph2, z)
    return z, np.asarray(d["nz_spec"], float) + ph2


def tracer_tables(closure=False):
    """``{build name: (z, n)}`` of the galaxy and AGN samples, from tier_xcorr/nz/*_nz.npz; with
    ``closure`` the galaxy tables carry the closure correction (:func:`closure_corrected`)."""
    out = {}
    for f in sorted((TIER / "nz").glob("*_nz.npz")):
        name = build_name(f.name[: -len("_nz.npz")])
        if name is not None:
            with np.load(f) as d:
                out[name] = _norm(*closure_corrected(d)) if closure else _norm(d["z"], d["nz"])
    return out


def write_nz_file(path, closure=False):
    """Write every table to ``path`` (an .npz) and return the names it holds."""
    tab = {**tracer_tables(closure), **shear_tables()}
    np.savez(path, **{f"{n}.{k}": v for n, (z, nz) in tab.items() for k, v in (("z", z), ("nz", nz))})
    return sorted(tab)
