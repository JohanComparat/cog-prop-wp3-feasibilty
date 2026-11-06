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


def tracer_tables():
    """``{build name: (z, n)}`` of the galaxy and AGN samples, from tier_xcorr/nz/*_nz.npz."""
    out = {}
    for f in sorted((TIER / "nz").glob("*_nz.npz")):
        name = build_name(f.name[: -len("_nz.npz")])
        if name is not None:
            with np.load(f) as d:
                out[name] = _norm(d["z"], d["nz"])
    return out


def write_nz_file(path):
    """Write every table to ``path`` (an .npz) and return the names it holds."""
    tab = {**tracer_tables(), **shear_tables()}
    np.savez(path, **{f"{n}.{k}": v for n, (z, nz) in tab.items() for k, v in (("z", z), ("nz", nz))})
    return sorted(tab)
