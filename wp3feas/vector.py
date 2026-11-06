"""Reader of the public vector public_2026 (sum_stat tier_ls11_v0.h5) and its jackknife covariance.

The file stores one flat vector: an index table (stat, a, b, survey, part, x), the measured
values `data` and the jackknife replicates `jk` (n_jk, n_rows).
x is the multipole for cl and cl_sh, the angle in deg for wtheta, r_p in Mpc/h for wp and
R in Mpc/h for dsigma (units as written by sum_stat/scripts/tier_assemble_vector.py).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

VECTOR = Path.home() / "data/legacysurvey/dr11/south/bgsr21s/results/tier_xcorr/vector/tier_ls11_v0.h5"
N_SHEAR = {"DES": 4, "KIDS": 5, "HSC": 4}
SLICES = {(0.1, 0.2): 0, (0.2, 0.3): 1, (0.3, 0.6): 2}  # slice index used in this study


def parse_tracer(name):
    """Kind, colour, stellar-mass or luminosity edge, redshift range and band of a tier tracer name."""
    out = {"name": name, "kind": None, "z": None, "slice": None}
    m = re.match(r"LS11_(RED|BLUE|ALL)_(\d+\.\d+)_Mstar_(\d+\.\d+)_(\d\.\d+)_z_(\d\.\d+)", name)
    if m:
        z = (float(m.group(4)), float(m.group(5)))
        out.update(kind="gal", colour=m.group(1).lower(), mstar=(float(m.group(2)), float(m.group(3))), z=z)
    elif m := re.match(r"ERO1_AGN_LX_(\d+\.\d+)_(\d\.\d+)_z_(\d\.\d+)", name):
        out.update(kind="agn", lx_min=float(m.group(1)), z=(float(m.group(2)), float(m.group(3))))
    elif name.startswith("ERO1_"):
        out.update(kind="xsrc")  # point sources or AGN without redshift cut
    elif m := re.match(r"EVT_(\d{4})_(\d{4})(_PSMASK)?", name):
        out.update(kind="xray", band_ev=(int(m.group(1)), int(m.group(2))), psmask=bool(m.group(3)))
    elif name in ("y", "y_std"):
        out.update(kind="y", cib_deproj=name == "y")  # sum_stat: y is the CIB-deprojected NILC map
    elif m := re.match(r"(DES|KIDS|HSC)_bin(\d)", name):
        out.update(kind="shear", survey=m.group(1), zbin=int(m.group(2)))
    if out["z"] is not None:
        out["slice"] = SLICES.get(tuple(round(v, 2) for v in out["z"]))
    return out


def slot_2031(name):
    """The 2031 tracers a tier tracer stands for (from sum_stat docs/scripts/plot_tier_ls11_matrix.py)."""
    t = parse_tracer(name)
    if t["kind"] == "gal" and t["colour"] != "all":
        shell = int(round((t["z"][0] - 0.1) / 0.1))
        return [f"g{t['colour'][0]}{t['mstar'][0]:g}{p}@{shell}" for p in ("", "p")]
    if t["kind"] == "agn":
        if t["lx_min"] == 44.0 and t["z"][0] == 0.3:
            return [f"a44@{s}" for s in (2, 3, 4)]
        if t["lx_min"] in (42.5, 43.0) and t["z"][1] <= 0.3:
            return [f"a{int(t['lx_min'])}@{int(round((t['z'][0] - 0.1) / 0.1))}"]
        return []
    if t["kind"] == "xray":
        return [f"x{t['band_ev'][0]:04d}"] if 500 <= t["band_ev"][0] < 2000 else []
    if t["kind"] == "y":
        return ["y"]
    if t["kind"] == "shear":
        return [f"ks{int(round(4 * t['zbin'] / (N_SHEAR[t['survey']] - 1)))}"]
    return []


@dataclass
class Vector:
    """The flat vector: index table, data (n,), jackknife replicates (n_jk, n)."""

    index: np.ndarray
    data: np.ndarray
    jk: np.ndarray
    attrs: dict

    @property
    def n_jk(self):
        return self.jk.shape[0]

    def blocks(self):
        """Ordered list of (stat, a, b, survey, part) keys with their row indices."""
        key = np.array([b"|".join(r) for r in zip(*(self.index[k] for k in ("stat", "a", "b", "survey", "part")))])
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        out = []
        for j in np.argsort(first):
            rows = np.flatnonzero(inv == j)
            r = self.index[rows[0]]
            out.append(((r["stat"].decode(), r["a"].decode(), r["b"].decode(), r["survey"].decode(), r["part"].decode()), rows))
        return out

    def select(self, mask):
        """Sub-vector on a boolean or integer row selection."""
        return Vector(self.index[mask], self.data[mask], self.jk[:, mask], self.attrs)


def load_vector(path=VECTOR):
    """Read the whole vector of `path` into memory (about 80 MB of replicates)."""
    with h5py.File(path, "r") as f:
        return Vector(f["index"][:], f["data"][:], f["jk"][:], dict(f.attrs))


def jk_cov(jk):
    """Jackknife covariance (n-1)/n D^T D of replicates (n_jk, p), pairwise over the finite replicates.

    Same estimator as cov() in sum_stat/scripts/tier_assemble_vector.py; NaN where fewer than two remain.
    """
    fin = np.isfinite(jk)
    d = np.where(fin, jk - np.nanmean(np.where(fin, jk, np.nan), axis=0), 0.0)
    n = fin.astype(float).T @ fin.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (n - 1) / n * (d.T @ d)
    c[n < 2] = np.nan
    return c


def hartlap(n_jk, p):
    """Factor (n - p - 2)/(n - 1) on the inverse covariance (Hartlap et al. 2007, Eq. 17); 0 when p >= n - 2."""
    return max(n_jk - p - 2, 0) / (n_jk - 1)
