r"""Igo et al. (2026) :math:`\log_{10}p(\log\lambda_{\rm SAR}|M_*, z)` as an external data term of
the slice-0 MAP fit (``scripts/26_map_fit.py``, ``WP3FEAS_EXTRA=igo26``).

The data are ggah_cal's :func:`~ggah_cal.datasets.erosita_c030.load_igo26` (diagonal errors, the
band half-width plus a 0.15-dex floor); the model is ggah_cal's
:func:`~ggah_cal.predict.agn_cross_igo.igo26_eval`, evaluated with the joint model's own AGN
sector (``_Joint.agn_all``, Zu & Mandelbaum 2015 galaxies, Duras et al. 2020 bolometric
correction) and its parameters at ``theta`` (``params_of(theta)["pa"]``, ``["po"]``).

The stellar-mass weights of Igo's bins (``igo26_prep``) are computed at the galaxy parameters of
the point where the term is built (numpy, not traced): the Jacobian holds the AGN and obscuration
columns, and the galaxy columns only through the black-hole mass, not through the weights.

``WP3FEAS_IGO_Z`` keeps a comma-separated subset of the redshift bins (0: 0.030-0.064,
1: 0.064-0.098, 2: 0.098-0.164); all three by default.
"""

import os
import types

import numpy as np

FILE = "igo26.npz"


def enabled():
    return os.environ.get("WP3FEAS_EXTRA", "") == "igo26"


def data():
    """``(x, y, sigma)`` of the Igo rows kept."""
    from ggah_cal.datasets.erosita_c030 import load_igo26

    dv = load_igo26()
    x, y, s = np.asarray(dv.x), np.asarray(dv.y), np.sqrt(np.diag(dv.cov))
    zs = os.environ.get("WP3FEAS_IGO_Z")
    if zs:
        keep = np.isin(x[:, 0].astype(int), [int(v) for v in zs.split(",")])
        x, y, s = x[keep], y[keep], s[keep]
    return x, y, s


class Igo26Model:
    """``theta -> log10 p`` at the Igo rows, with the joint model of ``layout``."""

    def __init__(self, cfg, theta):
        import jax.numpy as jnp
        from ggah_cal.predict.agn_cross_igo import igo26_prep
        from ggah_mod.sectors.galaxies import galaxy_defaults
        from ggah_sens_study.fiducial import load_fiducial
        from ggah_sens_study.joint import model as M

        fid = load_fiducial()
        self.J = M._Joint(fid, cfg, "closure")
        self.x, self.y, self.sigma = data()
        p = self.J.params_of(jnp.asarray(theta))
        gp = dict(galaxy_defaults("zumandelbaum15"))
        gp.update({k: float(v) for k, v in p["gp13"].items()})
        self.pred = types.SimpleNamespace(agn=self.J.agn_all, galaxy_params=gp, fiducial=fid,
                                          cosmology=fid.cosmology, catalogue_nh="comparat19",
                                          catalogue_scatter=None)
        self.prep = igo26_prep(self.pred)

    def __call__(self, theta):
        from ggah_cal.predict.agn_cross_igo import igo26_eval

        p = self.J.params_of(theta)
        return igo26_eval(self.pred, p["pa"], p["po"], self.prep, self.x)

    def value_and_jacobian(self, theta):
        import jax
        import jax.numpy as jnp

        th = jnp.asarray(theta)
        m = np.asarray(jax.jit(self.__call__)(th))
        jac = np.asarray(jax.jit(jax.jacfwd(self.__call__))(th))
        return m, jac


def write(path, model, theta, names):
    m, jac = model.value_and_jacobian(theta)
    np.savez(path, m=m, J=jac, d=model.y, sigma=model.sigma, x=model.x, names=np.array(names),
             theta=np.asarray(theta))
    return m, jac


def terms(itdir):
    """``(r, J, iC)`` of the Igo rows at the iterate in ``itdir`` (its ``igo26.npz``)."""
    with np.load(itdir / FILE) as z:
        r = z["d"] - z["m"]
        return r, z["J"], np.diag(1.0 / z["sigma"] ** 2)


def chi2(itdir):
    r, _, iC = terms(itdir)
    return float(r @ iC @ r), int(r.size)
