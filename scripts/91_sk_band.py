r"""The matter power suppression :math:`S(k, z) = P_{\rm mm}/P_{\rm mm}^{\rm DMO}` of the fitted
slice 0.1-0.2 (script 26, one layout and chain), with the band of the Laplace covariance.

    sbatch --export=ALL,<the chain's WP3FEAS_*> cc/cpu.sbatch scripts/91_sk_band.py cov --it 18   # at CC
    source ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh
    PYTHONPATH=~/software/ggah_sens_study_wt_v170:$PYTHONPATH python scripts/91_sk_band.py jac --it 18
    PYTHONPATH=... python scripts/91_sk_band.py band --it 18

``jac`` evaluates S of the build's matter field (the closure, as in the fit) at the iterate's
point and at the defaults, at the redshifts ``--z``, and its Jacobian in the parameters
(forward mode), into ``WP3FEAS_DATA/91_sk_band<layout><tag>_it<it>.npz``.
``cov`` (at CC, where the iterate's Jacobian is) writes the Laplace covariance of script 26's
``summary`` to the iterate's ``laplace_cov.npz``.
``band`` reads it and propagates it to S:
:math:`\sigma_S^2 = G C G^{\rm T}` on the free parameters, cosmology fixed.
It writes ``results/91_sk_band<layout><tag>_it<it>.json``.

The band is first order around the point and conditional on the fixed cosmology; the
covariance is the block-diagonal jackknife of script 26, so the band is a lower limit.
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import ggah_sens_study  # noqa: E402,F401  (x64 first)
from wp3feas.model import check_stack, public_config  # noqa: E402
from wp3feas.provenance import DATA, Record  # noqa: E402

S22 = importlib.import_module("22_fisher_public")


def tag_of(a):
    return f"{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', '')}_it{a.it}"


def it_dir(a):
    return DATA / "cc" / f"26_map{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', '')}" / f"it{a.it}"


def cmd_jac(a):
    import jax
    import jax.numpy as jnp
    from ggah_mod.spectra.pk import pk_cross
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import model as M
    from ggah_sens_study.joint.model import Z_GAS_SWITCH

    t0 = time.time()
    stack = check_stack()
    fid = load_fiducial()
    cfg = public_config(**S22.LAYOUTS[a.layout])
    J = M._Joint(fid, cfg, "closure")
    names = list(M.names(cfg))
    th0 = np.asarray(M.theta0(fid, cfg))
    point = json.loads((it_dir(a) / "theta.json").read_text())
    assert point["names"] == names, "the iterate's parameters differ from the build's"
    x = np.asarray(point["theta"])
    opts = fid.pk_options

    def S(theta, z):
        p = J.params_of(theta)
        cosmo = J.cosmology_at(theta)
        field = J.los_field(cosmo, z)
        dp = p["dp"] if z <= Z_GAS_SWITCH else J.dp_far(p, cosmo)
        wm, _ = J.matter_weights(field, cosmo, p, dp)
        w0, _ = J.matter_weights(field, cosmo, p, dp, matter="dmo")
        return (pk_cross(field, wm, wm, overlap="identical", options=opts).total
                / pk_cross(field, w0, w0, overlap="identical", options=opts).total)

    k = np.asarray(J.los_field(J.cosmology_at(jnp.asarray(x)), 0.0).k)
    out = {"names": np.array(names), "theta": x, "theta0": th0, "k": k, "z": np.array(a.z)}
    s_pt, s_def, jac = [], [], []
    for z in a.z:
        f = jax.jit(lambda t, z=z: S(t, z))
        s_pt.append(np.asarray(f(jnp.asarray(x))))
        s_def.append(np.asarray(f(jnp.asarray(th0))))
        jvp = jax.jit(lambda t, v, z=z: jax.jvp(lambda u: S(u, z), (t,), (v,))[1])
        cols = [np.asarray(jvp(jnp.asarray(x), jnp.zeros(x.size).at[j].set(1.0))) for j in range(x.size)]
        jac.append(np.column_stack(cols))
        print(f"z = {z}: S(k=1) {np.interp(0.0, np.log(k), s_pt[-1]):.4f}, "
              f"{time.time() - t0:.0f} s", flush=True)
        jax.clear_caches()
    out.update(S=np.array(s_pt), S_default=np.array(s_def), G=np.array(jac))
    path = DATA / f"91_sk_band{tag_of(a)}.npz"
    np.savez(path, **out)
    (DATA / f"91_sk_band{tag_of(a)}.stack.json").write_text(json.dumps(stack))
    print(path, f"in {time.time() - t0:.0f} s")


def cmd_cov(a):
    """At CC, with the chain's environment (WP3FEAS_MAP_TAG, _VECTOR, _MASK, _SHOT, _AGN_BAND):
    the full Laplace covariance of script 26's ``summary`` at the iterate, into its
    ``laplace_cov.npz`` (fetched to ``WP3FEAS_DATA/cc/`` for ``band``)."""
    S26 = importlib.import_module("26_map_fit")
    meta, Jd, r, blocks, x = S26.point(a.it, None)
    names = meta["names"]
    th0 = np.asarray(meta["theta0"])
    sp = S26.prior_sigma(names)
    A = np.zeros((len(names),) * 2)
    for sel, iC in blocks:
        A += Jd[sel].T @ iC @ Jd[sel]
    Pr = np.where(np.isfinite(sp), 1.0 / sp ** 2, 0.0)
    f = np.flatnonzero(np.array([n not in S26.FIXED for n in names]) & (np.diag(A) + Pr > 0))
    C = np.linalg.pinv(A[np.ix_(f, f)] + np.diag(Pr[f]))
    chi2 = float(sum(r[s] @ iC @ r[s] for s, iC in blocks))
    out = S26.it_dir(a.it) / "laplace_cov.npz"
    np.savez(out, names=np.array(names), theta=np.asarray(x), theta0=th0, free=f, C=C, chi2=chi2,
             n_rows=int(sum(s.size for s, _ in blocks)))
    print(out, len(f), "free parameters; chi2", chi2)


def cmd_band(a):
    rec = Record(__file__)
    jpath = DATA / f"91_sk_band{tag_of(a)}.npz"
    cpath = it_dir(a) / "laplace_cov.npz"
    J = np.load(jpath)
    L = np.load(cpath)
    assert list(J["names"]) == list(L["names"]) and np.allclose(J["theta"], L["theta"])
    f, C = L["free"], L["C"]
    names = list(J["names"])
    k, zs = J["k"], J["z"]
    res = {}
    for iz, z in enumerate(zs):
        G = J["G"][iz][:, f]
        sig = np.sqrt(np.clip(np.einsum("ki,ij,kj->k", G, C, G), 0, None))
        # the parameters that carry the variance at k = 1 and 10 h/Mpc
        top = {}
        for kk in (1.0, 10.0):
            i = int(np.argmin(np.abs(np.log(k / kk))))
            g = G[i]
            var = g * (C @ g)  # each parameter's share of g C g
            order = np.argsort(-np.abs(var))[:6]
            top[str(kk)] = {names[f[j]]: float(var[j] / max(sig[i] ** 2, 1e-300)) for j in order}
        res[f"{z:g}"] = {"S": J["S"][iz].tolist(), "S_default": J["S_default"][iz].tolist(),
                         "sigma": sig.tolist(), "share": top}
        for kk in (0.3, 1.0, 3.0, 10.0):
            i = int(np.argmin(np.abs(np.log(k / kk))))
            print(f"z {z:g} k {k[i]:6.2f}: S {J['S'][iz][i]:.4f} +- {sig[i]:.4f} "
                  f"(default {J['S_default'][iz][i]:.4f})")
    rec.inputs += [str(jpath), str(cpath)]
    rec.data.update(it=a.it, layout=a.layout, map_tag=os.environ.get("WP3FEAS_MAP_TAG", ""),
                    chi2=float(L["chi2"]), n_rows=int(L["n_rows"]), n_free=int(f.size),
                    k=k.tolist(), z=zs.tolist(), cosmology="fixed", band=res)
    print(rec.write(f"91_sk_band{tag_of(a)}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("cov", "jac", "band"))
    ap.add_argument("--it", type=int, required=True)
    ap.add_argument("--layout", default="_s0", choices=list(S22.LAYOUTS))
    ap.add_argument("--z", type=float, nargs="+", default=[0.0, 0.15])
    a = ap.parse_args(argv)
    S22.set_layout(a.layout)
    {"cov": cmd_cov, "jac": cmd_jac, "band": cmd_band}[a.cmd](a)


if __name__ == "__main__":
    main()
