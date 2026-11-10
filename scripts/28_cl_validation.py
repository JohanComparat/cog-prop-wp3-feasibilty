"""Part D1: validation of the angular path (C_ell) of the model against its projected path (w_p) and the data.

    source ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh cpu
    PYTHONPATH=~/software/ggah_sens_study_wt_v170:$PYTHONPATH python scripts/28_cl_validation.py v1
    PYTHONPATH=... python scripts/28_cl_validation.py v2

For each slice-0 galaxy sample, at the 1.7.0 defaults of the one-shell layout (``_s0``):

``v1``  the 3D spectrum the joint build holds: the bias sqrt(P_gg/P_mm) on 0.02-0.08 h/Mpc in each
        zone of the sample, and w_p from the same P_gg (``ggah_mod.observables.real_space.wp``,
        the path ggah_cal validated) against the data w_p, as the ratio on 0.1-1, 1-10 and 10-30 Mpc.
``v2``  the projection: C_ell of the sample's auto-spectrum from the same zone spectra and
        photometric kernels, by an independent Limber integral (numpy) and by
        ``ggah_mod.observables.limber.c_ell``, against the rows the build wrote (script 22,
        iterate 0); the ratio build / independent per ell node.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ggah_sens_study  # noqa: E402,F401  (x64 first)
from wp3feas.model import NZ_FILE, _in_build, check_stack, public_config  # noqa: E402
from wp3feas.nz import build_name  # noqa: E402
from wp3feas.provenance import DATA, Record  # noqa: E402
from wp3feas.vector import VECTOR, load_vector, parse_tracer  # noqa: E402

LAYOUT = {"shells": ((0.1, 0.2),), "spec_mstar_min": ((13.0, 13.0),)}
D22 = DATA / "22_jacobian_s0"
PI_MAX = 100.0                     # Mpc, sum_stat's w_p
RP_RANGES = ((0.1, 1.0), (1.0, 10.0), (10.0, 30.0))   # Mpc
K_LIN = (0.02, 0.08)               # h/Mpc


def setup():
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import model as M

    stack = check_stack()
    cfg = public_config(**LAYOUT)
    fid = load_fiducial()
    J = M._Joint(fid, cfg, "closure")
    th = np.asarray(M.theta0(fid, cfg))
    meta = json.loads((D22 / "meta.json").read_text())
    pair_of = {}
    for i, (a, b) in enumerate(meta["pairs"]):
        pair_of[(a, b)] = pair_of[(b, a)] = i
    vec = load_vector()
    names = sorted({n.decode() for k in ("a", "b") for n in vec.index[k]} - {""})
    samples = {}
    for n in names:
        t = parse_tracer(n)
        if t["kind"] == "gal" and t["slice"] == 0 and t["colour"] != "all":
            samples[n] = _in_build(build_name(n), pair_of)
    return stack, cfg, fid, J, th, meta, pair_of, vec, samples, M


def zone_spectra(J, fid, M, th, name):
    """``{u: (z, k, P_gg, P_mm, chi, W)}`` on every zone of the support of ``name``."""
    import jax.numpy as jnp

    th = jnp.asarray(th)
    cosmo = J.cosmology_at(th)
    p = J.params_of(th, 0)
    sup = sorted(J.support[name])
    chis = J.zone_chi(cosmo, sup)
    out = {}
    for u in sup:
        field, _, _, W, _ = J.zone(th, u, maps={"m"})
        pgg = M.pk_sum(field, W[name], W[name], J.ov(name, name), fid.pk_options)
        pmm = M.pk_sum(field, W["m"], W["m"], "identical", fid.pk_options)
        wk, _ = J.photo_kernel(cosmo, p, name, u, chis)
        out[u] = (float(np.asarray(field.z).ravel()[0]), np.asarray(field.k), np.asarray(pgg).ravel(),
                  np.asarray(pmm).ravel(), np.asarray(chis[u]), np.asarray(wk), np.asarray(J.z_zone[u]))
    return out


def cmd_v1(a):
    from ggah_mod.observables.real_space import wp

    rec = Record(__file__)
    for p in (VECTOR, NZ_FILE, D22 / "meta.json"):
        rec.add_input(p)
    stack, cfg, fid, J, th, meta, pair_of, vec, samples, M = setup()
    h = float(fid.cosmology.h) if hasattr(fid.cosmology, "h") else 0.6774
    blocks = {(k[0], frozenset((k[1], k[2] or k[1]))): i for k, i in vec.blocks()}
    out = {}
    for tier, name in samples.items():
        zs = zone_spectra(J, fid, M, th, name)
        row = {"build_name": name, "zones": {}}
        for u, (z, k, pgg, pmm, *_ ) in zs.items():
            s = (k >= K_LIN[0]) & (k <= K_LIN[1])
            row["zones"][int(u)] = {"z": z, "b_3d": float(np.sqrt(np.mean(pgg[s] / pmm[s])))}
        # w_p from the zone holding the slice centre (the shell's own zone)
        u0 = min(zs, key=lambda u: abs(zs[u][0] - 0.15))
        z, k, pgg = zs[u0][:3]
        idx = blocks[("wp", frozenset((tier,)))]
        rp = vec.index["x"][idx]
        good = np.isfinite(vec.data[idx]) & (rp >= 0.1) & (rp <= 40)
        wm = np.asarray(wp(rp[good] * h, (k, pgg), pi_max=PI_MAX * h)) / h
        wd = vec.data[idx][good]
        row["wp_z"] = z
        row["wp"] = {"rp_mpc": rp[good].tolist(), "data": wd.tolist(), "model": wm.tolist()}
        for lo, hi in RP_RANGES:
            s = (rp[good] >= lo) & (rp[good] <= hi)
            row[f"wp_model_over_data_{lo:g}_{hi:g}"] = float(np.median(wm[s] / wd[s]))
        out[tier] = row
        zb = ", ".join(f"z {v['z']:.3f}: {v['b_3d']:.2f}" for v in row["zones"].values())
        print(f"{tier[:40]:40s} b_3d [{zb}]  w_p model/data "
              + " ".join(f"{row[f'wp_model_over_data_{lo:g}_{hi:g}']:.2f}" for lo, hi in RP_RANGES), flush=True)
    rec.data.update(stack=stack, layout="_s0", k_lin=list(K_LIN), pi_max_mpc=PI_MAX, h=h, samples=out)
    print(rec.write("28_cl_validation_v1"))


def limber_np(ell, z, k, pk, chi, w1, w2):
    lk, lp = np.log(k), np.log(np.clip(pk, 1e-300, None))
    out = []
    for l in ell:
        kq = (l + 0.5) / chi
        pq = np.exp(np.interp(np.log(kq), lk, lp))
        out.append(np.trapezoid(w1 * w2 / chi ** 2 * pq, chi))
    return np.array(out)


def cmd_v2(a):
    import jax.numpy as jnp
    from ggah_mod.observables.kernels import RadialKernel
    from ggah_mod.observables.limber import c_ell

    rec = Record(__file__)
    for p in (NZ_FILE, D22 / "meta.json"):
        rec.add_input(p)
    stack, cfg, fid, J, th, meta, pair_of, vec, samples, M = setup()
    ell = np.asarray(meta["ell"])
    n_disc = len(meta["discrete"])
    D = np.zeros(n_disc + len(meta["pairs"]) * ell.size)
    for b in meta["blocks"]:
        with np.load(D22 / f"{b}.npz") as z:
            D[z["rows"]] += z["D"]
    out = {}
    sel = (ell >= 30) & (ell <= 2048)
    for tier, name in samples.items():
        zs = zone_spectra(J, fid, M, th, name)
        ind = np.zeros(ell.size)
        lib = np.zeros(ell.size)
        per_zone = {}
        for u, (z, k, pgg, pmm, chi, w, zg) in zs.items():
            c1 = limber_np(ell, z, k, pgg, chi, w, w)
            ker = RadialKernel(jnp.asarray(zg), jnp.asarray(chi), jnp.asarray(w), "photo")
            c2 = np.asarray(c_ell(jnp.asarray(ell), jnp.asarray(k),
                                  jnp.broadcast_to(jnp.asarray(pgg), (chi.size, k.size)), ker, ker))
            ind += c1
            lib += c2
            per_zone[int(u)] = {"z": z, "kernel_integral": float(np.trapezoid(w, chi)),
                                "cl_independent": c1.tolist()}
        j = n_disc + pair_of[(name, name)] * ell.size
        bld = D[j:j + ell.size]
        r_b, r_l = bld / ind, lib / ind
        out[tier] = {"build_name": name, "ell": ell.tolist(), "build": bld.tolist(), "independent": ind.tolist(),
                     "ggah_mod_c_ell": lib.tolist(), "zones": per_zone,
                     "build_over_independent_median": float(np.median(r_b[sel])),
                     "c_ell_over_independent_median": float(np.median(r_l[sel])),
                     "sum_kernel_integral": float(sum(v["kernel_integral"] for v in per_zone.values()))}
        print(f"{tier[:40]:40s} build/independent {np.median(r_b[sel]):.3f} "
              f"[{r_b[sel].min():.3f}, {r_b[sel].max():.3f}]  c_ell/independent {np.median(r_l[sel]):.3f}  "
              f"sum of kernel integrals {out[tier]['sum_kernel_integral']:.3f} over {len(per_zone)} zones",
              flush=True)
    rec.data.update(stack=stack, layout="_s0", samples=out)
    print(rec.write("28_cl_validation_v2"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("v1", "v2"))
    a = ap.parse_args(argv)
    {"v1": cmd_v1, "v2": cmd_v2}[a.cmd](a)


if __name__ == "__main__":
    main()
