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


NZ_SUMMARY = DATA / "tier_nz_summary.json"   # f_spec and the widths of each sample's n(z) (sum_stat tier_nz.py)
TOPHAT_WIDTH = 0.1 / np.sqrt(12)              # standard deviation of a flat slice of width 0.1


def xi_of(k, pk, r, k_damp=20.0):
    """xi(r) [Mpc/h] of P(k) by a damped spherical Bessel transform."""
    w = k ** 3 * pk * np.exp(-(k / k_damp) ** 2)
    return np.array([np.trapezoid(w * np.sinc(k * ri / np.pi), np.log(k)) for ri in r]) / (2 * np.pi ** 2)


def wp_diluted(rp, k, pk, pi_max, sigmas, weights, dpi=0.5):
    """w_p [Mpc/h] at rp [Mpc/h]: xi(rp, pi) convolved along the line of sight with a Gaussian of
    width sigma [Mpc/h] for each pair class (weights), then integrated over |pi_obs| < pi_max."""
    smax = max(sigmas)
    ext = pi_max + 6 * smax + 10
    pi = np.arange(-ext, ext + dpi, dpi)
    r = np.logspace(-2, np.log10(np.hypot(rp.max(), ext)) + 0.01, 800)
    xr = xi_of(k, pk, r)
    inside = np.abs(pi) <= pi_max
    out = np.zeros(rp.size)
    for i, x in enumerate(rp):
        f = np.interp(np.hypot(x, pi), r, xr)
        for s, w in zip(sigmas, weights):
            if s > 0:
                g = np.exp(-0.5 * (np.arange(-6 * s, 6 * s + dpi, dpi) / s) ** 2)
                fs = np.convolve(f, g / g.sum(), mode="same")
            else:
                fs = f
            out[i] += w * np.trapezoid(fs[inside], pi[inside])
    return out


def cmd_v1b(a):
    """w_p of the model with the photo-z scatter of the members: pairs spec-spec, spec-photo and
    photo-photo, scattered along the line of sight by 0, sigma_chi and sqrt(2) sigma_chi."""
    rec = Record(__file__)
    for p in (VECTOR, NZ_SUMMARY, D22 / "meta.json"):
        rec.add_input(p)
    stack, cfg, fid, J, th, meta, pair_of, vec, samples, M = setup()
    nzs = json.loads(NZ_SUMMARY.read_text())
    h = 0.6774
    om = 0.3089
    blocks = {(k[0], frozenset((k[1], k[2] or k[1]))): i for k, i in vec.blocks()}
    out = {}
    print(f"{'sample':40s} f_spec sigma_z sigma_chi  model/data w_p (0.1-1, 1-10, 10-30 Mpc): undiluted | diluted")
    for tier, name in samples.items():
        s = nzs[tier]
        sz = float(np.sqrt(max(s["width_phot"] ** 2 - TOPHAT_WIDTH ** 2, 0.0)))
        zc = 0.15
        sig = 2997.92458 * sz / np.sqrt(om * (1 + zc) ** 3 + 1 - om)   # Mpc/h
        fs = s["f_spec"]
        zs = zone_spectra(J, fid, M, th, name)
        u0 = min(zs, key=lambda u: abs(zs[u][0] - zc))
        z, k, pgg = zs[u0][:3]
        idx = blocks[("wp", frozenset((tier,)))]
        rp = vec.index["x"][idx]
        good = np.isfinite(vec.data[idx]) & (rp >= 0.1) & (rp <= 40)
        x = rp[good] * h
        w0 = wp_diluted(x, k, pgg, PI_MAX * h, [0.0], [1.0]) / h
        w1 = wp_diluted(x, k, pgg, PI_MAX * h, [0.0, sig, np.sqrt(2) * sig],
                        [fs ** 2, 2 * fs * (1 - fs), (1 - fs) ** 2]) / h
        wd = vec.data[idx][good]
        row = {"f_spec": fs, "sigma_z": sz, "sigma_chi_mpch": sig, "rp_mpc": rp[good].tolist(),
               "data": wd.tolist(), "model": w0.tolist(), "model_diluted": w1.tolist()}
        txt = []
        for tag, wm in (("undiluted", w0), ("diluted", w1)):
            vals = []
            for lo, hi in RP_RANGES:
                q = (rp[good] >= lo) & (rp[good] <= hi)
                row[f"{tag}_over_data_{lo:g}_{hi:g}"] = float(np.median(wm[q] / wd[q]))
                vals.append(f"{row[f'{tag}_over_data_{lo:g}_{hi:g}']:.2f}")
            txt.append(" ".join(vals))
        out[tier] = row
        print(f"{tier[:40]:40s} {fs:5.2f} {sz:7.4f} {sig:8.1f}   {txt[0]} | {txt[1]}", flush=True)
    rec.data.update(stack=stack, pi_max_mpc=PI_MAX, h=h, tophat_width=TOPHAT_WIDTH, samples=out,
                    method="Gaussian line-of-sight scatter per pair class; sigma_z from the true-z width of the photo-z members")
    print(rec.write("28_cl_validation_v1b"))


def cmd_v3(a):
    """w(theta) of the model, from the build's C_ell (ggah_mod.observables.transforms.cl_to_wtheta),
    against the data w(theta) of each slice-0 galaxy auto."""
    import jax.numpy as jnp
    from ggah_mod.observables.transforms import cl_to_wtheta

    rec = Record(__file__)
    for p in (VECTOR, D22 / "meta.json"):
        rec.add_input(p)
    stack, cfg, fid, J, th, meta, pair_of, vec, samples, M = setup()
    ell = np.asarray(meta["ell"])
    n_disc = len(meta["discrete"])
    D = np.zeros(n_disc + len(meta["pairs"]) * ell.size)
    for b in meta["blocks"]:
        with np.load(D22 / f"{b}.npz") as z:
            D[z["rows"]] += z["D"]
    blocks = {(k[0], frozenset((k[1], k[2] or k[1]))): i for k, i in vec.blocks()}
    lf = np.unique(np.geomspace(2, 20000, 4000).round())
    out = {}
    for tier, name in samples.items():
        j = n_disc + pair_of[(name, name)] * ell.size
        cl = np.exp(np.interp(np.log(lf), np.log(ell), np.log(np.clip(D[j:j + ell.size], 1e-300, None))))
        tail = lf > ell[-1]
        cl[tail] *= (lf[tail] / ell[-1]) ** -1.5  # beyond the last node: a falling power law (theta >= 12' needs ell < 1000)
        idx = blocks[("wtheta", frozenset((tier,)))]
        th_arc = vec.index["x"][idx]
        wm = np.asarray(cl_to_wtheta(jnp.asarray(np.radians(th_arc / 60)), jnp.asarray(lf), jnp.asarray(cl)))
        wd = vec.data[idx]
        q = (th_arc >= 12) & (th_arc <= 120)
        out[tier] = {"theta_arcmin": th_arc.tolist(), "data": wd.tolist(), "model": wm.tolist(),
                     "data_over_model_12_120": float(np.median(wd[q] / wm[q]))}
        print(f"{tier[:40]:40s} w(theta) data/model on 12-120 arcmin {out[tier]['data_over_model_12_120']:.2f}",
              flush=True)
    rec.data.update(stack=stack, samples=out)
    print(rec.write("28_cl_validation_v3"))


def cmd_xslice(a):
    """The slice-0 x slice-1 galaxy crosses: r = C_01 / sqrt(C_00 C_11) on 30-200, model (three-shell
    Jacobian run of script 22, iterate 0) against data, for the samples present in both slices."""
    from wp3feas.model import _in_build as inb

    rec = Record(__file__)
    d3 = DATA / "22_jacobian"
    for p in (VECTOR, d3 / "meta.json"):
        rec.add_input(p)
    meta = json.loads((d3 / "meta.json").read_text())
    ell = np.asarray(meta["ell"])
    n_disc = len(meta["discrete"])
    pair_of = {}
    for i, (p_, q_) in enumerate(meta["pairs"]):
        pair_of[(p_, q_)] = pair_of[(q_, p_)] = i
    D = np.zeros(n_disc + len(meta["pairs"]) * ell.size)
    for b in meta["blocks"]:
        with np.load(d3 / f"{b}.npz") as z:
            D[z["rows"]] += z["D"]
    vec = load_vector()
    by = {(k[0], k[1], k[2] or k[1]): i for k, i in vec.blocks()}
    names = sorted({n.decode() for k in ("a", "b") for n in vec.index[k]} - {""})
    s0 = [n for n in names if parse_tracer(n)["kind"] == "gal" and parse_tracer(n)["slice"] == 0]
    s1 = [n for n in names if parse_tracer(n)["kind"] == "gal" and parse_tracer(n)["slice"] == 1]

    def model(a_, b_):
        A, B = inb(build_name(a_), pair_of), inb(build_name(b_), pair_of)
        if (A, B) not in pair_of:
            return None
        j = n_disc + pair_of[(A, B)] * ell.size
        return D[j:j + ell.size]

    def data(a_, b_):
        i = by.get(("cl", a_, b_), by.get(("cl", b_, a_)))
        return i

    out = {}
    for x in s0:
        for y in s1:
            px, py = parse_tracer(x), parse_tracer(y)
            if px["colour"] != py["colour"] or px["colour"] == "all":
                continue
            ix, iy, ixy = data(x, x), data(y, y), data(x, y)
            mx, my, mxy = model(x, x), model(y, y), model(x, y)
            if None in (ix, iy, ixy) or mxy is None or mx is None or my is None:
                continue
            l = vec.index["x"][ixy]
            q = (l >= 30) & (l <= 200)
            lx = vec.index["x"][ix]
            qx = (lx >= 30) & (lx <= 200)
            rd = vec.data[ixy][q].mean() / np.sqrt(vec.data[ix][qx].mean() * vec.data[iy][qx].mean())
            rj = vec.jk[:, ixy][:, q].mean(1) / np.sqrt(vec.jk[:, ix][:, qx].mean(1) * vec.jk[:, iy][:, qx].mean(1))
            lq = vec.index["x"][ixy][q]
            mi = lambda m: np.interp(np.log(lq), np.log(ell), m).mean()
            rm = mi(mxy) / np.sqrt(mi(mx) * mi(my))
            sd = float(np.sqrt(0.99 * np.sum((rj - rj.mean()) ** 2)))
            key = f"{x.split('_N_')[0][5:]} x {y.split('_N_')[0][5:]}"
            out[key] = {"r_data": float(rd), "sigma": sd, "r_model": float(rm)}
            print(f"{key:70s} r data {rd:.3f} +- {sd:.3f}   model {rm:.3f}", flush=True)
    rec.data.update(ell_range=[30, 200], crosses=out)
    print(rec.write("28_cl_validation_xslice"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("v1", "v1b", "v2", "v3", "xslice"))
    a = ap.parse_args(argv)
    {"v1": cmd_v1, "v1b": cmd_v1b, "v2": cmd_v2, "v3": cmd_v3, "xslice": cmd_xslice}[a.cmd](a)


if __name__ == "__main__":
    main()
