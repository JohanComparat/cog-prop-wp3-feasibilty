"""Part D: the scale cuts and exclusions of public_2026, decided before any fit of S8.

    python scripts/25_scale_cuts.py mask                 # the row mask and its reasons
    python scripts/25_scale_cuts.py chi2 --model-dir DIR --layout _s0   # chi2 by block type, masked and not

``mask`` removes a row of the vector when
  (a) l < L_MIN (mask mode coupling);
  (b) l > k_max chi(z) for any discrete tracer of the pair (k_max = 1 h/Mpc; 3 with --kmax 3),
      chi at the mean redshift of its slice;
  (c) the pair is y x y (the noise of the y map is not subtracted in sum_stat);
  (d) the pair involves an X-ray field of v0 (to be replaced by the re-measurement on a clean
      footprint, sum_stat tier_xray_background.py) when --drop-xray-auto is given: X-ray x X-ray only.
The mask goes to WP3FEAS_DATA/25_mask.npz (keep, bool over the vector rows), which
scripts/26_map_fit.py reads through WP3FEAS_MASK.
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import DATA, Record  # noqa: E402
from wp3feas.vector import VECTOR, load_vector, parse_tracer  # noqa: E402

L_MIN = 30.0
H = 0.6774


def chi_mpch(z):
    from astropy.cosmology import FlatLambdaCDM
    return FlatLambdaCDM(H0=100 * H, Om0=0.3089).comoving_distance(z).value * H


def kind(n):
    return parse_tracer(n)["kind"] if n else None


def cmd_mask(a):
    rec = Record(__file__)
    rec.add_input(VECTOR)
    vec = load_vector()
    ix = vec.index
    ell = ix["x"].astype(float)
    a_ = np.char.decode(ix["a"]) if ix["a"].dtype.kind == "S" else ix["a"]
    b_ = np.char.decode(ix["b"]) if ix["b"].dtype.kind == "S" else ix["b"]
    st = np.char.decode(ix["stat"]) if ix["stat"].dtype.kind == "S" else ix["stat"]
    is_cl = np.char.startswith(st.astype(str), "cl")
    keep = np.ones(ell.size, bool)
    why = collections.Counter()
    lmax_of = {}
    for n in set(a_) | set(b_):
        t = parse_tracer(n) if n else {"z": None}
        if t.get("z") is not None and t["kind"] in ("gal", "agn"):
            lmax_of[n] = a.kmax * chi_mpch(0.5 * sum(t["z"]))
    for i in np.flatnonzero(is_cl):
        A, B = a_[i], b_[i] or a_[i]
        r = None
        if ell[i] < L_MIN:
            r = "l < 30"
        elif min(lmax_of.get(A, np.inf), lmax_of.get(B, np.inf)) < ell[i]:
            r = f"l > {a.kmax:g} h/Mpc chi(z)"
        elif kind(A) == "y" and kind(B) == "y":
            r = "y x y (noise not subtracted)"
        elif a.drop_xray_auto and kind(A) == "xray" and kind(B) == "xray":
            r = "X-ray x X-ray of v0 (re-measured on a clean footprint)"
        if r:
            keep[i] = False
            why[r] += 1
    out = DATA / f"25_mask{a.tag}.npz"
    np.savez(out, keep=keep)
    n_cl = int(is_cl.sum())
    print(f"{n_cl} C_l rows, {int(keep[is_cl].sum())} kept; removed: {dict(why)} -> {out}")
    rec.data.update(kmax_h_mpc=a.kmax, l_min=L_MIN, drop_xray_auto=a.drop_xray_auto, n_rows=int(ell.size),
                    n_cl_rows=n_cl, n_cl_kept=int(keep[is_cl].sum()), removed=dict(why), mask_file=str(out),
                    lmax_by_tracer={k: float(v) for k, v in sorted(lmax_of.items())})
    print(rec.write(f"25_scale_cuts_mask{a.tag}"))


def cmd_chi2(a):
    """chi2 by block type at the model of --model-dir, on all rows and on the kept rows."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s26", ROOT / "scripts/26_map_fit.py")
    s26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s26)
    s26.S22.set_layout(a.layout)
    rec = Record(__file__)
    keep = np.load(DATA / f"25_mask{a.tag}.npz")["keep"]
    _, D, _ = s26.S22._assemble(Path(a.model_dir), jac=False)
    vec = load_vector()
    P, r, blocks = s26.system(D)
    meta = json.loads((s26.S22.OUT / "meta.json").read_text())
    from wp3feas.model import projection
    _, rows = projection(vec, {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]],
                               "ell": meta["ell"]})

    def typ(i):
        A, B = vec.index["a"][i].decode(), vec.index["b"][i].decode() or vec.index["a"][i].decode()
        return vec.index["stat"][i].decode() + ":" + "x".join(sorted((kind(A) or "?", kind(B) or "?")))

    tot = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])
    for sel, iC in blocks:
        t = typ(rows[sel[0]])
        tot[t][0] += float(r[sel] @ iC @ r[sel])
        tot[t][1] += sel.size
        k = keep[rows[sel]]
        if k.any():
            s2 = sel[k]
            C = s26.jk_cov(vec.jk[:, rows[s2]])
            h = s26.hartlap(vec.n_jk, s2.size)
            tot[t][2] += float(r[s2] @ (h * np.linalg.pinv(C)) @ r[s2])
            tot[t][3] += s2.size
    rows_out = {t: {"chi2_all": v[0], "n_all": v[1], "chi2_kept": v[2], "n_kept": v[3]} for t, v in tot.items()}
    for t, v in sorted(rows_out.items(), key=lambda kv: -kv[1]["chi2_all"]):
        print(f"{t:18s} all {v['chi2_all']:9.0f} / {v['n_all']:5d}   kept {v['chi2_kept']:9.0f} / {v['n_kept']:5d}")
    s_all = sum(v["chi2_all"] for v in rows_out.values())
    s_k = sum(v["chi2_kept"] for v in rows_out.values())
    n_k = sum(v["n_kept"] for v in rows_out.values())
    print(f"total: all {s_all:.0f}, kept {s_k:.0f} on {n_k} rows")
    rec.data.update(model_dir=a.model_dir, layout=a.layout, by_type=rows_out, chi2_all=s_all,
                    chi2_kept=s_k, n_kept=n_k)
    print(rec.write(f"25_scale_cuts_chi2{a.tag}{a.out_tag}"))


def cmd_xshot(a):
    """X-ray x X-ray blocks: chi2 of the model of --model-dir as it is, with a free amplitude,
    and with a free amplitude plus a constant C_l (a shot noise of the resolved sources),
    each fitted by generalised least squares on the block's kept rows."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s26", ROOT / "scripts/26_map_fit.py")
    s26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s26)
    s26.S22.set_layout(a.layout)
    rec = Record(__file__)
    keep = np.load(DATA / f"25_mask{a.tag}.npz")["keep"]
    _, D, _ = s26.S22._assemble(Path(a.model_dir), jac=False)
    vec = load_vector()
    meta = json.loads((s26.S22.OUT / "meta.json").read_text())
    from wp3feas.model import projection
    P, rows = projection(vec, {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]],
                               "ell": meta["ell"]})
    m = P @ D
    ix = vec.index
    A_, B_ = ix["a"][rows].astype(str), ix["b"][rows].astype(str)
    out, tot = {}, np.zeros(3)
    meas = json.loads(Path(a.shot_json).read_text())["samples"] if a.shot_json else {}
    pick = {"xx": lambda x, y: x.startswith("EVT") and y.startswith("EVT"),
            "agnx": lambda x, y: (x.startswith("ERO1_AGN") and y.startswith("EVT"))
            or (y.startswith("ERO1_AGN") and x.startswith("EVT")),
            "galx": lambda x, y: (x.startswith("LS11") and y.startswith("EVT"))
            or (y.startswith("LS11") and x.startswith("EVT"))}[a.pairs]
    for name in sorted({f"{x}__{y}" for x, y in zip(A_, B_) if pick(x, y)}):
        x, y = name.split("__")
        sel = np.flatnonzero((A_ == x) & (B_ == y) & keep[rows])
        if sel.size < 4:
            continue
        d = vec.data[rows[sel]]
        iC = s26.hartlap(vec.n_jk, sel.size) * np.linalg.pinv(s26.jk_cov(vec.jk[:, rows[sel]]))
        res = []
        if meas:
            dsc = x if not x.startswith("EVT") else y
            sid, band = dsc.split("_N_")[0], (y if dsc == x else x)[:13]
            c = next((v[band]["c_shot"] for k, v in meas.items() if k.startswith(sid) and band in v), None)
        for X in (None, m[sel][:, None], np.c_[m[sel], np.ones(sel.size)]):
            if X is None:
                r = d - m[sel]
                coef = []
            else:
                coef = np.linalg.solve(X.T @ iC @ X, X.T @ iC @ d)
                r = d - X @ coef
            res.append((float(r @ iC @ r), [float(c) for c in coef]))
        if meas and c is not None:
            r = d - m[sel] - c
            mm = m[sel]
            amp = float(mm @ iC @ (d - c) / (mm @ iC @ mm))
            ra = d - c - amp * mm
            res += [(float(r @ iC @ r), []), (float(ra @ iC @ ra), [amp])]
        out[name] = {"n": int(sel.size), "chi2_model": res[0][0], "chi2_amp": res[1][0], "amp": res[1][1][0],
                     "chi2_amp_shot": res[2][0], "amp_with_shot": res[2][1][0], "shot": res[2][1][1],
                     "shot_over_model_l1000": res[2][1][1] / float(np.interp(1000, ix["x"][rows[sel]], m[sel]))}
        if len(res) == 5:
            out[name].update(shot_measured=c, chi2_model_meas=res[3][0], chi2_amp_meas=res[4][0], amp_meas=res[4][1][0])
        tot += [res[0][0], res[1][0], res[2][0]]
    n = sum(v["n"] for v in out.values())
    if meas:
        cm = [v for v in out.values() if "chi2_model_meas" in v]
        print(f"measured shot added to {len(cm)} blocks ({sum(v['n'] for v in cm)} rows): chi2 "
              f"{sum(v['chi2_model'] for v in cm):.0f} -> {sum(v['chi2_model_meas'] for v in cm):.0f}, "
              f"with a free amplitude {sum(v['chi2_amp_meas'] for v in cm):.0f} "
              f"(median amplitude {np.median([v['amp_meas'] for v in cm]):.2f})")
    print(f"{len(out)} {a.pairs} blocks, {n} rows: chi2 model {tot[0]:.0f}, + amplitude {tot[1]:.0f}, "
          f"+ amplitude and shot noise {tot[2]:.0f}")
    au = {k: v for k, v in out.items() if a.pairs == "agnx" or k.split("__")[0] == k.split("__")[1]}
    for k, v in list(au.items())[:30]:
        print(f"{k:30s} n {v['n']:3d}  chi2 {v['chi2_model']:8.0f} {v['chi2_amp']:8.0f} {v['chi2_amp_shot']:8.0f}  "
              f"A {v['amp_with_shot']:.2f}  N/model(l=1000) {v['shot_over_model_l1000']:.2f}")
    rec.data.update(model_dir=a.model_dir, n_blocks=len(out), n_rows=n, chi2_model=tot[0], chi2_amp=tot[1],
                    chi2_amp_shot=tot[2], blocks=out)
    print(rec.write(f"25_scale_cuts_xshot{'' if a.pairs == 'xx' else '_' + a.pairs}{a.tag}{a.out_tag}"))


AGN_TIER = Path.home() / "data/legacysurvey/dr11/south/bgsr21s/results/tier_xcorr/samples/agn"
ERO_PROD = Path.home() / "data/erosita/eRASS1/products"
AREA_LS11 = 16515.8  # deg2, the area scripts/20_vector_audit.py divided every count by


def cmd_agnx(a):
    """AGN abundance of slice 0 (z 0.1-0.2) with the eRASS1 selection put into the model.

    Data: the counts of the tier samples over the volume of the AGN footprint (eRASS1-DE x LS11,
    the HEALPix nside-128 pixels holding their randoms), not over the LS11 area.
    Model, at the ggah_mod 1.7.0 defaults (FluxBinnedAgnSector, population "mixed", comparat19):
      built  - catalogue band 2-10 keV and no flux limit, at the slice's z_eff (the build of 22/26);
      band   - catalogue band 0.2-2.3 keV (the band the eRASS1 L_X comes from), at z_eff;
      fcut   - band, and the flux cut f_cut of the catalogue, averaged over the slice volume;
      sens   - fcut, and the eRASS1 completeness c(f) (the summed tile area curves over their
               maximum), averaged over the slice volume."""
    import fitsio
    import healpy as hp
    import yaml
    from ggah_cal.predict.erosita_agn import FluxBinnedAgnSector
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import model as M
    from wp3feas.model import public_config

    rec = Record(__file__)
    man = yaml.safe_load((AGN_TIER / "manifest.yaml").read_text())
    f_cut = float(man["f_cut"])
    samples = {k: v for k, v in man["samples"].items() if v["z"] == [0.1, 0.2]}
    ra, dec = [], []
    for k in samples:
        r = fitsio.read(AGN_TIER / f"{k}_RAND.fits", columns=["RA", "DEC"])
        ra.append(r["RA"]); dec.append(r["DEC"])
        rec.add_input(AGN_TIER / f"{k}_RAND.fits")
    pix = np.unique(hp.ang2pix(128, np.concatenate(ra), np.concatenate(dec), lonlat=True))
    area = pix.size * hp.nside2pixarea(128, degrees=True)
    curves = sorted((ERO_PROD / "area").glob("*_area.npz"))
    with np.load(curves[0]) as z:
        fgrid = z["flux"]
    acurve = np.sum([np.load(c)["area"] for c in curves], axis=0)
    comp = acurve / acurve.max()
    c_of = lambda f: float(np.interp(np.log10(f), np.log10(fgrid), comp, left=0.0, right=1.0))  # noqa: E731

    def vol(z0, z1, deg2):
        return deg2 * (np.pi / 180) ** 2 / 3 * (chi_mpch(z1) ** 3 - chi_mpch(z0) ** 3)

    cfg = public_config(shells=((0.1, 0.2),), spec_mstar_min=((13.0, 13.0),),
                        agn_bins=tuple((v["lx_min"], None) for v in samples.values()))
    fid = load_fiducial()
    J = M._Joint(fid, cfg, "closure")
    th = np.asarray(M.theta0(fid, cfg))
    p = J.params_of(th, 0)
    cosmo = J.cosmology_at(th)
    F0 = J._shell_field(th, 0)
    zg = np.linspace(0.1, 0.2, a.nz)
    fields = [J.los_field(cosmo, z) for z in zg]
    chi = np.array([chi_mpch(z) for z in zg])
    wz = chi ** 2 * np.gradient(chi, zg)
    wz /= np.trapezoid(wz, zg)
    lf = np.log10(f_cut) + np.arange(0, a.flux_dex + 1e-9, a.flux_step)

    def sector(lo, band, flim):
        return FluxBinnedAgnSector(J.gal13, lo, None, flux_lim=flim, band=band, gamma=1.8, population="mixed",
                                   nh="comparat19", calibration="off", kbol="duras20")

    def nbar(s, field):
        return float(s.weights(field, p["pa"], p["gp13"], obsc=p["po"]).norm)

    out = {}
    for k, v in samples.items():
        lo = float(v["lx_min"])
        n_data = v["n"] / vol(0.1, 0.2, area)
        mod = {"built": nbar(sector(lo, (2.0, 10.0), 1e-30), F0),
               "band": nbar(sector(lo, (0.2, 2.3), 1e-30), F0)}
        s = sector(lo, (0.2, 2.3), f_cut)
        n_fc, n_se = [], []
        for fld in fields:
            nf = []
            for x in lf:
                s.flux_lim = float(10 ** x)
                nf.append(nbar(s, fld))
            nf = np.array(nf)
            cmid = np.array([c_of(10 ** (0.5 * (lf[i] + lf[i + 1]))) for i in range(lf.size - 1)])
            n_fc.append(nf[0])
            n_se.append(float(np.sum(-np.diff(nf) * cmid) + nf[-1] * c_of(10 ** lf[-1])))
        mod["fcut"] = float(np.trapezoid(np.array(n_fc) * wz, zg))
        mod["sens"] = float(np.trapezoid(np.array(n_se) * wz, zg))
        out[k] = {"lx_min": lo, "N": v["n"], "nbar_data_ls11_area": v["n"] / vol(0.1, 0.2, AREA_LS11),
                  "nbar_data": n_data, "nbar_model": mod,
                  "data_over_model": {m: n_data / x for m, x in mod.items()},
                  "n_of_z_sens": n_se, "n_of_z_fcut": n_fc}
        print(f"L_X >= {lo}: data {n_data:.3e} | " + " ".join(f"{m} {n_data / x:.2f}" for m, x in mod.items()))
    rec.add_input(AGN_TIER / "manifest.yaml")
    rec.data.update(units="(Mpc/h)^-3", area_agn_deg2=area, area_ls11_deg2=AREA_LS11, f_cut=f_cut,
                    completeness_at_fcut=c_of(f_cut), z_grid=zg.tolist(), log10_flux_grid=lf.tolist(),
                    z_eff=float(J.z0), samples=out)
    print(rec.write("25_scale_cuts_agnx"))


def cmd_agneval(a):
    """The AGN blocks of slice 0 with the catalogue band of eRASS1 (agn_band 0.2-2.3 keV in place
    of the build's 2-10 keV), at the point of --model-dir (an iterate of script 26, its theta.json),
    and the chi2 of every block with an AGN against the build as it is.  The model blocks holding
    an AGN go to WP3FEAS_DATA/25_agnband<out_tag>/ (resumable); the other blocks are the iterate's."""
    import importlib.util
    import jax
    import jax.numpy as jnp
    from wp3feas.model import build_public, public_config
    spec = importlib.util.spec_from_file_location("s26", ROOT / "scripts/26_map_fit.py")
    s26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s26)
    s26.S22.set_layout(a.layout)
    rec = Record(__file__)
    mdir = Path(a.model_dir)
    th = np.asarray(json.loads((mdir / "theta.json").read_text())["theta"])
    meta = json.loads((s26.S22.OUT / "meta.json").read_text())
    n_disc, n_ell = len(meta["discrete"]), len(meta["ell"])
    agn = {n for n in meta["discrete"] if n.startswith("a")}
    out = DATA / f"25_agnband{a.out_tag}"
    out.mkdir(exist_ok=True)
    band = tuple(a.agn_band)
    f, info, _, _ = build_public(public_config(**s26.S22.LAYOUTS[a.layout], agn_band=band), jit=False,
                                 block_pairs=40, block_pairs_los=10)
    assert [b["name"] for b in info["blocks"]] == meta["blocks"], "block layout differs from script 22"

    def touches(rows):
        rows = np.asarray(rows)
        if np.any(np.isin(np.asarray(meta["discrete"])[rows[rows < n_disc]], list(agn))):
            return True
        pr = np.unique((rows[rows >= n_disc] - n_disc) // n_ell)
        return any(meta["pairs"][k][0] in agn or meta["pairs"][k][1] in agn for k in pr)

    todo = [b for b in info["blocks"] if touches(b["rows"])]
    print(f"{len(todo)} of {len(info['blocks'])} blocks hold an AGN", flush=True)
    done = 0
    for b in todo:
        path = out / f"{b['name']}.npz"
        if path.is_file():
            continue
        if a.max_blocks and done >= a.max_blocks:
            print("stopping: --max-blocks reached", flush=True)
            return
        done += 1
        t1 = time.time()
        np.savez(path, rows=np.asarray(b["rows"]), D=np.asarray(jax.jit(b["f"])(jnp.asarray(th))))
        print(f"  {b['name']}: {time.time() - t1:.1f} s", flush=True)
        jax.clear_caches()
    _, D0, _ = s26.S22._assemble(mdir, jac=False)
    D1 = D0.copy()
    for b in todo:
        with np.load(mdir / f"{b['name']}.npz") as z0, np.load(out / f"{b['name']}.npz") as z1:
            D1[z0["rows"]] -= z0["D"]
            D1[z1["rows"]] += z1["D"]
    vec = load_vector()
    res = {}
    for tag, D in (("built", D0), ("band", D1)):
        P, r, blocks = s26.system(D)
        res[tag] = (P, r, blocks, P @ D)
    from wp3feas.model import projection
    _, prow = projection(vec, {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]],
                               "ell": meta["ell"]})
    blocks_out, tot = {}, collections.defaultdict(lambda: np.zeros(4))
    for sel, iC in res["built"][2]:
        r0 = prow[sel[0]]
        A_, B_ = vec.index["a"][r0].decode(), vec.index["b"][r0].decode()
        if not (A_.startswith("ERO1_AGN") or B_.startswith("ERO1_AGN")):
            continue
        name = "|".join(vec.index[k][r0].decode() for k in ("stat", "a", "b", "survey"))
        d = vec.data[prow[sel]]
        row = {"n": int(sel.size)}
        for tag in ("built", "band"):
            r, m = res[tag][1][sel], res[tag][3][sel]
            amp = float(m @ iC @ d / (m @ iC @ m))
            ra = d - amp * m
            row[tag] = {"chi2": float(r @ iC @ r), "amp": amp, "chi2_amp": float(ra @ iC @ ra)}
        other = B_ if A_.startswith("ERO1_AGN") else A_
        typ = "AGN x " + (kind(other) or "?") if A_ != B_ else "AGN x AGN"
        tot[typ] += [row["built"]["chi2"], row["band"]["chi2"], row["n"], 1]
        row["type"] = typ
        blocks_out[name] = row
    for t, v in sorted(tot.items()):
        amps = [(r["built"]["amp"], r["band"]["amp"]) for r in blocks_out.values() if r["type"] == t]
        md = np.median(amps, axis=0)
        print(f"{t:14s} {int(v[3]):3d} blocks {int(v[2]):5d} rows: chi2 {v[0]:8.0f} -> {v[1]:8.0f}; "
              f"median amplitude {md[0]:.2f} -> {md[1]:.2f}")
    rec.data.update(model_dir=str(mdir), agn_band=band, n_blocks_eval=len(todo),
                    by_type={t: {"chi2_built": v[0], "chi2_band": v[1], "n_rows": int(v[2]), "n_blocks": int(v[3])}
                             for t, v in tot.items()}, blocks=blocks_out)
    print(rec.write(f"25_scale_cuts_agneval{a.out_tag}"))
    print(f"done in {rec.data.get('wall_time_s', 0)}")


XFIELDS = Path.home() / "data/legacysurvey/dr11/south/bgsr21s/results/tier_xcorr/inputs/xray_fields"


def cmd_agnshot(a):
    """The Poisson term of discrete x X-ray, measured from the photons: for the field
    delta_X = g / (alpha r) - 1 of sum_stat (g events, r randoms per pixel, alpha = sum g / sum r)
    and a sample of N members, C_shot = Omega_pix / N * sum_i (excess photons around member i)
    / (alpha r at member i), the excess summed over the member's pixel and its 8 neighbours
    (--rings 1; nside 1024, 3.4 arcmin pixels; --rings 0: the pixel alone), the expected count of
    each pixel subtracted.  --samples agn (slice-0 eRASS1 AGN) or gal (slice-0 LS11 samples);
    --field-tag picks the fields (v0: none; v1: _Mb30_p55).  For the AGN of v0 the constant of
    xshot --pairs agnx (--xshot JSON) is printed alongside."""
    import fitsio
    import healpy as hp
    rec = Record(__file__)
    nside = 1024
    om = hp.nside2pixarea(nside)
    root = AGN_TIER if a.samples == "agn" else AGN_TIER.parent / "gal"
    pat = "ERO1_AGN_LX_*_0.10_z_0.20_*_DATA.fits" if a.samples == "agn" else "LS11_*_0.10_z_0.20_*_DATA.fits"
    samples = sorted(root.glob(pat))
    fit = json.loads(Path(a.xshot).read_text())["blocks"] if (a.xshot and a.samples == "agn"
                                                              and not a.field_tag) else {}
    tag = a.field_tag
    bands = sorted({f.name[:13] for f in XFIELDS.glob(f"EVT_*{tag}_nside1024.npz")
                    if f.name == f"{f.name[:13]}{tag}_nside1024.npz" and int(f.name[4:8]) >= 500})
    fields = {}
    for band in bands:
        with np.load(XFIELDS / f"{band}{tag}_nside1024.npz") as z:
            g, r = z["counts_g"].astype(float), z["counts_r"].astype(float)
        on = r > 0
        alpha = g[on].sum() / r[on].sum()
        fields[band] = (np.where(on, g - alpha * r, 0.0), alpha * r, on)
        rec.add_input(XFIELDS / f"{band}{tag}_nside1024.npz")
    out = {}
    for sf in samples:
        sid = sf.name[:-len("_DATA.fits")]
        d = fitsio.read(sf, columns=["RA", "DEC"])
        pix0 = hp.ang2pix(nside, d["RA"], d["DEC"], nest=True, lonlat=True)
        nb = pix0[:, None]
        if a.rings >= 1:
            nb = np.c_[nb, hp.get_all_neighbours(nside, d["RA"], d["DEC"], nest=True, lonlat=True).T]
        rec.add_input(sf)
        res = {}
        for band, (ex, mu, on) in fields.items():
            ok = on[pix0]
            nbo = nb[ok]
            e = np.where(nbo >= 0, ex[np.maximum(nbo, 0)], 0.0).sum(axis=1)
            v = e / mu[pix0[ok]]
            c, err = om * float(v.mean()), om * float(v.std() / np.sqrt(v.size))
            fk = next((k for k in fit if k.startswith(sid.split("_N_")[0]) and k.endswith(band)), None)
            res[band] = {"n": int(ok.sum()), "c_shot": c, "err": err,
                         "c_shot_fit": fit[fk]["shot"] if fk else None}
        out[sid] = res
        b0 = res[bands[len(bands) // 2]]
        print(f"{sid[:42]} n {b0['n']:7d}  {bands[len(bands) // 2]}: {b0['c_shot']:.3e} +- {b0['err']:.1e}", flush=True)
    rec.data.update(nside=nside, rings=a.rings, samples_kind=a.samples, field_tag=tag, xshot=a.xshot or None,
                    samples=out)
    print(rec.write(f"25_scale_cuts_agnshot_{a.samples}{tag}_r{a.rings}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("mask", "chi2", "xshot", "agnx", "agneval", "agnshot"))
    ap.add_argument("--kmax", type=float, default=1.0, help="h/Mpc")
    ap.add_argument("--drop-xray-auto", action="store_true")
    ap.add_argument("--tag", default="", help="mask variant (file 25_mask<tag>.npz)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--layout", default="_s0")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--agn-band", type=float, nargs=2, default=(0.2, 2.3), help="agneval: keV")
    ap.add_argument("--max-blocks", type=int, default=0, help="agneval: blocks per process (0: all)")
    ap.add_argument("--pairs", default="xx", choices=("xx", "agnx", "galx"), help="xshot: X-ray x X-ray, AGN or galaxies x X-ray")
    ap.add_argument("--samples", default="agn", choices=("agn", "gal"), help="agnshot")
    ap.add_argument("--field-tag", default="", help="agnshot: X-ray field tag (v1: _Mb30_p55)")
    ap.add_argument("--rings", type=int, default=1, help="agnshot: pixel rings around each AGN")
    ap.add_argument("--xshot", default="results/25_scale_cuts_xshot_agnx_d3it8.json")
    ap.add_argument("--shot-json", default=None, help="xshot: the agnshot JSON, the measured Poisson term added")
    ap.add_argument("--nz", type=int, default=5, help="agnx: redshift nodes in the slice")
    ap.add_argument("--flux-dex", type=float, default=2.5, help="agnx: flux range above f_cut, dex")
    ap.add_argument("--flux-step", type=float, default=0.25, help="agnx: flux step, dex")
    a = ap.parse_args(argv)
    {"mask": cmd_mask, "chi2": cmd_chi2, "xshot": cmd_xshot, "agnx": cmd_agnx, "agneval": cmd_agneval, "agnshot": cmd_agnshot}[a.cmd](a)


if __name__ == "__main__":
    main()
