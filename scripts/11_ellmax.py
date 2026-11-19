r"""Part B: the multipole to which each map contributes in each shell (Referee 1, pt 3).

    ~/software/miniforge3/envs/dev-full/bin/python scripts/11_ellmax.py      (pyccl)

For each shell of S5_public_2031, :math:`\ell_k = k\,\chi(z_{\rm eff})` at k = 1, 3, 10 h/Mpc
(χ in Mpc/h), compared with the multipole at which each map loses its signal:

* beams: :math:`B_\ell = 1/2` for the ACT DR6 y beam (``tier4.kernels.act_beam``), the eROSITA
  King PSF (``king_beam``) and the Gaussian archive beam (FWHM from the run's ``x_maps``);
* shot noise of each galaxy and AGN sample: :math:`b^2 C_\ell^{mm} = 1/\bar n`, with the
  shell's top-hat n(z) (convolved with :math:`\sigma_z = 0.0305(1+z)` for the photometric
  samples), halofit, and a nominal linear bias per sample (``BIAS``, a bracket: ×0.75 and ×1.25
  are reported too);
* shape noise of each source bin: :math:`C_\ell^{\kappa\kappa} = \sigma_e^2/n_{{\rm eff},i}`,
  with n_eff split equally over the five bins;
* the y map: the GNFW (Arnaud et al. 2010) halo-model :math:`C_\ell^{yy}` against the measured
  ACT DR6 noise of the run (power law through N(1000), N(3000), floored at the white level
  1.8e-18 sr) divided by :math:`B_\ell^2`.

The X-ray maps carry only their PSF here; ``--xray`` adds their photon noise (below).
The fraction of the Fisher information on S8 above k is :math:`1 - F(k)/F(10)` with
:math:`F = \sigma(S_8)^{-2}` of the run's T5, C1+SSC, E1 table (all maps together; the
information per map needs the per-block Jacobians, at CC).

Writes results/11_ellmax.json.

``--split`` (at CC, ``sbatch cc/cpu.sbatch scripts/11_ellmax.py --split``; needs the per-block
Jacobian ``jacobian.npz`` and ``ssc.npz`` of the run) splits the information by map family:
for each family (gal, agn, ks, y, x, x_arch) the T5 Fisher matrix (C1+SSC, E1, the headline
calibration) is recomputed without the pairs that involve it (``analysis.tier_pairs`` patched),
at k_max = 1, 3, 10 h/Mpc, and the family's share is :math:`1 - F_{\rm without}/F_{\rm all}`
on S8.  Writes results/11_ellmax_split.json.

``--xray`` (dev env, seconds: reads only ``v0.npy``, ``info.npz`` and ``config.json`` of the run)
adds the photon noise of the X-ray maps.  The run's model vector at the defaults, binned on its
24 bands (``covariance.binned``), gives :math:`C_\ell^{XX}` of each map and :math:`C_\ell^{gX}`
of each galaxy sample; ``covariance.noise_power`` gives the noise, :math:`1/\bar n_\gamma` [sr]
for an X-ray map (its photons per sr) and :math:`1/(\bar n V_s/\Omega)` for a galaxy sample.
For each map the multipole where :math:`C_\ell^{XX}` falls below :math:`1/\bar n_\gamma`;
for each shell and map, with the densest galaxy sample of the shell, the signal-to-noise of the
cross per band, :math:`\sqrt{N_{\rm modes}}\,|C^{gX}|/\sqrt{(C^{gg}+N_g)(C^{XX}+N_X)+(C^{gX})^2}`
with :math:`N_{\rm modes} = (2\ell+1)\Delta\ell f_{\rm sky}` (``cfg.area(g, X)``), and the
largest band centre where it is at least 1.  Writes results/11_ellmax_xray.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wp3feas.provenance import SENS, Record  # noqa: E402

RUN = SENS / "shells_2031" / "S5_public_2031"
KS = (1.0, 3.0, 10.0)
SQDEG_PER_SR = (180.0 / np.pi) ** 2
ARCMIN2_PER_SR = SQDEG_PER_SR * 3600.0
SIGMA_E = 0.26                       # lsst_euclid_public_2031 (ggah_sens_study tier2/lensing.py)
Y_WHITE = 1.8e-18                    # act_dr6_measured (joint/footprints.py MAP_NOISE)
ELL = np.geomspace(10, 1e5, 400)


def nominal_bias(name):
    """Linear bias of a sample from its name: g{r,b}<log M*>[p]@shell or a<log L_X>@shell."""
    tag = name.split("@")[0]
    if tag.startswith("a"):
        return 1.5
    m = float(tag[2:].rstrip("p"))
    red = tag[1] == "r"
    base = 1.1 if m < 10 else 1.3 if m < 10.5 else 1.6 if m < 11 else 2.0 if m < 11.5 else 2.6
    return base * (1.1 if red else 0.9)


def crossing(ell, signal, noise):
    """First ell at which signal falls below noise (None if never in range, ell[0] if always)."""
    r = np.log(signal / noise)
    if r[0] < 0:
        return float(ell[0])
    i = np.where(r < 0)[0]
    if not len(i):
        return None
    j = i[0]
    return float(np.exp(np.interp(0, [r[j], r[j - 1]], np.log([ell[j], ell[j - 1]]))))


def main():
    import pyccl as ccl
    from scipy.optimize import brentq
    from ggah_sens_study.tier4.kernels import act_beam, king_beam

    rec = Record(__file__)
    for f in ("info.npz", "samples.json", "shells_table.json"):
        rec.add_input(RUN / f)
    info = np.load(RUN / "info.npz")
    smp = json.loads((RUN / "samples.json").read_text())
    tab = json.loads((RUN / "shells_table.json").read_text())
    xm = json.loads(str(info["x_maps"]))
    cosmo = ccl.Cosmology(Omega_c=0.31 - 0.049, Omega_b=0.049, h=0.677, sigma8=0.8028, n_s=0.965,
                                     matter_power_spectrum="halofit")
    h = cosmo["h"]

    beams = {"y (ACT DR6)": brentq(lambda l: act_beam([l])[0] - 0.5, 10, 1e5),
             "X-ray eROSITA (King)": brentq(lambda l: king_beam([l])[0] - 0.5, 10, 1e6)}
    for n, b in xm.items():
        if b["beam"] == "gauss":
            sig = b["beam_fwhm_arcmin"] / 60 * np.pi / 180 / np.sqrt(8 * np.log(2))
            beams[f"X-ray archive {n}"] = float(np.sqrt(2 * np.log(2)) / sig)

    # y: GNFW halo model
    mdef = ccl.halos.MassDef200c
    hmf = ccl.halos.MassFuncTinker08(mass_def=mdef)
    hbf = ccl.halos.HaloBiasTinker10(mass_def=mdef)
    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf, mass_def=mdef)
    prof = ccl.halos.HaloProfilePressureGNFW(mass_def=mdef)
    tsz = ccl.tSZTracer(cosmo, z_max=3.0)
    pk_yy = ccl.halos.halomod_Pk2D(cosmo, hmc, prof, lk_arr=np.log(np.geomspace(1e-4, 1e2, 256)),
                                   a_arr=np.linspace(0.25, 1, 32))
    cl_yy = ccl.angular_cl(cosmo, tsz, tsz, ELL, p_of_k_a=pk_yy)
    n1, n3 = smp["y"]["N_1000"], smp["y"]["N_3000"]
    slope = np.log(n3 / n1) / np.log(3.0)
    n_y = np.maximum(n1 * (ELL / 1000.0) ** slope, Y_WHITE) / np.asarray(act_beam(ELL)) ** 2
    ell_y = crossing(ELL, cl_yy, n_y)

    # shear: shape noise per source bin
    src = []
    n_eff_bin = smp["sources"]["n_eff"] / info["n_src_bins"]
    for i, nz in enumerate(info["src_nz"]):
        t = ccl.WeakLensingTracer(cosmo, dndz=(info["src_z"], nz))
        cl = ccl.angular_cl(cosmo, t, t, ELL)
        src.append({"bin": str(info["src_names"][i]),
                    "ell_noise": crossing(ELL, cl, np.full_like(ELL, SIGMA_E ** 2 / (n_eff_bin * ARCMIN2_PER_SR)))})

    shells = []
    for s, (zlo, zhi) in enumerate(info["shells"]):
        chi = float(info["chi_eff_shells"][s])
        zg = np.linspace(max(zlo - 0.15, 0.005), zhi + 0.15, 400)
        tophat = ((zg >= zlo) & (zg < zhi)).astype(float)
        sz = 0.0305 * (1 + 0.5 * (zlo + zhi))
        photo = np.convolve(tophat, np.exp(-0.5 * ((zg - zg.mean()) / sz) ** 2), mode="same")
        samples = []
        for name, t in smp["tracers"].items():
            if int(name.split("@")[1]) != s or t["dropped"]:
                continue
            nz = photo if t["photometric"] else tophat
            tr = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(zg, nz), bias=(zg, np.ones_like(zg)))
            cl_mm = ccl.angular_cl(cosmo, tr, tr, ELL)
            noise = np.full_like(ELL, 1.0 / (t["deg2"] * SQDEG_PER_SR))
            b = nominal_bias(name)
            samples.append({"name": name, "bias": b, "deg2": t["deg2"],
                            "ell_noise": {f: crossing(ELL, (f * b) ** 2 * cl_mm, noise) for f in (0.75, 1.0, 1.25)}})
        ell_k = {str(k): k * chi for k in KS}
        shells.append({"z": [float(zlo), float(zhi)], "chi_eff_Mpc_h": chi, "ell_k": ell_k, "samples": samples})
        print(f"shell {zlo:.1f}-{zhi:.1f}: chi {chi:.0f} Mpc/h, ell_k " +
              " ".join(f"{v:.0f}" for v in ell_k.values()))
        for sm in samples:
            print(f"   {sm['name']:10s} b {sm['bias']:.2f}  ell(S/N=1) {sm['ell_noise'][1.0]}")

    t5 = sorted((r["k_max"], r["S8_rel"]) for r in tab["table"]
                if r["tier"] == "T5" and r["level"] == "C1+SSC" and r["scheme"] == "E1")
    f_top = t5[-1][1] ** -2
    info_above = [{"k_max": k, "S8_rel": s, "frac_info_above": 1 - s ** -2 / f_top} for k, s in t5]

    rec.data.update(ks_h_Mpc=list(KS), beams_ell_half=beams, y={"ell_noise": ell_y, "noise_slope": float(slope)},
                    shear={"sigma_e": SIGMA_E, "n_eff_per_bin_arcmin2": float(n_eff_bin), "bins": src},
                    shells=shells, fisher_S8_T5_C1SSC_E1=info_above,
                    caveats=["galaxy biases nominal (bracket x0.75, x1.25)",
                             "X-ray photon-noise crossing not computed (needs the gas-model C_ell)",
                             "information above k_max for all maps together, not per map"])
    print("beams (B=1/2):", {k: round(v) for k, v in beams.items()})
    print("y noise crossing:", ell_y, " shear:", [b["ell_noise"] for b in src])
    print("fraction of S8 info above k:", [(d["k_max"], round(d["frac_info_above"], 3)) for d in info_above])
    print(rec.write())


def split():
    """Leave-one-family-out S8 information of S5_public_2031 at CC."""
    import importlib.util

    import ggah_sens_study
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import analysis as AN
    from ggah_sens_study.joint.survey import Noise

    rec = Record(__file__)
    root = Path(ggah_sens_study.__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("s90", root / "scripts" / "90_shells_2031.py")
    S90 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S90)
    S90._setup()
    S77 = S90.S77
    S77.CALIBS = {S90.CALIB_MAIN: dict(S90.S78.CALIBS)[S90.CALIB_MAIN]}
    fid = load_fiducial()
    name = "S5_public_2031"
    run = S90.S78.Run(name, fid, drop=S90.dropped(name, S90._info(name)["discrete"]))
    rec.add_input(run.src_dir)
    wid = run.widths(calib=S90.CALIB_MAIN)
    ssc = run.ssc()
    orig = AN.tier_pairs
    xarch = {n for n, b in json.loads(str(run.info["x_maps"])).items() if b.get("family") == "x_arch"} \
        if "x_maps" in run.info else set()

    def fam(t):
        k = AN._kind(t)
        return "x_arch" if k == "x" and str(t).split("@")[0] in xarch else k

    def sigma(k, drop=None):
        AN.tier_pairs = (orig if drop is None else
                         (lambda pairs, tier: [p for p in orig(pairs, tier) if drop not in map(fam, p)]))
        try:
            F, st = AN.cell_fisher(run.v0, run.J, run.info, run.cfg, Noise(), fid.cosmology, tier="T5",
                                   k_max=k, ssc=ssc, area_scales=(1.0,), T=run.T, y_noise=run.y_noise)
        finally:
            AN.tier_pairs = orig
        r = run.constraints(F["C1+SSC"][0], "E1", widths=wid)
        return float(r["S8_rel"]), int(st["rows"])

    out = []
    for k in (1.0, 3.0, 10.0):
        s_all, n_all = sigma(k)
        cell = {"k_max": k, "S8_rel": s_all, "rows": n_all, "without": {}}
        for f in ("gal", "agn", "ks", "y", "x", "x_arch"):
            s_, n_ = sigma(k, f)
            cell["without"][f] = {"S8_rel": s_, "rows": n_, "share": 1 - (s_all / s_) ** 2}
            print(f"k_max {k:g}: all {s_all:.3e} ({n_all} rows); without {f}: {s_:.3e} ({n_} rows), "
                  f"share {cell['without'][f]['share']:.3f}", flush=True)
        out.append(cell)
    rec.data.update(run=name, run_dir=str(run.src_dir), level="C1+SSC", scheme="E1", tier="T5",
                    calib=str(S90.CALIB_MAIN), cells=out,
                    convention="share = 1 - F_without/F_all on S8 (leave one family out)")
    print(rec.write("11_ellmax_split"))


def xray():
    """Photon-noise limits of the X-ray maps and of the galaxy x X-ray crosses (S5_public_2031)."""
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import covariance as CV
    from ggah_sens_study.joint.bias import load_config
    from ggah_sens_study.joint.survey import Noise

    rec = Record(__file__)
    for f in ("v0.npy", "info.npz", "config.json"):
        rec.add_input(RUN / f)
    v0 = np.load(RUN / "v0.npy")
    raw = np.load(RUN / "info.npz", allow_pickle=True)
    info = {k: raw[k] for k in raw.files}
    info["pairs"] = [tuple(map(str, p)) for p in raw["pairs"]]
    info["discrete"] = [str(d) for d in raw["discrete"]]
    info["src_names"] = [str(n) for n in raw["src_names"]]
    info["photons_sr"] = {str(n): float(v) for n, v in zip(raw["x_names"], raw["x_photons"])}
    xmaps = json.loads(str(raw["x_maps"]))
    cfg = load_config(RUN)
    fid = load_fiducial()
    nbar, C = CV.binned(v0, info)
    N = CV.noise_power(cfg, Noise(), info, nbar, fid.cosmology)
    ell = np.asarray(info["centres"], float)
    edges = np.sqrt(ell[1:] * ell[:-1])
    edges = np.concatenate([[ell[0] ** 2 / edges[0]], edges, [ell[-1] ** 2 / edges[-1]]])
    dell = np.diff(edges)

    def cl(a, b):
        return C.get((a, b), C.get((b, a)))

    def nz(name):
        v = N.get(name, 0.0)
        return np.broadcast_to(np.asarray(v, float), ell.shape)

    autos = {}
    for b, nph in info["photons_sr"].items():
        c = cl(b, b)
        if c is None:
            continue
        autos[b] = {"family": xmaps.get(b, {}).get("family", "x"), "photons_sr": nph, "noise_sr": 1.0 / nph,
                    "ell_noise": crossing(ell, np.abs(c), nz(b)),
                    "signal_over_noise": {f"{L:.0f}": float(np.interp(np.log(L), np.log(ell), np.abs(c) / nz(b)))
                                          for L in (100.0, 1000.0, 3000.0)}}
        print(f"{b:6s} C_XX = N_X at ell {autos[b]['ell_noise']}", flush=True)

    shells = np.asarray(info["shells"], float)
    tsh = np.asarray(info["tracer_shell"])
    tnames = [str(t) for t in info["tracer_names"]]
    cross = []
    for s, (zlo, zhi) in enumerate(shells):
        gals = [t for t, sh in zip(tnames, tsh) if sh == s and t in info["discrete"]
                and not t.startswith("a") and cl(t, t) is not None]
        g = min(gals, key=lambda t: float(nz(t)[0]))       # the densest sample of the shell
        row = {"shell": f"{zlo:.1f}-{zhi:.1f}", "galaxy": g, "chi_eff": float(info["chi_eff_shells"][s]),
               "maps": {}}
        for b in autos:
            cgx = cl(g, b)
            if cgx is None:
                continue
            fsky = cfg.area(g, b) / (4 * np.pi * SQDEG_PER_SR)
            nm = (2 * ell + 1) * dell * fsky
            sn = np.sqrt(nm) * np.abs(cgx) / np.sqrt((cl(g, g) + nz(g)) * (cl(b, b) + nz(b)) + cgx ** 2)
            ok = np.nonzero(sn >= 1.0)[0]
            row["maps"][b] = {"ell_last_sn1": float(ell[ok[-1]]) if ok.size else None,
                              "k_last_sn1": float(ell[ok[-1]] / info["chi_eff_shells"][s]) if ok.size else None,
                              "sn_total": float(np.sqrt(np.sum(sn ** 2))), "fsky": float(fsky)}
        ks = [m["k_last_sn1"] for m in row["maps"].values() if m["k_last_sn1"]]
        print(f"shell {row['shell']} ({g}): last band with S/N >= 1 at k = "
              f"{min(ks):.2f}-{max(ks):.2f} h/Mpc" if ks else f"shell {row['shell']}: none", flush=True)
        cross.append(row)
    rec.data.update(run="S5_public_2031", sens=str(SENS), bands_ell=ell.tolist(), autos=autos, cross=cross,
                    rule="auto: first band where C_XX < 1/nbar_gamma; cross: last band with per-band S/N >= 1, "
                         "densest galaxy sample of the shell, model at the defaults (v0)")
    print(rec.write("11_ellmax_xray"))


if __name__ == "__main__":
    if "--split" in sys.argv:
        split()
    elif "--xray" in sys.argv:
        xray()
    else:
        main()
