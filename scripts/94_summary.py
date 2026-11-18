r"""Part G: the WP3 feasibility numbers in one file, ``results/wp3_feasibility_summary.json``,
read by ``collect_wp3feas()`` in ``erc-cog-prop/scripts/make_proposal_figures.py``.

    python scripts/94_summary.py

It reads only result JSONs of this repository and the N1 and N7 outputs of ggah_sens_study
(v1.7.0, read only); it computes nothing beyond interpolation and sums.
A section whose input is absent is written as ``null`` and named in ``missing``.
Every number keeps the path of the JSON it comes from in ``sources``.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wp3feas.provenance import RESULTS, SENS, Record  # noqa: E402

RUN = "S5_public_2031"
SHELLS = SENS / "shells_2031"
KS = (0.3, 1.0, 3.0, 10.0)


def load(rec, path):
    path = Path(path)
    if not path.exists():
        return None
    rec.add_input(path)
    return json.loads(path.read_text())


def at_k(k, y, k0):
    return float(np.interp(np.log(k0), np.log(k), y))


def section_inputs(d):
    return {"shells": len(d["shells"]), "n_params": d["n_params"], "spectra": d["spectra"],
            "rows_T5_k10": d["rows_T5_k10"], "check": d["check"]}


def section_ellmax(d, split):
    out = {"ell_k": {f"{s['z'][0]:g}-{s['z'][1]:g}": s["ell_k"] for s in d["shells"]},
           "beams_ell_half": d["beams_ell_half"],
           "shear_ell_noise": d["shear"],
           "y_ell_noise": d["y"],
           "frac_info_above": {f"{c['k_max']:g}": c["frac_info_above"] for c in d["fisher_S8_T5_C1SSC_E1"]},
           "S8_rel": {f"{c['k_max']:g}": c["S8_rel"] for c in d["fisher_S8_T5_C1SSC_E1"]}}
    if split is not None:
        out["share_by_family"] = {f"{c['k_max']:g}": {f: v["share"] for f, v in c["without"].items()}
                                  for c in split["cells"]}
        out["share_convention"] = split["convention"]
    return out


def section_outliers(by_k):
    out = {}
    for k, d in by_k.items():
        cells = {}
        for r in d["results"]:
            cells[f"{r['level']}_{r['scheme']}"] = {
                "sigma_S8_fixed": r["sigma_S8_fixed"],
                "f_max_own": {t: b["f_max_own"] for t, b in r["bias"].items()},
                "f_max_target": {t: b["f_max_target"] for t, b in r["bias"].items()},
                "free_ratio": {t: v["ratio"] for t, v in r["free"].items()}}
        out[k] = {"rows": d["rows"], "cells": cells}
    return out


def section_covariance(lc, comp1, comp5):
    out = {"rows_full": lc["rows_full"], "n_theta": lc["n_theta"],
           "light_cones": {route: {t: v["light_cones"] for t, v in r["targets"].items()}
                           for route, r in lc["routes"].items() if "targets" in r},
           "light_cones_subblock": {n: v["light_cones"] for n, v in lc["routes"]["analytic_validated"].items()}}
    for tag, c in (("compression_one_shell", comp1), ("compression_five_shell", comp5)):
        if c is None:
            out[tag] = None
            continue
        out[tag] = {"sens_run": c["inputs"][0] if c.get("inputs") else None, "cells": {
            k: {"rows": v["rows"], "S8_rel_full": v["S8_rel_full"], "loss": v["loss"],
                "sim_cov_ratio_median": {n: s["ratio_median"] for n, s in v.get("sim_cov", {}).items()}}
            for k, v in c["results"].items()}}
    return out


def section_timing(nuts, blocks):
    out = {"nuts_gaussian": None, "layouts": {}}
    if nuts is not None:
        out["nuts_gaussian"] = {n: {"n": t["n"], "grads_per_ess_min": t["grads_per_ess_min"],
                                    "warmup_grads_est": t["warmup_grads_est"]}
                                for n, t in nuts["targets"].items()}
    for lay, d in blocks.items():
        if d is None:
            continue
        b = list(d["blocks"].values())
        grad = np.mean([x["grad_s"] for x in b])
        val = np.mean([x["value_s"] for x in b])
        out["layouts"][lay] = {
            "device": d["device"], "n_params": d["n_params"], "n_blocks": d["n_blocks"], "blocks_timed": len(b),
            "complete": len(b) == d["n_blocks"],
            "value_s_est": float(val * d["n_blocks"]), "grad_s_est": float(grad * d["n_blocks"]),
            "grad_compile_s_est": float(np.mean([x["grad_compile_s"] for x in b]) * d["n_blocks"]),
            "rule": "mean over the timed blocks times n_blocks, blocks run one after another"}
    s5 = out["layouts"].get(RUN)
    g = (out["nuts_gaussian"] or {}).get("embedded_212")
    if s5 and g:
        n_grad = g["grads_per_ess_min"] * 1000 + g["warmup_grads_est"]
        out["nuts_chain_212"] = {"ess": 1000, "gradients": n_grad, "gpu_hours": n_grad * s5["grad_s_est"] / 3600,
                                 "note": "Gaussian target: a lower limit on the gradients"}
    return out


def section_public(fis, mapsum, chi2, band, fc):
    out = {}
    if fis is not None:
        out["fisher_S8"] = {s: v["sigma"]["S8"] for s, v in fis["sigma"].items()
                            if isinstance(v, dict) and "sigma" in v and "S8" in v["sigma"]}
    if mapsum is not None:
        o = mapsum["objective"]
        out["map"] = {"it": mapsum["it"], "chi2": o["chi2"], "n_rows": o["n_rows"], "chi2_per_row": o["chi2"] / o["n_rows"],
                      "shifts_beyond_3sigma": {n: p["shift_in_error"] for n, p in mapsum["parameters"].items()
                                               if abs(p.get("shift_in_error", 0)) > 3}}
    if chi2 is not None:
        out["chi2_by_type"] = {t: {"chi2": v["chi2_kept"], "n": v["n_kept"]} for t, v in chi2["by_type"].items()}
    for tag, d in (("sk_today", band), ("sk_2030", fc)):
        if d is None:
            continue
        b = d["band"]["0.15"]
        out[tag] = {"z": 0.15, "k": list(KS),
                    "S": [at_k(d["k"], b["S"], k) for k in KS],
                    "S_default": [at_k(d["k"], b["S_default"], k) for k in KS],
                    "sigma": [at_k(d["k"], b["sigma"], k) for k in KS]}
    return out


def section_sens(n1, n7, n7m):
    out = {"N1": None, "N6": None, "N7": None}
    if n1 is not None:
        cal = n1["runs"][f"{RUN}_calib"]["S8_1e3"]
        out["N1"] = {"level": n1["level"], "scheme": n1["scheme"], "unit": "1e3 sigma(S8)/S8",
                     "calibs": n1["runs"][f"{RUN}_calib"]["calibs"],
                     "T5": {c: {k: v["T5"][k] for k in ("1", "3", "10") if k in v.get("T5", {})} for c, v in cal.items()},
                     "variants_T5_k10": {r: {c: v["T5"].get("10") for c, v in d["S8_1e3"].items() if "T5" in v}
                                         for r, d in n1["runs"].items()}}
    if n7 is not None:
        out["N7"] = {"tolerances_eps_max_own": n1["tolerances"]["eps_max_own"] if n1 else None,
                     "requirements": n7["requirements"], "shifts": n7["shifts"]}
        if n7m is not None:
            out["N7"]["S8_rel_marg"] = n7m["S8_rel"]
    return out


def main():
    rec = Record(__file__)
    R = lambda name: load(rec, RESULTS / name)  # noqa: E731
    missing = []

    def need(name, val):
        if val is None:
            missing.append(name)
        return val

    a = need("01_inputs", R("01_inputs.json"))
    b = need("11_ellmax", R("11_ellmax.json"))
    bs = need("11_ellmax_split", R("11_ellmax_split.json"))
    c = {k: R(f"12_photoz_outliers_k{k}.json") for k in ("1", "3", "10")}
    for k, v in c.items():
        need(f"12_photoz_outliers_k{k}", v)
    lc = need("31_lightcones", R("31_lightcones.json"))
    comp1 = need("32_compression", R("32_compression.json"))
    comp5 = need("32_compression_S5_public_2031", R(f"32_compression_{RUN}.json"))
    if comp5 is not None and "v1.7.0" not in json.dumps(comp5.get("inputs", [])):
        missing.append("32_compression_S5_public_2031 (on v1.6.0, v1.7.0 rerun pending)")
    nuts = need("41_timing_nuts", R("41_timing_nuts.json"))
    blocks = {lay: need(f"41_timing_blocks{lay}", R(f"41_timing_blocks{lay}.json")) for lay in ("_s0", RUN)}
    fis = need("22_fisher_public", R("22_fisher_public.json"))
    mapsum = need("26_map_summary_s0_cuts_it21", R("26_map_summary_s0_cuts_it21.json"))
    chi2 = need("25_scale_cuts_chi2_cuts_cuts_it21", R("25_scale_cuts_chi2_cuts_cuts_it21.json"))
    band = need("91_sk_band_s0_cuts_it21", R("91_sk_band_s0_cuts_it21.json"))
    fc = need("91_sk_band_forecast", R("91_sk_band_forecast_forecast_S5_lsst_2030_k10.json"))
    n1 = need("N1 public_2031_summary", load(rec, SHELLS / "public_2031_summary.json"))
    n7 = need("N7 systematics", load(rec, SHELLS / RUN / "systematics.json"))
    n7m = need("N7 n7_mstar_marg", load(rec, SHELLS / RUN / "n7_mstar_marg.json"))
    missing.append("N6 (intrinsic alignments: no output in ggah_sens_study v1.7.0)")

    c_ok = {k: v for k, v in c.items() if v is not None}
    rec.data.update(
        run=RUN, sens=str(SENS),
        inputs_A=section_inputs(a) if a else None,
        ellmax_B=section_ellmax(b, bs) if b else None,
        photoz_outliers_C=section_outliers(c_ok) if c_ok else None,
        public_D=section_public(fis, mapsum, chi2, band, fc),
        covariance_E=section_covariance(lc, comp1, comp5) if lc else None,
        timing_F=section_timing(nuts, blocks),
        sens_study=section_sens(n1, n7, n7m),
        missing=missing)
    print(rec.write("wp3_feasibility_summary"))
    print("missing:", missing)


if __name__ == "__main__":
    main()
