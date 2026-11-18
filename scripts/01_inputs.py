"""Part A: the layout, row and spectra counts of the forecast run S5_public_2031, and what it assumes public.

    python scripts/01_inputs.py

Reads the outputs of ggah_sens_study (never reruns them): shells_2031_summary.json and, under
shells_2031/S5_public_2031/, config.json, samples.json, shells_table.json and info.npz.
Checks the row and spectra counts against \\nRowsFive and \\nSpectraFive of
erc-cog-prop/shared/numbers.tex.

Writes results/01_inputs.json: the shells, their effective redshift and comoving distance
(Mpc/h), the samples per shell (number, density per deg2, area, photometric or spectroscopic),
the maps (X-ray bands, y, shear source bins) with their areas and noise, the rows and σ(S8)/S8
per k_max (T5, C1+SSC, E1), and the spectra count.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wp3feas.provenance import SENS, Record  # noqa: E402

SHELLS = SENS / "shells_2031"
RUN = SHELLS / "S5_public_2031"
NUMBERS = Path.home() / "software" / "erc-cog-prop" / "shared" / "numbers.tex"


def tex_number(name):
    """The integer value of the macro \\name in numbers.tex (thin spaces removed)."""
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", NUMBERS.read_text())
    return int(m.group(1).replace("\\,", "").replace(",", "")) if m else None


def main():
    rec = Record(__file__)
    for f in ("config.json", "samples.json", "shells_table.json", "info.npz"):
        rec.add_input(RUN / f)
    rec.add_input(SHELLS / "shells_2031_summary.json")
    rec.add_input(NUMBERS)
    cfg = json.loads((RUN / "config.json").read_text())
    smp = json.loads((RUN / "samples.json").read_text())
    tab = json.loads((RUN / "shells_table.json").read_text())
    summ = json.loads((SHELLS / "shells_2031_summary.json").read_text())["runs"]["S5_public_2031"]
    info = np.load(RUN / "info.npz")

    dropped = set(summ["dropped"])
    shells = [{"z": list(map(float, s)), "z_eff": float(z), "chi_eff_Mpc_h": float(c)}
              for s, z, c in zip(info["shells"], info["z_eff_shells"], info["chi_eff_shells"])]
    per_shell = defaultdict(list)
    for name, t in smp["tracers"].items():
        sh = int(name.split("@")[1])
        per_shell[sh].append({"name": name, **t})
    for sh, s in enumerate(shells):
        kept = [t for t in per_shell[sh] if not t["dropped"]]
        s["samples"] = per_shell[sh]
        s["n_samples"] = len(kept)
        s["n_photometric"] = sum(t["photometric"] for t in kept)
        s["n_objects"] = float(sum(t["number"] for t in kept))

    pairs = [tuple(p) for p in info["pairs"]]
    kept_pairs = [p for p in pairs if not (set(p) & dropped)]
    t5 = [r for r in tab["table"] if r["tier"] == "T5" and r["level"] == "C1+SSC" and r["scheme"] == "E1"]
    rows_k10 = summ["checks"]["min_corr_eig_T5_k10"]["rows"]
    tex_rows, tex_spec = tex_number("nRowsFive"), tex_number("nSpectraFive")

    rec.data.update(
        run="S5_public_2031",
        survey=cfg["survey"],
        sources={"scenario": cfg["sources"], "n_src_bins": cfg["n_src_bins"], **smp["sources"],
                 "public_assumption": "Rubin DR1 (the first year, world-public two years after release) "
                                      "and Euclid DR2 over their union (ggah_sens_study tier2/lensing.py); "
                                      "Rubin Y3 and the Euclid Consortium data to 2030 are the private case"},
        areas_deg2=cfg["areas"],
        n_params=int(len(info["names"])),
        shells=shells,
        dropped=sorted(dropped),
        x_maps=smp["x_maps"],
        y=smp["y"],
        ell_range=[cfg["ell_min"], cfg["ell_max"]],
        n_bands=cfg["n_bands"],
        pairs_all=len(pairs),
        spectra=len(kept_pairs),
        rows_T5_k10=rows_k10,
        per_kmax_T5_C1SSC_E1=[{"k_max": r["k_max"], "rows": r["rows"], "S8_rel": r["S8_rel"]} for r in t5],
        check={"nRowsFive": tex_rows, "rows_match": tex_rows == rows_k10,
               "nSpectraFive": tex_spec, "spectra_match": tex_spec == len(kept_pairs)},
    )
    print(f"{len(shells)} shells, {rec.data['n_params']} parameters; spectra {len(kept_pairs)} "
          f"(numbers.tex {tex_spec}); rows T5 k10 {rows_k10} (numbers.tex {tex_rows})")
    for s in shells:
        print(f"  z {s['z']}: chi {s['chi_eff_Mpc_h']:.0f} Mpc/h, {s['n_samples']} samples "
              f"({s['n_photometric']} photometric), {s['n_objects']:.3g} objects")
    print(rec.write())


if __name__ == "__main__":
    main()
