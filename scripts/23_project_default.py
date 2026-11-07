"""Part D, step 3: the ggah_mod 1.7.0 default model of public_2026 against the data, block by block.

    source ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh cpu
    PYTHONPATH=~/software/ggah_sens_study_wt_v170:$PYTHONPATH python scripts/23_project_default.py

The model at the defaults is the ``D`` that script 22 stored with each Jacobian block.
Per data block (stat, a, b, survey), on ell >= 30: chi^2 with the jackknife covariance
(Hartlap), its PTE, the median data/model ratio, and the largest whitened residual.
A block is flagged when PTE < 0.01; it is called a systematic candidate when its
chi^2 stays above that threshold after the best amplitude rescaling of the model
(a free factor absorbs bias, baryon or AGN amplitude, which the model can produce).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from wp3feas.model import NZ_FILE, projection  # noqa: E402
from wp3feas.provenance import Record  # noqa: E402
from wp3feas.vector import VECTOR, hartlap, jk_cov, load_vector  # noqa: E402

import importlib  # noqa: E402

S22 = importlib.import_module("22_fisher_public")
PTE_MIN = 0.01


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="blocks of D at another point (script 26)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--layout", default="", choices=list(S22.LAYOUTS))
    a = ap.parse_args(argv)
    S22.set_layout(a.layout)
    rec = Record(__file__)
    for p in (VECTOR, NZ_FILE, S22.OUT / "meta.json"):
        rec.add_input(p)
    meta, D, _ = S22._assemble(Path(a.model_dir) if a.model_dir else None, jac=False)
    info = {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]], "ell": meta["ell"]}
    vec = load_vector()
    P, rows = projection(vec, info)
    m = P @ D
    d = vec.data[rows]
    ell = vec.index["x"][rows]
    fin = np.isfinite(d) & np.isfinite(m) & np.all(np.isfinite(vec.jk[:, rows]), axis=0) & (ell >= S22.SCALE_LMIN)
    key = np.array(["|".join(vec.index[k][r].decode() for k in ("stat", "a", "b", "survey")) for r in rows])
    slices = S22._slice_of(vec, rows)
    blocks = []
    for k in np.unique(key[fin]):
        sel = np.flatnonzero((key == k) & fin)
        C = jk_cov(vec.jk[:, rows[sel]])
        h = hartlap(vec.n_jk, sel.size)
        if h <= 0 or not np.all(np.isfinite(C)):
            continue
        iC = h * np.linalg.pinv(C)
        r = d[sel] - m[sel]
        chi2 = float(r @ iC @ r)
        A = float((m[sel] @ iC @ d[sel]) / (m[sel] @ iC @ m[sel]))  # best amplitude
        ra = d[sel] - A * m[sel]
        chi2a = float(ra @ iC @ ra)
        w, V = np.linalg.eigh(C)
        wr = (V.T @ r) / np.sqrt(np.clip(w, w.max() * 1e-12, None))
        n = sel.size
        sl = slices[sel[0]]
        blocks.append({"block": k, "slice": "los" if not sl else "+".join(map(str, sl)), "n": n,
                       "chi2": chi2, "pte": float(stats.chi2.sf(chi2, n)),
                       "amp": A, "chi2_amp": chi2a, "pte_amp": float(stats.chi2.sf(chi2a, n - 1)),
                       "ratio_med": float(np.median(d[sel] / m[sel])),
                       "max_whitened": float(np.max(np.abs(wr)))})
    for b in blocks:
        b["flag"] = ("ok" if b["pte"] >= PTE_MIN else
                     "amplitude" if b["pte_amp"] >= PTE_MIN else "systematic_candidate")
    by_stat = {}
    for b in blocks:
        s = b["block"].split("|")[0] + ":" + b["slice"]
        e = by_stat.setdefault(s, {"blocks": 0, "rows": 0, "ok": 0, "amplitude": 0, "systematic_candidate": 0})
        e["blocks"] += 1
        e["rows"] += b["n"]
        e[b["flag"]] += 1
    chi2_tot = sum(b["chi2"] for b in blocks)
    n_tot = sum(b["n"] for b in blocks)
    worst = sorted(blocks, key=lambda b: b["pte_amp"])[:30]
    rec.data.update(stack=meta["stack"], ell_min=S22.SCALE_LMIN, pte_min=PTE_MIN,
                    covariance="jackknife per block, Hartlap", n_blocks=len(blocks), n_rows=n_tot,
                    chi2_sum=chi2_tot, summary=by_stat, blocks=blocks)
    print(json.dumps(by_stat, indent=1))
    print(f"chi2 {chi2_tot:.0f} over {n_tot} rows in {len(blocks)} blocks")
    for b in worst[:15]:
        print(f"  {b['block']:60s} n={b['n']:3d} chi2={b['chi2']:9.1f} A={b['amp']:6.2f} chi2_A={b['chi2_amp']:9.1f}")
    print(rec.write(f"23_project_default{a.tag}"))


if __name__ == "__main__":
    main()
