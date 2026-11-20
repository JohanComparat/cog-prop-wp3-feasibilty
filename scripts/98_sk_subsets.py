r"""The S(k) band of the fitted slice 0.1-0.2 with groups of blocks removed from the Laplace covariance.

    source ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh
    WP3FEAS_MAP_TAG=_cuts WP3FEAS_VECTOR=<v1> WP3FEAS_MASK=<DATA>/25_mask_cuts.npz \
    WP3FEAS_SHOT=results/25_scale_cuts_agnshot_agn_Mb30_p55_r1.json WP3FEAS_AGN_BAND=0.2:2.3 \
    PYTHONPATH=~/software/ggah_sens_study_wt_v170:$PYTHONPATH python scripts/98_sk_subsets.py --it 21

Script 91 ``cov`` builds :math:`C = (\sum_b J_b^{\rm T} C_b^{-1} J_b + P)^{-1}` from every block of
the chain at an iterate.  Here the same sum runs over subsets of the blocks, each block classified
by the kinds of its two tracers:

- ``all``: every block (script 91's band);
- ``no_agn``: without a block that holds an AGN sample or the eRASS1 point sources;
- ``no_hot_gas``: without a block that holds an X-ray map or the y map;
- ``no_agn_no_hot_gas``: galaxies and shear only.

Galaxy clustering, galaxy-galaxy lensing and cosmic shear stay in every subset.  The point, the
Jacobian and the priors are those of the iterate (the parameters that lose every constraint keep
their prior), and the band is :math:`\sigma_S^2 = G C G^{\rm T}`, cosmology fixed, with ``G`` from
script 91 ``jac`` (``WP3FEAS_DATA/91_sk_band<layout><tag>_it<it>.npz``).
The iterate's model files are read from ``WP3FEAS_DATA/cc/26_map<layout><tag>/it<it>`` (the copy
of the CC run).  Writes ``results/98_sk_subsets<layout><tag>_it<it>.json``.
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import ggah_sens_study  # noqa: E402,F401  (x64 first)
from wp3feas.provenance import DATA, Record  # noqa: E402
from wp3feas.vector import load_vector, parse_tracer  # noqa: E402

S22 = importlib.import_module("22_fisher_public")
S26 = importlib.import_module("26_map_fit")

SUBSETS = {"all": set(),
           "no_agn": {"agn", "xsrc"},
           "no_hot_gas": {"xray", "y"},
           "no_agn_no_hot_gas": {"agn", "xsrc", "xray", "y"}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--it", type=int, required=True)
    ap.add_argument("--layout", default="_s0", choices=list(S22.LAYOUTS))
    a = ap.parse_args(argv)
    S22.set_layout(a.layout)
    tag = f"{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', '')}_it{a.it}"
    local = DATA / "cc" / f"26_map{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', '')}" / f"it{a.it}"
    S26.it_dir = lambda k: local
    rec = Record(__file__)

    meta, Jd, r, blocks, x = S26.point(a.it, None)
    names = meta["names"]
    vec = load_vector()
    from wp3feas.model import projection
    _, rows = projection(vec, {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]],
                               "ell": meta["ell"]})

    def kinds(sel):
        i = rows[sel[0]]
        A = vec.index["a"][i].decode()
        B = vec.index["b"][i].decode() or A
        return {parse_tracer(A)["kind"], parse_tracer(B)["kind"]}

    bk = [kinds(sel) for sel, _ in blocks]
    sp = S26.prior_sigma(names)
    Pr = np.where(np.isfinite(sp), 1.0 / sp ** 2, 0.0)
    free = np.array([n not in S26.FIXED for n in names])
    J = np.load(DATA / f"91_sk_band{tag}.npz")
    assert list(J["names"]) == list(names)
    k, zs = J["k"], J["z"]

    out = {}
    for name, drop in SUBSETS.items():
        A = np.zeros((len(names),) * 2)
        n_rows = 0
        for (sel, iC), kk in zip(blocks, bk):
            if kk & drop:
                continue
            A += Jd[sel].T @ iC @ Jd[sel]
            n_rows += sel.size
        f = np.flatnonzero(free & (np.diag(A) + Pr > 0))
        C = np.linalg.pinv(A[np.ix_(f, f)] + np.diag(Pr[f]))
        res = {"n_rows": n_rows, "n_free": int(f.size), "sigma": {}}
        for iz, z in enumerate(zs):
            G = J["G"][iz][:, f]
            sig = np.sqrt(np.clip(np.einsum("ki,ij,kj->k", G, C, G), 0, None))
            res["sigma"][f"{z:g}"] = sig.tolist()
            i1 = int(np.argmin(np.abs(np.log(k / 1.0))))
            print(f"{name:18s} {n_rows:6d} rows  z {z:g}: sigma_S(k=1) {sig[i1]:.4f}")
        out[name] = res
    rec.inputs += [str(local), str(DATA / f"91_sk_band{tag}.npz")]
    rec.data.update(it=a.it, layout=a.layout, map_tag=os.environ.get("WP3FEAS_MAP_TAG", ""),
                    k=k.tolist(), z=zs.tolist(), S={f"{z:g}": J["S"][iz].tolist() for iz, z in enumerate(zs)},
                    S_default={f"{z:g}": J["S_default"][iz].tolist() for iz, z in enumerate(zs)},
                    subsets={n: sorted(d) for n, d in SUBSETS.items()}, cosmology="fixed", band=out)
    print(rec.write(f"98_sk_subsets{tag}"))


if __name__ == "__main__":
    main()
