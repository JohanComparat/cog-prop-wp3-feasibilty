"""Part D, step 2: the Jacobian of the ggah_mod 1.7.0 default model of public_2026 and its Fisher matrix.

    source ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh gpu
    PYTHONPATH=~/software/ggah_sens_study_wt_v170:$PYTHONPATH python scripts/22_fisher_public.py jac [--block-pairs 40] [--block-pairs-los 10]
    PYTHONPATH=... python scripts/22_fisher_public.py fisher

``jac`` writes, per block of ``joint.model.build`` (``info["blocks"]``), ``D`` (its rows of the
model at the defaults) and ``J`` (forward mode, one column per parameter) under
``WP3FEAS_DATA/22_jacobian/``; an existing block is skipped, so a crashed run resumes.
``fisher`` assembles them, projects them on the modelled C_ℓ rows of the vector
(``wp3feas.model.projection``) and writes results/22_fisher_public.json.

The covariance of this first pass is the jackknife of each data block with the Hartlap factor,
block-diagonal across blocks (the analytic covariance of Part E replaces it).
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ggah_sens_study  # noqa: E402,F401  (x64 first)
from wp3feas.model import NZ_FILE, build_public, check_stack, projection, public_config  # noqa: E402
from wp3feas.provenance import DATA, Record  # noqa: E402
from wp3feas.vector import VECTOR, hartlap, jk_cov, load_vector, parse_tracer  # noqa: E402

OUT = DATA / "22_jacobian"
#: the one-shell layout: z 0.1-0.2 only, no shell copies of the parameters (the model has
#: no redshift evolution of its own, so each extra shell only adds free offsets)
LAYOUT = ""
LAYOUTS = {"": {}, "_s0": {"shells": ((0.1, 0.2),), "spec_mstar_min": ((13.0, 13.0),)}}


def set_layout(tag):
    global OUT, LAYOUT
    OUT, LAYOUT = DATA / f"22_jacobian{tag}", tag
SCALE_LMIN = 30.0  # below it the mask couples the multipoles and Limber fails
KEY = ("S8", "S8_rel", "Omega_m", "sigma8", "h", "log10_eps_agn_p")


def shard_of(a):
    """``(i, n)`` of ``--shard i/n`` (every n-th block, from block i), ``(0, 1)`` without it."""
    sh = getattr(a, "shard", None)
    if not sh:
        return 0, 1
    i, n = (int(v) for v in sh.split("/"))
    assert 0 <= i < n, sh
    return i, n


def cmd_jac(a):
    import jax
    import jax.numpy as jnp

    out = Path(getattr(a, "out", None) or OUT)  # another point: script 26's iterates
    out.mkdir(parents=True, exist_ok=True)
    stack = check_stack()
    t0 = time.time()
    cfg = public_config(**LAYOUTS[LAYOUT])
    f, info, th0, fid = build_public(cfg, jit=False, block_pairs=a.block_pairs,
                                     block_pairs_los=a.block_pairs_los)
    meta = {"names": list(info["names"]), "theta0": th0.tolist(), "stack": stack,
            "n_rows": int(sum(len(b["rows"]) for b in info["blocks"])),
            "pairs": [list(p) for p in info["pairs"]], "discrete": list(info["discrete"]),
            "ell": np.asarray(info["ell"]).tolist(), "config": cfg.to_dict(),
            "blocks": [b["name"] for b in info["blocks"]]}
    at = getattr(a, "at", None)
    if at is None:
        (OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    else:
        old = json.loads((OUT / "meta.json").read_text())
        assert old["blocks"] == meta["blocks"] and old["names"] == meta["names"], "layout differs"
        th0 = np.asarray(at, float)
    th = jnp.asarray(th0)
    n = th.size
    print(f"{n} parameters, {len(info['pairs'])} pairs, {len(info['blocks'])} blocks, "
          f"built in {time.time() - t0:.1f} s", flush=True)
    done = 0
    si, sn = shard_of(a)
    for ib, b in enumerate(info["blocks"]):
        path = out / f"{b['name']}.npz"
        if path.is_file() or ib % sn != si:
            continue
        if a.max_blocks and done >= a.max_blocks:
            print("stopping: --max-blocks reached (a fresh process frees the GPU)", flush=True)
            return
        done += 1
        t1 = time.time()
        jvp = jax.jit(lambda t, v, g=b["f"]: jax.jvp(g, (t,), (v,)))
        cols, D = [], None
        for j in range(n):
            y, dy = jvp(th, jnp.zeros(n).at[j].set(1.0))
            D = np.asarray(y) if D is None else D
            cols.append(np.asarray(dy))
        np.savez(path, rows=np.asarray(b["rows"]), D=D, J=np.column_stack(cols))
        print(f"  {b['name']}: {len(b['rows'])} rows in {time.time() - t1:.1f} s", flush=True)
        del jvp, cols, y, dy
        jax.clear_caches()  # each executable holds its constants on the GPU
        gc.collect()
    print(f"done in {time.time() - t0:.1f} s")


def _assemble(out=None, jac=True):
    out = out or OUT
    meta = json.loads((OUT / "meta.json").read_text())
    n = len(meta["names"])
    n_rows = len(meta["discrete"]) + len(meta["pairs"]) * len(meta["ell"])  # blocks overlap
    D, J = np.zeros(n_rows), np.zeros((n_rows, n))
    for name in meta["blocks"]:
        with np.load(out / f"{name}.npz") as z:
            D[z["rows"]] += z["D"]       # a photometric sample's rows sum the zones' blocks
            if jac:
                J[z["rows"]] += z["J"]
    return meta, D, J


def _slice_of(vec, rows):
    """The slices a row's two tracers live in, as a sorted tuple."""
    out = []
    for r in rows:
        s = {parse_tracer(vec.index[k][r].decode() or vec.index["a"][r].decode())["slice"] for k in ("a", "b")}
        out.append(tuple(sorted(x for x in s if x is not None)))
    return out


def cmd_fisher(a):
    rec = Record(__file__)
    for p in (VECTOR, NZ_FILE, OUT / "meta.json"):
        rec.add_input(p)
    meta, D, J = _assemble()
    names = meta["names"]
    info = {"discrete": meta["discrete"], "pairs": [tuple(p) for p in meta["pairs"]], "ell": meta["ell"]}
    vec = load_vector()
    P, rows = projection(vec, info)
    m, Jd = P @ D, P @ J
    ell = vec.index["x"][rows]
    fin = np.isfinite(vec.data[rows]) & np.all(np.isfinite(vec.jk[:, rows]), axis=0) & (ell >= SCALE_LMIN)
    # the inverse covariance, block by block (jackknife, Hartlap)
    key = np.array([b"|".join((vec.index["stat"][r], vec.index["a"][r], vec.index["b"][r],
                               vec.index["survey"][r])) for r in rows])
    slices = _slice_of(vec, rows)
    F_by = {}
    n_used = 0
    for k in np.unique(key[fin]):
        sel = np.flatnonzero((key == k) & fin)
        C = jk_cov(vec.jk[:, rows[sel]])
        h = hartlap(vec.n_jk, sel.size)
        if h <= 0 or not np.all(np.isfinite(C)):
            continue
        iC = h * np.linalg.pinv(C)
        Fk = Jd[sel].T @ iC @ Jd[sel]
        sl = slices[sel[0]]
        tag = "los" if not sl else "+".join(map(str, sl))
        F_by[tag] = F_by.get(tag, 0.0) + Fk
        n_used += sel.size
    groups = {"slice0": ["0", "los"], "slice1": ["1", "los"], "slice2": ["2", "los"],
              "joint": list(F_by)}
    # the reported basis (sigma8 for ln10A_s) and the study's priors: the parameter boxes,
    # Planck/BBN on h, n_s and Omega_b, the stage-4 source priors (ggah_sens_study.joint)
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import analysis as AN
    from ggah_sens_study.joint import decompose as DC

    fid = load_fiducial()
    T, rep, centre, _ = DC.reported_basis({"names": names, "theta0": meta["theta0"]}, fid)
    cfg = public_config(**LAYOUTS[LAYOUT])
    widths = AN.photo_widths(rep, cfg, [0.5 * (lo + hi) for lo, hi in cfg.shells])
    out = {}
    for g, tags in groups.items():
        F = T.T @ sum(F_by[t] for t in tags if t in F_by) @ T
        sig, _ = AN.constraints_joint(F, rep, centre, meta["config"].get("sources", "stage4"),
                                      cosmo_raw=fid.raw["cosmology"], widths=widths)
        out[g] = {"sigma": {n: v for n, v in sig.items() if n in KEY or g == "joint"}}
    rec.data.update(stack=meta["stack"], n_params=len(names), rows_modelled=int(rows.size),
                    rows_used=int(n_used), ell_min=SCALE_LMIN,
                    covariance="jackknife per block, Hartlap, block-diagonal",
                    sigma=out, theta0=dict(zip(names, meta["theta0"])),
                    priors="box + Planck/BBN (h, n_s, Omega_b) + stage4 sources", centre=centre)
    print(json.dumps({g: {k: v for k, v in d["sigma"].items() if k in KEY or k == "S8_rel"}
                      for g, d in out.items()}, indent=1))
    print(rec.write(f"22_fisher_public{LAYOUT}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("jac", "fisher"))
    ap.add_argument("--block-pairs", type=int, default=40)
    ap.add_argument("--block-pairs-los", type=int, default=10)
    ap.add_argument("--max-blocks", type=int, default=0, help="blocks per process (0: all)")
    ap.add_argument("--layout", default="", choices=list(LAYOUTS))
    ap.add_argument("--shard", default=None, help="i/n: every n-th block from block i (parallel jobs)")
    a = ap.parse_args(argv)
    set_layout(a.layout)
    {"jac": cmd_jac, "fisher": cmd_fisher}[a.cmd](a)


if __name__ == "__main__":
    main()
