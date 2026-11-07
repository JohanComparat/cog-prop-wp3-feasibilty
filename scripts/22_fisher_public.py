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
SCALE_LMIN = 30.0  # below it the mask couples the multipoles and Limber fails
KEY = ("S8", "Omega_m", "sigma8", "omega_cdm", "ln10A_s", "log10_eps_agn_p")


def cmd_jac(a):
    import jax
    import jax.numpy as jnp

    OUT.mkdir(parents=True, exist_ok=True)
    stack = check_stack()
    t0 = time.time()
    cfg = public_config()
    f, info, th0, fid = build_public(cfg, jit=False, block_pairs=a.block_pairs,
                                     block_pairs_los=a.block_pairs_los)
    meta = {"names": list(info["names"]), "theta0": th0.tolist(), "stack": stack,
            "n_rows": int(sum(len(b["rows"]) for b in info["blocks"])),
            "pairs": [list(p) for p in info["pairs"]], "discrete": list(info["discrete"]),
            "ell": np.asarray(info["ell"]).tolist(), "config": cfg.to_dict(),
            "blocks": [b["name"] for b in info["blocks"]]}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    th = jnp.asarray(th0)
    n = th.size
    print(f"{n} parameters, {len(info['pairs'])} pairs, {len(info['blocks'])} blocks, "
          f"built in {time.time() - t0:.1f} s", flush=True)
    done = 0
    for b in info["blocks"]:
        path = OUT / f"{b['name']}.npz"
        if path.is_file():
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


def _assemble():
    meta = json.loads((OUT / "meta.json").read_text())
    n = len(meta["names"])
    n_rows = len(meta["discrete"]) + len(meta["pairs"]) * len(meta["ell"])  # blocks overlap
    D, J = np.zeros(n_rows), np.zeros((n_rows, n))
    for name in meta["blocks"]:
        with np.load(OUT / f"{name}.npz") as z:
            D[z["rows"]] += z["D"]       # a photometric sample's rows sum the zones' blocks
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
    out = {}
    for g, tags in groups.items():
        F = sum(F_by[t] for t in tags if t in F_by)
        live = np.flatnonzero(np.diag(F) > 0)
        cov = np.full((len(names), len(names)), np.nan)
        cov[np.ix_(live, live)] = np.linalg.pinv(F[np.ix_(live, live)])
        sig = np.sqrt(np.diag(cov))
        out[g] = {n: float(sig[i]) for i, n in enumerate(names) if n in KEY or g == "joint"}
        out[g]["n_constrained"] = int(live.size)
    rec.data.update(stack=meta["stack"], n_params=len(names), rows_modelled=int(rows.size),
                    rows_used=int(n_used), ell_min=SCALE_LMIN,
                    covariance="jackknife per block, Hartlap, block-diagonal",
                    sigma=out, theta0=dict(zip(names, meta["theta0"])))
    print(json.dumps({g: {k: v for k, v in d.items() if k in KEY or k == "n_constrained"} for g, d in out.items()}, indent=1))
    print(rec.write())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("jac", "fisher"))
    ap.add_argument("--block-pairs", type=int, default=40)
    ap.add_argument("--block-pairs-los", type=int, default=10)
    ap.add_argument("--max-blocks", type=int, default=0, help="blocks per process (0: all)")
    a = ap.parse_args(argv)
    {"jac": cmd_jac, "fisher": cmd_fisher}[a.cmd](a)


if __name__ == "__main__":
    main()
