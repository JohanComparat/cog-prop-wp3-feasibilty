"""Part F: the cost of one likelihood value and gradient, and of a NUTS chain.

    source cc/env.sh   (or the laptop stack: ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh gpu)
    python scripts/41_timing.py blocks --layout _s0              # slice 0 of public_2026 (script 26)
    python scripts/41_timing.py blocks --layout S5_public_2031   # the five-shell reference run
    python scripts/41_timing.py nuts                             # Gaussian targets, any machine
    python scripts/41_timing.py summary

``blocks`` builds the model (``joint.model.build``, split into ``info["blocks"]`` as in
script 22) and, per block, times the value ``f(theta)`` and the gradient of a scalar of it
(``jax.grad`` of ``c . f``, reverse mode; forward mode ``jacfwd`` when reverse mode fails), each
as compile time (first call) and steady time (median of three later calls).
The likelihood value and gradient are the sums over blocks: the blocks run one after another,
as in the MAP chains, so the sum is an upper limit for a device that holds the whole program.
It resumes from ``results/41_timing_blocks<layout>.json`` and prints "done in" only when every
block is timed (``cc/gpu.sbatch`` reruns it with ``--max-blocks``).

``nuts`` runs numpyro NUTS (dense mass matrix) on Gaussian targets with the Laplace covariance
of script 26 (79 free parameters at ``_cuts`` it 21) and on a 212-parameter Gaussian with that
covariance embedded and the remaining directions spread over the same range of widths:
gradient evaluations per effective sample, which do not depend on the cost of the model
once the target is close to Gaussian (a lower limit for the real posterior).

``summary`` combines them into ``results/41_timing.json``: the wall time of a chain of
``--ess`` effective samples per parameter at 212 parameters = gradient evaluations per
effective sample x ESS x the steady gradient time of the five-shell model.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from wp3feas.provenance import DATA, RESULTS, Record  # noqa: E402

LAPLACE = Path(os.environ.get("WP3FEAS_LAPLACE", DATA / "cc/26_map_s0_cuts/it21/laplace_cov.npz"))
N_FULL = 212


def _state_path(layout):
    return RESULTS / f"41_timing_blocks{layout}.json"


def _config(layout):
    """``(cfg, fid)`` of the layout: script 22's for public_2026, ggah_sens_study's script 90 for
    the 2031 runs."""
    from ggah_sens_study.fiducial import load_fiducial

    if layout.endswith("_2031"):
        import ggah_sens_study
        sd = Path(ggah_sens_study.__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(sd))
        spec = importlib.util.spec_from_file_location("s90", sd / "90_shells_2031.py")
        s90 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s90)
        cfg = s90.config(layout)
    else:
        import importlib as il
        from wp3feas.model import public_config
        S22 = il.import_module("22_fisher_public")
        cfg = public_config(**S22.LAYOUTS[layout])
    return cfg, load_fiducial()


def _median_time(fn, x, n=3):
    import jax
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(fn(x))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def cmd_blocks(a):
    import jax
    import jax.numpy as jnp
    from ggah_sens_study.joint import model as M

    t_start = time.time()
    path = _state_path(a.layout)
    st = json.loads(path.read_text()) if path.is_file() else {"layout": a.layout, "blocks": {}, "builds": []}
    t0 = time.perf_counter()
    cfg, fid = _config(a.layout)
    _, info = M.build(fid, cfg, matter="closure", jit=False, block_pairs=a.block_pairs,
                      block_pairs_los=a.block_pairs_los)
    th = jnp.asarray(np.asarray(M.theta0(fid, cfg)))
    st["builds"].append(time.perf_counter() - t0)
    st.update(n_params=int(th.size), n_blocks=len(info["blocks"]), n_pairs=len(info["pairs"]),
              n_ell=int(np.size(info["ell"])), device=str(jax.devices()[0]),
              block_pairs=a.block_pairs, block_pairs_los=a.block_pairs_los)
    print(f"{a.layout}: {th.size} parameters, {len(info['blocks'])} blocks, build {st['builds'][-1]:.1f} s",
          flush=True)
    done = 0
    for b in info["blocks"]:
        if b["name"] in st["blocks"]:
            continue
        if a.max_blocks and done >= a.max_blocks:
            print("stopping: --max-blocks reached", flush=True)
            path.write_text(json.dumps(st, indent=1) + "\n")
            return
        done += 1
        r = {"rows": int(np.size(b["rows"]))}
        f = jax.jit(b["f"])
        t = time.perf_counter()
        y = jax.block_until_ready(f(th))
        r["value_compile_s"] = time.perf_counter() - t
        r["value_s"] = _median_time(f, th)
        c = jnp.asarray(np.random.default_rng(0).standard_normal(np.shape(y)))
        for mode in ("rev", "fwd"):
            if mode == "rev":
                g = jax.jit(jax.grad(lambda x: jnp.sum(c * b["f"](x))))
            else:
                g = jax.jit(lambda x: c.ravel() @ jax.jacfwd(b["f"])(x).reshape(c.size, -1))
            try:
                t = time.perf_counter()
                jax.block_until_ready(g(th))
                r["grad_compile_s"] = time.perf_counter() - t
                r["grad_s"] = _median_time(g, th)
                r["grad_mode"] = mode
                break
            except Exception as e:  # noqa: BLE001  (reverse mode through a non-differentiable loop)
                r[f"grad_{mode}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        st["blocks"][b["name"]] = r
        print(f"  {b['name']}: {r['rows']} rows, value {r['value_compile_s']:.1f} / {r['value_s']:.3f} s, "
              f"grad ({r.get('grad_mode')}) {r.get('grad_compile_s', np.nan):.1f} / {r.get('grad_s', np.nan):.3f} s",
              flush=True)
        path.write_text(json.dumps(st, indent=1) + "\n")
        jax.clear_caches()
    path.write_text(json.dumps(st, indent=1) + "\n")
    print(f"done in {time.time() - t_start:.1f} s")


def _gaussian_targets():
    """``{name: covariance}``: the Laplace covariance of script 26 and its 212-parameter extension."""
    z = np.load(LAPLACE, allow_pickle=True)
    C = np.asarray(z["C"])
    n = C.shape[0]
    sd = np.sqrt(np.diag(C))
    rng = np.random.default_rng(1)
    extra = np.exp(rng.uniform(np.log(sd.min()), np.log(sd.max()), N_FULL - n))
    C2 = np.zeros((N_FULL, N_FULL))
    C2[:n, :n] = C
    C2[n:, n:] = np.diag(extra**2)
    return {f"laplace_{n}": C, f"embedded_{N_FULL}": C2}


def cmd_nuts(a):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import effective_sample_size
    from numpyro.infer import MCMC, NUTS

    rec = Record(__file__)
    rec.add_input(LAPLACE)
    out = {}
    for name, C in _gaussian_targets().items():
        n = C.shape[0]
        L = jnp.asarray(np.linalg.cholesky(C + 1e-12 * np.eye(n) * np.trace(C) / n))

        def model():
            numpyro.sample("x", dist.MultivariateNormal(jnp.zeros(n), scale_tril=L))

        kern = NUTS(model, dense_mass=True, target_accept_prob=0.8)
        mc = MCMC(kern, num_warmup=a.warmup, num_samples=a.samples, num_chains=1, progress_bar=False)
        t = time.perf_counter()
        mc.run(jax.random.PRNGKey(0), extra_fields=("num_steps",))
        wall = time.perf_counter() - t
        x = np.asarray(mc.get_samples()["x"])
        steps = np.asarray(mc.get_extra_fields()["num_steps"])
        ess = np.asarray(effective_sample_size(x[None]))
        ess_min = float(ess.min())
        out[name] = {"n": n, "warmup": a.warmup, "samples": a.samples, "wall_s": wall,
                     "grads_per_sample_mean": float(steps.mean()), "grads_sampling": int(steps.sum()),
                     "ess_min": ess_min, "ess_median": float(np.median(ess)),
                     "grads_per_ess_min": float(steps.sum() / ess_min),
                     "warmup_grads_est": float(steps.mean() * a.warmup)}
        print(name, json.dumps(out[name]), flush=True)
    rec.data.update(targets=out)
    print(rec.write("41_timing_nuts"))
    print("done in", f"{time.time() - rec.t0:.1f} s")


def _sum(layout):
    st = json.loads(_state_path(layout).read_text())
    bl = st["blocks"].values()
    s = {k: float(sum(b.get(k, 0.0) for b in bl)) for k in ("value_s", "grad_s", "value_compile_s", "grad_compile_s")}
    s.update(n_blocks_timed=len(st["blocks"]), n_blocks=st.get("n_blocks"), n_params=st.get("n_params"),
             device=st.get("device"), build_s=float(np.median(st["builds"])),
             grad_modes=sorted({b.get("grad_mode") or "failed" for b in bl}),
             complete=len(st["blocks"]) == st.get("n_blocks"))
    return s


def cmd_summary(a):
    rec = Record(__file__)
    lay = {}
    for layout in ("_s0", "S5_public_2031"):
        if _state_path(layout).is_file():
            rec.add_input(_state_path(layout))
            lay[layout] = _sum(layout)
    nuts = json.loads((RESULTS / "41_timing_nuts.json").read_text())["targets"]
    rec.add_input(RESULTS / "41_timing_nuts.json")
    g = nuts[f"embedded_{N_FULL}"]
    chain = {}
    for layout, s in lay.items():
        tg = s["grad_s"]
        n_grad = g["grads_per_ess_min"] * a.ess + g["warmup_grads_est"]
        chain[layout] = {"grad_s": tg, "n_grad": n_grad, "chain_wall_h": n_grad * tg / 3600.0,
                         "ess": a.ess, "n_params": N_FULL}
        print(f"{layout}: value {s['value_s']:.2f} s, gradient {tg:.2f} s ({s['device']}); "
              f"NUTS {a.ess} ESS at {N_FULL} parameters: {n_grad:.3g} gradients, {n_grad * tg / 3600:.1f} h",
              flush=True)
    rec.data.update(layouts=lay, nuts=nuts, chain=chain)
    print(rec.write("41_timing"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("blocks", "nuts", "summary"))
    ap.add_argument("--layout", default="_s0")
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--block-pairs", type=int, default=40)
    ap.add_argument("--block-pairs-los", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--ess", type=int, default=1000)
    a = ap.parse_args(argv)
    {"blocks": cmd_blocks, "nuts": cmd_nuts, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
