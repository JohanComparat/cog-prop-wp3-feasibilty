"""Part F: the cost of one likelihood value and gradient, and of a NUTS chain.

    source cc/env.sh   (or the laptop stack: ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh gpu)
    python scripts/41_timing.py blocks --layout _s0              # slice 0 of public_2026 (script 26)
    python scripts/41_timing.py blocks --layout S5_public_2031   # the five-shell reference run
    python scripts/41_timing.py nuts                             # Gaussian targets, any machine
    python scripts/41_timing.py summary
    python scripts/41_timing.py ckpt --layout _s0 --out-tag _v100   # checkpointed line of sight

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

``fused`` measures what a production gradient would cost, on a subset of blocks (those already
timed by ``blocks``, spread over the list, so their sequential times are read, not rerun):
(a) the gradient of the sum over ``G`` blocks as one jitted program, against the sum of their
sequential times, for each ``--group-sizes``; (b) the largest group split over 2, 4, ... GPUs
(each device holds one jitted group, all dispatched before any is awaited), when the job has
more than one device; (c) the compile time of the smallest group in a fresh process that reads
the persistent compilation cache filled by this one (``--cache-dir``).
It writes ``results/41_timing_fused<layout>.json`` after each stage and resumes from it.

``ckpt`` measures the two routes for the line-of-sight blocks whose gradient does not fit a
V100: ``plain`` (as ``blocks``) and ``remat`` (``build(..., remat_los=True)``: a
``jax.checkpoint`` per line-of-sight node, so reverse mode keeps the spectra of each node and
recomputes the rest node by node).  The blocks are, by default, the first line-of-sight block
whose gradient failed in ``results/41_timing_blocks<layout>.json`` and the largest one whose
gradient ran (to compare times with plain); ``--ckpt-blocks`` names them.  Each (block,
variant) runs in a fresh process (``ckpt-one``), which reports the compile and steady time of
the reverse-mode gradient, the peak device memory (``memory_stats``) and the gradient itself;
the parent compares the remat and plain gradients where both ran.  It writes
``results/41_timing_ckpt<layout><--out-tag>.json`` after each measurement and resumes from it.

``mgpu`` repeats the multi-GPU part of ``fused`` with a dispatch that can overlap.  In ``fused``
the model was built once on the first device and every device's program closed over its
arrays, so each call moved them to the other devices and the calls ran one after another
(2 and 4 V100s: 0.82 and 1.07 times one).  Here each device gets its own build under
``jax.default_device`` (every constant lives on that device), its share of the subset of
``fused`` (``blocks[i::D]``, one jitted reverse-mode gradient per block, as in the MAP chains),
and its own Python thread; the threads compile together, then run together, each timing its
own busy interval.  The same programs are also run one device at a time, so the speed-up
compares identical work: (sum of the per-device times run alone) / (wall time run together).
Each ``D`` runs in a fresh process (``mgpu-one``), and a second fresh process for the largest
``D`` reads the persistent compilation cache filled by the first (the cache test of ``fused``
ran out of memory because its child shared the device with the parent).
It writes ``results/41_timing_mgpu<layout><--out-tag>.json`` after each process.

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


def _select_blocks(info, layout, n, fixed=None):
    """The ``n`` blocks of the subset: those already timed by ``blocks``, evenly spread over the
    list; untimed blocks fill in when fewer are timed. ``fixed`` (names) keeps an earlier choice."""
    names = [b["name"] for b in info["blocks"]]
    p = _state_path(layout)
    timed = json.loads(p.read_text())["blocks"] if p.is_file() else {}
    if fixed:
        by = {b["name"]: b for b in info["blocks"]}
        return [by[k] for k in fixed], {k: timed[k] for k in fixed if k in timed}
    pool = [k for k in names if k in timed and "grad_s" in timed[k]] or names
    if len(pool) < n:
        pool = pool + [k for k in names if k not in pool]
    idx = np.unique(np.linspace(0, len(pool) - 1, n).round().astype(int))
    sel = [pool[i] for i in idx]
    by = {b["name"]: b for b in info["blocks"]}
    return [by[k] for k in sel], {k: timed[k] for k in sel if k in timed}


def _group_grad(blocks, th):
    """The jitted reverse-mode gradient of sum_b c_b . f_b over a group of blocks."""
    import jax
    import jax.numpy as jnp
    cs = []
    for i, b in enumerate(blocks):
        shape = jax.eval_shape(b["f"], th).shape
        cs.append(jnp.asarray(np.random.default_rng(i).standard_normal(shape)))
    fs = [b["f"] for b in blocks]

    def scalar(x):
        return sum(jnp.sum(c * f(x)) for c, f in zip(cs, fs))

    return jax.jit(jax.grad(scalar))


def _enable_cache(d):
    import jax
    Path(d).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(d))
    for k, v in (("jax_persistent_cache_min_compile_time_secs", 0.0),
                 ("jax_persistent_cache_min_entry_size_bytes", -1)):
        try:
            jax.config.update(k, v)
        except Exception:  # noqa: BLE001  (option absent in this jax)
            pass


def _build(layout, a):
    import jax.numpy as jnp
    from ggah_sens_study.joint import model as M
    cfg, fid = _config(layout)
    t0 = time.perf_counter()
    _, info = M.build(fid, cfg, matter="closure", jit=False, block_pairs=a.block_pairs,
                      block_pairs_los=a.block_pairs_los)
    th = jnp.asarray(np.asarray(M.theta0(fid, cfg)))
    return info, th, time.perf_counter() - t0


def cmd_cache_probe(a):
    """Child process of ``fused``: the first call of the smallest group with the cache filled."""
    import jax
    _enable_cache(a.cache_dir)
    info, th, tb = _build(a.layout, a)
    st = json.loads((RESULTS / f"41_timing_fused{a.layout}.json").read_text())
    blocks, _ = _select_blocks(info, a.layout, max(a.group_sizes), st["selected"])
    g = _group_grad(blocks[:min(a.group_sizes)], th)
    t = time.perf_counter()
    jax.block_until_ready(g(th))
    print(json.dumps({"build_s": tb, "first_call_s": time.perf_counter() - t}), flush=True)


def cmd_fused(a):
    import subprocess

    import jax

    t_start = time.time()
    _enable_cache(a.cache_dir)
    out_path = RESULTS / f"41_timing_fused{a.layout}.json"
    st = json.loads(out_path.read_text()) if out_path.is_file() else {"layout": a.layout}

    def save():
        out_path.write_text(json.dumps(st, indent=1) + "\n")

    info, th, tb = _build(a.layout, a)
    devs = jax.devices()
    gmax = max(a.group_sizes)
    blocks, seq = _select_blocks(info, a.layout, gmax, st.get("selected"))
    st.update(n_blocks=len(info["blocks"]), n_params=int(th.size), build_s=tb,
              devices=[str(d) for d in devs], selected=[b["name"] for b in blocks],
              sequential={k: {x: v.get(x) for x in ("rows", "grad_s", "grad_compile_s", "grad_mode")}
                          for k, v in seq.items()},
              cache_dir=str(a.cache_dir))
    print(f"{a.layout}: {len(info['blocks'])} blocks, {len(devs)} device(s), subset {len(blocks)}", flush=True)
    save()
    # (a) one program per group on one device
    st.setdefault("fused", {})
    for G in sorted(a.group_sizes):
        if str(G) in st["fused"]:
            continue
        if st.get("running") == f"fused{G}":  # the process died in this stage (rerun by cc/gpu.sbatch)
            st["fused"][str(G)] = {"error": "process died (out of memory?)"}
            st.pop("running")
            save()
            continue
        st["running"] = f"fused{G}"
        save()
        grp = blocks[:G]
        r = {"blocks": [b["name"] for b in grp]}
        try:
            g = _group_grad(grp, th)
            t = time.perf_counter()
            jax.block_until_ready(g(th))
            r["compile_s"] = time.perf_counter() - t
            r["grad_s"] = _median_time(g, th)
        except Exception as e:  # noqa: BLE001  (out of memory)
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        ss = [seq[b["name"]]["grad_s"] for b in grp if b["name"] in seq and "grad_s" in seq[b["name"]]]
        if len(ss) == len(grp):
            r["sequential_grad_s"] = float(sum(ss))
            r["sequential_compile_s"] = float(sum(seq[b["name"]].get("grad_compile_s", 0.0) for b in grp))
            if "grad_s" in r:
                r["speedup"] = r["sequential_grad_s"] / r["grad_s"]
        st["fused"][str(G)] = r
        st.pop("running", None)
        print(f"  fused G={G}:", json.dumps({k: v for k, v in r.items() if k != "blocks"}), flush=True)
        save()
        jax.clear_caches()
    # (b) the largest group over several devices
    st.setdefault("multi_gpu", {})
    for D in [d for d in (2, 4, 8) if d <= len(devs) and d <= gmax]:
        if str(D) in st["multi_gpu"]:
            continue
        if st.get("running") == f"multi{D}":
            st["multi_gpu"][str(D)] = {"error": "process died (out of memory?)"}
            st.pop("running")
            save()
            continue
        st["running"] = f"multi{D}"
        save()
        r = {}
        try:
            parts = [blocks[i::D] for i in range(D)]
            ths = [jax.device_put(th, devs[i]) for i in range(D)]
            gs = [_group_grad(parts[i], ths[i]) for i in range(D)]
            t = time.perf_counter()
            for i in range(D):
                jax.block_until_ready(gs[i](ths[i]))
            r["compile_s_sum"] = time.perf_counter() - t

            def run_all(_):
                return [gs[i](ths[i]) for i in range(D)]

            r["grad_s"] = _median_time(run_all, None)
            f1 = st["fused"].get(str(gmax), {})
            if "grad_s" in f1:
                r["speedup_vs_one_device"] = f1["grad_s"] / r["grad_s"]
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        st["multi_gpu"][str(D)] = r
        st.pop("running", None)
        print(f"  {D} devices:", json.dumps(r), flush=True)
        save()
        jax.clear_caches()
    # (c) the persistent cache, read by a fresh process
    if "cache" not in st:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "cache-probe", "--layout", a.layout,
               "--cache-dir", str(a.cache_dir), "--block-pairs", str(a.block_pairs),
               "--block-pairs-los", str(a.block_pairs_los), "--group-sizes", *map(str, a.group_sizes)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        r = json.loads(lines[-1]) if lines else {"error": p.stderr[-500:]}
        f0 = st["fused"].get(str(min(a.group_sizes)), {})
        if "compile_s" in f0 and "first_call_s" in r:
            r["compile_s_cold"] = f0["compile_s"]
        st["cache"] = r
        print("  cache:", json.dumps(r), flush=True)
        save()
    # extrapolation to every block: the per-block fused time of the largest group that ran
    ok = [int(G) for G, r in st["fused"].items() if "grad_s" in r]
    if ok:
        G = max(ok)
        per = st["fused"][str(G)]["grad_s"] / G
        ex = {"group": G, "grad_s_per_block": per, "grad_s_all_one_device": per * st["n_blocks"]}
        for D, r in st["multi_gpu"].items():
            if "speedup_vs_one_device" in r:
                ex[f"grad_s_all_{D}_devices"] = per * st["n_blocks"] / r["speedup_vs_one_device"]
        st["extrapolation"] = ex
        print("  extrapolation:", json.dumps(ex), flush=True)
    save()
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


def _ckpt_blocks(layout, names):
    """The blocks of ``ckpt``: ``names``, or the first line-of-sight block whose gradient failed
    in the ``blocks`` timings and the largest one whose gradient ran."""
    if names:
        return list(names)
    st = json.loads(_state_path(layout).read_text())["blocks"]
    los = [k for k in st if k.startswith("los")]
    failed = [k for k in los if "grad_s" not in st[k]]
    ran = sorted((k for k in los if "grad_s" in st[k]), key=lambda k: (-st[k]["rows"], -st[k]["grad_s"]))
    return failed[:1] + ran[:1]


def _peak_bytes():
    import jax
    try:
        m = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001  (CPU device)
        m = {}
    return {k: m.get(k) for k in ("peak_bytes_in_use", "bytes_limit")}


def cmd_ckpt_one(a):
    """Child of ``ckpt``: one block, one variant, in a fresh process; prints one JSON line."""
    import jax
    import jax.numpy as jnp
    from ggah_sens_study.joint import model as M

    r = {"block": a.block, "variant": a.variant, "device": str(jax.devices()[0])}
    cfg, fid = _config(a.layout)
    t = time.perf_counter()
    _, info = M.build(fid, cfg, matter="closure", jit=False, block_pairs=a.block_pairs,
                      block_pairs_los=a.block_pairs_los, remat_los=a.variant == "remat")
    th = jnp.asarray(np.asarray(M.theta0(fid, cfg)))
    r["build_s"] = time.perf_counter() - t
    b = next(b for b in info["blocks"] if b["name"] == a.block)
    r["rows"] = int(np.size(b["rows"]))
    f = jax.jit(b["f"])
    t = time.perf_counter()
    y = jax.block_until_ready(f(th))
    r["value_compile_s"] = time.perf_counter() - t
    r["value_s"] = _median_time(f, th)
    r["after_value"] = _peak_bytes()
    c = jnp.asarray(np.random.default_rng(0).standard_normal(np.shape(y)))
    g = jax.jit(jax.grad(lambda x: jnp.sum(c * b["f"](x))))
    try:
        t = time.perf_counter()
        gv = jax.block_until_ready(g(th))
        r["grad_compile_s"] = time.perf_counter() - t
        r["grad_s"] = _median_time(g, th)
        r["grad"] = np.asarray(gv).tolist()
        r["value_norm"] = float(jnp.linalg.norm(y))
    except Exception as e:  # noqa: BLE001  (out of memory)
        r["grad_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    r["peak"] = _peak_bytes()
    print(json.dumps(r), flush=True)


def cmd_ckpt(a):
    import subprocess

    t_start = time.time()
    out_path = RESULTS / f"41_timing_ckpt{a.layout}{a.out_tag}.json"
    st = json.loads(out_path.read_text()) if out_path.is_file() else {"layout": a.layout, "runs": {}}
    st.setdefault("blocks", _ckpt_blocks(a.layout, a.ckpt_blocks))
    st["variants"] = a.variants

    def save():
        out_path.write_text(json.dumps(st, indent=1) + "\n")

    save()
    for bn in st["blocks"]:
        for v in a.variants:
            key = f"{bn}|{v}"
            if key in st["runs"]:
                continue
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "ckpt-one", "--layout", a.layout,
                   "--block", bn, "--variant", v, "--block-pairs", str(a.block_pairs),
                   "--block-pairs-los", str(a.block_pairs_los)]
            t = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            r = json.loads(lines[-1]) if lines else {"error": f"exit {p.returncode}: {p.stderr[-600:]}"}
            r["wall_s"] = time.perf_counter() - t
            st["runs"][key] = r
            print(f"  {key}:", json.dumps({k: x for k, x in r.items() if k != "grad"}), flush=True)
            save()
    # remat against plain, where both gradients ran
    cmp = {}
    for bn in st["blocks"]:
        pl, rm = st["runs"].get(f"{bn}|plain", {}), st["runs"].get(f"{bn}|remat", {})
        c = {}
        if "grad" in pl and "grad" in rm:
            gp, gr = np.asarray(pl["grad"]), np.asarray(rm["grad"])
            c["grad_max_rel_diff"] = float(np.max(np.abs(gr - gp)) / np.max(np.abs(gp)))
            c["grad_time_ratio"] = rm["grad_s"] / pl["grad_s"]
            c["compile_time_ratio"] = rm["grad_compile_s"] / pl["grad_compile_s"]
        for v, r in (("plain", pl), ("remat", rm)):
            pk = (r.get("peak") or {}).get("peak_bytes_in_use")
            if pk:
                c[f"peak_gb_{v}"] = pk / 1e9
            c[f"grad_ok_{v}"] = "grad" in r
        cmp[bn] = c
    st["comparison"] = cmp
    save()
    print("  comparison:", json.dumps(cmp), flush=True)
    print(f"done in {time.time() - t_start:.1f} s")


def cmd_mgpu_one(a):
    """Child of ``mgpu``: ``--ndev`` devices, one build, thread and set of block programs each;
    prints one JSON line."""
    import concurrent.futures as cf

    import jax
    import jax.numpy as jnp
    from ggah_sens_study.joint import model as M

    _enable_cache(a.cache_dir)
    devs = jax.devices()[:a.ndev]
    D = len(devs)
    fz = json.loads((RESULTS / f"41_timing_fused{a.layout}.json").read_text())
    names = fz["selected"]
    parts = [names[i::D] for i in range(D)]
    cfg, fid = _config(a.layout)
    r = {"ndev": D, "devices": [str(d) for d in devs], "parts": parts,
         "env": {k: os.environ.get(k) for k in ("XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
                                                 "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR")}}

    def setup(i):
        """Build on device i and compile the gradients of its blocks; returns (fns, theta, times)."""
        d = devs[i]
        with jax.default_device(d):
            t0 = time.perf_counter()
            _, info = M.build(fid, cfg, matter="closure", jit=False, block_pairs=a.block_pairs,
                              block_pairs_los=a.block_pairs_los)
            th = jax.device_put(jnp.asarray(np.asarray(M.theta0(fid, cfg))), d)
            tb = time.perf_counter() - t0
            by = {b["name"]: b for b in info["blocks"]}
            fns, comp = [], {}
            for k, bn in enumerate(parts[i]):
                b = by[bn]
                shape = jax.eval_shape(b["f"], th).shape
                c = jax.device_put(jnp.asarray(np.random.default_rng(k).standard_normal(shape)), d)
                g = jax.jit(jax.grad(lambda x, f=b["f"], c=c: jnp.sum(c * f(x))))
                t = time.perf_counter()
                jax.block_until_ready(g(th))
                comp[bn] = time.perf_counter() - t
                fns.append(g)
        return fns, th, {"build_s": tb, "first_call_s": comp}

    def run(i, fns, th):
        """One pass over device i's blocks; returns (start, end) on the shared clock."""
        t = time.perf_counter()
        out = [g(th) for g in fns]  # dispatched without waiting, then awaited together
        jax.block_until_ready(out)
        return t, time.perf_counter()

    t = time.perf_counter()
    with cf.ThreadPoolExecutor(D) as ex:
        built = list(ex.map(setup, range(D)))
    r["setup_wall_s"] = time.perf_counter() - t
    r["per_device"] = [b[2] for b in built]
    for i in range(D):  # warm call, also the memory check
        run(i, built[i][0], built[i][1])
    # alone: each device in turn, nothing else running
    alone = []
    for i in range(D):
        ts = [run(i, built[i][0], built[i][1]) for _ in range(3)]
        alone.append(float(np.median([e - s for s, e in ts])))
    r["alone_s"] = alone
    # together: all threads started at once
    walls, busy = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(D) as ex:
            se = list(ex.map(lambda i: run(i, built[i][0], built[i][1]), range(D)))
        walls.append(time.perf_counter() - t0)
        busy.append([(s - t0, e - t0) for s, e in se])
    k = int(np.argsort(walls)[len(walls) // 2])
    r["together_wall_s"] = walls[k]
    r["together_intervals_s"] = busy[k]
    r["speedup_same_work"] = float(sum(alone) / walls[k])
    r["ideal_speedup"] = float(sum(alone) / max(alone))
    f8 = fz.get("fused", {}).get(str(len(names)), {})
    seq = fz.get("sequential", {})
    if len(seq) == len(names):
        r["sequential_blocks_grad_s"] = float(sum(v["grad_s"] for v in seq.values()))
        r["speedup_vs_sequential_one_device"] = r["sequential_blocks_grad_s"] / walls[k]
    if "grad_s" in f8:
        r["speedup_vs_fused_one_device"] = f8["grad_s"] / walls[k]
    try:
        r["peak_gb"] = [(d.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e9 for d in devs]
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(r), flush=True)


def cmd_mgpu(a):
    import subprocess

    t_start = time.time()
    out_path = RESULTS / f"41_timing_mgpu{a.layout}{a.out_tag}.json"
    st = json.loads(out_path.read_text()) if out_path.is_file() else {"layout": a.layout, "runs": {}}
    # count the GPUs without opening them: the children hold the devices, the parent never computes
    q = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    n = sum(ln.startswith("GPU ") for ln in q.stdout.splitlines()) or 1
    st["n_devices"] = n

    def save():
        out_path.write_text(json.dumps(st, indent=1) + "\n")

    Ds = [d for d in a.ndevs if d <= n]
    todo = [(str(D), D) for D in Ds] + ([(f"{max(Ds)}_cache", max(Ds))] if Ds else [])
    save()
    for key, D in todo:
        if key in st["runs"]:
            continue
        if st.get("running") == key:  # the process died in this stage (rerun by cc/gpu.sbatch)
            st["runs"][key] = {"error": "process died"}
            st.pop("running")
            save()
            continue
        st["running"] = key
        save()
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "mgpu-one", "--layout", a.layout,
               "--ndev", str(D), "--cache-dir", str(a.cache_dir), "--block-pairs", str(a.block_pairs),
               "--block-pairs-los", str(a.block_pairs_los)]
        t = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        r = json.loads(lines[-1]) if lines else {"error": f"exit {p.returncode}: {p.stderr[-800:]}"}
        r["wall_s"] = time.perf_counter() - t
        st["runs"][key] = r
        st.pop("running", None)
        print(f"  {key}:", json.dumps({k: v for k, v in r.items() if k not in ("parts", "per_device")}), flush=True)
        save()
    c0, c1 = st["runs"].get(str(max(Ds)) if Ds else "", {}), st["runs"].get(f"{max(Ds)}_cache" if Ds else "", {})
    if "per_device" in c0 and "per_device" in c1:
        cold = sum(sum(x["first_call_s"].values()) for x in c0["per_device"])
        warm = sum(sum(x["first_call_s"].values()) for x in c1["per_device"])
        st["cache"] = {"first_calls_cold_s": cold, "first_calls_cached_s": warm, "ratio": warm / cold}
    ok = {k: v for k, v in st["runs"].items() if "speedup_same_work" in v and not k.endswith("_cache")}
    if ok:
        fz = json.loads((RESULTS / f"41_timing_fused{a.layout}.json").read_text())
        nb = fz["n_blocks"]
        G = len(fz["selected"])
        st["extrapolation"] = {k: {"grad_s_all_blocks": v["together_wall_s"] * nb / G,
                                   "rule": f"wall time for the {G} blocks x n_blocks / {G}"} for k, v in ok.items()}
    save()
    print(f"done in {time.time() - t_start:.1f} s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("blocks", "nuts", "summary", "fused", "cache-probe", "ckpt", "ckpt-one",
                                       "mgpu", "mgpu-one"))
    ap.add_argument("--layout", default="_s0")
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--block-pairs", type=int, default=40)
    ap.add_argument("--block-pairs-los", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--ess", type=int, default=1000)
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[4, 8], help="fused: blocks per program")
    ap.add_argument("--cache-dir", type=Path, default=DATA / "jax_cache_41", help="fused: persistent compilation cache")
    ap.add_argument("--ckpt-blocks", nargs="+", default=None, help="ckpt: block names (default: see the docstring)")
    ap.add_argument("--variants", nargs="+", default=["remat", "plain"], choices=("plain", "remat"),
                    help="ckpt: the routes, in this order")
    ap.add_argument("--out-tag", default="", help="ckpt, mgpu: suffix of the output JSON (e.g. _v100, _h100)")
    ap.add_argument("--block", default=None, help="ckpt-one: the block")
    ap.add_argument("--variant", default="plain", choices=("plain", "remat"), help="ckpt-one: the route")
    ap.add_argument("--ndevs", type=int, nargs="+", default=[2, 4], help="mgpu: device counts")
    ap.add_argument("--ndev", type=int, default=1, help="mgpu-one: device count")
    a = ap.parse_args(argv)
    {"blocks": cmd_blocks, "nuts": cmd_nuts, "summary": cmd_summary, "fused": cmd_fused,
     "cache-probe": cmd_cache_probe, "ckpt": cmd_ckpt, "ckpt-one": cmd_ckpt_one,
     "mgpu": cmd_mgpu, "mgpu-one": cmd_mgpu_one}[a.cmd](a)


if __name__ == "__main__":
    main()
