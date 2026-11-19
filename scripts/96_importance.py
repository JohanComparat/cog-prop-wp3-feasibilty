r"""Part D: the posterior of the slice-0 fit by importance sampling of its Laplace approximation.

    sbatch --export=ALL,<the chain's WP3FEAS_*> cc/cpu.sbatch scripts/96_importance.py draw --it 21
    sbatch --array=0-19 --export=... cc/gpu.sbatch scripts/96_importance.py eval --it 21 --nshard 20
    sbatch --export=... cc/cpu.sbatch scripts/96_importance.py chi2 --it 21
    python scripts/96_importance.py combine --it 21          (laptop, after cc/fetch.sh and a copy of the run directory)

``draw`` takes ``N`` points from a proposal around the iterate of script 26 (chain
``WP3FEAS_MAP_TAG``): a Student-t with ``--nu`` degrees of freedom (default 5) and the Laplace
covariance ``laplace_cov.npz`` (or ``--cov``, any npz with ``names``, ``theta``, ``free``, ``C``),
or, with ``--nu 0``, a Gaussian with that covariance times ``--inflate``² (default 1.2).
Only the free parameters move; the cosmology of the chain stays fixed unless ``--cov`` frees it.

``eval`` evaluates the exact model (value only) at every draw, block by block: a task (``--nshard``
tasks, the task index from ``SLURM_ARRAY_TASK_ID`` or ``--task``) compiles each of its blocks
once and loops over all the draws, so no block is compiled twice across the array; it resumes
block by block (``cc/gpu.sbatch`` reruns it until "done in").

``chi2`` assembles the blocks and gives per draw the chi^2 of script 26 (same rows, same
block-diagonal jackknife covariance with the Hartlap factor, the chain's mask, Poisson term and
band) and its Gaussian prior term; a draw outside the parameter ranges or with a non-finite model
on a used row has zero weight.

``combine`` gives the importance weights :math:`w \propto p/q` with ``p = exp(-(chi^2 + prior)/2)``,
smoothed by Pareto (PSIS, Vehtari et al. 2024) with their shape :math:`\hat k`
(:math:`\hat k > 0.7`: the proposal is not good enough), the effective sample size, the weighted
16/50/84 per cent of every free parameter against the Laplace ones, the S(k) band of the weighted
draws (first order with the Jacobian of script 91, at z = 0.15), and corner plots per parameter
group in ``WP3FEAS_DATA/96_corner_<group>.pdf``.
It writes ``results/96_importance<layout><tag>_it<it>.json``.
"""

import argparse
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from wp3feas.provenance import DATA, Record  # noqa: E402

GROUPS = {
    "galaxy": ("lg_m1h", "lg_m0star", "beta", "delta", "gamma", "sigma_lnmstar", "eta", "fc", "bsat", "beta_sat",
               "bcut", "beta_cut", "alpha_sat", "lg_mh_qc", "mu_c", "lg_mh_qs", "mu_s"),
    "agn": ("mu_bh", "al_bh", "sig_bh", "log10_lstar", "log10_lstar_mslope", "delta1", "delta2", "log10_ferdf",
            "log10_ferdf_mslope", "gamma_x", "f_duty_sat", "f_faint_norm", "l_blend_0", "l_blend_z_amp",
            "l_blend_m_amp", "log10_eps_agn_p", "eta_ej"),
    "gas": ("log10_ne_anchor", "alpha_in_n", "alpha_tr_n", "alpha_out_n", "alpha_in_n_var", "alpha_tr_n_var",
            "alpha_out_n_var", "beta_n", "gamma_n", "log10_pe_anchor", "alpha_in_p", "alpha_tr_p", "alpha_out_p",
            "alpha_in_p_var", "alpha_tr_p_var", "alpha_out_var", "beta_p", "gamma_p", "z_anchor", "alpha_in_z",
            "alpha_tr_z", "alpha_out_z", "log10_conc_ratio", "log10_conc_ratio_var", "sigma_scatter", "nh_over_ne",
            "gamma_n_far", "gamma_p_far"),
    "nuisance": ("a_ia", "beta_m", "dzp@0", "szp@0") + tuple(f"dz_s{i}" for i in range(13)),
    "cosmology": ("Omega_m", "ln10A_s", "h", "n_s", "Omega_b"),
}


def _s26(layout):
    S22 = importlib.import_module("22_fisher_public")
    S22.set_layout(layout)
    S26 = importlib.import_module("26_map_fit")
    return S22, S26


def run_dir(a):
    return DATA / f"96_importance{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', '')}_it{a.it}"


def cov_path(a, S26):
    return Path(a.cov) if a.cov else S26.it_dir(a.it) / "laplace_cov.npz"


def cmd_draw(a):
    S22, S26 = _s26(a.layout)
    L = np.load(cov_path(a, S26))
    names, x, f, C = list(L["names"]), np.asarray(L["theta"]), np.asarray(L["free"]), np.asarray(L["C"])
    meta = json.loads((S22.OUT / "meta.json").read_text())
    assert names == meta["names"], "the covariance's parameters differ from the layout's"
    th0 = np.asarray(meta["theta0"])
    sp = S26.prior_sigma(names)
    lo, hi = S26.bounds(names, th0, sp)
    lo, hi = S26.shell_limits(names, x, lo, hi)
    rng = np.random.default_rng(a.seed)
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    w = np.clip(w, 1e-300, None)
    Ls = V * np.sqrt(w)
    z = rng.standard_normal((a.n, f.size))
    p = f.size
    if a.nu > 0:
        u = rng.chisquare(a.nu, a.n)
        dx = (z * np.sqrt(a.nu / u)[:, None]) @ Ls.T
        m = np.einsum("ij,ij->i", z * np.sqrt(a.nu / u)[:, None], z * np.sqrt(a.nu / u)[:, None])
        logq = -0.5 * (a.nu + p) * np.log1p(m / a.nu)
        kind = f"student-t nu={a.nu}"
    else:
        dx = a.inflate * z @ Ls.T
        m = np.einsum("ij,ij->i", z, z)
        logq = -0.5 * m
        kind = f"gaussian x{a.inflate}"
    d = np.tile(x, (a.n, 1))
    d[:, f] += dx
    inside = np.all((d[:, f] >= lo[f]) & (d[:, f] <= hi[f]), axis=1)
    out = run_dir(a)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "draws.npz", names=np.array(names), centre=x, theta0=th0, free=f, C=C, draws=d, logq=logq,
             inside=inside, kind=kind)
    print(out / "draws.npz", a.n, "draws,", kind, f"{inside.mean():.3f} inside the ranges")
    print("done in draw")


def _task(a):
    if a.task is not None:
        return a.task
    return int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


def cmd_eval(a):
    import jax
    import jax.numpy as jnp
    from wp3feas.model import build_public, public_config

    S22, S26 = _s26(a.layout)
    rd = run_dir(a)
    D = np.load(rd / "draws.npz")
    draws, inside = D["draws"], D["inside"]
    ti, tn = _task(a), a.nshard
    bd = rd / "blocks"
    bd.mkdir(exist_ok=True)
    t0 = time.time()
    _, info, _, _ = build_public(public_config(**S22.LAYOUTS[S22.LAYOUT]), jit=False, block_pairs=a.block_pairs,
                                 block_pairs_los=a.block_pairs_los)
    meta = json.loads((S22.OUT / "meta.json").read_text())
    assert [b["name"] for b in info["blocks"]] == meta["blocks"], "block layout differs from script 22"
    print(f"task {ti}/{tn}: build {time.time() - t0:.0f} s", flush=True)
    done = 0
    for ib, b in enumerate(info["blocks"]):
        path = bd / f"{b['name']}.npz"
        if ib % tn != ti or path.is_file():
            continue
        if a.max_blocks and done >= a.max_blocks:
            print("stopping: --max-blocks reached", flush=True)
            return
        done += 1
        t1 = time.time()
        g = jax.jit(b["f"])
        y0 = np.asarray(g(jnp.asarray(draws[0])))
        tc = time.time() - t1
        Y = np.full((len(draws), y0.size), np.nan)
        Y[0] = y0
        for i in range(1, len(draws)):
            if inside[i]:
                Y[i] = np.asarray(g(jnp.asarray(draws[i])))
        np.savez(path, rows=np.asarray(b["rows"]), D=Y)
        print(f"  {b['name']}: compile {tc:.0f} s, {len(draws)} draws in {time.time() - t1 - tc:.0f} s", flush=True)
        del g
        jax.clear_caches()
        gc.collect()
    missing = [b["name"] for ib, b in enumerate(info["blocks"]) if ib % tn == ti and not (bd / f"{b['name']}.npz").is_file()]
    if not missing:
        print(f"done in {time.time() - t0:.1f} s")


def cmd_chi2(a):
    S22, S26 = _s26(a.layout)
    rec = Record(__file__)
    rd = run_dir(a)
    Dr = np.load(rd / "draws.npz")
    draws, inside = Dr["draws"], Dr["inside"]
    meta, Dmap, _ = S22._assemble(S26.it_dir(a.it), jac=False)
    P, r0, blocks = S26.system(Dmap)
    used = np.unique(np.concatenate([s for s, _ in blocks]))
    names = meta["names"]
    th0 = np.asarray(meta["theta0"])
    sp = S26.prior_sigma(names)
    pr = np.where(np.isfinite(sp), 1.0 / sp ** 2, 0.0)
    n = len(draws)
    chi2 = np.full(n, np.inf)
    n_rows = len(meta["discrete"]) + len(meta["pairs"]) * len(meta["ell"])
    for c0 in range(0, n, a.chunk):
        c1 = min(n, c0 + a.chunk)
        Dc = np.zeros((c1 - c0, n_rows))
        for name in meta["blocks"]:
            with np.load(rd / "blocks" / f"{name}.npz") as z:
                Dc[:, z["rows"]] += z["D"][c0:c1]
        M = np.asarray((P @ (Dc - Dmap).T).T)  # the model change on the projected rows
        for j in range(c1 - c0):
            r = r0 - M[j]
            if not inside[c0 + j] or not np.all(np.isfinite(r[used])):
                continue
            chi2[c0 + j] = float(sum(r[s] @ iC @ r[s] for s, iC in blocks))
        print(f"draws {c0}-{c1}: chi2 median {np.median(chi2[c0:c1]):.1f}", flush=True)
    prior = np.sum(pr * (draws - th0) ** 2, axis=1)
    chi2_map = float(sum(r0[s] @ iC @ r0[s] for s, iC in blocks))
    np.savez(rd / "chi2.npz", chi2=chi2, prior=prior, chi2_map=chi2_map,
             prior_map=float(np.sum(pr * (np.asarray(Dr["centre"]) - th0) ** 2)),
             n_rows=int(used.size))
    rec.data.update(it=a.it, layout=a.layout, n=int(n), finite=int(np.isfinite(chi2).sum()), chi2_map=chi2_map,
                    n_rows=int(used.size))
    print(rec.write(f"96_importance_chi2{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', '')}_it{a.it}"))
    print("done in chi2")


def gpdfit(x):
    """Shape and scale of the generalised Pareto fitted to exceedances ``x`` (Zhang & Stephens 2009,
    with the weakly informative prior of Vehtari et al.)."""
    x = np.sort(x)
    n = x.size
    prior = 3
    m = 30 + int(np.sqrt(n))
    bs = 1 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5))
    bs /= prior * x[int(n / 4 + 0.5) - 1]
    bs += 1 / x[-1]
    ks = -bs
    temp = ks[:, None] * x
    ks = np.log1p(temp).mean(axis=1)
    L = n * (np.log(-(bs / ks)) - ks - 1)
    w = 1 / np.exp(L - L[:, None]).sum(axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, bs = w[keep], bs[keep]
    w /= w.sum()
    b = np.sum(bs * w)
    k = np.log1p(-b * x).mean()
    sigma = -k / b
    k = k * n / (n + 10) + 10 * 0.5 / (n + 10)
    return k, sigma


def psis(logw):
    """Pareto-smoothed log weights and k-hat (Vehtari et al. 2024)."""
    lw = logw - np.max(logw)
    n = lw.size
    M = int(min(0.2 * n, 3 * np.sqrt(n)))
    order = np.argsort(lw)
    cut = lw[order[-M - 1]]
    tail = order[-M:]
    x = np.exp(lw[tail]) - np.exp(cut)
    k, sigma = gpdfit(x)
    if np.isfinite(k):
        p = (np.arange(1, M + 1) - 0.5) / M
        q = np.exp(cut) + sigma / k * ((1 - p) ** (-k) - 1) if k != 0 else np.exp(cut) - sigma * np.log(1 - p)
        lw = lw.copy()
        lw[tail] = np.minimum(np.log(q), 0.0)
    lw -= np.log(np.sum(np.exp(lw)))
    return lw, float(k)


def wquant(v, w, qs):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    c /= c[-1]
    return [float(np.interp(q, c, v[o])) for q in qs]


def cmd_combine(a):
    rec = Record(__file__)
    rd = run_dir(a) if a.local_dir is None else Path(a.local_dir)
    Dr, X = np.load(rd / "draws.npz"), np.load(rd / "chi2.npz")
    names, f, x, C = list(Dr["names"]), Dr["free"], Dr["centre"], Dr["C"]
    draws, logq = Dr["draws"], Dr["logq"]
    logp = -0.5 * (X["chi2"] + X["prior"])
    ok = np.isfinite(logp)
    lw, khat = psis(logp[ok] - logq[ok])
    w = np.exp(lw)
    ess = float(1 / np.sum(w ** 2))
    d = draws[ok]
    par = {}
    for j, i in enumerate(f):
        n_ = names[i]
        q = wquant(d[:, i], w, (0.16, 0.5, 0.84))
        s_l = float(np.sqrt(C[j, j]))
        par[n_] = {"map": float(x[i]), "laplace_sigma": s_l, "is_q16_q50_q84": q,
                   "is_mean": float(np.sum(w * d[:, i])), "is_sigma": float(np.sqrt(np.sum(w * (d[:, i] - np.sum(w * d[:, i])) ** 2))),
                   "shift_in_laplace_sigma": float((q[1] - x[i]) / s_l), "width_ratio": float((q[2] - q[0]) / (2 * s_l))}
    # S(k) of the weighted draws, first order with script 91's Jacobian at the iterate
    sk = None
    jp = DATA / f"91_sk_band{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', '')}_it{a.it}.npz"
    if jp.is_file():
        J = np.load(jp)
        if list(J["names"]) == names:
            iz = int(np.argmin(np.abs(np.asarray(J["z"]) - 0.15)))
            S = J["S"][iz] + (d - x) @ J["G"][iz].T
            k = J["k"]
            sk = {"z": float(J["z"][iz]), "k": [], "S_map": [], "q16_q50_q84": [], "laplace_sigma": []}
            Gf = J["G"][iz][:, f]
            sl = np.sqrt(np.clip(np.einsum("ki,ij,kj->k", Gf, C, Gf), 0, None))
            for kk in (0.3, 1.0, 3.0, 10.0):
                i = int(np.argmin(np.abs(np.log(k / kk))))
                sk["k"].append(float(k[i]))
                sk["S_map"].append(float(J["S"][iz][i]))
                sk["q16_q50_q84"].append(wquant(S[:, i], w, (0.16, 0.5, 0.84)))
                sk["laplace_sigma"].append(float(sl[i]))
    # corner plots per group
    plots = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import corner
        free_names = [names[i] for i in f]
        for g, members in GROUPS.items():
            sel = [names.index(m) for m in members if m in free_names]
            if len(sel) < 2:
                continue
            fig = corner.corner(d[:, sel], weights=w, labels=[names[i] for i in sel], truths=x[sel],
                                quantiles=(0.16, 0.5, 0.84), smooth=1.0, plot_datapoints=False)
            p = DATA / f"96_corner_{g}{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', '')}_it{a.it}.pdf"
            fig.savefig(p)
            plots[g] = str(p)
    except Exception as e:  # the plots are a convenience
        plots["error"] = repr(e)
    verdict = ("the Laplace proposal is adequate" if khat <= 0.5 else
               "usable, with care" if khat <= 0.7 else
               "k-hat > 0.7: the Gaussian is not good enough; fall back to a surrogate or a sampler")
    worst = sorted(par.items(), key=lambda t: -abs(t[1]["shift_in_laplace_sigma"]))[:8]
    rec.inputs += [str(rd / "draws.npz"), str(rd / "chi2.npz")]
    rec.data.update(it=a.it, layout=a.layout, map_tag=os.environ.get("WP3FEAS_MAP_TAG", ""),
                    proposal=str(Dr["kind"]), n=int(len(draws)), n_finite=int(ok.sum()), khat=khat, ess=ess,
                    verdict=verdict, chi2_map=float(X["chi2_map"]), n_rows=int(X["n_rows"]),
                    chi2_draws_q16_q50_q84=np.percentile(X["chi2"][ok], [16, 50, 84]).tolist(),
                    largest_shifts={n_: v["shift_in_laplace_sigma"] for n_, v in worst},
                    parameters=par, sk_band=sk, corner=plots)
    print(f"k-hat {khat:.2f}, ESS {ess:.0f} of {ok.sum()} ({verdict})")
    print(rec.write(f"96_importance{a.layout}{os.environ.get('WP3FEAS_MAP_TAG', '')}_it{a.it}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("draw", "eval", "chi2", "combine"))
    ap.add_argument("--it", type=int, default=21)
    ap.add_argument("--layout", default="_s0")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--nu", type=float, default=5.0, help="draw: Student-t degrees of freedom (0: Gaussian)")
    ap.add_argument("--inflate", type=float, default=1.2, help="draw, Gaussian: scale of the covariance's square root")
    ap.add_argument("--seed", type=int, default=96)
    ap.add_argument("--cov", default=None, help="draw: an npz with names, theta, free, C (default: the iterate's laplace_cov.npz)")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--block-pairs", type=int, default=40)
    ap.add_argument("--block-pairs-los", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=200, help="chi2: draws assembled at a time")
    ap.add_argument("--local-dir", default=None, help="combine: where draws.npz and chi2.npz are")
    a = ap.parse_args(argv)
    {"draw": cmd_draw, "eval": cmd_eval, "chi2": cmd_chi2, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    main()
