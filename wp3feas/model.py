"""The ggah_mod 1.7.0 default model of public_2026, through ggah_sens_study's joint.model.build.

Run with the v1.7.0 stack of ggah_sens_study (branch rerun-v170):
``source ~/software/ggah_sens_study_wt_v170/local/stack_v170.sh cpu`` and
``PYTHONPATH=~/software/ggah_sens_study_wt_v170:$PYTHONPATH``.

The build evaluates every pair of its tracers on its Limber nodes; :func:`projection`
gives the sparse matrix P from that output to the rows of the vector it models
(the C_ℓ and shear C_ℓ blocks whose two tracers have a model counterpart), linear in ln ℓ.
"""

from pathlib import Path

import numpy as np

from .nz import build_name, write_nz_file
from .provenance import DATA

#: the layout of public_2026: three shells (the third holds only the AGN with
#: L_X >= 10^43.5 erg/s), every LS11 sample photometric, the AGN as cumulative
#: thresholds (open bins), the 19 bands of 0.5-2 keV, 13 source bins
PUBLIC = dict(
    shells=((0.1, 0.2), (0.2, 0.3), (0.3, 0.6)),
    mstar_edges=(10.0, 10.5, 11.0, 11.5, 11.75),
    colours=("red", "blue"),
    agn_bins=((42.5, None), (43.0, None), (43.5, None), (44.0, None)),
    spec_mstar_min=((13.0, 13.0),) * 3,
    n_src_bins=13,
    sources="stage4",
    xray_bands="x19",
    ell_min=2.0,
    ell_max=2048.0,
    n_bands=23,
)
NZ_FILE = DATA / "public_2026_nz.npz"


def public_config(nz_file=NZ_FILE, **kw):
    """The JointConfig of public_2026 over the LS10 footprints, with the measured n(z)."""
    from ggah_sens_study.joint.survey import JointConfig

    nz_file = Path(nz_file)
    if not nz_file.exists():
        nz_file.parent.mkdir(parents=True, exist_ok=True)
        write_nz_file(nz_file)
    return JointConfig.for_survey("ls10", **{**PUBLIC, **kw}, nz_file=str(nz_file))


GGAH_MOD_VERSION = "1.7.0"


def check_stack():
    """Refuse any stack but ggah_mod 1.7.0 with the eps_dpl AGN closure; return its description."""
    import ggah_mod
    from ggah_mod.sectors.energetics import EnergeticsParams
    from ggah_sens_study.joint import model as M

    e = EnergeticsParams()
    names = list(M.names(None))
    if ggah_mod.__version__ != GGAH_MOD_VERSION:
        raise RuntimeError(f"ggah_mod {ggah_mod.__version__} loaded from {ggah_mod.__file__}; "
                           f"source ggah_sens_study_wt_v170/local/stack_v170.sh for {GGAH_MOD_VERSION}")
    if e.agn_energy != "eps_dpl" or "log10_eps_agn_p" not in names or "log10_eps_agn" in names:
        raise RuntimeError(f"AGN closure {e.agn_energy!r} with {[n for n in names if 'eps' in n]}: "
                           "the 1.7.0 eps_dpl closure needs ggah_sens_study's rerun-v170 branch")
    return {"ggah_mod": ggah_mod.__version__, "ggah_mod_path": ggah_mod.__file__,
            "agn_energy": e.agn_energy, "eps_sn": float(e.eps_sn),
            "log10_eps_agn_p": float(e.log10_eps_agn_p), "log10_m_eps_p": float(e.log10_m_eps_p),
            "a_eps_lo": float(e.a_eps_lo), "a_eps_hi": float(e.a_eps_hi)}


def build_public(cfg=None, *, matter="closure", jit=True):
    """``(f, info, theta0, fid)`` of the build at the v1.7.0 defaults (``eps_dpl`` closure)."""
    from ggah_sens_study.fiducial import load_fiducial
    from ggah_sens_study.joint import model as M

    check_stack()
    cfg = cfg or public_config()
    fid = load_fiducial()
    f, info = M.build(fid, cfg, matter=matter, jit=jit)
    return f, info, np.asarray(M.theta0(fid, cfg)), fid


def projection(vector, info):
    """``(P, rows)``: the sparse matrix (scipy CSR) from the build's output to the modelled
    rows ``rows`` of ``vector``, and those rows (indices into the vector)."""
    from scipy.sparse import csr_matrix

    n_disc = len(info["discrete"])
    ell = np.asarray(info["ell"], float)
    n_ell = ell.size
    pair_of = {}
    for i, (a, b) in enumerate(info["pairs"]):
        pair_of[(a, b)] = pair_of[(b, a)] = i
    lnl = np.log(ell)
    r_i, c_i, w_i, rows = [], [], [], []
    for (stat, a, b, survey, part), idx in vector.blocks():
        if stat not in ("cl", "cl_sh"):
            continue
        A, B = build_name(a), build_name(b or a)
        if A is None or B is None or (A, B) not in pair_of:
            continue
        base = n_disc + pair_of[(A, B)] * n_ell
        x = np.log(np.clip(vector.index["x"][idx], ell[0], ell[-1]))
        j = np.clip(np.searchsorted(lnl, x) - 1, 0, n_ell - 2)
        t = (x - lnl[j]) / (lnl[j + 1] - lnl[j])
        for k, r in enumerate(idx):
            n = len(rows)
            rows.append(r)
            r_i += [n, n]
            c_i += [base + j[k], base + j[k] + 1]
            w_i += [1.0 - t[k], t[k]]
    n_out = n_disc + len(info["pairs"]) * n_ell
    P = csr_matrix((w_i, (r_i, c_i)), shape=(len(rows), n_out))
    return P, np.asarray(rows, int)
