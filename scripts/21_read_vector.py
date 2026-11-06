"""Part D, step 1: inventory of public_2026 (tier_ls11_v0.h5) and its map onto the S5_public_2031 layout.

Writes results/21_read_vector.json: rows and blocks per statistic, per slice and per tracer kind,
the blocks a jackknife covariance can invert (p < n_jk - 2), and the S5_public_2031 tracer pairs
the C_ℓ blocks cover.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wp3feas.provenance import SENS, Record  # noqa: E402
from wp3feas.vector import VECTOR, load_vector, parse_tracer, slot_2031  # noqa: E402

RUN = SENS / "shells_2031" / "S5_public_2031"


def main():
    rec = Record(__file__)
    rec.add_input(VECTOR)
    rec.add_input(RUN / "samples.json")
    rec.add_input(RUN / "config.json")
    v = load_vector(VECTOR)
    blocks = v.blocks()
    smp = json.loads((RUN / "samples.json").read_text())
    n_src = json.loads((RUN / "config.json").read_text())["n_src_bins"]
    public = set(smp["tracers"]) | set(smp["x_maps"]) | {"y"} | {f"ks{i}" for i in range(n_src)}

    per_stat = defaultdict(lambda: {"blocks": 0, "rows": 0, "finite_rows": 0, "jk_invertible_blocks": 0})
    per_slice = Counter()
    per_pair_kind = Counter()
    rows_max = 0
    covered = set()
    unmapped = set()
    for (stat, a, b, survey, part), rows in blocks:
        s = per_stat[stat]
        fin = np.isfinite(v.data[rows]) & np.all(np.isfinite(v.jk[:, rows]), axis=0)
        s["blocks"] += 1
        s["rows"] += len(rows)
        s["finite_rows"] += int(fin.sum())
        s["jk_invertible_blocks"] += int(fin.sum() < v.n_jk - 2)
        rows_max = max(rows_max, len(rows))
        ta, tb = parse_tracer(a), parse_tracer(b) if b else parse_tracer(a)
        sl = sorted({t["slice"] for t in (ta, tb) if t["slice"] is not None})
        per_slice["+".join(map(str, sl)) or "none"] += len(rows)
        per_pair_kind[f"{stat}:{'x'.join(sorted((ta['kind'] or '?', tb['kind'] or '?')))}"] += 1
        if stat in ("cl", "cl_sh"):
            sa, sb = slot_2031(a), slot_2031(b or a)
            for sl in (sa, sb):
                hit = [t for t in sl if t in public]
                covered.update(hit)
                if sl and not hit:
                    unmapped.update(sl)
            for x in sa:
                for y in sb:
                    if x in public and y in public:
                        covered.add(tuple(sorted((x, y))))

    pairs = {c for c in covered if isinstance(c, tuple)}
    tracers = {c for c in covered if isinstance(c, str)}
    n_pub = len(public)
    rec.data.update(
        n_rows=int(len(v.data)),
        n_finite=int(np.isfinite(v.data).sum()),
        n_jk=int(v.n_jk),
        n_blocks=len(blocks),
        rows_per_block_max=rows_max,
        per_stat=dict(per_stat),
        rows_per_slice=dict(per_slice),
        blocks_per_pair_kind=dict(sorted(per_pair_kind.items())),
        s5_public_2031={
            "tracers": n_pub,
            "tracers_covered": len(tracers),
            "pairs": n_pub * (n_pub + 1) // 2,
            "pairs_covered": len(pairs),
            "tracers_not_in_public_layout": sorted(unmapped),
        },
    )
    print(json.dumps({k: rec.data[k] for k in ("n_rows", "n_blocks", "per_stat", "rows_per_slice", "s5_public_2031")}, indent=1))
    print(rec.write())


if __name__ == "__main__":
    main()
