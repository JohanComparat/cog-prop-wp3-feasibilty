"""Seed a MAP chain of scripts/26_map_fit.py from an iterate of another chain.

    python cc/seed_chain.py SRC_ITER_DIR DST_CHAIN_DIR [AGNBAND_DIR]

Copies SRC_ITER_DIR (blocks with D and J, theta.json) to DST_CHAIN_DIR/it<k>, replaces the D of
the blocks found in AGNBAND_DIR (scripts/25_scale_cuts.py agneval, the same theta) and writes a
placeholder objective (chi2 1e30) for the base of that iterate, so that the chain's first step
accepts it<k> as its start: a base objective computed under another model is not comparable.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
band = Path(sys.argv[3]) if len(sys.argv) > 3 else None
k = int(src.name[2:])
it = dst / f"it{k}"
if it.exists():
    sys.exit(f"{it} exists")
shutil.copytree(src, it)
if band:
    for f in sorted(band.glob("*.npz")):
        z0, z1 = dict(np.load(it / f.name)), np.load(f)
        assert (z0["rows"] == z1["rows"]).all(), f.name
        z0["D"] = z1["D"]
        np.savez(it / f.name, **z0)
        print("replaced", f.name)
h = json.loads((it / "theta.json").read_text())
base = int(h.get("base", k - 1))
ob = json.loads((src.parent / f"it{base}" / "objective.json").read_text())
ob.update(chi2=1e30, prior=0.0, placeholder=True, note=f"placeholder: the chain starts at it{k}, seeded from {src}")
(dst / f"it{base}").mkdir(parents=True, exist_ok=True)
(dst / f"it{base}" / "objective.json").write_text(json.dumps(ob))
print(f"seeded {it} (base it{base} placeholder)")
