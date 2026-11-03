"""Provenance for every result JSON: input files, git hash, wall time."""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = Path(os.environ.get("WP3FEAS_DATA", Path.home() / "data" / "cog-prop-wp3-feasibility"))
# The forecast runs of ggah_sens_study that this study reads and never writes.
SENS = Path(os.environ.get("WP3FEAS_SENS", Path.home() / "data" / "ggah_sens_study_results_v1.6.0"))


def git_hash():
    """Hash of HEAD, with '-dirty' when the work tree has changes."""
    h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return h + ("-dirty" if dirty else "")


class Record:
    """Collects inputs and results, and writes them as one JSON under results/."""

    def __init__(self, script):
        self.script = Path(script).name
        self.t0 = time.time()
        self.inputs = []
        self.data = {}

    def add_input(self, path):
        self.inputs.append(str(path))

    def write(self, name=None):
        RESULTS.mkdir(exist_ok=True)
        name = name or Path(self.script).stem
        out = {
            "script": self.script,
            "git_hash": git_hash(),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_time_s": round(time.time() - self.t0, 3),
            "inputs": self.inputs,
            **self.data,
        }
        path = RESULTS / f"{name}.json"
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")
        return path
