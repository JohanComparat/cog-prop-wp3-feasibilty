"""Self-describing product files: every array with its unit and meaning, the provenance, a format version.

A product is an .npz file (numpy, no pickle) holding its arrays and three JSON strings:
  _format    {"product": <name>, "version": <int>, "reader": "wp3feas.store.load"}
  _meta      the run, output, cosmology, choices and provenance (git hash, date, code module)
  _columns   for each array: {"unit": ..., "description": ...}
so that a user without this repository can read it with numpy alone:

    f = np.load("bnl_048.npz")
    meta, columns = json.loads(str(f["_meta"])), json.loads(str(f["_columns"]))

and with it, load() returns the arrays, the metadata and the descriptions, and checks that every
array is described.  docs/13_data_products.md lists every product, its arrays and units.
"""

import json
import time
from pathlib import Path

import numpy as np

from wp3feas.provenance import git_hash


def save(path, product, version, arrays, columns, meta, compress=True):
    """Write `arrays` to `path` with their `columns` descriptions; every array must be described.

    columns: {name: (unit, description)}; a name ending in "*" describes every array with that
    prefix, and one starting with "*" every array with that suffix (the same quantity at three
    mass definitions, "vir_P_hh", "200m_P_hh", "500c_P_hh", is described once as "*_P_hh").
    """
    head = header(product, version, list(arrays), columns, meta)
    (np.savez_compressed if compress else np.savez)(path, **arrays, **head)
    return Path(path)


def header(product, version, names, columns, meta):
    """The three header arrays of save(), for a writer that keeps its own np.savez call and keys.

    names are the keys of the arrays written; every one must be described in columns.
    """
    missing = [k for k in names if describe(k, columns) is None]
    if missing:
        raise ValueError(f"{product}: arrays without a description: {missing}")
    meta = {**meta, "git_hash": git_hash(), "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return {"_format": json.dumps({"product": product, "version": version, "reader": "wp3feas.store.load"}),
            "_meta": json.dumps(meta, default=_json_default),
            "_columns": json.dumps({k: dict(zip(("unit", "description"), describe(k, columns))) for k in names})}


def describe(name, columns):
    if name in columns:
        return columns[name]
    for k, v in columns.items():
        if (k.endswith("*") and name.startswith(k[:-1])) or (k.startswith("*") and name.endswith(k[1:])):
            return v
    return None


def load(path):
    """(arrays, meta, columns, format) of a product written by save()."""
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files if not k.startswith("_")}
        fmt = json.loads(str(f["_format"]))
        meta = json.loads(str(f["_meta"]))
        cols = json.loads(str(f["_columns"]))
    return arrays, meta, cols, fmt


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)
