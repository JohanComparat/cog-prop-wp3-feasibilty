import h5py
import numpy as np

from wp3feas.vector import hartlap, jk_cov, load_vector, parse_tracer, slot_2031


def test_parse_and_slot():
    t = parse_tracer("LS11_RED_10.50_Mstar_11.00_0.20_z_0.30_N_1749793")
    assert t["kind"] == "gal" and t["slice"] == 1 and t["mstar"] == (10.5, 11.0)
    assert slot_2031("LS11_RED_10.50_Mstar_11.00_0.20_z_0.30_N_1749793") == ["gr10.5@1", "gr10.5p@1"]
    assert slot_2031("ERO1_AGN_LX_44.00_0.30_z_0.60_N_0009843") == ["a44@2", "a44@3", "a44@4"]
    assert parse_tracer("EVT_0500_0600_PSMASK")["psmask"]
    assert slot_2031("KIDS_bin4") == ["ks4"]


def test_jk_cov_matches_numpy():
    rng = np.random.default_rng(1)
    jk = rng.normal(size=(100, 5))
    ref = 99 / 100 * np.cov(jk.T, bias=True) * 100
    assert np.allclose(jk_cov(jk), ref)
    assert hartlap(100, 98) == 0 and np.isclose(hartlap(100, 10), 88 / 99)


def test_roundtrip(tmp_path):
    p = tmp_path / "v.h5"
    dt = [("stat", "S8"), ("a", "S48"), ("b", "S48"), ("survey", "S8"), ("part", "S8"), ("x", "<f8")]
    ix = np.array([(b"cl", b"y", b"y", b"", b"LS11", 100.0), (b"cl", b"y", b"y", b"", b"LS11", 200.0),
                   (b"wp", b"g", b"g", b"", b"LS11", 1.0)], dtype=dt)
    with h5py.File(p, "w") as f:
        f["index"], f["data"], f["jk"] = ix, np.arange(3.0), np.ones((4, 3))
        f.attrs["n_jk"] = 4
    v = load_vector(p)
    b = v.blocks()
    assert [k[0] for k, _ in b] == ["cl", "wp"] and list(b[0][1]) == [0, 1]
