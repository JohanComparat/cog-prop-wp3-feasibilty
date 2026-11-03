import json

from wp3feas import provenance


def test_record_writes_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "RESULTS", tmp_path)
    rec = provenance.Record("scripts/99_test.py")
    rec.add_input("a.json")
    rec.data["x"] = 1.5
    d = json.loads(rec.write().read_text())
    assert d["script"] == "99_test.py"
    assert d["inputs"] == ["a.json"]
    assert d["x"] == 1.5
    assert {"git_hash", "date", "wall_time_s"} <= d.keys()
