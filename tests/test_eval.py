"""Offline tests for the eval bookkeeping (no API calls)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("eval_script", Path(__file__).parent.parent / "eval" / "eval.py")
eval_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_script)

EXPECTED = [
    {"file": "a.txt", "fit": "yes", "location_ok": "yes", "hours_ok": "yes", "language_ok": "yes"},
    {"file": "b.txt", "fit": "no", "location_ok": "no", "hours_ok": "yes", "language_ok": "yes"},
    {"file": "c.txt", "fit": "no", "location_ok": "yes", "hours_ok": "no", "language_ok": "yes"},
]


def result(file, model="m1", date="2026-11-18", **labels):
    base = {"fit": "yes", "location_ok": "yes", "hours_ok": "yes", "language_ok": "yes"}
    return {"file": file, "model": model, "date": date, "score": "80", "verdict": "strong fit", **base, **labels}


def test_summary_counts_only_finished_ads_of_this_model():
    results = [
        result("a.txt"),  # 4/4 correct
        result("b.txt", date="2026-11-19", fit="no"),  # location wrong -> 3/4
        result("c.txt", model="other-model"),  # different model, ignored
    ]
    lines = eval_script.summarize(EXPECTED, results, "m1")
    text = "\n".join(lines)
    assert "ads evaluated: 2 of 3 (still missing: c.txt)" in text
    assert "run on: 2026-11-18, 2026-11-19" in text
    assert "**overall: 7/8 (88%)**" in text
    assert "**yes** (expected no)" in text


def test_summary_without_results():
    assert eval_script.summarize(EXPECTED, [], "m1")[-1] == "No results yet."


def test_results_round_trip(tmp_path):
    path = tmp_path / "results.csv"
    eval_script.save_results([result("b.txt"), result("a.txt")], path)
    assert [r["file"] for r in eval_script.load_results(path)] == ["a.txt", "b.txt"]
    assert eval_script.load_results(tmp_path / "missing.csv") == []
