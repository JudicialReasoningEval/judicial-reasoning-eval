"""Regression tests for the frontier-reference validation pipeline.

All remote clients are mocked; these tests spend no money and use no network.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import metrics.frontier_agreement as fa
import metrics.m5_faithfulness as m5
import pipeline.run_frontier_validation as runner
from metrics.premise_labeler import StopRequested


def _judgment():
    return {
        "citations_total": 2,
        "citations_unsupported": 1,
        "facts_total": 2,
        "facts_ungrounded": 0,
        "groundedness": 0.75,
        "notes": "",
    }


def test_agreement_loader_rejects_duplicate_case(tmp_path):
    path = tmp_path / "judge.jsonl"
    row = {"filename": "case.md"}
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate filename"):
        fa._load(path)


def test_case_clustered_kappa_interval_is_deterministic():
    pairs = {
        "a": ([1, 1, 0], [1, 0, 0]),
        "b": ([0, 0], [0, 0]),
        "c": ([1, 0], [1, 0]),
    }
    a = fa._case_bootstrap_kappa(pairs, n_resamples=100, seed=7)
    b = fa._case_bootstrap_kappa(pairs, n_resamples=100, seed=7)
    assert a == b
    assert a["unit"] == "case"
    assert a["lo"] <= a["hi"]


def test_case_clustered_rank_interval_is_deterministic():
    scores = {
        "low": {"a": (0.1, 0.2), "b": (0.2, 0.1), "c": (0.1, 0.2)},
        "mid": {"a": (0.5, 0.6), "b": (0.6, 0.5), "c": (0.5, 0.5)},
        "high": {"a": (0.9, 0.8), "b": (0.8, 0.9), "c": (0.9, 0.8)},
    }
    a = fa._case_bootstrap_rank(scores, n_resamples=100, seed=7)
    b = fa._case_bootstrap_rank(scores, n_resamples=100, seed=7)
    assert a == b
    assert a["unit"] == "case"
    assert a["n_models"] == 3
    assert a["pearson_r"]["lo"] <= a["pearson_r"]["hi"]


def test_m5_flushes_completed_rows_before_cooperative_stop(tmp_path, monkeypatch):
    results_dir = tmp_path / "model"
    results_dir.mkdir()
    rows = [
        {"filename": "a.md", "sample_idx": 0, "considerato_predicted": "A"},
        {"filename": "b.md", "sample_idx": 0, "considerato_predicted": "B"},
    ]
    (results_dir / "inference_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )

    def fake_call(*args, filename="", **kwargs):
        if filename == "a.md":
            return _judgment()
        raise StopRequested("budget reached")

    monkeypatch.setattr(m5, "_call_judge", fake_call)
    monkeypatch.setattr(m5, "full_case_facts", lambda filename: "")
    out = tmp_path / "frontier" / "judge.jsonl"

    with pytest.raises(StopRequested, match="budget reached"):
        m5.score_model(
            results_dir,
            base_url="http://unused.invalid/v1",
            model="stub",
            workers=1,
            resume=False,
            out=out,
        )

    saved = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["filename"] for r in saved] == ["a.md"]


def test_truncated_m5_response_is_not_retried(monkeypatch):
    calls = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None),
                        finish_reason="length",
                    )
                ],
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    assert m5._call_judge(client, "stub", "testo", "fatti", retries=3) is None
    assert calls == 1


def test_meter_keeps_paid_response_then_stops_next_request(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "LEDGER", tmp_path / "ledger.jsonl")
    usage = SimpleNamespace(
        cost=0.2,
        prompt_tokens=10,
        completion_tokens=10,
        completion_tokens_details=None,
    )
    meter = runner.Meter(0.1, "C-smoke")
    meter.tag = "judge"
    meter.sink("model", "case.md", 0, usage)
    assert meter.spent == pytest.approx(0.2)
    with pytest.raises(runner.BudgetStop):
        meter.check()


def test_m5_agreement_requires_all_three_reference_judges(tmp_path, monkeypatch):
    frontier = tmp_path / "frontier"
    frontier.mkdir()
    (frontier / "model.m5.claude-sonnet-5.jsonl").write_text(
        json.dumps({"filename": "case.md"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(fa, "FRONTIER_M5", frontier)
    with pytest.raises(SystemExit, match="missing frontier judges"):
        fa.m5_agreement({"case.md"})
