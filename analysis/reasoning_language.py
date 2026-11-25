"""Reasoning-trace language vs per-premise M3 (Discussion hypothesis test).

Hypothesis: models deliberating in Italian qualify Italian facts in Italian
legal categories (the premessa minore) better than models deliberating in
English. With
eleven models and panel-consensus labels we can test this properly — but
first the "reasoning language" must be MEASURED, not assumed: each model's
seed-42 traces are language-identified by Italian vs English function-word
counts, then per-premise M3 (from metrics.json) is compared across the
resulting groups.

Output: results/_reasoning_language.json + printed table.

Usage:
    python -m analysis.reasoning_language
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

IT_WORDS = {"di", "che", "il", "la", "per", "non", "del", "della", "con",
            "una", "sono", "come", "anche", "nel", "alla", "gli", "delle",
            "questo", "essere", "sulla"}
EN_WORDS = {"the", "of", "to", "and", "in", "that", "is", "for", "with",
            "as", "on", "are", "this", "be", "it", "we", "need", "which",
            "should", "must"}

_TOKEN_RE = re.compile(r"[a-zàèéìòù]+")

PREMISES = ("maggiore", "minore", "conclusione")


def _trace_it_share(text: str) -> float | None:
    toks = _TOKEN_RE.findall(text.lower())
    it = sum(1 for t in toks if t in IT_WORDS)
    en = sum(1 for t in toks if t in EN_WORDS)
    if it + en < 20:  # too short to classify
        return None
    return it / (it + en)


def main() -> None:
    payload: dict[str, dict] = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        inf = d / "inference_results.jsonl"
        mj = d / "metrics.json"
        if not (d.is_dir() and inf.exists() and inf.stat().st_size and mj.exists()):
            continue
        m3 = json.loads(mj.read_text()).get("M3_semantic")
        if not m3 or "per_premise" not in m3:
            continue
        shares = []
        seen: set[str] = set()
        for line in inf.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("sample_idx", 0) != 0 or r["filename"] in seen:
                continue
            seen.add(r["filename"])
            s = _trace_it_share(r.get("judge_reasoning_trace") or "")
            if s is not None:
                shares.append(s)
        if not shares:
            lang, it_share = "none", float("nan")
        else:
            it_share = float(np.mean(shares))
            lang = "it" if it_share >= 0.7 else ("en" if it_share <= 0.3 else "mixed")
        payload[d.name] = {
            "n_traces": len(shares),
            "italian_share": round(it_share, 3),
            "reasoning_language": lang,
            "m3_per_premise": {p: round(m3["per_premise"][p]["f1_mean"], 4)
                               for p in PREMISES},
            "minore_minus_maggiore": round(
                m3["per_premise"]["minore"]["f1_mean"]
                - m3["per_premise"]["maggiore"]["f1_mean"], 4),
        }

    # group means (exclude degenerate/none)
    groups: dict[str, list[dict]] = {}
    for name, v in payload.items():
        if v["reasoning_language"] in ("it", "en", "mixed") and v["n_traces"] >= 50:
            groups.setdefault(v["reasoning_language"], []).append(v)
    summary = {}
    for g, vs in sorted(groups.items()):
        summary[g] = {
            "n_models": len(vs),
            "m3_per_premise_mean": {
                p: round(float(np.mean([v["m3_per_premise"][p] for v in vs])), 4)
                for p in PREMISES},
        }

    out = RESULTS_DIR / "_reasoning_language.json"
    out.write_text(json.dumps({"models": payload, "groups": summary},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'model':26s} {'lang':6s} {'it%':>5s} {'n':>4s} {'magg':>6s} {'min':>6s} {'conc':>6s} {'min-magg':>8s}")
    for name, v in payload.items():
        pp = v["m3_per_premise"]
        print(f"{name:26s} {v['reasoning_language']:6s} {v['italian_share']:5.2f} "
              f"{v['n_traces']:4d} {pp['maggiore']:6.3f} {pp['minore']:6.3f} "
              f"{pp['conclusione']:6.3f} {v['minore_minus_maggiore']:+8.3f}")
    print("\ngroup means:", json.dumps(summary, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
