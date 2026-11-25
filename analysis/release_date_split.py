"""Do models score higher on judgments published before their release?

    python -m analysis.release_date_split

For every model the 91 evaluated test cases are split by the judgment's
publication date (the date folder of its italgiure URL in
data/corpus_manifest.json) against the model's release month (`release_date`
in pipeline/model_backends.PROFILES). A judgment published in or after the
release month is "unseen"; an earlier one is "possible" (it existed before the
weights were published, so it could be in the training data). Reported per
model: the two group sizes, M1w pooled over five seeds, the seed-42 M3 mean,
and delta = possible - unseen (positive would point to memorisation), with a
two-sample permutation p-value on the per-case scores (case-level article F1
averaged over seeds; per-case M3). A model whose smaller group has fewer than
MIN_GROUP cases is marked as having no power.

Documented training cut-offs are not published for every model, so the split
uses release months only.

Reads results/<model>/inference_results_scored.jsonl and m3_semantic.jsonl
(written by metrics.evaluate). Writes results/_release_date_split.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from metrics.stats_utils import multi_label_eval
from pipeline.model_backends import PROFILES

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "corpus_manifest.json"
EVALUATED = ROOT / "splits" / "evaluated_91.json"
LABELS = ["318", "319", "320", "321", "322"]
MIN_GROUP = 10

# The eleven models reported in the paper (profile keys).
REPORTED_MODELS = [
    "qwen3.5-397b-a17b", "qwen3.5-122b-a10b", "minimax-m2.7-nvfp4", "nemotron-3-super-120b",
    "gpt-oss-120b", "mistral-medium-3.5-128b", "nemotron-3-nano-30b-a3b", "qwen3.5-9b",
    "gpt-oss-20b", "qwen3.6-35b-a3b", "minicpm5-1b",
]

_DATE = re.compile(r"id=\./(\d{8})/")


def read_jsonl(path: Path) -> list[dict]:
    """Split on newlines only: generated text can contain U+2028/U+2029."""
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def perm_p(a: list[float], b: list[float], n: int = 10000, seed: int = 42) -> float:
    """Two-sided permutation test on the difference of two independent group means."""
    x = np.asarray(a + b, dtype=float)
    k = len(a)
    obs = abs(x[:k].mean() - x[k:].mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        rng.shuffle(x)
        hits += abs(x[:k].mean() - x[k:].mean()) >= obs
    return (hits + 1) / (n + 1)


def case_f1(rows: list[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for r in rows:
        gt, pred = set(r.get("gt_articles") or []), set(r.get("predicted_articles") or [])
        if not gt:
            continue
        tp, fp, fn = len(gt & pred), len(pred - gt), len(gt - pred)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        acc.setdefault(r["filename"], []).append(2 * p * rc / (p + rc) if p + rc else 0.0)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def publication_dates() -> dict[str, str]:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = {}
    for j in man["judgments"]:
        m = _DATE.search(j["url_pdf"])
        if m:
            d = m.group(1)
            out[j["filename"]] = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=REPORTED_MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS / "_release_date_split.json")
    args = ap.parse_args()

    dates = publication_dates()
    evaluated = set(json.loads(EVALUATED.read_text(encoding="utf-8"))["evaluated_test"])
    payload: dict = {"definition": "unseen = published in or after the release month",
                     "min_group": MIN_GROUP, "models": {}}
    for m in args.models:
        rel = str(PROFILES[m]["release_date"])
        scored = read_jsonl(RESULTS / m / "inference_results_scored.jsonl")
        m3 = {r["filename"]: r["m3"] for r in read_jsonl(RESULTS / m / "m3_semantic.jsonl")
              if r.get("m3") is not None}
        cases = sorted({r["filename"] for r in scored})
        if set(cases) != evaluated:
            raise SystemExit(f"{m}: expected the 91 evaluated cases, found {len(cases)}")
        if len(rel) < 7:                      # year only: no month to split on
            payload["models"][m] = {"release": rel, "status": "release month unknown"}
            continue
        unseen = {c for c in cases if dates[c][:7] >= rel}
        possible = set(cases) - unseen
        row = {"release": rel, "n_unseen": len(unseen), "n_possible": len(possible)}
        if min(len(unseen), len(possible)) < MIN_GROUP:
            row["status"] = "no power"
        else:
            def m1w(group):
                ev = multi_label_eval([r for r in scored if r["filename"] in group], LABELS,
                                      bootstrap=False)
                return ev["weighted"]["f1"]

            def m3mean(group):
                v = [m3[c] for c in sorted(group) if c in m3 and not np.isnan(m3[c])]
                return float(np.mean(v)) if v else None

            cf = case_f1(scored)
            row.update({
                "status": "ok",
                # Sorted, so the permutation draws do not depend on set order.
                "p_case_f1": perm_p([cf[c] for c in sorted(possible) if c in cf],
                                    [cf[c] for c in sorted(unseen) if c in cf]),
                "p_m3": perm_p([m3[c] for c in sorted(possible) if c in m3],
                               [m3[c] for c in sorted(unseen) if c in m3]),
                "m1w_possible": m1w(possible), "m1w_unseen": m1w(unseen),
                "delta_m1w_possible_minus_unseen": m1w(possible) - m1w(unseen),
                "m3_possible": m3mean(possible), "m3_unseen": m3mean(unseen),
                "delta_m3_possible_minus_unseen": m3mean(possible) - m3mean(unseen),
            })
        payload["models"][m] = row
    histogram: dict[str, int] = {}
    for c in sorted(evaluated):
        histogram[dates[c][:7]] = histogram.get(dates[c][:7], 0) + 1
    payload["publication_month_histogram"] = histogram
    payload["n_with_power"] = sum(1 for v in payload["models"].values() if v.get("status") == "ok")
    payload["n_without_power"] = sum(1 for v in payload["models"].values() if v.get("status") != "ok")
    payload["any_positive_delta_m1w"] = sorted(
        m for m, v in payload["models"].items()
        if v.get("status") == "ok" and v["delta_m1w_possible_minus_unseen"] > 0)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for m, v in payload["models"].items():
        print(f"{m:26s} {v}")
    print(histogram)


if __name__ == "__main__":
    main()
