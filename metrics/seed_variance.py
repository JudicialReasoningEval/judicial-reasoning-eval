"""Cross-seed variance analysis for the multi-seed runs.

Motivation (see the reporting literature): single-generation evaluations at
temperature > 0 conflate model quality with sampling luck. Reasoning
benchmarks show double-digit Pass@1 swings across seeds on small test sets
(Hochlehnert et al. 2025, "A Sober Look at Progress in Language Model
Reasoning"); best practice is multi-seed evaluation with mean +/- std and an
explicit within-vs-between variance decomposition (Miller 2024, "Adding
Error Bars to Evals"; Zhang et al. 2025, "Beyond the Singular"; Song et al.
NAACL 2025, "Evaluation of LLMs Should Not Ignore Non-Determinism";
Dodge et al. EMNLP 2019, "Show Your Work").

For every model whose inference_results.jsonl carries multiple seeds
(seeds 42-46), this script computes PER SEED:

    M1  corpus weighted/macro F1 (stats_utils.multi_label_eval, no bootstrap)
    M4  coarse Cohen's kappa + coarse accuracy (rows of m4_dispositive.jsonl
        are written in inference_results order -> positional seed join)
    dual-format compliance rate

and reports across seeds: mean, sample std (ddof=1), min/max, and a 95%
t-interval (n=5 -> t=2.776). It also decomposes the per-case M1 variance
into between-case and within-case (sampling) components: the within-share
answers "how much of the score spread is sampling luck vs case difficulty".

Output: results/_seed_variance.json + a printed table.

Usage:
    python -m metrics.seed_variance
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metrics.m4_dispositive import cohen_kappa
from metrics.stats_utils import multi_label_eval

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

LABELS = ["318", "319", "320", "321", "322"]
T_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}  # t_{0.975, n-1}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # \n-only split: splitlines() breaks records on embedded U+2028/U+2029
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def _per_case_f1(rec: dict) -> float | None:
    gt = set(rec.get("gt_articles") or [])
    if not gt:
        return None
    pred = set(rec.get("predicted_articles") or [])
    tp = len(gt & pred); fp = len(pred - gt); fn = len(gt - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def _stats(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    n = len(a)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    half = T_95.get(n, 1.96) * std / np.sqrt(n) if n > 1 else 0.0
    return {"n_seeds": n, "per_seed": [round(v, 4) for v in vals],
            "mean": round(mean, 4), "std": round(std, 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "ci95": [round(mean - half, 4), round(mean + half, 4)]}


def analyze_model(model_dir: Path) -> dict | None:
    inf = _read_jsonl(model_dir / "inference_results.jsonl")
    if not inf:
        return None
    seeds = sorted({r.get("seed", 42) for r in inf})
    if len(seeds) < 2:
        return None
    scored = _read_jsonl(model_dir / "inference_results_scored.jsonl")
    m4 = _read_jsonl(model_dir / "m4_dispositive.jsonl")
    if len(m4) == len(inf):
        for row, rec in zip(m4, inf):
            row["seed"] = rec.get("seed", 42)
    else:
        m4 = []  # cannot align — skip M4 rather than misattribute

    m1w, m1m, m4k, m4acc, dual = [], [], [], [], []
    percase: dict[str, dict[int, float]] = {}
    for seed in seeds:
        recs = [r for r in scored if r.get("seed", 42) == seed]
        ev = multi_label_eval(recs, LABELS, bootstrap=False)
        m1w.append(ev["weighted"]["f1"]); m1m.append(ev["macro"]["f1"])
        for r in recs:
            f1 = _per_case_f1(r)
            if f1 is not None:
                percase.setdefault(r["filename"], {})[seed] = f1
        if m4:
            rows = [r for r in m4 if r["seed"] == seed
                    and r.get("judge_coarse") is not None and r.get("llm_coarse") is not None]
            if rows:
                m4k.append(cohen_kappa([r["judge_coarse"] for r in rows],
                                       [r["llm_coarse"] for r in rows]))
                m4acc.append(float(np.mean([1.0 if r["coarse_agree"] else 0.0 for r in rows])))
        inf_recs = [r for r in inf if r.get("seed", 42) == seed]
        dual.append(float(np.mean([1.0 if r.get("raw_dual") else 0.0 for r in inf_recs])))

    # Variance decomposition on the per-case M1 matrix (cases x seeds):
    # between-case variance of case means vs mean within-case variance.
    full = {fn: sv for fn, sv in percase.items() if len(sv) == len(seeds)}
    case_means = np.array([np.mean(list(sv.values())) for sv in full.values()])
    within = np.array([np.var(list(sv.values()), ddof=1) for sv in full.values()])
    var_between = float(case_means.var(ddof=1)) if len(case_means) > 1 else float("nan")
    var_within = float(within.mean()) if len(within) else float("nan")
    within_share = var_within / (var_within + var_between) if var_within + var_between > 0 else float("nan")

    out = {"seeds": seeds,
           "M1_weighted_f1": _stats(m1w),
           "M1_macro_f1": _stats(m1m),
           "dual_format": _stats(dual),
           "M1_variance_decomposition": {
               "n_cases_full": len(full),
               "var_between_cases": round(var_between, 5),
               "var_within_case_sampling": round(var_within, 5),
               "within_share": round(within_share, 4)}}
    if m4k:
        out["M4_coarse_kappa"] = _stats(m4k)
        out["M4_coarse_acc"] = _stats(m4acc)
    return out


def main() -> None:
    payload: dict[str, dict] = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and (d / "inference_results.jsonl").exists():
            res = analyze_model(d)
            if res:
                payload[d.name] = res

    out = RESULTS_DIR / "_seed_variance.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'model':26s} {'M1w mean±std':>16s} {'M4k mean±std':>16s} "
          f"{'dual mean±std':>15s} {'within-share':>12s}")
    for name, r in payload.items():
        m1 = r["M1_weighted_f1"]; m4 = r.get("M4_coarse_kappa"); du = r["dual_format"]
        ws = r["M1_variance_decomposition"]["within_share"]
        m4s = f"{m4['mean']:.3f}±{m4['std']:.3f}" if m4 else "  —  "
        print(f"{name:26s} {m1['mean']:.3f}±{m1['std']:.3f}      {m4s:>10s}   "
              f"{du['mean']:.3f}±{du['std']:.3f}   {ws:>10.1%}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
