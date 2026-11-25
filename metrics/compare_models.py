"""Cross-model comparison: significance + orthogonality for the 5-metric system.

Two outputs that make the evaluation defensible:

  1. SIGNIFICANCE — every cross-model difference, per metric, is tested with a
     paired permutation test (cases are matched by filename) and the whole grid
     is Benjamini-Hochberg FDR-corrected. Without this a "model A beats B" claim
     has no p-value.

  2. ORTHOGONALITY — the Spearman correlation matrix among the per-case metric
     scores. Low off-diagonal correlation = each metric adds information (we are
     not measuring the same thing five times).

Per-case arrays are read from the files the metric modules already write:
  M1  inference_results_scored.jsonl   (per-case article micro-F1, computed here)
  M2  m2_structural.jsonl              ("m2")
  M3  m3_semantic.jsonl               ("m3")          (skipped if absent)
  M4  m4_dispositive.jsonl            ("coarse_agree" -> 0/1)
  M5  m5_faithfulness.jsonl           ("groundedness")  (skipped if absent)

CLI:
  python compare_models.py --models gpt-oss-20b qwen3.5-9b
    Writes results/_comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metrics.stats_utils import benjamini_hochberg, paired_permutation

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

METRIC_FILES = {
    "M2_structural": ("m2_structural.jsonl", "m2"),
    "M3_semantic":   ("m3_semantic.jsonl", "m3"),
    "M5_faithfulness": ("m5_faithfulness.jsonl", "groundedness"),
}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _m1_per_case(results_dir: Path) -> dict[str, float]:
    """Per-case article micro-F1 keyed by filename (mean across seeds)."""
    acc: dict[str, list[float]] = {}
    for r in _read_jsonl(results_dir / "inference_results_scored.jsonl"):
        if not r.get("gt_articles"):
            continue
        gt = set(r.get("gt_articles", []))
        pred = set(r.get("predicted_articles", []))
        tp = len(gt & pred); fp = len(pred - gt); fn = len(gt - pred)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * rec / (p + rec)) if (p + rec) else 0.0
        acc.setdefault(r["filename"], []).append(f1)
    return {fn: float(np.mean(v)) for fn, v in acc.items()}


def _m4_per_case(results_dir: Path) -> dict[str, float]:
    """Per-case outcome-agreement rate keyed by filename (mean across seeds)."""
    acc: dict[str, list[float]] = {}
    for r in _read_jsonl(results_dir / "m4_dispositive.jsonl"):
        # only cases where both sides parsed contribute a 0/1 outcome-agreement
        if r.get("judge_coarse") is not None and r.get("llm_coarse") is not None:
            acc.setdefault(r["filename"], []).append(1.0 if r.get("coarse_agree") else 0.0)
    return {fn: float(np.mean(v)) for fn, v in acc.items()}


def _generic_per_case(results_dir: Path, fname: str, key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in _read_jsonl(results_dir / fname):
        v = r.get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[r["filename"]] = float(v)
    return out


def per_case_metrics(model: str) -> dict[str, dict[str, float]]:
    d = RESULTS_DIR / model
    metrics: dict[str, dict[str, float]] = {"M1_normative": _m1_per_case(d),
                                            "M4_dispositive": _m4_per_case(d)}
    for name, (fname, key) in METRIC_FILES.items():
        m = _generic_per_case(d, fname, key)
        if m:
            metrics[name] = m
    return metrics


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def orthogonality(models: list[str]) -> dict:
    """Spearman correlation among metrics, pooled over all models (per-case).

    Each pairwise correlation is computed over the (model, filename) keys
    present in BOTH metrics. Metrics are not uniformly available (M5 exists
    only for some models; M3/M4 drop unparsable cases), so a single shared
    pooled vector per metric would either crash on mismatched lengths or
    silently misalign pairs. Per-pair intersection
    uses every genuinely paired observation and nothing else.
    """
    pooled: dict[str, dict[tuple[str, str], float]] = {}
    for model in models:
        for metric, vals in per_case_metrics(model).items():
            for fn, v in vals.items():
                pooled.setdefault(metric, {})[(model, fn)] = v
    names = list(pooled)
    n = len(names)
    matrix: list[list[float]] = []
    n_pairs: dict[str, int] = {}
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
                continue
            keys = sorted(set(pooled[names[i]]) & set(pooled[names[j]]))
            if j > i:
                n_pairs[f"{names[i]}|{names[j]}"] = len(keys)
            x = np.array([pooled[names[i]][k] for k in keys])
            y = np.array([pooled[names[j]][k] for k in keys])
            row.append(_spearman(x, y))
        matrix.append(row)
    return {"metrics": names,
            # kept for the report caption: the largest genuinely-paired n
            "n_pooled": max(n_pairs.values(), default=0),
            "n_pairs": n_pairs,
            "spearman": matrix}


def significance(models: list[str]) -> dict:
    """Paired permutation per metric for every model pair, BH-FDR over the grid."""
    pm_by_model = {m: per_case_metrics(m) for m in models}
    all_metrics = sorted({k for pm in pm_by_model.values() for k in pm})
    tests: list[dict] = []
    for metric in all_metrics:
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                va, vb = pm_by_model[a].get(metric, {}), pm_by_model[b].get(metric, {})
                common = sorted(set(va) & set(vb))
                if len(common) < 3:
                    continue
                arr_a = np.array([va[fn] for fn in common])
                arr_b = np.array([vb[fn] for fn in common])
                res = paired_permutation(arr_a, arr_b, n_resamples=10000, seed=42)
                tests.append({"metric": metric, "a": a, "b": b, "n_pairs": len(common),
                              "mean_a": float(arr_a.mean()), "mean_b": float(arr_b.mean()),
                              "diff_mean": res["diff_mean"], "p_raw": res["p_value"]})
    if tests:
        bh = benjamini_hochberg([t["p_raw"] for t in tests], alpha=0.05)
        for t, adj, rej in zip(tests, bh["adjusted"], bh["rejected"]):
            t["p_adj"] = adj
            t["significant"] = bool(rej)
    return {"n_tests": len(tests), "tests": tests}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    sig = significance(args.models)
    ortho = orthogonality(args.models)
    payload = {"models": args.models, "significance": sig, "orthogonality": ortho}
    out = RESULTS_DIR / "_comparison.json"

    def _json_safe(o):  # NaN/Inf -> None (valid JSON for the browser)
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _json_safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_json_safe(v) for v in o]
        return o

    out.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
                   encoding="utf-8")

    print(f"Significance ({sig['n_tests']} tests, BH-FDR @0.05):")
    for t in sig["tests"]:
        star = " *" if t.get("significant") else ""
        print(f"  {t['metric']:16s} {t['a']} ({t['mean_a']:.3f}) vs {t['b']} ({t['mean_b']:.3f})  "
              f"diff={t['diff_mean']:+.3f}  p_adj={t.get('p_adj', float('nan')):.3f}{star}")
    print(f"\nOrthogonality — Spearman among metrics (pooled n={ortho['n_pooled']}):")
    names = ortho["metrics"]
    print("  " + " " * 16 + "".join(f"{n.split('_')[0]:>8}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:16s}" + "".join(f"{ortho['spearman'][i][j]:>8.2f}" for j in range(len(names))))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
