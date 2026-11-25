"""
Statistical utilities for the Cassazione evaluation pipeline.

Goal: every numerical claim in the paper should come with a bootstrap CI and a
significance test, FDR-corrected when we make many comparisons. This module
centralises that machinery so every metric and analysis script shares the
same implementation.

Provides:
    bootstrap_ci          : non-parametric bootstrap mean + percentile CI
    paired_permutation    : paired permutation test for two matched samples
    benjamini_hochberg    : BH-FDR adjustment on a vector of p-values
    cross_model_significance : run paired tests for every model pair × metric
                              and FDR-adjust over the resulting grid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: Iterable[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Non-parametric bootstrap of the mean with percentile CI.

    Returns {mean, lo, hi, n, n_resamples, ci}.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "n_resamples": n_resamples, "ci": ci}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return {
        "mean":        float(arr.mean()),
        "lo":          float(np.quantile(means, alpha)),
        "hi":          float(np.quantile(means, 1 - alpha)),
        "n":           int(n),
        "n_resamples": n_resamples,
        "ci":          ci,
    }


# ── Paired permutation test ──────────────────────────────────────────────────

def paired_permutation(
    values_a: Iterable[float],
    values_b: Iterable[float],
    n_resamples: int = 10000,
    seed: int = 42,
    alternative: str = "two-sided",
) -> dict:
    """
    Paired permutation test of mean(A − B) = 0.

    Pairs are kept (same case for A and B). For each resample, each pair's
    sign is flipped with probability 0.5; we count how often the absolute
    permuted mean equals/exceeds the observed absolute mean.

    Returns {diff_mean, p_value, n_pairs, n_resamples}.
    """
    a = np.asarray(list(values_a), dtype=np.float64)
    b = np.asarray(list(values_b), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = a.size
    if n == 0:
        return {"diff_mean": float("nan"), "p_value": float("nan"),
                "n_pairs": 0, "n_resamples": n_resamples}

    diffs = a - b
    observed = diffs.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_resamples, n))
    permuted_means = (signs * diffs).mean(axis=1)

    if alternative == "two-sided":
        count = int((np.abs(permuted_means) >= abs(observed)).sum())
    elif alternative == "greater":
        count = int((permuted_means >= observed).sum())
    elif alternative == "less":
        count = int((permuted_means <= observed).sum())
    else:
        raise ValueError(f"unknown alternative: {alternative!r}")

    # +1 to numerator and denominator for the standard exact-permutation correction.
    p_value = (count + 1) / (n_resamples + 1)
    return {
        "diff_mean":   float(observed),
        "p_value":     float(p_value),
        "n_pairs":     int(n),
        "n_resamples": n_resamples,
        "alternative": alternative,
    }


# ── Benjamini–Hochberg FDR correction ────────────────────────────────────────

def benjamini_hochberg(p_values: Iterable[float], alpha: float = 0.05) -> dict:
    """
    BH-FDR correction. Returns {adjusted, rejected, alpha} where:
      adjusted : list of BH-adjusted p-values, parallel to input
      rejected : list of bool — True if adjusted p ≤ alpha
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    n = p.size
    if n == 0:
        return {"adjusted": [], "rejected": [], "alpha": alpha}
    order = np.argsort(p)
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    # Adjusted = min over k≥i of (n/k) · p_{(k)}, then unsort.
    sorted_p = p[order]
    adj_sorted = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = adj_sorted
    rejected = adjusted <= alpha
    return {
        "adjusted": adjusted.tolist(),
        "rejected": rejected.tolist(),
        "alpha":    alpha,
    }


# ── Cross-model comparison grid ──────────────────────────────────────────────

def cross_model_significance(
    per_model_metrics: dict[str, dict[str, list[float]]],
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict:
    """
    For every (metric, model_a, model_b) triple with model_a < model_b,
    run a paired permutation test. Then BH-correct across the whole grid.

    Input shape:
        per_model_metrics = {
            "qwen3.5-9b": {"anchor_alignment": [...per-case floats], "category_kl": [...]},
            "gpt-oss-20b": {"anchor_alignment": [...], "category_kl": [...]},
            ...
        }
        The per-case arrays must be the same length and aligned to the same
        case ordering across models (the caller is responsible for this).

    Returns:
        {
          "metrics": {
            "anchor_alignment": {
              "pairs": [
                {"a": "qwen3.5-9b", "b": "gpt-oss-20b",
                 "diff_mean": ..., "p_value_raw": ..., "p_value_adj": ...,
                 "n_pairs": ..., "significant": bool},
                ...
              ]
            },
            ...
          },
          "n_total_tests": N,
          "alpha": 0.05,
          "seed": 42,
        }
    """
    models = sorted(per_model_metrics.keys())
    metric_names = sorted({m for v in per_model_metrics.values() for m in v.keys()})

    pairs = []
    for metric in metric_names:
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                va = per_model_metrics.get(a, {}).get(metric, [])
                vb = per_model_metrics.get(b, {}).get(metric, [])
                if not va or not vb:
                    continue
                # Align by truncation if lengths differ (caller should pre-align;
                # this is a safety net).
                k = min(len(va), len(vb))
                res = paired_permutation(va[:k], vb[:k], n_resamples=n_resamples, seed=seed)
                pairs.append({
                    "metric":    metric,
                    "a":         a,
                    "b":         b,
                    **res,
                    "p_value_raw": res["p_value"],
                })

    if not pairs:
        return {"metrics": {}, "n_total_tests": 0, "alpha": alpha, "seed": seed}

    bh = benjamini_hochberg([p["p_value_raw"] for p in pairs], alpha=alpha)
    for p, adj, rej in zip(pairs, bh["adjusted"], bh["rejected"]):
        p["p_value_adj"] = adj
        p["significant"] = bool(rej)
        del p["p_value"]  # avoid ambiguity

    by_metric: dict = {}
    for p in pairs:
        m = p.pop("metric")
        by_metric.setdefault(m, {"pairs": []})["pairs"].append(p)

    return {
        "metrics":       by_metric,
        "n_total_tests": len(pairs),
        "alpha":         alpha,
        "seed":          seed,
    }


# ── Multi-label classification evaluation ────────────────────────────────────

def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def multi_label_eval(
    source: "Path | str | list[dict]",
    labels: list[str],
    bootstrap: bool = True,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Multi-label evaluation against `gt_articles` / `predicted_articles`.

    `source` may be a JSONL path or an already-loaded list of records. Each
    record needs `filename`, `gt_articles` (list[str]), `predicted_articles`
    (list[str]). Records missing GT (empty list) are dropped from the count
    so they don't artificially inflate true-negative counts.

    Returns a dict with:
        n              : number of cases scored
        per_class      : {label: {tp, fp, fn, precision, recall, f1, support}}
        micro          : {precision, recall, f1}    (pool all cells)
        macro          : {precision, recall, f1}    (unweighted class mean)
        weighted       : {precision, recall, f1}    (support-weighted mean)
        subset_acc     : exact-match rate on the GT set
        jaccard_mean   : mean |gt ∩ pred| / |gt ∪ pred|
        bootstrap      : (if bootstrap=True) bootstrap CIs on micro_f1,
                         macro_f1, subset_acc, jaccard_mean

    Subset accuracy is the strictest measure: it only counts a case correct
    if every predicted article is in GT AND every GT article was predicted.
    """
    if isinstance(source, (str, Path)):
        records: list[dict] = []
        with open(source) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        records = list(source)

    records = [r for r in records if r.get("gt_articles")]
    n = len(records)
    if n == 0:
        return {"n": 0}

    per_class = {lbl: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for lbl in labels}
    subset_correct = 0
    jaccards: list[float] = []
    case_f1s: list[float] = []  # per-case micro-F1 for bootstrap

    for r in records:
        gt = set(r.get("gt_articles", []))
        pred = set(r.get("predicted_articles", []))
        if gt == pred:
            subset_correct += 1
        inter = gt & pred
        union = gt | pred
        jaccards.append(_safe_div(len(inter), len(union)))
        # Per-case micro: pooled cells across labels for this case only.
        tp = len(inter)
        fp = len(pred - gt)
        fn = len(gt - pred)
        p = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        case_f1s.append(_safe_div(2 * p * rec, p + rec))
        for lbl in labels:
            cell = per_class[lbl]
            in_gt = lbl in gt
            in_pred = lbl in pred
            if in_gt:
                cell["support"] += 1
            if in_gt and in_pred:
                cell["tp"] += 1
            elif in_pred and not in_gt:
                cell["fp"] += 1
            elif in_gt and not in_pred:
                cell["fn"] += 1

    # Per-class P/R/F1
    for lbl in labels:
        c = per_class[lbl]
        c["precision"] = _safe_div(c["tp"], c["tp"] + c["fp"])
        c["recall"]    = _safe_div(c["tp"], c["tp"] + c["fn"])
        c["f1"]        = _safe_div(2 * c["precision"] * c["recall"], c["precision"] + c["recall"])

    # Micro: pooled across all (case, label) cells
    tp_sum = sum(c["tp"] for c in per_class.values())
    fp_sum = sum(c["fp"] for c in per_class.values())
    fn_sum = sum(c["fn"] for c in per_class.values())
    micro_p = _safe_div(tp_sum, tp_sum + fp_sum)
    micro_r = _safe_div(tp_sum, tp_sum + fn_sum)
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r)

    # Macro: unweighted mean of per-class F1
    macro_p  = sum(c["precision"] for c in per_class.values()) / len(labels)
    macro_r  = sum(c["recall"]    for c in per_class.values()) / len(labels)
    macro_f1 = sum(c["f1"]        for c in per_class.values()) / len(labels)

    # Weighted: support-weighted mean
    total_support = sum(c["support"] for c in per_class.values()) or 1
    w_p  = sum(c["precision"] * c["support"] for c in per_class.values()) / total_support
    w_r  = sum(c["recall"]    * c["support"] for c in per_class.values()) / total_support
    w_f1 = sum(c["f1"]        * c["support"] for c in per_class.values()) / total_support

    out = {
        "n":            n,
        "per_class":    per_class,
        "micro":        {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro":        {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted":     {"precision": w_p,     "recall": w_r,     "f1": w_f1},
        "subset_acc":   subset_correct / n,
        "jaccard_mean": sum(jaccards) / n,
    }

    if bootstrap:
        # Bootstrap on case-level metrics. Micro-F1 doesn't decompose
        # cleanly over cases (it pools cells, not per-case scores), but
        # case-level micro-F1 is the standard surrogate.
        out["bootstrap"] = {
            "case_f1":      bootstrap_ci(case_f1s, n_resamples=n_resamples, seed=seed),
            "subset_acc":   bootstrap_ci([1.0 if set(r["gt_articles"]) == set(r["predicted_articles"]) else 0.0 for r in records], n_resamples=n_resamples, seed=seed),
            "jaccard_mean": bootstrap_ci(jaccards, n_resamples=n_resamples, seed=seed),
        }
    return out


def format_multi_label_table(
    results_by_model: dict[str, dict],
    labels: list[str],
) -> str:
    """Pretty-print a side-by-side comparison of model scorings."""
    models = list(results_by_model)
    if not models:
        return "(no results)"
    lines: list[str] = []
    head = f"{'metric':<24}" + "".join(f"{m:>14}" for m in models)
    lines.append(head)
    lines.append("-" * len(head))
    rows: list[tuple[str, list[float]]] = []
    rows.append(("micro_f1",        [results_by_model[m]["micro"]["f1"]    for m in models]))
    rows.append(("macro_f1",        [results_by_model[m]["macro"]["f1"]    for m in models]))
    rows.append(("weighted_f1",     [results_by_model[m]["weighted"]["f1"] for m in models]))
    rows.append(("subset_acc",      [results_by_model[m]["subset_acc"]     for m in models]))
    rows.append(("jaccard_mean",    [results_by_model[m]["jaccard_mean"]   for m in models]))
    for name, vals in rows:
        lines.append(f"{name:<24}" + "".join(f"{v:>14.3f}" for v in vals))
    lines.append("")
    lines.append("per-class F1:")
    for lbl in labels:
        vals = [results_by_model[m]["per_class"][lbl]["f1"] for m in models]
        sups = [results_by_model[m]["per_class"][lbl]["support"] for m in models]
        sup_str = f"(n={sups[0]})"
        lines.append(f"  {lbl} {sup_str:<8}" + " " * (24 - 2 - len(lbl) - 1 - len(sup_str)) + "".join(f"{v:>14.3f}" for v in vals))
    return "\n".join(lines)
