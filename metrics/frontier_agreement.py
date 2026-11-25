"""Panel-vs-frontier agreement.

Consumes what pipeline/run_frontier_validation.py produced and answers the
question the paper promises to answer: *how well does the open-weight panel
agree with a higher-capability reference panel?*

Reference standard, not gold standard. The frontier trio is itself an LLM panel
with the same strict 2-of-3 majority rule, so what we measure is agreement with
a stronger set of labellers — not correctness against legally trained
annotators, and the paper reports it as such.

Three families of number are produced, and the second two are the ones that
actually matter for the paper's claims:

  1. PANEL vs FRONTIER majority — Cohen's kappa, raw agreement, and P/R/F1 with
     the frontier majority as reference. The headline "is the silver standard
     any good" number.

  2. FRONTIER-INTERNAL Fleiss kappa, recomputed against the open-weight panel's
     kappa ON THE SAME SUBSET (not the full-corpus figure — that would be
     an unfair comparison). If the frontier trio also lands near 0.6, the
     panel's agreement reflects intrinsic task difficulty rather than
     open-weight weakness.

  3. M5 RANK PRESERVATION — Spearman rho between per-model mean groundedness
     under the panel and under the frontier judges. M5 is used in the paper only
     as a comparative gate, so rank preservation is what the comparative claims
     actually rest on; absolute level shifts matter much less.

Usage:
    python -m metrics.frontier_agreement --n 20
    python -m metrics.frontier_agreement --n 20 --premise-only
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from metrics.consensus_labels import _fleiss_kappa

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SPLITS = ROOT / "splits"

PREMISES = ["maggiore", "minore", "conclusione"]
CHECKS = ("citations", "facts")

PANEL_TAGS = ["gpt-oss-120b", "nemotron-3-super-120b", "qwen3.6-27b"]
FRONTIER_TAGS = ["claude-sonnet-5", "gemini-3.6-flash", "glm-5.2"]
PANEL_LABELS = RESULTS / "panel_labels"
PANEL_M5 = RESULTS / "panel_m5"
FRONTIER_LABELS = RESULTS / "frontier_labels"
FRONTIER_M5 = RESULTS / "frontier_m5"


# ── helpers ──────────────────────────────────────────────────────────────────
def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line:
            r = json.loads(line)
            fn = r["filename"]
            if fn in out:
                raise ValueError(
                    f"duplicate filename {fn!r} in {path}; agreement inputs "
                    "must contain exactly one frozen response per judge/case"
                )
            out[fn] = r
    return out


def _fleiss_or_none(rows: list[list[int]]) -> float | None:
    """Fleiss kappa needs >=2 raters — it divides by n(n-1). Guard rather than
    crash, so a partially-complete run still reports what it can."""
    if not rows or sum(rows[0]) < 2:
        return None
    return _fleiss_kappa(rows)


def _cohen_kappa(a: list[int], b: list[int]) -> float:
    """Binary Cohen's kappa between two label vectors."""
    if not a:
        return float("nan")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _prf(ref: list[int], hyp: list[int]) -> dict:
    """Precision/recall/F1 of `hyp` against reference `ref` on the positive class."""
    tp = sum(1 for r, h in zip(ref, hyp) if r == 1 and h == 1)
    fp = sum(1 for r, h in zip(ref, hyp) if r == 0 and h == 1)
    fn = sum(1 for r, h in zip(ref, hyp) if r == 1 and h == 0)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    if np.isnan(p) or np.isnan(r):
        f = float("nan")
    elif p + r == 0:
        f = 0.0  # both zero is a real F1 of 0, not undefined
    else:
        f = 2 * p * r / (p + r)
    return {"precision": _r(p), "recall": _r(r), "f1": _r(f),
            "support_ref": tp + fn, "support_hyp": tp + fp}


def _r(x: float, nd: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _f(x) -> str:
    """Format a possibly-None metric for the console table (None is legitimate:
    e.g. frontier Fleiss kappa before all three judges have run)."""
    return "n/a" if x is None else f"{x}"


def _case_bootstrap_kappa(
    case_pairs: dict[str, tuple[list[int], list[int]]],
    *,
    n_resamples: int = 5000,
    seed: int = 42,
) -> dict:
    """Case-clustered percentile interval for sentence-level Cohen kappa."""
    names = sorted(case_pairs)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        sampled = rng.choice(names, size=len(names), replace=True)
        a = [x for fn in sampled for x in case_pairs[str(fn)][0]]
        b = [x for fn in sampled for x in case_pairs[str(fn)][1]]
        values.append(_cohen_kappa(a, b))
    lo, hi = np.nanquantile(np.asarray(values, dtype=float), [0.025, 0.975])
    return {
        "lo": _r(float(lo)),
        "hi": _r(float(hi)),
        "unit": "case",
        "n_cases": len(names),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def _percentile_interval(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(finite):
        return {"lo": None, "hi": None}
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return {"lo": _r(float(lo)), "hi": _r(float(hi))}


def _case_bootstrap_rank(
    model_case_scores: dict[str, dict[str, tuple[float, float]]],
    *,
    n_resamples: int = 5000,
    seed: int = 42,
) -> dict:
    """Resample shared cases while retaining every model within each case.

    The number of models is fixed (three in the representative M5 check), so
    these intervals describe case-sampling uncertainty; they do not make a
    three-model rank correlation a population-level validation statistic.
    """
    models = sorted(model_case_scores)
    common = set.intersection(*(set(model_case_scores[m]) for m in models))
    cases = sorted(common)
    rng = np.random.default_rng(seed)
    spearman, pearson, shift = [], [], []
    for _ in range(n_resamples):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        panel_means = [
            float(np.mean([model_case_scores[m][str(fn)][0] for fn in sampled]))
            for m in models
        ]
        frontier_means = [
            float(np.mean([model_case_scores[m][str(fn)][1] for fn in sampled]))
            for m in models
        ]
        spearman.append(_spearman(panel_means, frontier_means))
        pearson.append(_pearson(panel_means, frontier_means))
        shift.append(float(np.mean(np.asarray(frontier_means) - np.asarray(panel_means))))
    return {
        "spearman_rho": _percentile_interval(spearman),
        "pearson_r": _percentile_interval(pearson),
        "mean_level_shift_frontier_minus_panel": _percentile_interval(shift),
        "unit": "case",
        "n_cases": len(cases),
        "n_models": len(models),
        "n_resamples": n_resamples,
        "seed": seed,
        "note": ("Case-sampling uncertainty only; n_models remains fixed and "
                 "small, so the rank correlations are descriptive."),
    }


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho via Pearson on ranks (average ranks for ties)."""
    if len(x) < 3:
        return float("nan")

    def rank(v):
        a = np.asarray(v, dtype=float)
        out = np.empty(len(a), dtype=float)
        srt = np.argsort(a, kind="mergesort")
        i = 0
        while i < len(a):  # average ranks within tied groups
            j = i
            while j + 1 < len(a) and a[srt[j + 1]] == a[srt[i]]:
                j += 1
            out[srt[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return out

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _subset(n: int) -> set[str]:
    p = SPLITS / f"frontier_validation_{n}.json"
    if not p.exists():
        raise SystemExit(f"{p} missing — run pipeline.run_frontier_validation --task subset")
    return {c["filename"] for c in json.loads(p.read_text(encoding="utf-8"))["cases"]}


def _discover_tags(directory: Path, prefix: str, infix: str) -> list[str]:
    tags = []
    for p in sorted(directory.glob("*.jsonl")):
        stem = p.name[: -len(".jsonl")]
        if infix and infix in stem:
            tags.append(stem.split(infix, 1)[1])
        elif stem.startswith(prefix):
            tags.append(stem[len(prefix):])
    return sorted(set(tags))


# ── premise labels ───────────────────────────────────────────────────────────
def premise_agreement(keep: set[str], allow_partial: bool = False) -> dict:
    discovered = _discover_tags(FRONTIER_LABELS, "sillogismo_judge.", "")
    if not discovered:
        raise SystemExit(f"no frontier label files in {FRONTIER_LABELS}")
    # A missing judge silently redefines the reference standard: strict majority
    # of 2 is unanimity, and of 1 is a single judge. Refuse rather than report
    # that as if it were the 3-judge design.
    missing_tags = [t for t in FRONTIER_TAGS if t not in discovered]
    if missing_tags and not allow_partial:
        raise SystemExit(
            f"[frontier-agreement] premise labels are missing frontier judges "
            f"{missing_tags}; the design is a 3-judge 2-of-3 majority. A run still in "
            f"flight will look like this. Pass --allow-partial for a provisional "
            f"look, but do NOT put those numbers in the paper."
        )
    ftags = [t for t in FRONTIER_TAGS if t in discovered]

    panel = {t: _load(PANEL_LABELS / f"sillogismo_judge.{t}.jsonl") for t in PANEL_TAGS}
    front = {t: _load(FRONTIER_LABELS / f"sillogismo_judge.{t}.jsonl") for t in ftags}

    common = set(keep)
    for d in list(panel.values()) + list(front.values()):
        common &= set(d)
    common = sorted(common)
    if not common:
        raise SystemExit("no cases labelled by BOTH panels — nothing to compare")

    # Per-sentence binary vectors, per premise.
    vec_panel: dict[str, dict[str, list[int]]] = {t: {p: [] for p in PREMISES} for t in PANEL_TAGS}
    vec_front: dict[str, dict[str, list[int]]] = {t: {p: [] for p in PREMISES} for t in ftags}
    maj_panel: dict[str, list[int]] = {p: [] for p in PREMISES}
    maj_front: dict[str, list[int]] = {p: [] for p in PREMISES}
    rows_panel: dict[str, list[list[int]]] = {p: [] for p in PREMISES}
    rows_front: dict[str, list[list[int]]] = {p: [] for p in PREMISES}
    case_pairs: dict[str, dict[str, tuple[list[int], list[int]]]] = {
        p: {} for p in PREMISES
    }

    mismatched, used = [], []
    for fn in common:
        recs_p = [panel[t][fn] for t in PANEL_TAGS]
        recs_f = [front[t][fn] for t in ftags]
        lens = {len(r["sentences"]) for r in recs_p + recs_f}
        if len(lens) != 1:
            mismatched.append(fn)
            continue
        n_sent = lens.pop()
        used.append(fn)
        starts = {p: len(maj_panel[p]) for p in PREMISES}
        for i in range(n_sent):
            for p in PREMISES:
                for t in PANEL_TAGS:
                    vec_panel[t][p].append(1 if p in (panel[t][fn]["premise_labels"][i] or []) else 0)
                for t in ftags:
                    vec_front[t][p].append(1 if p in (front[t][fn]["premise_labels"][i] or []) else 0)
                vp = sum(vec_panel[t][p][-1] for t in PANEL_TAGS)
                vf = sum(vec_front[t][p][-1] for t in ftags)
                maj_panel[p].append(1 if vp >= len(PANEL_TAGS) // 2 + 1 else 0)
                maj_front[p].append(1 if vf >= len(ftags) // 2 + 1 else 0)
                rows_panel[p].append([len(PANEL_TAGS) - vp, vp])
                rows_front[p].append([len(ftags) - vf, vf])
        for p in PREMISES:
            start = starts[p]
            case_pairs[p][fn] = (maj_panel[p][start:], maj_front[p][start:])

    out: dict = {
        "cases": len(used), "sentences": len(maj_panel[PREMISES[0]]),
        "split_mismatches": mismatched,
        "panel_labellers": PANEL_TAGS, "frontier_judges": ftags,
        "per_premise": {},
        "note": ("Frontier trio is a higher-capability REFERENCE panel, not a human "
                 "gold standard. Panel kappa is recomputed on this subset so it is "
                 "comparable with the frontier kappa."),
    }
    for p in PREMISES:
        cross = [_cohen_kappa(vec_panel[a][p], vec_front[b][p])
                 for a, b in product(PANEL_TAGS, ftags)]
        out["per_premise"][p] = {
            "majority_vs_majority": {
                "cohen_kappa": _r(_cohen_kappa(maj_panel[p], maj_front[p])),
                "raw_agreement": _r(sum(1 for x, y in zip(maj_panel[p], maj_front[p]) if x == y)
                                    / max(1, len(maj_panel[p]))),
                "panel_vs_frontier_prf": _prf(maj_front[p], maj_panel[p]),
                "prevalence_panel": _r(sum(maj_panel[p]) / max(1, len(maj_panel[p]))),
                "prevalence_frontier": _r(sum(maj_front[p]) / max(1, len(maj_front[p]))),
                "case_clustered_bootstrap_95": _case_bootstrap_kappa(case_pairs[p]),
            },
            "fleiss_kappa_within": {
                "open_weight_panel_this_subset": _r(_fleiss_or_none(rows_panel[p])),
                "frontier_panel": _r(_fleiss_or_none(rows_front[p])),
            },
            "panel_member_vs_frontier_majority": {
                t: _r(_cohen_kappa(vec_panel[t][p], maj_front[p])) for t in PANEL_TAGS},
            "frontier_judge_vs_panel_majority": {
                t: _r(_cohen_kappa(vec_front[t][p], maj_panel[p])) for t in ftags},
            "mean_pairwise_cross_kappa": _r(float(np.nanmean(cross))),
        }
    return out


# ── M5 ───────────────────────────────────────────────────────────────────────
def m5_agreement(keep: set[str], allow_partial: bool = False) -> dict:
    discovered = _discover_tags(FRONTIER_M5, "", ".m5.")
    if not discovered:
        raise SystemExit(f"no frontier M5 files in {FRONTIER_M5}")
    missing_tags = [t for t in FRONTIER_TAGS if t not in discovered]
    if missing_tags and not allow_partial:
        raise SystemExit(
            f"[frontier-agreement] M5 is missing frontier judges {missing_tags}; "
            "the reference design requires all three. Use --allow-partial only "
            "for diagnostics, never for paper figures."
        )
    ftags = [t for t in FRONTIER_TAGS if t in discovered]
    ptags = PANEL_TAGS

    models = sorted({p.name.split(".m5.")[0] for p in FRONTIER_M5.glob("*.m5.*.jsonl")})
    per_model: dict[str, dict] = {}
    panel_means, front_means, rank_names = [], [], []
    names: list[str] = []
    pooled_panel = {c: [] for c in CHECKS}
    pooled_front = {c: [] for c in CHECKS}
    pooled_case_pairs: dict[str, dict[str, tuple[list[int], list[int]]]] = {
        c: {} for c in CHECKS
    }
    rank_case_scores: dict[str, dict[str, tuple[float, float]]] = {}

    for m in models:
        front = {t: _load(FRONTIER_M5 / f"{m}.m5.{t}.jsonl") for t in ftags}
        panel = {t: _load(PANEL_M5 / f"{m}.m5.{t}.jsonl") for t in ptags}
        if any(not d for d in front.values()) or any(not d for d in panel.values()):
            per_model[m] = {"skipped": "incomplete judge coverage"}
            continue
        panel_common = set(keep)
        for d in panel.values():
            panel_common &= set(d)
        missing_by_judge = {
            t: sorted(panel_common - (set(front[t]) & set(keep))) for t in ftags
        }
        if any(missing_by_judge.values()) and not allow_partial:
            per_model[m] = {
                "skipped": "incomplete frontier case coverage",
                "expected_cases": len(panel_common),
                "missing_by_judge": missing_by_judge,
            }
            continue
        common = set(panel_common)
        for d in front.values():
            common &= set(d)
        common = sorted(common)
        if len(common) < 3:
            per_model[m] = {"skipped": f"only {len(common)} shared cases"}
            continue

        gp, gf = [], []
        rows_p = {c: [] for c in CHECKS}
        rows_f = {c: [] for c in CHECKS}
        maj_p = {c: [] for c in CHECKS}
        maj_f = {c: [] for c in CHECKS}
        for fn in common:
            gp.append(float(np.median([panel[t][fn]["groundedness"] for t in ptags])))
            gf.append(float(np.median([front[t][fn]["groundedness"] for t in ftags])))
            for c in CHECKS:
                key = "citations_unsupported" if c == "citations" else "facts_ungrounded"
                vp = sum(1 for t in ptags if int(panel[t][fn].get(key, 0) or 0) > 0)
                vf = sum(1 for t in ftags if int(front[t][fn].get(key, 0) or 0) > 0)
                rows_p[c].append([len(ptags) - vp, vp])
                rows_f[c].append([len(ftags) - vf, vf])
                maj_p[c].append(1 if vp >= 2 else 0)
                maj_f[c].append(1 if vf >= 2 else 0)

        gate_p = [1 if g >= 0.90 else 0 for g in gp]
        gate_f = [1 if g >= 0.90 else 0 for g in gf]
        per_model[m] = {
            "n_cases": len(common),
            "mean_groundedness_panel": _r(float(np.mean(gp))),
            "mean_groundedness_frontier": _r(float(np.mean(gf))),
            "mean_signed_diff_frontier_minus_panel": _r(float(np.mean(np.array(gf) - np.array(gp)))),
            "mean_abs_diff": _r(float(np.mean(np.abs(np.array(gf) - np.array(gp))))),
            "pearson_r": _r(_pearson(gp, gf)),
            "spearman_rho": _r(_spearman(gp, gf)),
            "gate_0.90_agreement": _r(sum(1 for a, b in zip(gate_p, gate_f) if a == b) / len(gate_p)),
            "gate_0.90_passes": {"panel": sum(gate_p), "frontier": sum(gate_f)},
            "majority_vs_majority": {
                c: {
                    "cohen_kappa": _r(_cohen_kappa(maj_p[c], maj_f[c])),
                    "raw_agreement": _r(
                        sum(a == b for a, b in zip(maj_p[c], maj_f[c])) / len(common)
                    ),
                    "panel_vs_frontier_prf": _prf(maj_f[c], maj_p[c]),
                    "case_clustered_bootstrap_95": _case_bootstrap_kappa({
                        fn: ([maj_p[c][i]], [maj_f[c][i]])
                        for i, fn in enumerate(common)
                    }),
                }
                for c in CHECKS
            },
            "fleiss_kappa_within": {
                c: {"open_weight_panel_this_subset": _r(_fleiss_or_none(rows_p[c])),
                    "frontier_panel": _r(_fleiss_or_none(rows_f[c]))} for c in CHECKS},
        }
        names.append(m)
        for c in CHECKS:
            pooled_panel[c].extend(maj_p[c])
            pooled_front[c].extend(maj_f[c])
            for i, fn in enumerate(common):
                pair = pooled_case_pairs[c].setdefault(fn, ([], []))
                pair[0].append(maj_p[c][i])
                pair[1].append(maj_f[c][i])
        if len(common) == len(keep):
            panel_means.append(float(np.mean(gp)))
            front_means.append(float(np.mean(gf)))
            rank_names.append(m)
            rank_case_scores[m] = {
                fn: (gp[i], gf[i]) for i, fn in enumerate(common)
            }

    if not names and not allow_partial:
        raise SystemExit(
            "[frontier-agreement] no model has complete three-judge M5 coverage"
        )
    out = {
        "panel_judges": ptags, "frontier_judges": ftags,
        "models_compared": names,
        "per_model": per_model,
        "pooled_majority_vs_majority": {
            c: {
                "n_model_cases": len(pooled_panel[c]),
                "cohen_kappa": _r(_cohen_kappa(pooled_panel[c], pooled_front[c])),
                "raw_agreement": _r(
                    sum(a == b for a, b in zip(pooled_panel[c], pooled_front[c]))
                    / len(pooled_panel[c])
                ) if pooled_panel[c] else None,
                "panel_vs_frontier_prf": (
                    _prf(pooled_front[c], pooled_panel[c]) if pooled_panel[c] else None
                ),
                "case_clustered_bootstrap_95": (
                    _case_bootstrap_kappa(pooled_case_pairs[c])
                    if pooled_case_pairs[c] else None
                ),
            }
            for c in CHECKS
        },
        "rank_preservation": {
            "models": rank_names,
            "excluded_reduced_case_models": sorted(set(names) - set(rank_names)),
            "n_models": len(rank_names),
            "spearman_rho_of_per_model_means": _r(_spearman(panel_means, front_means)),
            "pearson_r_of_per_model_means": _r(_pearson(panel_means, front_means)),
            "mean_level_shift_frontier_minus_panel": _r(
                float(np.mean(np.array(front_means) - np.array(panel_means)))
                if rank_names else float("nan")),
            "case_clustered_bootstrap_95": (
                _case_bootstrap_rank(rank_case_scores)
                if len(rank_case_scores) >= 3 else None
            ),
            "note": ("Only a representative high/mid/low three-model subset was "
                     "cross-judged. Rank correlations at n=3 are descriptive; the "
                     "case-clustered bootstrap captures case sampling, not model "
                     "sampling."),
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--premise-only", action="store_true")
    ap.add_argument("--m5-only", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report with fewer than 3 frontier judges (provisional only)")
    args = ap.parse_args()

    keep = _subset(args.n)
    payload: dict = {"subset": f"splits/frontier_validation_{args.n}.json",
                     "subset_size": len(keep)}

    if not args.m5_only:
        pa = premise_agreement(keep, allow_partial=args.allow_partial)
        payload["premise"] = pa
        print(f"\n=== PREMISE LABELS — {pa['cases']} cases, {pa['sentences']} sentences ===")
        print(f"{'premise':13s} {'kappa(maj)':>10s} {'raw':>6s} {'F1':>6s} "
              f"{'kappa_within: panel':>20s} {'frontier':>9s}")
        for p in PREMISES:
            d = pa["per_premise"][p]
            m, w = d["majority_vs_majority"], d["fleiss_kappa_within"]
            print(f"{p:13s} {_f(m['cohen_kappa']):>10} {_f(m['raw_agreement']):>6} "
                  f"{_f(m['panel_vs_frontier_prf']['f1']):>6} "
                  f"{_f(w['open_weight_panel_this_subset']):>20} "
                  f"{_f(w['frontier_panel']):>9}")

    if not args.premise_only:
        ma = m5_agreement(keep, allow_partial=args.allow_partial)
        payload["m5"] = ma
        print(f"\n=== M5 — {len(ma['models_compared'])} models ===")
        print(f"{'model':28s} {'n':>3s} {'g_panel':>8s} {'g_front':>8s} {'r':>6s} {'gate':>6s}")
        for m in ma["models_compared"]:
            d = ma["per_model"][m]
            print(f"{m:28s} {d['n_cases']:>3} {_f(d['mean_groundedness_panel']):>8} "
                  f"{_f(d['mean_groundedness_frontier']):>8} {_f(d['pearson_r']):>6} "
                  f"{_f(d['gate_0.90_agreement']):>6}")
        rp = ma["rank_preservation"]
        print(f"rank preservation across models: spearman="
              f"{rp['spearman_rho_of_per_model_means']} "
              f"level shift={rp['mean_level_shift_frontier_minus_panel']}")

    out = RESULTS / "_frontier_agreement.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[frontier-agreement] wrote {out}")


if __name__ == "__main__":
    main()
