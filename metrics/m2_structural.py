"""M2 — Fedeltà strutturale del sillogismo.

Validates: *"the LLM reasons with the right structure"* — the SHAPE of the
reasoning (presence and proportion of the three premises), independently of
what each premise says (that is M3's job).

For each case i, from the 3-premise sentence labels (`by_premise`) on both
sides we build the premise-proportion distributions q^J_i and q^L_i over
P = {maggiore, minore, conclusione} and combine two orthogonal sub-scores:

    StructSim_i  = 1 - JSD(q^J_i || q^L_i)        (JSD in base 2 -> [0,1])
    PresenceF1_i = F1( 1[q^J>0], 1[q^L>0] )       (which premises are present)
    M2_i         = (StructSim_i + PresenceF1_i) / 2

StructSim catches wrong *proportions* (e.g. over-weighting the conclusione);
PresenceF1 catches *missing* premises (e.g. skipping the maggiore entirely).

Cases with no premise content on the judge side are skipped (no reference
structure to compare against).

CLI:
  python m2_structural.py --results-dir results/<model>
    Reads:  results/sillogismo_judge.jsonl           (judge side)
            <results-dir>/sillogismo_labels.jsonl    (LLM side)
    Writes: <results-dir>/m2_structural.jsonl        (per-case)
            <results-dir>/m2_structural.json         (aggregate + CI)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metrics.premise_labeler import PREMISES
from metrics.stats_utils import bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
JUDGE_SILLOGISMO = RESULTS_DIR / "sillogismo_judge.jsonl"


def _read_jsonl_by_filename(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a half-written line from an interrupted run
            out[r["filename"]] = r
    return out


def _premise_counts(record: dict) -> np.ndarray:
    """Membership count per premise (a sentence can map to >1 premise)."""
    by = record.get("by_premise", {})
    return np.array([len(by.get(p, [])) for p in PREMISES], dtype=np.float64)


def _jsd_base2(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in base 2, bounded in [0, 1].

    Assumes p, q are non-negative and at least one entry positive each.
    """
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _presence_f1(j: np.ndarray, l: np.ndarray) -> float:
    jset = j > 0
    lset = l > 0
    tp = int(np.sum(jset & lset))
    fp = int(np.sum(lset & ~jset))
    fn = int(np.sum(jset & ~lset))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def score_case(judge_rec: dict, llm_rec: dict) -> dict | None:
    """Per-case M2; returns None when the judge side has no structure."""
    j = _premise_counts(judge_rec)
    l = _premise_counts(llm_rec)
    if j.sum() == 0:
        return None  # no reference structure

    if l.sum() == 0:
        struct_sim = 0.0  # LLM produced no premise structure at all
    else:
        struct_sim = 1.0 - _jsd_base2(j, l)

    presence_f1 = _presence_f1(j, l)
    m2 = 0.5 * (struct_sim + presence_f1)
    return {
        "filename": judge_rec["filename"],
        "judge_counts": {p: int(j[i]) for i, p in enumerate(PREMISES)},
        "llm_counts": {p: int(l[i]) for i, p in enumerate(PREMISES)},
        "struct_sim": struct_sim,
        "presence_f1": presence_f1,
        "m2": m2,
    }


def score_model(results_dir: Path, judge_path: Path = JUDGE_SILLOGISMO) -> dict:
    llm_path = results_dir / "sillogismo_labels.jsonl"
    if not judge_path.exists():
        raise SystemExit(f"judge sillogismo not found: {judge_path}")
    if not llm_path.exists():
        raise SystemExit(f"LLM sillogismo not found: {llm_path}")

    judge_by_fn = _read_jsonl_by_filename(judge_path)
    llm_by_fn = _read_jsonl_by_filename(llm_path)

    per_case: list[dict] = []
    for fn, judge_rec in judge_by_fn.items():
        if fn not in llm_by_fn:
            continue
        scored = score_case(judge_rec, llm_by_fn[fn])
        if scored is not None:
            per_case.append(scored)

    out_jsonl = results_dir / "m2_structural.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in per_case:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    m2_vals = [r["m2"] for r in per_case]
    ci = bootstrap_ci(m2_vals, n_resamples=1000, seed=42)
    agg = {
        "n_cases": len(per_case),
        "m2": {"mean": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]},
        "struct_sim_mean": float(np.mean([r["struct_sim"] for r in per_case])) if per_case else None,
        "presence_f1_mean": float(np.mean([r["presence_f1"] for r in per_case])) if per_case else None,
    }
    (results_dir / "m2_structural.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    def _f(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"
    if agg["n_cases"] == 0:
        print(f"WARN: M2 structural — no overlapping cases for {results_dir.name} (empty/missing labels?)")
    print(f"M2 structural (n={agg['n_cases']}): {_f(agg['m2']['mean'])} "
          f"[{_f(agg['m2']['ci_lo'])}, {_f(agg['m2']['ci_hi'])}]  "
          f"(struct_sim={_f(agg['struct_sim_mean'])}, presence_f1={_f(agg['presence_f1_mean'])})")
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--judge", type=Path, default=JUDGE_SILLOGISMO)
    args = parser.parse_args()
    score_model(args.results_dir, judge_path=args.judge)


if __name__ == "__main__":
    main()
