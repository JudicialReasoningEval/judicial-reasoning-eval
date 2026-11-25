"""M3 — Allineamento semantico del ragionamento.

Validates: *"the LLM reasons about the right content"* — within each premise,
the LLM says semantically the same things the judge does.

Embeddings: Qwen/Qwen3-Embedding-8B (see embeddings.py). Soft cosine, NO hard
threshold (a hard threshold would be a free parameter). For premise p in
case i, with judge sentences J_p and LLM sentences L_p:

    Rec_{i,p}  = mean_{s in J_p}  max_{t in L_p} cos(s, t)   (judge covered)
    Prec_{i,p} = mean_{t in L_p}  max_{s in J_p} cos(s, t)   (LLM grounded)
    F1_{i,p}   = 2·Prec·Rec / (Prec + Rec)
    M3_i       = mean over judge-observed premises of F1_{i,p}

The precision term (LLM axis) stops a verbose model from scoring well on
recall alone. Premises absent on the judge side are not
counted against the model.

CLI:
  python m3_semantic.py --results-dir results/<model>
    Reads:  results/sillogismo_judge.jsonl
            <results-dir>/sillogismo_labels.jsonl
    Writes: <results-dir>/m3_semantic.jsonl    (per-case)
            <results-dir>/m3_semantic.json     (aggregate + per-premise CIs)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metrics.embeddings import DEFAULT_MODEL, cosine_matrix, embed
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


def _premise_sentences(record: dict, premise: str) -> list[str]:
    idxs = record.get("by_premise", {}).get(premise, [])
    sents = record.get("sentences", [])
    return [sents[i] for i in idxs if 0 <= i < len(sents) and sents[i].strip()]


def _soft_f1(sim: np.ndarray) -> tuple[float, float, float]:
    """sim has shape (n_judge, n_llm). Returns (precision, recall, f1).

    Cosine is clipped to [0, 1]: a sentence with no positively-similar match on
    the other side contributes 0 (unrelated), never a negative score — otherwise
    F1 = 2PR/(P+R) is ill-defined for negative similarities.
    """
    sim = np.clip(sim, 0.0, 1.0)
    recall = float(np.mean(sim.max(axis=1)))     # per judge sentence: best LLM match
    precision = float(np.mean(sim.max(axis=0)))  # per LLM sentence: best judge match
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def score_case(judge_rec: dict, llm_rec: dict, model_name: str = DEFAULT_MODEL) -> dict:
    per_premise: dict[str, dict] = {}
    observed: list[str] = []
    for p in PREMISES:
        j_sents = _premise_sentences(judge_rec, p)
        l_sents = _premise_sentences(llm_rec, p)
        if not j_sents:
            per_premise[p] = {"judge_present": False, "f1": None}
            continue
        observed.append(p)
        if not l_sents:
            per_premise[p] = {"judge_present": True, "llm_present": False,
                              "precision": 0.0, "recall": 0.0, "f1": 0.0,
                              "n_judge": len(j_sents), "n_llm": 0}
            continue
        sim = cosine_matrix(embed(j_sents, model_name), embed(l_sents, model_name))
        precision, recall, f1 = _soft_f1(sim)
        per_premise[p] = {"judge_present": True, "llm_present": True,
                          "precision": precision, "recall": recall, "f1": f1,
                          "n_judge": len(j_sents), "n_llm": len(l_sents)}

    m3 = float(np.mean([per_premise[p]["f1"] for p in observed])) if observed else float("nan")
    return {
        "filename": judge_rec["filename"],
        "premises": per_premise,
        "observed_premises": observed,
        "m3": m3,
    }


def score_model(results_dir: Path, judge_path: Path = JUDGE_SILLOGISMO,
                model_name: str = DEFAULT_MODEL) -> dict:
    llm_path = results_dir / "sillogismo_labels.jsonl"
    if not judge_path.exists():
        raise SystemExit(f"judge sillogismo not found: {judge_path}")
    if not llm_path.exists():
        raise SystemExit(f"LLM sillogismo not found: {llm_path}")

    judge_by_fn = _read_jsonl_by_filename(judge_path)
    llm_by_fn = _read_jsonl_by_filename(llm_path)

    per_case: list[dict] = []
    per_premise_f1: dict[str, list[float]] = {p: [] for p in PREMISES}
    out_jsonl = results_dir / "m3_semantic.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for fn, judge_rec in judge_by_fn.items():
            if fn not in llm_by_fn:
                continue
            scored = score_case(judge_rec, llm_by_fn[fn], model_name=model_name)
            fh.write(json.dumps(scored, ensure_ascii=False) + "\n")
            if not np.isnan(scored["m3"]):
                per_case.append(scored)
            for p in PREMISES:
                row = scored["premises"][p]
                if row.get("judge_present") and row.get("f1") is not None:
                    per_premise_f1[p].append(row["f1"])

    m3_vals = [r["m3"] for r in per_case]
    ci = bootstrap_ci(m3_vals, n_resamples=1000, seed=42)

    def _premise_block(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "f1_mean": None}
        pci = bootstrap_ci(vals, n_resamples=1000, seed=42)
        return {"n": len(vals), "f1_mean": pci["mean"], "f1_ci_lo": pci["lo"], "f1_ci_hi": pci["hi"]}

    agg = {
        "n_cases": len(per_case),
        "embedding_model": model_name,
        "m3": {"mean": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"]},
        "per_premise": {p: _premise_block(vals) for p, vals in per_premise_f1.items()},
    }
    (results_dir / "m3_semantic.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"M3 semantic (n={agg['n_cases']}, {model_name}): {agg['m3']['mean']:.3f} "
          f"[{agg['m3']['ci_lo']:.3f}, {agg['m3']['ci_hi']:.3f}]")
    for p in PREMISES:
        st = agg["per_premise"][p]
        if st["n"]:
            print(f"  {p:12s} n={st['n']:>3}  f1={st['f1_mean']:.3f}")
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--judge", type=Path, default=JUDGE_SILLOGISMO)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model name")
    args = parser.parse_args()
    score_model(args.results_dir, judge_path=args.judge, model_name=args.model)


if __name__ == "__main__":
    main()
