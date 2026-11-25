"""Export real M3-selected pairs for a human polarity audit.

This consumes the bidirectional argmax pairs written by ``negation_probe``.
It does *not* infer contradiction prevalence. It only prioritises pairs where
one side contains an explicit standalone ``non`` and the other does not.
Different cue counts are neither necessary nor sufficient for contradiction:
the cue can sit in citation metadata, a subsidiary clause, or content present
on only one side. Human labels are therefore left blank by construction.

Usage:
    python -m analysis.m3_polarity_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DEFAULT_MATCHED = RESULTS / "_negation_probe_matched_pairs.jsonl"
DEFAULT_SUMMARY = RESULTS / "_m3_polarity_audit.json"
DEFAULT_SAMPLE = RESULTS / "_m3_polarity_audit_sample.csv"

_NON_RE = re.compile(r"\bnon\b", re.IGNORECASE)


def normalise_matched_row(row: dict) -> dict:
    """Give both M3 directions the same judge/model orientation."""
    if row["direction"] == "recall":
        judge_sentence, model_sentence = row["source"], row["match"]
    elif row["direction"] == "precision":
        model_sentence, judge_sentence = row["source"], row["match"]
    else:
        raise ValueError(f"unknown M3 direction: {row['direction']!r}")
    return {
        "filename": row["filename"],
        "premise": row["premise"],
        "judge_sentence": judge_sentence,
        "model_sentence": model_sentence,
        "cosine": float(row["cosine"]),
        "direction": row["direction"],
    }


def deduplicate_selected(rows: list[dict]) -> list[dict]:
    """Collapse a judge/model pair selected in both M3 directions."""
    by_key: dict[tuple[str, str, str, str], dict] = {}
    for raw in rows:
        row = normalise_matched_row(raw)
        key = (
            row["filename"], row["premise"],
            row["judge_sentence"], row["model_sentence"],
        )
        if key not in by_key:
            by_key[key] = {**row, "directions": [row["direction"]]}
        elif row["direction"] not in by_key[key]["directions"]:
            by_key[key]["directions"].append(row["direction"])
        by_key[key]["cosine"] = max(by_key[key]["cosine"], row["cosine"])
    return list(by_key.values())


def is_explicit_non_mismatch(row: dict) -> bool:
    judge_has = bool(_NON_RE.search(row["judge_sentence"]))
    model_has = bool(_NON_RE.search(row["model_sentence"]))
    return judge_has != model_has


def write_audit_sample(rows: list[dict], path: Path, threshold: float) -> None:
    candidates = sorted(
        (row for row in rows
         if is_explicit_non_mismatch(row) and row["cosine"] >= threshold),
        key=lambda row: (-row["cosine"], row["filename"], row["premise"]),
    )
    fields = [
        "pair_id", "filename", "premise", "directions", "cosine",
        "judge_non_count", "model_non_count",
        "judge_sentence", "model_sentence",
        "relation", "aligned_negation_scope", "legal_materiality", "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(candidates, start=1):
            writer.writerow({
                "pair_id": f"MP{i:03d}",
                "filename": row["filename"],
                "premise": row["premise"],
                "directions": "|".join(sorted(row["directions"])),
                "cosine": f"{row['cosine']:.8f}",
                "judge_non_count": len(_NON_RE.findall(row["judge_sentence"])),
                "model_non_count": len(_NON_RE.findall(row["model_sentence"])),
                "judge_sentence": row["judge_sentence"],
                "model_sentence": row["model_sentence"],
                "relation": "",
                "aligned_negation_scope": "",
                "legal_materiality": "",
                "notes": "",
            })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    ap.add_argument("--out", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--sample-out", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--audit-threshold", type=float, default=0.80,
                    help="prioritisation threshold only; M3 itself has none")
    ap.add_argument("--reference-model", default="qwen3.5-397b-a17b")
    args = ap.parse_args()

    raw: list[dict] = []
    with args.matched.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    selected = deduplicate_selected(raw)
    mismatches = [row for row in selected if is_explicit_non_mismatch(row)]

    thresholds = [0.0, 0.6, 0.7, 0.8, 0.9]
    summary = {
        "reference_model": args.reference_model,
        "source": str(args.matched.relative_to(ROOT)),
        "interpretation": (
            "candidate explicit-negation cue mismatches only; contradiction "
            "prevalence remains unmeasured until the CSV is human-annotated"
        ),
        "counts": {
            "raw_bidirectional_argmax_records": len(raw),
            "unique_selected_pairs": len(selected),
            "unique_explicit_non_mismatches": len(mismatches),
            "cases_with_explicit_non_mismatch": len({
                row["filename"] for row in mismatches
            }),
            "by_minimum_cosine": {
                f"{threshold:.1f}": sum(
                    row["cosine"] >= threshold for row in mismatches
                )
                for threshold in thresholds
            },
            "audit_rows": sum(
                row["cosine"] >= args.audit_threshold for row in mismatches
            ),
        },
        "audit_threshold": args.audit_threshold,
        "audit_threshold_is_metric_threshold": False,
        "human_validation_status": "pending",
        "audit_sample": str(args.sample_out.relative_to(ROOT)),
    }
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_audit_sample(selected, args.sample_out, args.audit_threshold)

    print(json.dumps(summary["counts"], indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.sample_out} (blank human-validation fields)")


if __name__ == "__main__":
    main()
