"""Order of the premises in the court's own Considerato (court side, no model output).

    python -m analysis.premise_order

Reads the panel-consensus sentence labels of the court's text
(results/sillogismo_judge.jsonl, written by metrics.consensus_labels) and
reports, for every labelled judgment and for the 91 evaluated test cases:

  opens_with_conclusion   the first labelled sentence carries `conclusione`
  canonical_order         the label sequence never steps back
                          (major -> minor -> conclusion; a sentence with several
                          labels counts as its earliest premise; unlabelled
                          sentences are skipped; consecutive repeats collapse)
  canonical_all_three     canonical_order and all three premises occur
  returns_to_major        how many times a major-premise block starts after
                          another block

This is why M2 compares premise proportions rather than premise order: the
court itself rarely runs major -> minor -> conclusion.

Writes results/_premise_order.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
JUDGE = RESULTS / "sillogismo_judge.jsonl"
EVALUATED = ROOT / "splits" / "evaluated_91.json"

ORDER = {"maggiore": 0, "minore": 1, "conclusione": 2}


def read_jsonl(path: Path) -> list[dict]:
    """Split on newlines only: legal text can contain U+2028/U+2029."""
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def describe(rec: dict) -> dict | None:
    labelled = [l for l in rec["premise_labels"] if l]
    if not labelled:
        return None
    seq = [min(ORDER[x] for x in l) for l in labelled]
    runs = [seq[0]]
    for s in seq[1:]:
        if s != runs[-1]:
            runs.append(s)
    canonical = runs == sorted(runs)
    return {
        "opens_with_conclusion": "conclusione" in labelled[0],
        "canonical_order": canonical,
        "canonical_all_three": canonical and len(set(runs)) == 3,
        "returns_to_major": sum(1 for i in range(1, len(runs)) if runs[i] == 0),
        "blocks": len(runs),
    }


def summarise(recs: list[dict]) -> dict:
    rows = [d for d in (describe(r) for r in recs) if d]
    n = len(rows)
    out = {"n": n}
    for k in ("opens_with_conclusion", "canonical_order", "canonical_all_three"):
        c = sum(r[k] for r in rows)
        out[k] = {"count": c, "share": c / n}
    out["returns_to_major_mean"] = float(np.mean([r["returns_to_major"] for r in rows]))
    out["blocks_mean"] = float(np.mean([r["blocks"] for r in rows]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", type=Path, default=JUDGE)
    ap.add_argument("--out", type=Path, default=RESULTS / "_premise_order.json")
    args = ap.parse_args()

    recs = read_jsonl(args.judge)
    test = set(json.loads(EVALUATED.read_text(encoding="utf-8"))["evaluated_test"])
    payload = {
        "source": str(args.judge.relative_to(ROOT)) if args.judge.is_relative_to(ROOT) else str(args.judge),
        "all_labelled_judgments": summarise(recs),
        "test_91": summarise([r for r in recs if r["filename"] in test]),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
