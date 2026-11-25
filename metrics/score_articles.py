"""Extract cited corruption-article labels from LLM-generated text.

The generation step asks the LLM to produce both `CONSIDERATO IN DIRITTO`
and `P.Q.M.`. We need a `predicted_articles` list per case so the
already-validated `stats_utils.multi_label_eval()` can score article
recall/precision/F1.

`extract_articles(text)` reuses the regex patterns from
`baselines.KW_BY_LABEL` but drops the threshold/ratio gating that the
keyword *baseline* applies: for an LLM's own output we want every
article it actually cites, not a thresholded approximation.

CLI:
  python score_articles.py --results-dir results/<model>
    Reads `<dir>/inference_results.jsonl`, extracts predicted_articles
    from `considerato_predicted + pqm_predicted` (falls back to
    `pqm_predicted` alone if considerato_predicted is missing), refreshes
    `gt_articles` from metadata.json, and writes
    `<dir>/inference_results_scored.jsonl` next to the original.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.baselines import (
    ARTICLE_LABELS,
    KW_BY_LABEL,
    gt_articles_from_rnc,
    load_metadata_map,
)


def extract_articles(text: str) -> list[str]:
    """Return the sorted list of articles cited in `text`.

    No threshold filtering — any match is enough. Uses the same regex
    patterns as `baselines.KW_BY_LABEL` for cross-pipeline consistency.
    """
    if not text:
        return []
    hits = [lbl for lbl, pat in KW_BY_LABEL.items() if pat.search(text)]
    return sorted(hits, key=ARTICLE_LABELS.index)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    # split on \n only: generated text can embed U+2028/U+2029, which
    # str.splitlines() treats as record boundaries, corrupting the parse
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def score_inference_dir(results_dir: Path) -> Path:
    """Read `results_dir/inference_results.jsonl`, re-extract
    predicted_articles and gt_articles, write the scored file next to
    the original. Returns the output path.
    """
    src = results_dir / "inference_results.jsonl"
    if not src.exists():
        raise FileNotFoundError(src)
    out = results_dir / "inference_results_scored.jsonl"

    meta = load_metadata_map()
    rows = _read_jsonl(src)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            considerato = r.get("considerato_predicted", "") or ""
            pqm = r.get("pqm_predicted", "") or ""
            combined = (considerato + "\n" + pqm).strip()
            r["predicted_articles"] = extract_articles(combined)

            entry = meta.get(r.get("filename", ""))
            if entry is not None:
                r["gt_articles"] = sorted(
                    gt_articles_from_rnc(entry), key=ARTICLE_LABELS.index
                )
            else:
                r.setdefault("gt_articles", [])
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Per-model output directory containing inference_results.jsonl")
    parser.add_argument("--score", action="store_true",
                        help="Also print multi_label_eval over the scored output")
    args = parser.parse_args()

    out = score_inference_dir(args.results_dir)
    print(f"Wrote {out}")

    if args.score:
        try:
            from metrics.stats_utils import format_multi_label_table, multi_label_eval
        except ImportError as exc:
            sys.exit(f"--score requires stats_utils: {exc}")
        result = multi_label_eval(out, labels=ARTICLE_LABELS)
        print()
        print(format_multi_label_table({args.results_dir.name: result}, labels=ARTICLE_LABELS))


if __name__ == "__main__":
    main()
