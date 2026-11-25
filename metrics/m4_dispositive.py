"""M4 — Accordo sul dispositivo (gate).

Validates: *"the LLM reaches the right decision"* — the LLM's P.Q.M. lands on
the same outcome the judge did.

The Cassazione dispositive is a closed, formulaic vocabulary, so a
deterministic extractor has ~0 error and is far more defensible than an
LLM-as-judge. The SAME extractor is applied to both sides (no judge bias).

Two axes:
  * FINE   — the canonical class (annulla con/senza rinvio, rigetta,
             inammissibile, accoglie, estinto). Diagnostic.
  * COARSE — the legally meaningful macro-outcome from the appeal's point of
             view: accolto / respinto / estinto. This is what M4 GATES on,
             because inammissibile-vs-rigetta and annulla-vs-annulla-senza-rinvio
             are the *same* outcome for the parties.

Reject synonyms (rifiuta / respinge) are folded into the reject family: an LLM
that clearly rejects the appeal but uses non-court wording reaches the same
*outcome* — M4 measures the outcome, not the vocabulary. Genuinely garbled /
hallucinated P.Q.M. still fall through to `unparsable` (a real quality signal).

Judge side: the disposition is read from the P.Q.M. section of the judgment
file, falling back to the stored `pqm_ground_truth` field.

Agreement: accuracy AND Cohen's kappa on both axes. Unparsable LLM P.Q.M. are
reported separately (resolve by hand) and excluded from kappa.

CLI:
  python m4_dispositive.py --results-dir results/<model>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "sentenze_md_filtered"
PQM_HEADING = re.compile(r"P\.?\s*Q\.?\s*M\.?|PER\s+QUESTI\s+MOTIVI", re.I)

# Ordered: specific first; bare `annulla` before reject so mixed dispositions
# (annulla X / rigetta Y) resolve to the accoglimento. First match wins.
DISPOSITION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("annulla_senza_rinvio", re.compile(r"annull\w*\s+senza\s+rinvio", re.I)),
    ("annulla_con_rinvio",   re.compile(r"annull\w*\s+con\s+rinvio", re.I)),
    ("annulla",              re.compile(r"\bannull\w+", re.I)),
    ("accoglie",             re.compile(r"\baccogl\w+", re.I)),
    ("estinto",              re.compile(r"\bestin\w+", re.I)),
    ("inammissibile",        re.compile(r"inammissibil\w*", re.I)),
    ("rigetta",              re.compile(r"\brigett\w+|\brifiut\w+|\bresping\w+|\brespint\w+", re.I)),
]

DISPOSITIONS = [name for name, _ in DISPOSITION_PATTERNS]

COARSE_MAP: dict[str, str] = {
    "annulla_senza_rinvio": "accolto",
    "annulla_con_rinvio":   "accolto",
    "annulla":              "accolto",
    "accoglie":             "accolto",
    "inammissibile":        "respinto",
    "rigetta":              "respinto",
    "estinto":              "estinto",
}


def extract_disposition(text: str) -> str | None:
    """Return the canonical (fine) dispositive class, or None if unparsable."""
    if not text:
        return None
    for name, pat in DISPOSITION_PATTERNS:
        if pat.search(text):
            return name
    return None


def coarse(fine: str | None) -> str | None:
    return COARSE_MAP.get(fine) if fine else None


def judge_pqm_text(row: dict) -> str:
    """Authoritative judge P.Q.M.: the .md section, else the stored field."""
    fn = row.get("filename", "")
    p = CORPUS_DIR / fn
    if p.exists():
        text = p.read_text(encoding="utf-8", errors="ignore")
        parts = PQM_HEADING.split(text)
        if len(parts) > 1 and parts[-1].strip():
            return parts[-1]
    return row.get("pqm_ground_truth", "") or ""


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = {l: a.count(l) / n for l in labels}
    cb = {l: b.count(l) / n for l in labels}
    pe = sum(ca[l] * cb[l] for l in labels)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a half-written line from an interrupted run
    return rows


def _agreement(pairs_j: list[str], pairs_l: list[str]) -> dict:
    n = len(pairs_j)
    acc = (sum(1 for j, l in zip(pairs_j, pairs_l) if j == l) / n) if n else float("nan")
    return {"n_paired": n, "accuracy": acc, "cohen_kappa": cohen_kappa(pairs_j, pairs_l) if n else float("nan")}


def score_model(results_dir: Path) -> dict:
    src = results_dir / "inference_results.jsonl"
    if not src.exists():
        raise SystemExit(f"inference results not found: {src}")
    rows = _read_jsonl(src)

    per_case: list[dict] = []
    fine_j: list[str] = []; fine_l: list[str] = []
    coarse_j: list[str] = []; coarse_l: list[str] = []
    n_unparsable_llm = 0; n_unparsable_judge = 0

    for r in rows:
        jd = extract_disposition(judge_pqm_text(r))
        ld = extract_disposition(r.get("pqm_predicted", "") or "")
        if jd is None:
            n_unparsable_judge += 1
        if ld is None:
            n_unparsable_llm += 1
        per_case.append({
            "filename": r.get("filename"),
            "judge_fine": jd, "llm_fine": ld,
            "judge_coarse": coarse(jd), "llm_coarse": coarse(ld),
            "fine_agree": (jd is not None and jd == ld),
            "coarse_agree": (jd is not None and ld is not None and coarse(jd) == coarse(ld)),
            "llm_unparsable": ld is None,
        })
        if jd is not None and ld is not None:
            fine_j.append(jd); fine_l.append(ld)
            coarse_j.append(coarse(jd)); coarse_l.append(coarse(ld))

    with (results_dir / "m4_dispositive.jsonl").open("w", encoding="utf-8") as fh:
        for row in per_case:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    fine = _agreement(fine_j, fine_l)
    coarse_agg = _agreement(coarse_j, coarse_l)

    dist_fine: dict[str, int] = {}
    for j in fine_j:
        dist_fine[j] = dist_fine.get(j, 0) + 1
    dist_coarse: dict[str, int] = {}
    for j in coarse_j:
        dist_coarse[j] = dist_coarse.get(j, 0) + 1

    agg = {
        "n_cases": len(rows),
        "n_unparsable_llm": n_unparsable_llm,
        "n_unparsable_judge": n_unparsable_judge,
        "unparsable_llm_rate": n_unparsable_llm / len(rows) if rows else 0.0,
        "fine": fine,
        "coarse": coarse_agg,
        "cohen_kappa": coarse_agg["cohen_kappa"],   # the gate uses the coarse axis
        "accuracy": coarse_agg["accuracy"],
        "judge_fine_distribution": dist_fine,
        "judge_coarse_distribution": dist_coarse,
    }
    (results_dir / "m4_dispositive.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"M4 dispositive (paired={coarse_agg['n_paired']}/{len(rows)}, "
          f"judge-unparsable={n_unparsable_judge}, llm-unparsable={n_unparsable_llm}):")
    print(f"  COARSE (gate)  acc={coarse_agg['accuracy']:.3f}  kappa={coarse_agg['cohen_kappa']:.3f}  "
          f"dist={dist_coarse}")
    print(f"  fine           acc={fine['accuracy']:.3f}  kappa={fine['cohen_kappa']:.3f}")
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    score_model(args.results_dir)


if __name__ == "__main__":
    main()
