"""
Trivial baselines for the corruption-article classification task.

They give the LLM results a concrete floor: does a model beat random, majority
or a regex baseline? Every baseline runs on the 91 evaluated test cases and
reads the same facts section the models receive (pipeline/case_sections.py).

This is a **multi-label** task. Cassazione judgments commonly cite several
corruption articles at once (e.g. 319 + 321 = "propria corruption +
corruptor's liability"), and forcing a single label per case throws away
information and crushes the rare classes (320 has 9 raw GT cases but only 2
under a priority projection). So each case has a *set* of ground-truth articles and each
baseline emits a *set* of predicted articles.

Label space (5 corruption articles, no acquittal):
    318  corruzione per l'esercizio della funzione (impropria)
    319  corruzione per atto contrario ai doveri d'ufficio (propria) —
         absorbs 319-bis / -ter / -quater
    320  corruzione di persona incaricata di pubblico servizio
    321  pene per il corruttore
    322  istigazione alla corruzione

Acquittal is intentionally OUT of the article task: it is an *outcome*
dimension (did the Cassazione confirm / requalify / quash with full
acquittal), orthogonal to which article was charged; M4 measures it.

Baselines:
    random    Per-article independent Bernoulli(0.5). The "no information"
              floor — high recall, low precision.
    majority  Always predict the set of articles whose corpus prevalence
              ≥ 0.5 (so a class is predicted iff over half of cases carry
              it). On this corpus that yields {319, 321}.
    keyword   Per-article regex over the case text. An article is included
              in the predicted set when its hit count is ≥ KW_THRESHOLD
              (default 1) AND ≥ KW_RATIO × max_hit. Falls back to the
              single best-scoring article when no article hits threshold.

Ground truth is sourced from `sentenze_md/metadata.json` (`rnc-art` /
`rnc-gen` pulled from the italgiure Solr index — the rubricated articles
the Court itself tagged on each judgment). The lenient rule from
`filter_sentenze.doc_cites_target_articles` applies: prefer CP-tagged
entries, fall back to bare rnc-art prefix matches when the gen tag is
missing.

Usage:
    python -m pipeline.baselines --baseline random --seed 42
    python -m pipeline.baselines --baseline majority
    python -m pipeline.baselines --baseline keyword
    python -m pipeline.baselines --baseline all   # runs all three, prints F1 table

Output (per baseline):
    results/baseline-<name>/inference_results.jsonl

Record schema (compatible with the rest of the pipeline; the multi-label
fields sit alongside single-label projections):
    {filename, judge_reasoning_trace, pqm_predicted, pqm_ground_truth,
     finish_reason, seed, baseline,
     predicted_articles: list[str],   # the multi-label prediction
     gt_articles:        list[str],   # the multi-label ground truth
     gt_source:          str,         # 'metadata' | 'none'
     predicted_label:    str,         # the most-confident single label
     gt_label:           str | None}  # priority projection of gt_articles
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
INPUT_DIR     = ROOT / "sentenze_md_filtered"
RESULTS_ROOT  = ROOT / "results"
METADATA_PATH = ROOT / "sentenze_md" / "metadata.json"

ARTICLE_LABELS = ["318", "319", "320", "321", "322"]

# Single-label projection for the `gt_label` / `predicted_label` fields; the
# metrics use the multi-label fields.
LEGACY_PRIORITY = ["322", "319", "318", "320", "321"]

# Keyword patterns. Phrasings drawn from the codice penale rubrics
# (filter_sentenze.TITLE_KEYWORD_RES) and common Cassazione legalese.
# The 319 bucket absorbs 319-bis/ter/quater.
KW_318 = re.compile(
    r"\b(art(?:icolo)?\.?\s*318|corruzione\s+impropria|"
    r"corruzione\s+per\s+l['‘’`]esercizio\s+della\s+funzione|"
    r"esercizio\s+delle\s+funzioni|munificenza)\b",
    re.IGNORECASE,
)
KW_319 = re.compile(
    r"\b(art(?:icolo)?\.?\s*319(?:[\s\-](?:bis|ter|quater))?|"
    r"corruzione\s+propria|atto\s+contrario|specifico\s+atto|"
    r"corruzione\s+per\s+(?:un\s+)?atto\s+contrario|"
    r"corruzione\s+per\s+atti\s+giudiziari|"
    r"induzione\s+indebita\s+a\s+dare)\b",
    re.IGNORECASE,
)
KW_320 = re.compile(
    r"\b(art(?:icolo)?\.?\s*320|"
    r"corruzione\s+di\s+persona\s+incaricata|"
    r"incaricato\s+di\s+(?:un\s+)?pubblico\s+servizio)\b",
    re.IGNORECASE,
)
KW_321 = re.compile(
    r"\b(art(?:icolo)?\.?\s*321|pene\s+per\s+il\s+corruttore|corruttore)\b",
    re.IGNORECASE,
)
KW_322 = re.compile(
    r"\b(art(?:icolo)?\.?\s*322|istigazione\s+alla\s+corruzione)\b",
    re.IGNORECASE,
)
KW_BY_LABEL = {"318": KW_318, "319": KW_319, "320": KW_320, "321": KW_321, "322": KW_322}

# Keyword baseline thresholds — tuned to avoid both empty predictions
# (no article cleared the bar) and the inverse failure (every keyword hit
# counts so every case predicts everything).
KW_THRESHOLD = 1     # minimum absolute hit count to include an article
KW_RATIO     = 0.25  # minimum hit count as a fraction of the top scorer

# Map from zero-padded rnc-art prefix → our label.
RNC_TO_LABEL = {"0318": "318", "0319": "319", "0320": "320", "0321": "321", "0322": "322"}


def load_metadata_map() -> dict[str, dict]:
    """Return {md_filename: metadata_entry} for the scraped corpus."""
    if not METADATA_PATH.exists():
        return {}
    raw = json.loads(METADATA_PATH.read_text())
    out: dict[str, dict] = {}
    for e in raw:
        fn = e.get("filename", [])
        fn = fn[0] if isinstance(fn, list) and fn else (fn if isinstance(fn, str) else "")
        if not fn:
            continue
        md_name = Path(fn).name.replace(".pdf", ".clean.md")
        out[md_name] = e
    return out


def gt_articles_from_rnc(entry: dict) -> set[str]:
    """Return the set of corruption articles cited in this case.

    Lenient rule: prefer CP-tagged (Codice Penale) entries; fall back to
    any rnc-art prefix match when the gen tag is missing — matches
    filter_sentenze.doc_cites_target_articles.
    """
    rnc_art = entry.get("rnc-art", [])
    rnc_gen = entry.get("rnc-gen", [])
    cp_labels: set[str] = set()
    any_labels: set[str] = set()
    for a, g in zip(rnc_art, rnc_gen):
        prefix = a[:4]
        if prefix in RNC_TO_LABEL:
            any_labels.add(RNC_TO_LABEL[prefix])
            if g == "CP":
                cp_labels.add(RNC_TO_LABEL[prefix])
    return cp_labels or any_labels


def project_to_single_label(labels: set[str]) -> str | None:
    """Pick one label from a set via LEGACY_PRIORITY (for the
    `gt_label` / `predicted_label` fields)."""
    for cand in LEGACY_PRIORITY:
        if cand in labels:
            return cand
    return None


# ── Multi-label baseline predictors ──────────────────────────────────────────

def predict_random_multi(case_input: str, rng: random.Random) -> set[str]:
    """Per-article Bernoulli(0.5). Uninformed prior over the power set."""
    return {lbl for lbl in ARTICLE_LABELS if rng.random() < 0.5}


def predict_majority_multi(case_input: str, prior_set: set[str]) -> set[str]:
    """Always predict the set of articles whose corpus prevalence ≥ 0.5."""
    return set(prior_set)


def predict_keyword_multi(case_input: str) -> set[str]:
    """Per-article regex; include articles whose hit count clears
    KW_THRESHOLD and is at least KW_RATIO × the top scorer. If no article
    clears the bar, fall back to the single best-scoring article (so the
    set is never empty — empty predictions have undefined precision)."""
    scores = {lbl: len(pat.findall(case_input)) for lbl, pat in KW_BY_LABEL.items()}
    top = max(scores.values()) if scores else 0
    if top == 0:
        return set()
    cutoff = max(KW_THRESHOLD, KW_RATIO * top)
    chosen = {lbl for lbl, s in scores.items() if s >= cutoff}
    if not chosen:
        chosen = {max(scores, key=lambda k: (scores[k], -LEGACY_PRIORITY.index(k) if k in LEGACY_PRIORITY else -99))}
    return chosen


# ── Output rendering ─────────────────────────────────────────────────────────

ARTICLE_PHRASE = {
    "318": "corruzione impropria (art. 318 c.p.)",
    "319": "corruzione propria (art. 319 c.p.)",
    "320": "corruzione di persona incaricata di un pubblico servizio (art. 320 c.p.)",
    "321": "responsabilità del corruttore (art. 321 c.p.)",
    "322": "istigazione alla corruzione (art. 322 c.p.)",
}


def labels_to_pqm(labels: set[str]) -> str:
    """Render a label set as a P.Q.M.-shaped string. Empty → assolutio
    placeholder so downstream text-similarity scoring still gets something."""
    if not labels:
        return "P.Q.M.\n\nNessun reato di corruzione integrato dai fatti."
    ordered = [a for a in ARTICLE_LABELS if a in labels]
    body = "; ".join(ARTICLE_PHRASE[a] for a in ordered)
    return f"P.Q.M.\n\nDichiara che i fatti integrano i reati di {body}."


# ── Main loop ────────────────────────────────────────────────────────────────

def compute_corpus_prior(cases: list[dict], threshold: float = 0.5) -> set[str]:
    """Articles whose prevalence in the GT distribution is ≥ threshold."""
    n = sum(1 for c in cases if c["gt_articles"])
    if n == 0:
        return set()
    counts = Counter()
    for c in cases:
        for lbl in c["gt_articles"]:
            counts[lbl] += 1
    return {lbl for lbl, k in counts.items() if (k / n) >= threshold}


def run_baseline(baseline: str, seed: int, cases: list[dict], prior_set: set[str]) -> Path:
    rng = random.Random(seed)
    out_dir = RESULTS_ROOT / f"baseline-{baseline}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "inference_results.jsonl"

    with out_file.open("w", encoding="utf-8") as fh:
        for c in cases:
            if baseline == "random":
                pred_set = predict_random_multi(c["input"], rng)
            elif baseline == "majority":
                pred_set = predict_majority_multi(c["input"], prior_set)
            elif baseline == "keyword":
                pred_set = predict_keyword_multi(c["input"])
            else:
                sys.exit(f"unknown baseline: {baseline}")
            pred_list = [a for a in ARTICLE_LABELS if a in pred_set]
            gt_list   = [a for a in ARTICLE_LABELS if a in c["gt_articles"]]
            record = {
                "filename":              c["filename"],
                "judge_reasoning_trace": f"[baseline:{baseline}] no reasoning generated",
                "pqm_predicted":         labels_to_pqm(pred_set),
                "pqm_ground_truth":      c["pqm_ground_truth"],
                "finish_reason":         "stop",
                "seed":                  seed,
                "baseline":              baseline,
                "predicted_articles":    pred_list,
                "gt_articles":           gt_list,
                "gt_source":             c["gt_source"],
                # Single-label projections, derived via LEGACY_PRIORITY.
                "predicted_label":       project_to_single_label(pred_set) or "",
                "gt_label":              project_to_single_label(c["gt_articles"]),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_file


def _summarize_corpus(cases: list[dict]) -> None:
    source_counts = Counter(c["gt_source"] for c in cases)
    label_tally = Counter()
    set_size = Counter()
    for c in cases:
        for lbl in c["gt_articles"]:
            label_tally[lbl] += 1
        set_size[len(c["gt_articles"])] += 1
    n = len(cases)
    print(f"GT source: {dict(source_counts)}")
    print(f"GT article multi-label tally (out of {n} cases):")
    for lbl in ARTICLE_LABELS:
        k = label_tally.get(lbl, 0)
        print(f"  {lbl}: {k:4d}  ({k/n:.1%})")
    print(f"GT set-size distribution: {dict(set_size)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-label baselines for corruption-article classification")
    parser.add_argument("--baseline", choices=["random", "majority", "keyword", "all"],
                        default="all", help="Which baseline(s) to generate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random baseline")
    parser.add_argument("--no-score", action="store_true",
                        help="Skip the in-process self-scoring step")
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        sys.exit(f"Input directory not found: {INPUT_DIR}")
    from pipeline.case_sections import evaluated_test_cases, split_case

    files = [INPUT_DIR / name for name in evaluated_test_cases()]
    if not files:
        sys.exit(f"No evaluated test cases found in {INPUT_DIR}")

    meta_map = load_metadata_map()
    if not meta_map:
        print(f"WARN: {METADATA_PATH} missing — GT will be empty for every case.")

    cases: list[dict] = []
    for f in files:
        parsed, _rule, _considerato = split_case(f.read_text(encoding="utf-8"))
        c = {"input": parsed["input"], "pqm_ground_truth": parsed["pqm_ground_truth"]}
        c["filename"] = f.name
        meta = meta_map.get(f.name)
        if meta is not None:
            c["gt_articles"] = gt_articles_from_rnc(meta)
            c["gt_source"]   = "metadata" if c["gt_articles"] else "metadata-empty"
        else:
            c["gt_articles"] = set()
            c["gt_source"]   = "none"
        cases.append(c)

    _summarize_corpus(cases)
    prior_set = compute_corpus_prior(cases, threshold=0.5)
    print(f"Corpus prior (prevalence ≥ 0.5): {sorted(prior_set)}")

    baselines = ["random", "majority", "keyword"] if args.baseline == "all" else [args.baseline]
    out_paths: dict[str, Path] = {}
    for b in baselines:
        print(f"\n=== Baseline: {b} ===")
        out_paths[b] = run_baseline(b, args.seed, cases, prior_set)
        print(f"Wrote {out_paths[b]} ({len(cases)} cases)")

    if not args.no_score:
        try:
            from metrics.stats_utils import multi_label_eval, format_multi_label_table
        except ImportError as e:
            print(f"\n(Skipping self-score: {e})")
            return
        results = {b: multi_label_eval(out_paths[b], labels=ARTICLE_LABELS) for b in out_paths}
        print()
        print(format_multi_label_table(results, labels=ARTICLE_LABELS))


if __name__ == "__main__":
    main()
