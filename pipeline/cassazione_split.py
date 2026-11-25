"""Stratified split of the 181-case corruption corpus.

Single source of truth for which cases form the 100-case test set and
which 81 form the few-shot exemplar pool (91 and 79 of them are used; see
pipeline/case_sections.py).

Stratification key: the sorted article-set the case cites (e.g. "318|321").
Sampling: largest-remainder proportional allocation with these constraints:
  - any stratum with size >= 2 keeps at least 1 case in BOTH test and pool
  - size-1 strata are placed in the test set (preserves test diversity;
    few-shot fallback handles missing-exact-match pool cases)

Writes:
  splits/test_100.json
  splits/fewshot_pool_81.json

Public API:
  load_split("test"|"fewshot") -> list[dict]   # cases with filename, gt_articles, stratum
  build_splits(seed=42) -> (test_cases, pool_cases)   # in-memory, no IO

Run as a script to (re-)generate the files:
  python -m pipeline.cassazione_split
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.baselines import (
    ARTICLE_LABELS,
    INPUT_DIR,
    gt_articles_from_rnc,
    load_metadata_map,
)

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "splits"
TEST_PATH = SPLITS_DIR / "test_100.json"
POOL_PATH = SPLITS_DIR / "fewshot_pool_81.json"

TEST_SIZE = 100
TOTAL_SIZE = 181
SEED = 42


def _stratum_key(gt_articles: set[str]) -> str:
    """Canonical article-set string for stratification."""
    if not gt_articles:
        return "_empty"
    return "|".join(sorted(gt_articles, key=lambda x: ARTICLE_LABELS.index(x)))


def _load_all_cases() -> list[dict]:
    """Return every filtered case as {filename, gt_articles, stratum}."""
    files = sorted(INPUT_DIR.glob("*.md"))
    meta = load_metadata_map()
    out: list[dict] = []
    for f in files:
        entry = meta.get(f.name)
        gt = gt_articles_from_rnc(entry) if entry else set()
        out.append({
            "filename": f.name,
            "gt_articles": sorted(gt, key=lambda x: ARTICLE_LABELS.index(x)),
            "stratum": _stratum_key(gt),
        })
    return out


def _allocate_test_counts(stratum_sizes: dict[str, int], target_total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation with min/max guards.

    For each stratum of size N:
      - if N == 1: test count is 1 (size-1 goes to test, pool gets nothing for this combo)
      - if N >= 2: test count is in [1, N-1] (both sides keep at least one)
    """
    n_total = sum(stratum_sizes.values())
    raw = {k: v * target_total / n_total for k, v in stratum_sizes.items()}
    floor = {k: int(v) for k, v in raw.items()}
    remainder = {k: raw[k] - floor[k] for k in raw}

    alloc: dict[str, int] = {}
    for k, n in stratum_sizes.items():
        if n == 1:
            alloc[k] = 1
        else:
            lo, hi = 1, n - 1
            alloc[k] = max(lo, min(hi, floor[k]))

    diff = target_total - sum(alloc.values())
    if diff > 0:
        candidates = sorted(
            (k for k in stratum_sizes if alloc[k] < (stratum_sizes[k] if stratum_sizes[k] == 1 else stratum_sizes[k] - 1)),
            key=lambda k: (-remainder[k], k),
        )
        for k in candidates:
            if diff == 0:
                break
            alloc[k] += 1
            diff -= 1
    elif diff < 0:
        candidates = sorted(
            (k for k in stratum_sizes if alloc[k] > (1 if stratum_sizes[k] >= 1 else 0)),
            key=lambda k: (remainder[k], k),
        )
        for k in candidates:
            if diff == 0:
                break
            alloc[k] -= 1
            diff += 1
    if diff != 0:
        raise RuntimeError(f"could not balance allocation (residual diff={diff})")
    return alloc


def build_splits(seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Return (test_cases, pool_cases) for the corpus.

    Deterministic given seed. Use this if you want the split in-memory
    without touching disk.
    """
    cases = _load_all_cases()
    if len(cases) != TOTAL_SIZE:
        raise RuntimeError(
            f"expected {TOTAL_SIZE} filtered cases, found {len(cases)} "
            f"at {INPUT_DIR}"
        )

    by_stratum: dict[str, list[dict]] = {}
    for c in cases:
        by_stratum.setdefault(c["stratum"], []).append(c)
    sizes = {k: len(v) for k, v in by_stratum.items()}
    alloc = _allocate_test_counts(sizes, TEST_SIZE)

    rng = np.random.default_rng(seed)
    test_cases: list[dict] = []
    pool_cases: list[dict] = []
    for k in sorted(by_stratum):
        group = sorted(by_stratum[k], key=lambda c: c["filename"])
        idx = rng.permutation(len(group))
        n_test = alloc[k]
        for i, j in enumerate(idx):
            (test_cases if i < n_test else pool_cases).append(group[j])
    test_cases.sort(key=lambda c: c["filename"])
    pool_cases.sort(key=lambda c: c["filename"])
    return test_cases, pool_cases


def _dump(path: Path, cases: list[dict], split_name: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "split": split_name,
        "seed": seed,
        "rng": "numpy.random.default_rng",
        "size": len(cases),
        "cases": cases,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_split(which: str) -> list[dict]:
    """Return the case list for `which` in {"test", "fewshot"}.

    Each case dict has keys: filename, gt_articles, stratum. Raises if
    the split files don't exist — run `python -m pipeline.cassazione_split` first.
    """
    if which == "test":
        path = TEST_PATH
    elif which == "fewshot":
        path = POOL_PATH
    else:
        raise ValueError(f"which must be 'test' or 'fewshot', got {which!r}")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m pipeline.cassazione_split` to generate it"
        )
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def case_text(filename: str) -> dict:
    """Convenience: load a case's parsed sentenza by filename.

    Returns dict with input and pqm_ground_truth (pipeline.case_sections.split_case),
    plus the considerato; None for a judgment with no separable facts section.
    """
    from pipeline.case_sections import split_case

    text = (INPUT_DIR / filename).read_text(encoding="utf-8")
    parsed, _rule, considerato = split_case(text)
    if parsed is None:
        return None
    parsed["considerato"] = considerato
    parsed["filename"] = filename
    parsed["raw"] = text
    return parsed


def main() -> None:
    test_cases, pool_cases = build_splits(SEED)
    _dump(TEST_PATH, test_cases, "test", SEED)
    _dump(POOL_PATH, pool_cases, "fewshot", SEED)

    per_class_test = {lbl: 0 for lbl in ARTICLE_LABELS}
    per_class_pool = {lbl: 0 for lbl in ARTICLE_LABELS}
    for c in test_cases:
        for lbl in c["gt_articles"]:
            per_class_test[lbl] += 1
    for c in pool_cases:
        for lbl in c["gt_articles"]:
            per_class_pool[lbl] += 1

    print(f"Wrote {TEST_PATH} ({len(test_cases)} cases)")
    print(f"Wrote {POOL_PATH} ({len(pool_cases)} cases)")
    print()
    print("Per-class tally:")
    print(f"  {'label':<6} {'test':>5} {'pool':>5} {'total':>6}")
    for lbl in ARTICLE_LABELS:
        t, p = per_class_test[lbl], per_class_pool[lbl]
        print(f"  {lbl:<6} {t:>5} {p:>5} {t + p:>6}")

    strata_total = len({c["stratum"] for c in test_cases + pool_cases})
    strata_test = len({c["stratum"] for c in test_cases})
    print(f"\nStrata covered: test={strata_test}/{strata_total}")


if __name__ == "__main__":
    main()
