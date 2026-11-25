"""Pick few-shot exemplars from the pool for a test case.

An article-matched k=2 few-shot regime: the
exemplars shown to the model should be drawn from the held-out pool and
should cite the same articles as the test case, so the model sees how a
human judge handled the same statutory cluster.

Scoring (higher = better match):
  score = |gt_test ∩ gt_pool|  -  0.1 * |gt_pool \\ gt_test|

That is, exact-overlap dominates and extra articles in the pool case
pay a small penalty. Ties are broken by a deterministic RNG-shuffle.

Fallback chain when no exemplars have any article overlap:
  1. single-label overlap (any one article in common)
  2. nearest Hamming distance over the 5-article one-hot
  3. random from the pool

The fallback path is logged in the returned exemplar dicts so the
reproducibility appendix can quote how often each path was used.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pipeline.baselines import ARTICLE_LABELS, INPUT_DIR
from pipeline.cassazione_split import load_split

_CONS_RE = re.compile(r"\n\s*CONSIDERATO\s+IN\s+DIRITTO[^\n]*\n", re.IGNORECASE)
_PQM_RE = re.compile(r"\n\s*P\s*\.?\s*Q\s*\.?\s*M\s*\.?", re.IGNORECASE)


def _split_sentenza(text: str) -> dict:
    """Return {input, considerato, pqm} sliced from a raw .md file.

    `input` is everything up to the CONSIDERATO marker (header + RITENUTO).
    `considerato` is the legal-reasoning section. `pqm` starts at the
    P.Q.M. marker. Any section that doesn't match its marker is "".
    """
    cm = _CONS_RE.search(text)
    pm = _PQM_RE.search(text)
    if cm:
        input_part = text[:cm.start()].strip()
        rest = text[cm.end():]
    else:
        input_part, rest = text.strip(), ""
    if pm:
        offset = pm.start() - (cm.end() if cm else 0)
        if cm and offset > 0:
            considerato = rest[:offset].strip()
            pqm = text[pm.start():].strip()
        else:
            considerato = ""
            pqm = text[pm.start():].strip()
    else:
        considerato = rest.strip()
        pqm = ""
    return {"input": input_part, "considerato": considerato, "pqm": pqm}


def _load_pool_with_text() -> list[dict]:
    """Return the usable few-shot pool (79 of 81) with text fields attached.

    Exemplars without a separable reasoning section are left out
    (pipeline.case_sections.split_exemplar). Companion judgments are NOT
    removed here, because they depend on the test case; see
    pipeline.case_sections.prepare_cases and pipeline.inference_judge.
    """
    from pipeline.case_sections import split_exemplar

    cases = load_split("fewshot")
    out: list[dict] = []
    for c in cases:
        path = INPUT_DIR / c["filename"]
        parts, _rule = split_exemplar(path.read_text(encoding="utf-8"))
        if parts is None:
            continue
        out.append({
            "filename": c["filename"],
            "gt_articles": list(c["gt_articles"]),
            "stratum": c["stratum"],
            "input": parts["input"],
            "considerato": parts["considerato"],
            "pqm": parts["pqm"],
        })
    return out


def _overlap_score(test_gt: set[str], pool_gt: set[str]) -> float:
    inter = len(test_gt & pool_gt)
    extra = len(pool_gt - test_gt)
    return inter - 0.1 * extra


def _onehot(labels: set[str]) -> np.ndarray:
    return np.array([1 if lbl in labels else 0 for lbl in ARTICLE_LABELS], dtype=int)


def pick_exemplars(
    test_case: dict,
    pool: list[dict] | None = None,
    k: int = 2,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Return up to k exemplars matched to `test_case`'s article set.

    `test_case` must have at least {"filename", "gt_articles"}. `pool` is
    a list of dicts each with {"filename","gt_articles","input",
    "considerato","pqm"}. If omitted, the usable pool is loaded from
    disk (companion judgments are not removed; pipeline.inference_judge
    does that). The returned exemplars carry a `fewshot_path` field naming the
    fallback level that produced them.
    """
    if pool is None:
        pool = _load_pool_with_text()
    if rng is None:
        rng = np.random.default_rng(42)

    test_gt = set(test_case["gt_articles"])
    test_vec = _onehot(test_gt)

    scored = [(idx, _overlap_score(test_gt, set(p["gt_articles"]))) for idx, p in enumerate(pool)]
    positive = [item for item in scored if item[1] > 0]
    if positive:
        order = sorted(positive, key=lambda x: (-x[1], rng.random()))
        chosen = [pool[i] | {"fewshot_path": "article-overlap"} for i, _ in order[:k]]
        if len(chosen) >= k:
            return chosen
        excluded = {c["filename"] for c in chosen}
    else:
        chosen = []
        excluded = set()

    need = k - len(chosen)
    if need > 0:
        single_hits = [
            (idx, len(test_gt & set(pool[idx]["gt_articles"])))
            for idx in range(len(pool))
            if pool[idx]["filename"] not in excluded
            and len(test_gt & set(pool[idx]["gt_articles"])) >= 1
        ]
        if single_hits:
            order = sorted(single_hits, key=lambda x: (-x[1], rng.random()))
            chosen += [pool[i] | {"fewshot_path": "single-overlap"} for i, _ in order[:need]]
            excluded |= {c["filename"] for c in chosen}

    need = k - len(chosen)
    if need > 0:
        hams = []
        for idx in range(len(pool)):
            if pool[idx]["filename"] in excluded:
                continue
            d = int(np.sum(np.abs(_onehot(set(pool[idx]["gt_articles"])) - test_vec)))
            hams.append((idx, d))
        hams.sort(key=lambda x: (x[1], rng.random()))
        chosen += [pool[i] | {"fewshot_path": "hamming"} for i, _ in hams[:need]]
        excluded |= {c["filename"] for c in chosen}

    need = k - len(chosen)
    if need > 0:
        remaining = [pool[i] for i in range(len(pool)) if pool[i]["filename"] not in excluded]
        rng.shuffle(remaining)
        chosen += [r | {"fewshot_path": "random"} for r in remaining[:need]]

    return chosen[:k]


def main() -> None:
    """Smoke test: report the fewshot-path distribution over the test split."""
    test_cases = load_split("test")
    pool = _load_pool_with_text()
    rng = np.random.default_rng(0)
    path_tally: dict[str, int] = {}
    for tc in test_cases:
        exemplars = pick_exemplars(tc, pool=pool, k=2, rng=rng)
        for ex in exemplars:
            path_tally[ex["fewshot_path"]] = path_tally.get(ex["fewshot_path"], 0) + 1
    total = sum(path_tally.values())
    print(f"Few-shot exemplar selection over {len(test_cases)} test cases (k=2):")
    for path, n in sorted(path_tally.items(), key=lambda x: -x[1]):
        print(f"  {path:<18} {n:>4}  ({100 * n / total:.1f}%)")


if __name__ == "__main__":
    main()
