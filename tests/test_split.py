"""Regression tests for cassazione_split.py.

Guards the stratified-sampler invariants we rely on downstream:
  - test ∪ pool covers the full filtered corpus
  - test ∩ pool is empty
  - per-class tally in test is within ±2 of the proportional target
  - every stratum with size >= 2 is represented in both test and pool
  - the split is deterministic (same seed → same files)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.baselines import ARTICLE_LABELS, INPUT_DIR
from pipeline.cassazione_split import (
    POOL_PATH,
    TEST_PATH,
    TEST_SIZE,
    TOTAL_SIZE,
    _load_all_cases,
    build_splits,
)

# These tests re-derive the split from the corpus itself, so they need the
# judgments on disk. The corpus is not redistributed with the code (the Court
# publishes it unredacted); fetch it first with tools/fetch_corpus.py.
pytestmark = pytest.mark.skipif(
    not INPUT_DIR.exists() or not any(INPUT_DIR.glob("*.md")),
    reason=f"corpus not present at {INPUT_DIR}; run tools/fetch_corpus.py",
)


@pytest.fixture(scope="module")
def splits():
    return build_splits(seed=42)


def test_corpus_size_matches_expected():
    cases = _load_all_cases()
    assert len(cases) == TOTAL_SIZE, (
        f"sentenze_md_filtered/ has {len(cases)} files, expected {TOTAL_SIZE}"
    )


def test_total_and_split_sizes(splits):
    test_cases, pool_cases = splits
    assert len(test_cases) == TEST_SIZE
    assert len(pool_cases) == TOTAL_SIZE - TEST_SIZE


def test_union_equals_corpus_and_disjoint(splits):
    test_cases, pool_cases = splits
    test_names = {c["filename"] for c in test_cases}
    pool_names = {c["filename"] for c in pool_cases}
    all_names = {c["filename"] for c in _load_all_cases()}
    assert test_names | pool_names == all_names
    assert test_names & pool_names == set()


def test_per_class_tally_within_two(splits):
    test_cases, pool_cases = splits
    corpus = _load_all_cases()
    corpus_tally = {lbl: 0 for lbl in ARTICLE_LABELS}
    test_tally = {lbl: 0 for lbl in ARTICLE_LABELS}
    for c in corpus:
        for lbl in c["gt_articles"]:
            corpus_tally[lbl] += 1
    for c in test_cases:
        for lbl in c["gt_articles"]:
            test_tally[lbl] += 1
    for lbl in ARTICLE_LABELS:
        target = round(corpus_tally[lbl] * TEST_SIZE / TOTAL_SIZE)
        assert abs(test_tally[lbl] - target) <= 2, (
            f"label {lbl}: test_tally={test_tally[lbl]} target={target} "
            f"(corpus count={corpus_tally[lbl]})"
        )


def test_each_multi_stratum_in_both_sides(splits):
    test_cases, pool_cases = splits
    by_stratum: dict[str, int] = {}
    for c in _load_all_cases():
        by_stratum[c["stratum"]] = by_stratum.get(c["stratum"], 0) + 1
    test_strata = {c["stratum"] for c in test_cases}
    pool_strata = {c["stratum"] for c in pool_cases}
    for stratum, size in by_stratum.items():
        if size >= 2:
            assert stratum in test_strata, f"stratum {stratum!r} (size {size}) missing from test"
            assert stratum in pool_strata, f"stratum {stratum!r} (size {size}) missing from pool"


def test_deterministic_under_same_seed():
    a_test, a_pool = build_splits(seed=42)
    b_test, b_pool = build_splits(seed=42)
    assert [c["filename"] for c in a_test] == [c["filename"] for c in b_test]
    assert [c["filename"] for c in a_pool] == [c["filename"] for c in b_pool]


def test_split_files_exist_and_match_in_memory(splits):
    """If splits/*.json exist, they must match the in-memory build."""
    if not TEST_PATH.exists() or not POOL_PATH.exists():
        pytest.skip("split files not generated yet; run `python cassazione_split.py`")
    test_cases, pool_cases = splits
    on_disk_test = json.loads(TEST_PATH.read_text(encoding="utf-8"))["cases"]
    on_disk_pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))["cases"]
    assert [c["filename"] for c in on_disk_test] == [c["filename"] for c in test_cases]
    assert [c["filename"] for c in on_disk_pool] == [c["filename"] for c in pool_cases]
