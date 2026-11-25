"""Separate a judgment into what the model may read and what it must not.

A Cassazione judgment has three parts: the *Ritenuto in fatto* (facts,
lower-court findings, grounds of appeal), the *Considerato in diritto* (the
court's legal reasoning) and the *P.Q.M.* (the dispositive). The model is given
only the first; the other two are the gold standard it is scored against.

The *Considerato* heading is not typeset uniformly across the corpus. It may be
a bare line (``CONSIDERATO IN DIRITTO``), a markdown or bold heading
(``## **CONSIDERATO IN DIRITTO**``), carry an OCR casing (``CONSIDERato IN
DIRITTO``), or be fused with the facts heading (``RITENUTO IN FATTO E
CONSIDERATO IN DIRITTO``). This module defines the split once, for every
consumer:

  * ``split_case`` — a test judgment. A judgment whose facts cannot be separated
    from the reasoning (fused or missing heading) is set aside, never sent.
  * ``split_exemplar`` — a few-shot exemplar. One without a reasoning section
    after the heading is unusable and is dropped from the pool.
  * ``find_companions`` — test/pool pairs that are companion judgments of the
    same proceedings (one judgment's reasoning largely reproduced in the other).
    A companion is never offered as an exemplar for its test case.
  * ``check_prompts`` — verify, right before generation, that no prompt contains
    passages of the test case's own reasoning.

On the 181-judgment corpus this sets aside 9 of the 100 test judgments and 2 of
the 81 pool judgments, leaving 91 evaluated cases and 79 exemplars
(``splits/evaluated_91.json``).

The split is pure: whatever the court wrote before the heading is kept as
published, even where the facts section anticipates part of the decision.
"""
from __future__ import annotations

import re

from pipeline.cassazione_split import INPUT_DIR, load_split
from pipeline.fewshot_selector import _CONS_RE, _split_sentenza

# The plain heading, matched exactly: a line break followed by the uppercase
# heading. Most judgments use it.
PLAIN_HEADING = "\nCONSIDERATO IN DIRITTO"

# Dispositive marker used with the plain heading.
PQM_RE = re.compile(r"\n\s*P\.\s*Q\.\s*M\.", re.IGNORECASE)

# Any other CONSIDERATO IN DIRITTO heading at the start of its line, with
# optional markdown decoration (#, *, _, >). Uppercase may be followed by text
# on the same line; any other casing must stand alone on its line, so that a
# sentence opening "Considerato in diritto che ..." is never taken for the
# heading. Fused headings such as "RITENUTO IN FATTO E CONSIDERATO IN DIRITTO"
# or "CONSIDERATO IN FATTO E DIRITTO" deliberately do not match: they have no
# separable facts section.
CONS_HEADING = re.compile(
    r"^[ \t#>*_]*(?:CONSIDERATO\s+IN\s+DIRIT\s?TO\b[^\n]*|(?i:considerato\s+in\s+dirit\s?to)[ \t#>*_.:]*)$",
    re.MULTILINE,
)
PQM_HEADING = re.compile(r"^[ \t#>*_]*P\s*\.?\s*Q\s*\.?\s*M\s*\.?(?![A-Za-z])", re.MULTILINE)

# A pool judgment counts as a companion of a test judgment when at least this
# share of either one's reasoning 12-grams occurs in the other document.
COMPANION_THRESHOLD = 0.20


class PromptCheckFailed(SystemExit):
    """Raised before generation when a prompt contains the court's own reasoning."""


# ── Splitting ────────────────────────────────────────────────────────────────

def split_case(text: str) -> tuple[dict | None, str, str]:
    """Split a test judgment.

    Returns ``(parsed, rule, considerato)``. ``parsed`` holds ``input`` (header
    and *Ritenuto*, the model input) and ``pqm_ground_truth``; it is ``None``
    when the judgment has no separable facts section. ``rule`` names the
    heading form that was matched.
    """
    if PLAIN_HEADING in text:
        input_part, rest = text.split(PLAIN_HEADING, 1)
        m = PQM_RE.search(rest)
        parsed = {"input": input_part.strip(),
                  "pqm_ground_truth": rest[m.start():].strip() if m else ""}
        rule = "plain-heading"
    else:
        m = CONS_HEADING.search(text)
        if not m:
            return None, "no-separable-facts", ""
        parsed = {"input": text[:m.start()].strip(), "pqm_ground_truth": ""}
        rest = text[m.end():]
        rule = "decorated-heading"
    pm = PQM_HEADING.search(rest)
    considerato = (rest[:pm.start()] if pm else rest).strip()
    if not parsed["pqm_ground_truth"] and pm:
        parsed["pqm_ground_truth"] = rest[pm.start():].strip()
    return parsed, rule, considerato


def split_exemplar(text: str) -> tuple[dict | None, str]:
    """Split a pool judgment into ``{input, considerato, pqm}``.

    Returns ``(parts, rule)``; ``parts`` is ``None`` when the exemplar has no
    reasoning after its heading. An exemplar without a P.Q.M. heading is kept.
    """
    if _CONS_RE.search(text):
        parts = _split_sentenza(text)
        rule = "plain-heading"
    else:
        m = CONS_HEADING.search(text)
        if not m:
            return None, "no-separable-facts"
        rest = text[m.end():]
        pm = PQM_HEADING.search(rest)
        parts = {
            "input": text[:m.start()].strip(),
            "considerato": (rest[:pm.start()] if pm else rest).strip(),
            "pqm": rest[pm.start():].strip() if pm else "",
        }
        rule = "decorated-heading"
    if not parts["considerato"]:
        return None, f"{rule}-empty-considerato"
    return parts, (rule if parts["pqm"] else f"{rule}-no-pqm-heading")


def facts_and_withheld(text: str) -> tuple[str, str] | None:
    """``(model input, withheld reasoning + P.Q.M.)`` of a test judgment, or None."""
    parsed, _rule, considerato = split_case(text)
    if parsed is None:
        return None
    return parsed["input"], considerato + "\n" + (parsed.get("pqm_ground_truth") or "")


# ── Companion judgments ──────────────────────────────────────────────────────

def _ngrams(text: str, n: int = 12) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def find_companions(test_docs: dict[str, tuple[str, str]],
                    pool_docs: dict[str, tuple[str, str]],
                    threshold: float = COMPANION_THRESHOLD) -> dict[str, dict[str, float]]:
    """Test/pool pairs where >= ``threshold`` of either judgment's reasoning
    12-grams appear in the other document.

    Both arguments map filename -> (full text, reasoning). Returns
    ``{test filename: {pool filename: share}}``.
    """
    def prep(docs):
        return {f: (_ngrams(full), _ngrams(cons)) for f, (full, cons) in docs.items()}
    tests, pools = prep(test_docs), prep(pool_docs)
    pairs: dict[str, dict[str, float]] = {}
    for t, (t_full, t_cons) in tests.items():
        for p, (p_full, p_cons) in pools.items():
            scores = []
            if len(t_cons) >= 50:
                scores.append(len(t_cons & p_full) / len(t_cons))
            if len(p_cons) >= 50:
                scores.append(len(p_cons & t_full) / len(p_cons))
            if scores and max(scores) >= threshold:
                pairs.setdefault(t, {})[p] = round(max(scores), 3)
    return pairs


# ── Prompt check ─────────────────────────────────────────────────────────────

def reasoning_passages(considerato: str) -> list[str]:
    """Three 150-character probes taken at 25/50/75% of the court's reasoning."""
    if len(considerato) <= 600:
        return []
    return [considerato[int(len(considerato) * q): int(len(considerato) * q) + 150]
            for q in (0.25, 0.5, 0.75)]


def check_prompts(sentenze: list[dict], dev_list: list[str],
                  passages: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """Return ``(failures, warnings)`` for one generation pass.

    Two or more of the court's own reasoning probes anywhere in a prompt (system
    or user text) is a failure; a single one is a warning, since legal
    boilerplate recurs across judgments.
    """
    failures, warnings = [], []
    for s, dev in zip(sentenze, dev_list):
        fn = s["filename"]
        found = [p for p in passages.get(fn, []) if p.strip() and (p in s["input"] or p in dev)]
        if len(found) >= 2:
            failures.append(f"{fn}: {len(found)} passages of the court's reasoning in the prompt")
        elif found:
            warnings.append(f"{fn}: 1 passage of the court's reasoning in the prompt")
    return failures, warnings


# ── Everything generation needs, in one pass over the corpus ─────────────────

def prepare_cases(threshold: float = COMPANION_THRESHOLD) -> dict:
    """Split the test set and the pool, and find companion judgments.

    Returns ``{"report", "pool", "parsed", "passages", "companions"}``:
    ``report`` records the rule applied to every judgment, the test cases set
    aside and the exemplars dropped; ``pool`` is the usable few-shot pool with
    text fields; ``parsed`` maps each evaluated test case to its split.
    """
    report: dict = {"test_rules": {}, "pool_rules": {}, "excluded_test": [], "dropped_pool": []}
    test_docs, passages, parsed_by_file = {}, {}, {}
    for c in load_split("test"):
        text = (INPUT_DIR / c["filename"]).read_text(encoding="utf-8")
        parsed, rule, considerato = split_case(text)
        report["test_rules"][c["filename"]] = rule
        if parsed is None:
            report["excluded_test"].append(c["filename"])
            continue
        parsed_by_file[c["filename"]] = parsed
        test_docs[c["filename"]] = (text, considerato)
        passages[c["filename"]] = reasoning_passages(considerato)

    pool, pool_docs = [], {}
    for c in load_split("fewshot"):
        text = (INPUT_DIR / c["filename"]).read_text(encoding="utf-8")
        parts, rule = split_exemplar(text)
        report["pool_rules"][c["filename"]] = rule
        if parts is None:
            report["dropped_pool"].append(c["filename"])
            continue
        pool.append({"filename": c["filename"], "gt_articles": list(c["gt_articles"]),
                     "stratum": c["stratum"], **parts})
        pool_docs[c["filename"]] = (text, parts["considerato"])

    companions = find_companions(test_docs, pool_docs, threshold)
    report["companions"] = companions
    return {"report": report, "pool": pool, "parsed": parsed_by_file,
            "passages": passages, "companions": companions}


def evaluated_test_cases() -> list[str]:
    """Filenames of the test judgments with a separable facts section (91)."""
    out = []
    for c in load_split("test"):
        parsed, _rule, _cons = split_case((INPUT_DIR / c["filename"]).read_text(encoding="utf-8"))
        if parsed is not None:
            out.append(c["filename"])
    return out
