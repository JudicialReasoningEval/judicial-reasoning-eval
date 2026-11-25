"""Tests for the facts/reasoning split, companion detection and the prompt check."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.case_sections import (
    check_prompts,
    facts_and_withheld,
    find_companions,
    prepare_cases,
    reasoning_passages,
    split_case,
    split_exemplar,
)
from pipeline.cassazione_split import INPUT_DIR

ROOT = Path(__file__).resolve().parent.parent
EVALUATED = ROOT / "splits" / "evaluated_91.json"

FACTS = "RITENUTO IN FATTO\n\n1. Con sentenza la Corte di appello ha confermato la condanna."
REASONING = "1. Il ricorso è infondato. " * 40
PQM = "P.Q.M.\n\nRigetta il ricorso e condanna il ricorrente al pagamento delle spese processuali."


def judgment(heading: str) -> str:
    return f"REPUBBLICA ITALIANA\n\n{FACTS}\n{heading}\n\n{REASONING}\n\n{PQM}\n"


def test_plain_heading():
    parsed, rule, considerato = split_case(judgment("CONSIDERATO IN DIRITTO"))
    assert rule == "plain-heading"
    assert parsed["input"].endswith("confermato la condanna.")
    assert "CONSIDERATO" not in parsed["input"] and "Rigetta" not in parsed["input"]
    assert considerato.startswith("1. Il ricorso è infondato.")
    assert parsed["pqm_ground_truth"].startswith("P.Q.M.")


@pytest.mark.parametrize("heading", [
    "## **CONSIDERATO IN DIRITTO**",
    "**Considerato in diritto**",
    "CONSIDERato IN DIRITTO",
])
def test_decorated_heading(heading):
    parsed, rule, considerato = split_case(judgment(heading))
    assert rule == "decorated-heading"
    assert "Il ricorso è infondato" not in parsed["input"]
    assert considerato.startswith("1. Il ricorso è infondato.")
    assert parsed["pqm_ground_truth"].startswith("P.Q.M.")


@pytest.mark.parametrize("text", [
    "RITENUTO IN FATTO E CONSIDERATO IN DIRITTO\n\n" + REASONING + "\n" + PQM,
    "CONSIDERATO IN FATTO E DIRITTO\n\n" + REASONING + "\n" + PQM,
    FACTS + "\n\n" + REASONING,
])
def test_no_separable_facts_is_set_aside(text):
    parsed, rule, considerato = split_case(text)
    assert parsed is None and rule == "no-separable-facts" and considerato == ""
    assert facts_and_withheld(text) is None


def test_inline_sentence_is_not_a_heading():
    text = FACTS + "\nConsiderato in diritto che il ricorso è infondato, si osserva.\n" + REASONING
    parsed, _rule, _considerato = split_case(text)
    assert parsed is None


def test_exemplar_without_reasoning_is_dropped():
    parts, rule = split_exemplar(f"{FACTS}\n## CONSIDERATO IN DIRITTO\n\n{PQM}\n")
    assert parts is None and rule.endswith("empty-considerato")
    parts, rule = split_exemplar(judgment("CONSIDERATO IN DIRITTO"))
    assert parts is not None and parts["considerato"] and parts["pqm"]


def test_companions_found_and_threshold_respected():
    shared = " ".join(f"parola{i}" for i in range(200))
    other = " ".join(f"altro{i}" for i in range(200))
    test_docs = {"t.md": (FACTS + shared, shared)}
    pool_docs = {"same.md": (FACTS + shared, shared), "other.md": (FACTS + other, other)}
    pairs = find_companions(test_docs, pool_docs, threshold=0.20)
    assert set(pairs) == {"t.md"} and set(pairs["t.md"]) == {"same.md"}


def test_prompt_check():
    considerato = " ".join(f"motivo{i}" for i in range(400))
    probes = reasoning_passages(considerato)
    assert len(probes) == 3
    case = {"filename": "t.md", "input": FACTS}
    ok_fail, ok_warn = check_prompts([case], ["istruzioni"], {"t.md": probes})
    assert ok_fail == [] and ok_warn == []
    bad_fail, _ = check_prompts([case], ["istruzioni " + considerato], {"t.md": probes})
    assert len(bad_fail) == 1


@pytest.mark.skipif(not INPUT_DIR.exists(), reason="corpus not fetched (see README)")
def test_protocol_reproduces_evaluated_set():
    expected = json.loads(EVALUATED.read_text(encoding="utf-8"))
    prep = prepare_cases()
    report = prep["report"]
    assert sorted(report["excluded_test"]) == expected["set_aside_test"]
    assert sorted(report["dropped_pool"]) == expected["dropped_pool"]
    assert [p["filename"] for p in prep["pool"]] == expected["exemplar_pool"]
    assert sorted(prep["parsed"]) == sorted(expected["evaluated_test"])
    assert prep["companions"] == expected["companions"]
