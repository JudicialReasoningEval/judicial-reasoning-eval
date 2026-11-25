"""Rule correctness for the M3 polarity probe's perturbations.

The probe's whole claim rests on its minimal pairs being genuine polarity
inversions. These tests pin the linguistic edge cases that silently produce
INVALID pairs (and did, before the negative-concord blocklist: those pairs were
the ones scoring cosine 1.000).
"""
from __future__ import annotations

import csv
import re

from analysis.negation_probe import (
    auc,
    cluster_bootstrap_auc,
    cluster_bootstrap_ci,
    collect_negation_pairs,
    collect_paired_controls,
    strip_negation,
    strip_neutral,
    write_validation_sample,
)
from metrics.embeddings import _format_for_model
from analysis.m3_polarity_audit import (
    deduplicate_selected,
    is_explicit_non_mismatch,
    normalise_matched_row,
)

PAD = " Il collegio osserva quanto segue in ordine al motivo dedotto."


def _pad(s: str) -> str:
    """Bring a short clause above the probe's MIN_LEN window."""
    return s if len(s) >= 40 else s + PAD


def test_deletes_clause_negation():
    out = strip_negation(_pad("Il Tribunale non ha fatto corretta applicazione."))
    assert out is not None
    assert not re.search(r"\bnon\b", out, re.I)
    assert "ha fatto corretta applicazione" in out


def test_capitalisation_repaired_when_sentence_initial():
    out = strip_negation(_pad("Non sussiste il reato contestato al ricorrente."))
    assert out is not None
    assert out[0].isupper(), out


def test_rejects_multiple_negations():
    # two `non` — deleting one leaves an ambiguous polarity change
    assert strip_negation(_pad("Il motivo non e' fondato e non e' ammissibile.")) is None


def test_rejects_correlative_and_exceptive_non():
    for s in ["Il fatto non solo sussiste ma e' aggravato dalla condotta.",
              "Il ricorso e' inammissibile se non entro il termine di legge.",
              "Il giudice nonche' il pubblico ministero hanno concordato."]:
        assert strip_negation(_pad(s)) is None, s


def test_rejects_negative_concord():
    """`non ... nemmeno/nessuno/mai/ne'` — deletion does NOT invert polarity."""
    for s in ["La deduzione non risulta nemmeno prospettata dalla difesa.",
              "Non sussiste nessuna violazione della norma invocata.",
              "Il ricorrente non ha mai contestato la ricostruzione dei fatti.",
              "Non vi e' prova ne' del dolo ne' della colpa dell'imputato.",
              "Non risulta alcuna violazione delle norme processuali."]:
        assert strip_negation(_pad(s)) is None, s


def test_rejects_expletive_negation():
    """Pleonastic `non` after certain subordinators carries no negative meaning."""
    for s in ["Il pagamento e' valido a meno che il Comune non abbia disposto altro.",
              "Il termine decorre prima che il giudice non abbia provveduto in merito.",
              "Il sequestro resta salvo che non intervenga una revoca espressa."]:
        assert strip_negation(_pad(s)) is None, s


def test_rejects_conditionals_disjunctions_and_citation_metadata():
    for s in [
        "Il termine opera ove non sia intervenuta una tempestiva impugnazione.",
        "La regola vale qualora non ricorrano i presupposti previsti dalla legge.",
        "La regola vale qualora tali presupposti non ricorrano nel caso concreto.",
        "Il funzionario poteva fare o non fare uso del proprio potere pubblico.",
        "Si veda sul punto la decisione Cutrona, non massimata, citata in motivazione.",
        "Una volta restituito il bene, non può più essere disposta la conversione.",
    ]:
        assert strip_negation(_pad(s)) is None, s


def test_rejects_contrastive_corrective():
    s = "Costituiscono prezzo e non invece profitto del reato di corruzione."
    assert strip_negation(_pad(s)) is None


def test_rejects_litotes():
    assert strip_negation(_pad("La condotta non senza rilievo va valutata appieno.")) is None


def test_respects_length_window():
    assert strip_negation("Non sussiste.") is None          # below MIN_LEN
    assert strip_negation("Non " + "a" * 500) is None        # above MAX_LEN


def test_no_space_before_punctuation():
    out = strip_negation(_pad("Il motivo, che non sussiste, va rigettato oggi."))
    assert out is not None
    assert " ," not in out and " ." not in out, out


def test_neutral_edit_preserves_polarity():
    out = strip_neutral(_pad("Il motivo, tuttavia, non supera il vaglio di ammissibilita'."))
    assert out is not None
    # the control must NOT touch the negation
    assert re.search(r"\bnon\b", out, re.I), out
    assert "tuttavia" not in out.lower()


def test_neutral_rejects_multiple_cues():
    assert strip_neutral(_pad("Infatti, tuttavia, il motivo va rigettato dal collegio.")) is None


def test_neutral_rejects_repeated_same_cue():
    assert strip_neutral(_pad(
        "Il motivo, infatti, è infondato e infatti deve essere rigettato."
    )) is None


def test_neutral_removes_parenthetical_punctuation_as_one_span():
    out = strip_neutral(_pad(
        "Le condotte possono, peraltro, intervenire dopo la scelta del contraente."
    ))
    assert out is not None
    assert ",," not in out
    assert "possono intervenire" in out


def test_collection_deduplicates_multi_premise_sentences():
    sentence = _pad("Il Tribunale non ha fatto corretta applicazione.")
    judge = {
        "case.md": {
            "filename": "case.md",
            "sentences": [sentence],
            "by_premise": {
                "maggiore": [0],
                "minore": [0],
                "conclusione": [],
            },
        }
    }
    pairs = collect_negation_pairs(judge, None)
    assert len(pairs) == 1
    assert pairs[0]["premises"] == ["maggiore", "minore"]


def test_paired_control_uses_the_same_source_sentence():
    original = _pad(
        "Il Tribunale, pertanto, non ha fatto corretta applicazione."
    )
    pairs = [{
        "filename": "case.md",
        "premises": ["conclusione"],
        "original": original,
        "perturbed": strip_negation(original),
    }]
    controls = collect_paired_controls(pairs)
    assert len(controls) == 1
    assert controls[0]["original"] == original
    assert "non" in controls[0]["neutral_perturbed"].lower()
    assert "pertanto" not in controls[0]["neutral_perturbed"].lower()


def test_auc_bounds_and_direction():
    assert auc([1.0, 0.9, 0.8], [0.1, 0.2, 0.3]) == 1.0     # pos strictly above
    assert auc([0.1, 0.2, 0.3], [1.0, 0.9, 0.8]) == 0.0     # pos strictly below
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5               # all ties


def test_cluster_bootstrap_uses_case_as_the_unit():
    ci = cluster_bootstrap_ci([0.0, 0.0, 1.0], ["a", "a", "b"],
                              n_resamples=100, seed=7)
    assert ci["mean"] == 0.5
    assert ci["observation_mean"] == 1 / 3
    assert ci["n_observations"] == 3
    assert ci["n_cases"] == 2


def test_cluster_auc_reports_case_counts_and_bounds():
    result = cluster_bootstrap_auc(
        [0.9, 0.8], ["a", "b"],
        [0.2, 0.1], ["c", "d"],
        n_resamples=100,
        seed=7,
    )
    assert result["auc"] == 1.0
    assert result["lo"] == 1.0
    assert result["hi"] == 1.0
    assert result["n_pos_cases"] == result["n_neg_cases"] == 2


def test_empty_instruction_disables_qwen_prefix():
    texts = ["Una frase giuridica."]
    assert _format_for_model(texts, "Qwen/Qwen3-Embedding-8B", "") == texts
    assert _format_for_model(texts, "Qwen/Qwen3-Embedding-8B", None)[0].startswith(
        "Instruct:"
    )


def test_m3_selected_pair_directions_normalise_to_judge_model_order():
    recall = {
        "filename": "case.md", "premise": "maggiore", "direction": "recall",
        "source": "Il giudice non accoglie.", "match": "Il modello accoglie.",
        "cosine": 0.9,
    }
    precision = {
        **recall, "direction": "precision",
        "source": recall["match"], "match": recall["source"],
    }
    assert normalise_matched_row(recall)["judge_sentence"] == recall["source"]
    assert normalise_matched_row(precision)["judge_sentence"] == recall["source"]
    deduped = deduplicate_selected([recall, precision])
    assert len(deduped) == 1
    assert set(deduped[0]["directions"]) == {"recall", "precision"}
    assert is_explicit_non_mismatch(deduped[0])


def test_validation_sample_maximises_case_coverage(tmp_path):
    rows = [
        {
            "filename": f"case-{i}.md",
            "premises": ["maggiore"],
            "original": f"Il giudice non accoglie il motivo numero {i}.",
            "polarity_perturbed": f"Il giudice accoglie il motivo numero {i}.",
            "polarity_cosine": 0.9,
        }
        for i in range(6)
    ]
    out = tmp_path / "sample.csv"
    write_validation_sample(rows, out, n_rows=4, seed=42)
    with out.open() as fh:
        sample = list(csv.DictReader(fh))
    assert len(sample) == 4
    assert len({row["filename"] for row in sample}) == 4
