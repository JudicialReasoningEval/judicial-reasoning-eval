"""M3 polarity-sensitivity probe.

Does M3's soft-cosine alignment distinguish AGREEMENT from CONTRADICTION?
Cosine similarity may score
"e' necessario un atto contrario ai doveri" and "NON e' necessario un atto
contrario ai doveri" as near-identical. In this corpus that is not a marginal
worry: the 318/319 boundary turns on exactly that predicate, so a polarity
flip is precisely the doctrinal error M3 ought to catch.

Design follows HEROS (Chiang & Lee, Repl4NLP 2023): minimal pairs with high
lexical overlap, compared against anchors. A bare cos(s, not-s) is
uninterpretable on its own; it only becomes a finding when placed on a scale
whose endpoints are known.

Perturbation is RULE-BASED and LLM-FREE by design (the metric must stay
deterministic and auditable): we take judge sentences that already contain
exactly one candidate clause-negating `non` and DELETE it. Deletion avoids the
hard part of insertion (locating the finite verb), which would need a parser.
The resulting pairs are *candidate* polarity changes, not assumed gold
contradictions: a deterministic sample is exported for human validation.
Precedent for rule-based perturbation as eval-data construction: NegBLEURT /
CANNOT (Anschuetz et al., INLG 2023).

The primary control is paired: on every source sentence that contains both a
candidate `non` and exactly one discourse marker, compare deleting `non` with
deleting the discourse marker. This holds sentence, case, premise, length and
lexical content nearly fixed. Three anchors, all embedded through the SAME code path M3 uses
(metrics.embeddings.embed, instruct prefix included) so the numbers are about
the production metric, not a proxy:

  matched  both recall- and precision-direction argmax pairs selected by M3
  polarity cos(s, delete-non(s)) on rule-built candidate pairs
  neutral  cos(s, delete-discourse-marker(s)) on the SAME source sentences
  random   cos(s, t), t drawn from a different case (the floor)

The headline is the paired within-sentence cosine cost of changing polarity
relative to the control edit. Confidence intervals resample legal cases, not
sentences, because sentences from one judgment are not independent. AUCs are
reported as secondary, threshold-free summaries with case-clustered CIs.

Outputs:
  results/_negation_probe.json
  results/_negation_probe_matched_pairs.jsonl
  results/_negation_probe_validation_sample.csv

Usage:
    python -m analysis.negation_probe
    python -m analysis.negation_probe --n-pairs 500
    python -m analysis.negation_probe --disable-instruction \
        --out results/_negation_probe_no_instruction.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from metrics.embeddings import DEFAULT_INSTRUCT, DEFAULT_MODEL, cosine_matrix, embed
from metrics.premise_labeler import PREMISES

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
JUDGE_SILLOGISMO = RESULTS_DIR / "sillogismo_judge.jsonl"

# Default reference model for the `matched` anchor: the composite (Q) leader,
# i.e. the strongest case for M3 crediting real alignment.
DEFAULT_MODEL_DIR = RESULTS_DIR / "qwen3.5-397b-a17b"

_NON_RE = re.compile(r"\bnon\b", re.IGNORECASE)

# `non` occurrences that are NOT clause negation, or where deletion does not
# cleanly inverse polarity. Sentences matching any of these are skipped
# outright rather than repaired: the pool is large enough to afford purity.
_BLOCKLIST = [
    r"\bnon\s+solo\b",        # "non solo X ma anche Y" — correlative, not negation
    r"\bse\s+non\b",          # conditionals change an antecedent, not an assertion
    r"\bove\s+non\b",
    r"\bqualora\s+non\b",
    r"\bladdove\s+non\b",
    r"\bpurch[eé]\s+non\b",
    r"\ba\s+condizione\s+che\s+non\b",
    r"\ba\s+patto\s+che\s+non\b",
    r"\bsempre\s+che\s+non\b",
    r"\b(?:se|ove|qualora|laddove|purch[eé])\b[^,;:.]{0,100}\bnon\b",
    r"\b(?:a\s+condizione\s+che|a\s+patto\s+che|sempre\s+che)"
    r"[^,;:.]{0,100}\bnon\b",
    r"\bo\s+non\b",           # disjunct ("fare o non fare"), not sentence reversal
    r"\bnon\s+che\b",         # "non che" — concessive
    r"\bnon\s+appena\b",      # "non appena" — temporal ("as soon as")
    r"\bnonch[eé]\b",         # "nonché" — additive conjunction
    r"\bnon\s+ultimo\b",
    r"\bnon\s+meno\b",
    r"\bnon\s+pi[uù]\s+tardi\b",
    r"\bnon\s+senza\b",       # litotes — deletion flips to the same meaning
    r"\bnon\s+altro\b",
    r"\bnon\s+massimat\w*\b", # citation metadata, not the legal proposition
    # NEGATIVE CONCORD: Italian requires `non` alongside a negative quantifier
    # or adverb ("non ... nemmeno", "non ... nessuno", "non ... mai"). Deleting
    # `non` there leaves an ungrammatical fragment that is STILL semantically
    # negative, so the pair is not a polarity inversion at all. Found empirically:
    # such pairs were the ones scoring cosine 1.000.
    r"\bnessun\w*\b",
    r"\bnemmeno\b",
    r"\bneanche\b",
    r"\bneppure\b",
    r"\bniente\b",
    r"\bnulla\b",
    r"\bmai\b",
    r"\bpi[uù]\b",            # "non ... più": deletion leaves invalid/ambiguous concord
    r"\baffatto\b",
    r"\bmica\b",
    r"\bn[eé]\b",
    r"\balcun\w*\b",          # "non ... alcuno" — same concord pattern
    # EXPLETIVE (pleonastic) NEGATION: after these subordinators Italian takes a
    # semantically empty `non` ("a meno che un Comune NON avesse disposto"),
    # so deleting it does not change truth conditions at all.
    r"\ba\s+meno\s+che\b",
    r"\bprima\s+che\b",
    r"\bfinch[eé]\b",
    r"\bsalvo\s+che\b",
    r"\btranne\s+che\b",
    r"\beccetto\s+che\b",
    # CONTRASTIVE CORRECTIVE: "X e non invece Y" — deletion breaks the
    # construction rather than inverting a claim (same class as "non solo").
    r"\be\s+non\s+invece\b",
    r"\bnon\s+invece\b",
    r"\bnon\s+gi[aà]\b",
]
_BLOCK_RE = [re.compile(p, re.IGNORECASE) for p in _BLOCKLIST]

MIN_LEN, MAX_LEN = 40, 400


def _read_jsonl_by_filename(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a half-written line from an interrupted run
            out[r["filename"]] = r
    return out


def strip_negation(sentence: str) -> str | None:
    """Delete one candidate clause-negating ``non``.

    Returns None if the sentence is not a conservative single-negation
    candidate. Human validation is still required before treating the returned
    pair as a gold contradiction.
    Capitalisation is repaired when `non` was sentence-initial, so the pair
    differs ONLY in polarity and not in casing (which would otherwise leak a
    surface cue into the embedding).
    """
    s = sentence.strip()
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return None
    if any(rx.search(s) for rx in _BLOCK_RE):
        return None
    if len(_NON_RE.findall(s)) != 1:
        return None

    m = _NON_RE.search(s)
    initial = m.start() == 0
    out = (s[:m.start()] + s[m.end():])
    out = re.sub(r"\s{2,}", " ", out).strip()
    # drop a space left before punctuation, e.g. "sussiste , poiche" -> "sussiste, poiche"
    out = re.sub(r"\s+([,;.:])", r"\1", out)
    if not out:
        return None
    if initial and out[0].islower():
        out = out[0].upper() + out[1:]
    if out == s:
        return None
    return out


# Discourse adverbs whose deletion changes rhetorical framing but NOT truth
# conditions. These drive the `neutral_edit` control: an edit of comparable
# size to deleting `non`, but polarity-preserving. Without it, a high cosine on
# negation pairs is confounded with their ~99% lexical overlap; with it, the
# polarity effect is isolated at matched overlap.
_NEUTRAL_CUES = ["infatti", "tuttavia", "dunque", "peraltro", "invece",
                 "altres[iì]", "comunque", "quindi", "inoltre", "pertanto",
                 "difatti", "orbene", "del resto", "in effetti"]
_NEUTRAL_RE = [re.compile(rf"\b{c}\b", re.IGNORECASE) for c in _NEUTRAL_CUES]


def strip_neutral(sentence: str) -> str | None:
    """Delete one discourse marker — the polarity-preserving control edit.

    Same machinery as strip_negation (single occurrence, length window,
    capitalisation repair) so the two perturbations differ only in WHAT is
    removed, not in how.
    """
    s = sentence.strip()
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return None
    hits = [(rx, m) for rx in _NEUTRAL_RE for m in rx.finditer(s)]
    if len(hits) != 1:
        return None
    _, m = hits[0]
    initial = m.start() == 0
    left, right = s[:m.start()], s[m.end():]
    # Discourse markers are commonly parenthetical ("può, pertanto, agire").
    # Remove their punctuation span as a unit; deleting only the word would
    # leave malformed controls such as "può,, agire".
    left = re.sub(r"[,;:]\s*$", "", left)
    right = re.sub(r"^\s*[,;:]", "", right)
    out = left + right
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s+([,;.:])", r"\1", out)
    out = re.sub(r"([,;:])\1+", r"\1", out)
    out = re.sub(r"^[,;:]\s*", "", out)
    if not out or out == s:
        return None
    if initial and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


def _sentence_records(judge_by_fn: dict[str, dict]) -> list[dict]:
    """Return unique judge sentences with all of their premise memberships."""
    by_key: dict[tuple[str, str], dict] = {}
    for fn, rec in judge_by_fn.items():
        for p in PREMISES:
            for s in _premise_sentences(rec, p):
                original = s.strip()
                key = (fn, original)
                if key not in by_key:
                    by_key[key] = {
                        "filename": fn,
                        "premises": [],
                        "original": original,
                    }
                if p not in by_key[key]["premises"]:
                    by_key[key]["premises"].append(p)
    return list(by_key.values())


def _stratified_sample(records: list[dict], n_pairs: int | None,
                       seed: int) -> list[dict]:
    """Sample unique sentences across premises without multi-label duplicates."""
    if n_pairs is None or n_pairs <= 0 or n_pairs >= len(records):
        return records
    rng = np.random.default_rng(seed)
    target = max(1, n_pairs // len(PREMISES))
    chosen: set[int] = set()
    ordered: list[int] = []
    for premise in PREMISES:
        pool = [i for i, row in enumerate(records)
                if i not in chosen and premise in row["premises"]]
        if not pool:
            continue
        k = min(target, len(pool))
        picked = rng.choice(pool, size=k, replace=False)
        for i in sorted(int(v) for v in picked):
            chosen.add(i)
            ordered.append(i)
    if len(ordered) < n_pairs:
        rest = [i for i in range(len(records)) if i not in chosen]
        k = min(n_pairs - len(ordered), len(rest))
        picked = rng.choice(rest, size=k, replace=False)
        ordered.extend(sorted(int(v) for v in picked))
    return [records[i] for i in ordered[:n_pairs]]


def _premise_sentences(record: dict, premise: str) -> list[str]:
    idxs = record.get("by_premise", {}).get(premise, [])
    sents = record.get("sentences", [])
    return [sents[i] for i in idxs if 0 <= i < len(sents) and sents[i].strip()]


def collect_negation_pairs(judge_by_fn: dict[str, dict], n_pairs: int | None,
                           seed: int = 42) -> list[dict]:
    """Rule-built candidate polarity pairs, unique at sentence level."""
    pool: list[dict] = []
    for row in _sentence_records(judge_by_fn):
        perturbed = strip_negation(row["original"])
        if perturbed is not None:
            pool.append({**row, "perturbed": perturbed})
    return _stratified_sample(pool, n_pairs, seed)


def collect_paired_controls(negation_pairs: list[dict]) -> list[dict]:
    """Same-source polarity and discourse-marker edits."""
    paired: list[dict] = []
    for row in negation_pairs:
        neutral = strip_neutral(row["original"])
        if neutral is not None:
            paired.append({
                **row,
                "polarity_perturbed": row["perturbed"],
                "neutral_perturbed": neutral,
            })
    return paired


def matched_pair_scores(judge_by_fn: dict[str, dict], model_dir: Path,
                        model_name: str, max_cases: int | None = None,
                        instruct: str | None = None) -> list[dict]:
    """Return both directions of the argmax pairs selected inside M3."""
    llm_path = model_dir / "sillogismo_labels.jsonl"
    if not llm_path.exists():
        raise SystemExit(f"LLM sillogismo labels not found: {llm_path}")
    llm_by_fn = _read_jsonl_by_filename(llm_path)

    scores: list[dict] = []
    n_done = 0
    for fn, judge_rec in judge_by_fn.items():
        if fn not in llm_by_fn:
            continue
        if max_cases is not None and n_done >= max_cases:
            break
        n_done += 1
        for p in PREMISES:
            j_sents = _premise_sentences(judge_rec, p)
            l_sents = _premise_sentences(llm_by_fn[fn], p)
            if not j_sents or not l_sents:
                continue
            sim = cosine_matrix(
                embed(j_sents, model_name, instruct=instruct),
                embed(l_sents, model_name, instruct=instruct),
            )
            sim = np.clip(sim, 0.0, 1.0)
            recall_idx = sim.argmax(axis=1)
            for i, j in enumerate(recall_idx):
                scores.append({
                    "filename": fn,
                    "premise": p,
                    "direction": "recall",
                    "source_index": i,
                    "match_index": int(j),
                    "source": j_sents[i],
                    "match": l_sents[int(j)],
                    "cosine": float(sim[i, j]),
                })
            precision_idx = sim.argmax(axis=0)
            for j, i in enumerate(precision_idx):
                scores.append({
                    "filename": fn,
                    "premise": p,
                    "direction": "precision",
                    "source_index": j,
                    "match_index": int(i),
                    "source": l_sents[j],
                    "match": j_sents[int(i)],
                    "cosine": float(sim[i, j]),
                })
    return scores


def random_pair_scores(judge_by_fn: dict[str, dict], n_pairs: int,
                       model_name: str, seed: int = 42,
                       instruct: str | None = None) -> list[dict]:
    """Floor anchor: sentence vs sentence from a DIFFERENT case."""
    flat = [(fn, s.strip()) for fn, rec in judge_by_fn.items()
            for s in rec.get("sentences", [])
            if MIN_LEN <= len(s.strip()) <= MAX_LEN]
    if len(flat) < 2:
        return []
    rng = np.random.default_rng(seed)
    left: list[str] = []
    right: list[str] = []
    left_idx: list[int] = []
    right_idx: list[int] = []
    tries = 0
    while len(left) < n_pairs and tries < n_pairs * 20:
        tries += 1
        i, j = rng.integers(0, len(flat), size=2)
        if flat[i][0] == flat[j][0]:
            continue  # same case — not an independent floor
        left.append(flat[i][1])
        right.append(flat[j][1])
        left_idx.append(int(i))
        right_idx.append(int(j))
    if not left:
        return []
    a = embed(left, model_name, instruct=instruct)
    b = embed(right, model_name, instruct=instruct)
    sims = np.clip(np.sum(a * b, axis=1), 0.0, 1.0)
    return [
        {
            "left_filename": flat[i][0],
            "right_filename": flat[j][0],
            "cosine": float(v),
        }
        for i, j, v in zip(left_idx, right_idx, sims)
    ]


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a random `pos` scores above a random `neg`), ties at 0.5.

    Rank-based (Mann-Whitney U); hand-rolled to keep the metric dependency-free.
    Here `pos` = matched pairs, `neg` = negation pairs, so AUC is the
    probability that cosine correctly ranks a genuine alignment above a
    contradiction. 0.5 = no discrimination.
    """
    if not pos or not neg:
        return float("nan")
    a = np.asarray(pos, dtype=np.float64)
    b = np.asarray(neg, dtype=np.float64)
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks within ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:a.size].sum()
    return float((r_pos - a.size * (a.size + 1) / 2) / (a.size * b.size))


def cluster_bootstrap_ci(values: Iterable[float], clusters: Iterable[str],
                         n_resamples: int = 2000, ci: float = 0.95,
                         seed: int = 42) -> dict:
    """Bootstrap a case-weighted mean, treating each judgment as one unit."""
    grouped: dict[str, list[float]] = defaultdict(list)
    observations: list[float] = []
    for value, cluster in zip(values, clusters):
        value = float(value)
        if np.isnan(value):
            continue
        grouped[str(cluster)].append(value)
        observations.append(value)
    if not grouped:
        return {
            "mean": float("nan"), "observation_mean": float("nan"),
            "lo": float("nan"), "hi": float("nan"),
            "n_observations": 0, "n_cases": 0,
            "n_resamples": n_resamples, "ci": ci,
        }
    case_means = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, case_means.size,
                       size=(n_resamples, case_means.size))
    means = case_means[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return {
        "mean": float(case_means.mean()),
        "observation_mean": float(np.mean(observations)),
        "lo": float(np.quantile(means, alpha)),
        "hi": float(np.quantile(means, 1 - alpha)),
        "n_observations": len(observations),
        "n_cases": int(case_means.size),
        "n_resamples": n_resamples,
        "ci": ci,
    }


def cluster_bootstrap_auc(pos: list[float], pos_clusters: list[str],
                          neg: list[float], neg_clusters: list[str],
                          n_resamples: int = 2000, ci: float = 0.95,
                          seed: int = 42) -> dict:
    """Two-sample AUC with each side resampled by legal case."""
    pos_grouped: dict[str, list[float]] = defaultdict(list)
    neg_grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(pos, pos_clusters):
        if not np.isnan(value):
            pos_grouped[str(cluster)].append(float(value))
    for value, cluster in zip(neg, neg_clusters):
        if not np.isnan(value):
            neg_grouped[str(cluster)].append(float(value))
    if not pos_grouped or not neg_grouped:
        return {
            "auc": float("nan"), "lo": float("nan"), "hi": float("nan"),
            "n_pos": len(pos), "n_neg": len(neg),
            "n_pos_cases": len(pos_grouped), "n_neg_cases": len(neg_grouped),
            "n_resamples": n_resamples, "ci": ci,
        }
    pos_keys, neg_keys = sorted(pos_grouped), sorted(neg_grouped)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sampled_pos = rng.choice(pos_keys, size=len(pos_keys), replace=True)
        sampled_neg = rng.choice(neg_keys, size=len(neg_keys), replace=True)
        pos_values = [v for key in sampled_pos for v in pos_grouped[str(key)]]
        neg_values = [v for key in sampled_neg for v in neg_grouped[str(key)]]
        samples[i] = auc(pos_values, neg_values)
    alpha = (1 - ci) / 2
    return {
        "auc": auc(pos, neg),
        "lo": float(np.quantile(samples, alpha)),
        "hi": float(np.quantile(samples, 1 - alpha)),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_pos_cases": len(pos_grouped),
        "n_neg_cases": len(neg_grouped),
        "n_resamples": n_resamples,
        "ci": ci,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_revision(model_name: str) -> str | None:
    """Best-effort record of the Hub revision resolved at experiment time."""
    try:
        from huggingface_hub import model_info
        return model_info(model_name).sha
    except Exception:
        return None


def write_validation_sample(pairs: list[dict], path: Path, n_rows: int,
                            seed: int) -> None:
    """Export blank human-annotation fields; never fabricate validation."""
    n_rows = min(n_rows, len(pairs))
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    chosen_idx: set[int] = set()
    chosen_cases: set[str] = set()
    target = max(1, n_rows // len(PREMISES))
    # Maximise case coverage while retaining rough premise balance.
    for premise in PREMISES:
        pool = [i for i, row in enumerate(pairs)
                if i not in chosen_idx
                and row["filename"] not in chosen_cases
                and premise in row["premises"]]
        rng.shuffle(pool)
        added = 0
        for i in pool:
            if pairs[i]["filename"] in chosen_cases:
                continue
            chosen.append(i)
            chosen_idx.add(i)
            chosen_cases.add(pairs[i]["filename"])
            added += 1
            if added >= target:
                break
    if len(chosen) < n_rows:
        pool = [i for i, row in enumerate(pairs)
                if i not in chosen_idx and row["filename"] not in chosen_cases]
        rng.shuffle(pool)
        needed = n_rows - len(chosen)
        added = 0
        for i in pool:
            if pairs[i]["filename"] in chosen_cases:
                continue
            chosen.append(i)
            chosen_idx.add(i)
            chosen_cases.add(pairs[i]["filename"])
            added += 1
            if added >= needed:
                break
    if len(chosen) < n_rows:
        pool = [i for i in range(len(pairs)) if i not in chosen_idx]
        rng.shuffle(pool)
        chosen.extend(pool[:n_rows - len(chosen)])
    sample = [pairs[i] for i in chosen]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id", "filename", "premises", "mentions_318_319",
        "original", "polarity_perturbed", "cosine",
        "grammatical_original", "grammatical_perturbed",
        "relation", "negation_scope", "legal_materiality", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(sample, start=1):
            text = row["original"]
            writer.writerow({
                "pair_id": f"NP{i:03d}",
                "filename": row["filename"],
                "premises": "|".join(row["premises"]),
                "mentions_318_319": int(bool(re.search(r"\b31[89]\b", text))),
                "original": text,
                "polarity_perturbed": row["polarity_perturbed"],
                "cosine": f"{row['polarity_cosine']:.8f}",
                "grammatical_original": "",
                "grammatical_perturbed": "",
                "relation": "",
                "negation_scope": "",
                "legal_materiality": "",
                "notes": "",
            })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", type=Path, default=JUDGE_SILLOGISMO)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help="reference model for the `matched` anchor")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="embedding model")
    ap.add_argument("--n-pairs", type=int, default=0,
                    help="candidate polarity pairs; 0 uses the full unique pool")
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--max-cases", type=int, default=None,
                    help="cap cases for the matched anchor (speed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--disable-instruction", action="store_true",
                    help="diagnostic ablation: omit the common Qwen3 prefix")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "_negation_probe.json")
    ap.add_argument("--matched-out", type=Path, default=None)
    ap.add_argument("--validation-out", type=Path, default=None)
    ap.add_argument("--validation-n", type=int, default=120)
    ap.add_argument("--skip-auxiliary-artifacts", action="store_true",
                    help="do not export matched-pair JSONL or validation CSV")
    args = ap.parse_args()

    if not args.judge.exists():
        raise SystemExit(f"judge sillogismo not found: {args.judge}")
    judge_by_fn = _read_jsonl_by_filename(args.judge)
    instruct = "" if args.disable_instruction else None

    pairs = collect_negation_pairs(
        judge_by_fn,
        args.n_pairs if args.n_pairs > 0 else None,
        seed=args.seed,
    )
    if not pairs:
        raise SystemExit("no candidate polarity pairs found — check the filters")
    print(f"built {len(pairs)} unique candidate polarity pairs")

    orig = [d["original"] for d in pairs]
    pert = [d["perturbed"] for d in pairs]
    e_o = embed(orig, args.model, instruct=instruct)
    e_p = embed(pert, args.model, instruct=instruct)
    neg_cos = [float(v) for v in np.clip(np.sum(e_o * e_p, axis=1), 0.0, 1.0)]
    for d, c in zip(pairs, neg_cos):
        d["polarity_perturbed"] = d.pop("perturbed")
        d["polarity_cosine"] = c

    # Build controls from the same source sentences after polarity scores have
    # been attached, so each row supports a genuinely paired comparison.
    paired_input = [{**d, "perturbed": d["polarity_perturbed"]} for d in pairs]
    paired = collect_paired_controls(paired_input)
    for d in paired:
        d.pop("perturbed", None)
    if paired:
        n_o = embed([d["original"] for d in paired], args.model, instruct=instruct)
        n_p = embed([d["neutral_perturbed"] for d in paired],
                    args.model, instruct=instruct)
        neu_cos = [float(v) for v in
                   np.clip(np.sum(n_o * n_p, axis=1), 0.0, 1.0)]
        for d, c in zip(paired, neu_cos):
            d["neutral_cosine"] = c
        print(f"built {len(paired)} same-sentence control pairs")
    else:
        neu_cos = []
        print("WARNING: no paired controls built — polarity effect not isolated")

    print(f"embedding matched-pair anchor from {args.model_dir.name} ...")
    matched = matched_pair_scores(
        judge_by_fn,
        args.model_dir,
        args.model,
        max_cases=args.max_cases,
        instruct=instruct,
    )
    matched_cos = [d["cosine"] for d in matched]
    matched_recall = [d for d in matched if d["direction"] == "recall"]
    matched_precision = [d for d in matched if d["direction"] == "precision"]
    print(f"embedding random-pair floor ...")
    random_rows = random_pair_scores(
        judge_by_fn, args.n_random, args.model,
        seed=args.seed, instruct=instruct,
    )
    random_cos = [d["cosine"] for d in random_rows]

    def ci_for(rows: list[dict], key: str = "cosine",
               cluster_key: str = "filename") -> dict:
        return cluster_bootstrap_ci(
            [d[key] for d in rows],
            [d[cluster_key] for d in rows],
            n_resamples=args.n_resamples,
            seed=args.seed,
        )

    ci_neg = ci_for(pairs, "polarity_cosine")
    paired_neg = [d["polarity_cosine"] for d in paired]
    paired_clusters = [d["filename"] for d in paired]
    ci_paired_neg = cluster_bootstrap_ci(
        paired_neg, paired_clusters,
        n_resamples=args.n_resamples, seed=args.seed,
    )
    ci_neu = cluster_bootstrap_ci(
        neu_cos, paired_clusters,
        n_resamples=args.n_resamples, seed=args.seed,
    ) if paired else None
    deltas = [n - p for n, p in zip(neu_cos, paired_neg)]
    ci_delta = cluster_bootstrap_ci(
        deltas, paired_clusters,
        n_resamples=args.n_resamples, seed=args.seed,
    ) if paired else None
    ci_mat = ci_for(matched)
    ci_mat_rec = ci_for(matched_recall)
    ci_mat_pre = ci_for(matched_precision)
    ci_rnd = cluster_bootstrap_ci(
        random_cos,
        [d["left_filename"] for d in random_rows],
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    matched_clusters = [d["filename"] for d in matched]
    neg_clusters = [d["filename"] for d in pairs]
    auc_matched_neg = cluster_bootstrap_auc(
        matched_cos, matched_clusters, neg_cos, neg_clusters,
        n_resamples=args.n_resamples, seed=args.seed,
    )
    auc_matched_random = cluster_bootstrap_auc(
        matched_cos, matched_clusters,
        random_cos, [d["left_filename"] for d in random_rows],
        n_resamples=args.n_resamples, seed=args.seed,
    )
    auc_neutral_neg = cluster_bootstrap_auc(
        neu_cos, paired_clusters, paired_neg, paired_clusters,
        n_resamples=args.n_resamples, seed=args.seed,
    ) if paired else None

    # where negation sits on the [random -> matched] scale; 1.0 = indistinguishable
    # from a credited alignment, 0.0 = indistinguishable from an unrelated sentence.
    span = ci_mat["mean"] - ci_rnd["mean"]
    position = float((ci_neg["mean"] - ci_rnd["mean"]) / span) if span else float("nan")

    matched_out: Path | None = None
    validation_out: Path | None = None
    if not args.skip_auxiliary_artifacts:
        matched_out = args.matched_out or args.out.with_name(
            f"{args.out.stem}_matched_pairs.jsonl"
        )
        validation_out = args.validation_out or args.out.with_name(
            f"{args.out.stem}_validation_sample.csv"
        )
        matched_out.parent.mkdir(parents=True, exist_ok=True)
        with matched_out.open("w", encoding="utf-8") as fh:
            for row in matched:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_validation_sample(pairs, validation_out, args.validation_n, args.seed)

    payload = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "source_sha256": {
                "analysis/negation_probe.py": _sha256(Path(__file__)),
                "metrics/embeddings.py": _sha256(ROOT / "metrics" / "embeddings.py"),
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sentence_transformers": _package_version("sentence-transformers"),
            "torch": _package_version("torch"),
            "embedding_model": args.model,
            "embedding_model_revision": _model_revision(args.model),
            "instruction_mode": "disabled" if args.disable_instruction else "symmetric",
            "instruction": None if args.disable_instruction else DEFAULT_INSTRUCT,
            "reference_model_dir": args.model_dir.name,
            "seed": args.seed,
            "n_resamples": args.n_resamples,
            "max_cases": args.max_cases,
        },
        "design": {
            "perturbation": "rule-based deletion of one candidate clause-negating `non`; "
                            "automatically filtered but not assumed human-validated",
            "control": "deletion of one discourse marker from the same source sentence",
            "sampling_unit": "legal case (case-weighted means and case-cluster bootstrap)",
            "matched_anchor": "both recall- and precision-direction M3 argmax pairs",
            "human_validation_status": "pending; blank annotation sample exported",
        },
        "counts": {
            "candidate_polarity_pairs": len(pairs),
            "paired_controls": len(paired),
            "matched_bidirectional": len(matched),
            "matched_recall": len(matched_recall),
            "matched_precision": len(matched_precision),
            "random": len(random_rows),
            "candidate_cases": len(set(neg_clusters)),
            "paired_control_cases": len(set(paired_clusters)),
        },
        "distributions": {
            "matched_bidirectional": ci_mat,
            "matched_recall": ci_mat_rec,
            "matched_precision": ci_mat_pre,
            "candidate_polarity_all": ci_neg,
            "candidate_polarity_paired_subset": ci_paired_neg,
            "neutral_edit_paired_subset": ci_neu,
            "random": ci_rnd,
        },
        "auc": {
            "matched_vs_candidate_polarity": auc_matched_neg,
            "matched_vs_random": auc_matched_random,
            "neutral_vs_candidate_polarity_paired": auc_neutral_neg,
        },
        "paired_polarity_cost": ci_delta,
        "diagnostics": {
            "candidate_position_on_random_to_matched_scale": position,
            "fraction_candidates_above_matched_observation_mean": float(np.mean(
                np.asarray(neg_cos) > ci_mat["observation_mean"]
            )),
            "paired_controls_with_double_comma": sum(
                ",," in d["neutral_perturbed"] for d in paired
            ),
            "candidate_pairs_mentioning_318_319": sum(
                bool(re.search(r"\b31[89]\b", d["original"])) for d in pairs
            ),
        },
        "artifacts": {
            "matched_pairs_jsonl": (
                str(matched_out.relative_to(ROOT)) if matched_out else None
            ),
            "human_validation_csv": (
                str(validation_out.relative_to(ROOT)) if validation_out else None
            ),
        },
        "pairs": pairs,
        "paired_controls": paired,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    print()
    print(f"{'distribution':24s} {'n':>6s} {'cases':>6s} {'mean':>7s} {'95% CI':>18s}")
    rows = [
        ("matched_bidirectional", ci_mat),
        ("candidate_polarity_all", ci_neg),
        ("candidate_polarity_pair", ci_paired_neg),
    ]
    if ci_neu is not None:
        rows.append(("neutral_edit_paired", ci_neu))
    rows.append(("random", ci_rnd))
    for name, ci in rows:
        print(f"{name:24s} {ci['n_observations']:6d} {ci['n_cases']:6d} "
              f"{ci['mean']:7.3f} "
              f"  [{ci['lo']:.3f}, {ci['hi']:.3f}]")
    print()
    if auc_neutral_neg is not None and ci_delta is not None:
        print(f"AUC neutral vs polarity  : {auc_neutral_neg['auc']:.3f} "
              f"[{auc_neutral_neg['lo']:.3f}, {auc_neutral_neg['hi']:.3f}]")
        print(f"paired polarity cost     : {ci_delta['mean']:+.3f} "
              f"[{ci_delta['lo']:+.3f}, {ci_delta['hi']:+.3f}]")
    print(f"candidate edits sit at {position:.1%} of the [random -> matched] span")
    print(f"\nWrote {args.out}")
    if matched_out and validation_out:
        print(f"Wrote {matched_out}")
        print(f"Wrote {validation_out} (blank human-validation fields)")


if __name__ == "__main__":
    main()
