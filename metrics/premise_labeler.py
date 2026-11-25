"""Direct 3-premise labeller for the *considerato in diritto*.

An LLM served on an OpenAI-compatible endpoint assigns each sentence of a
considerato to one or more of the three syllogism premises:

    maggiore     — the rule of law in the abstract (norms, precedents)
    minore       — the case facts qualified in law (the legal question)
    conclusione  — subsumption + exception-handling + outcome

Multi-membership: a sentence may belong to several premises. The 9 functional
categories survive only as anchoring *examples* in the rubric.

Output schema (consumed unchanged by m2_structural.py / m3_semantic.py):
    {filename, source, sentences, premise_labels, by_premise}

CLI:
  # judge side (reads the .md corpus) -> results/sillogismo_judge.jsonl
  python -m metrics.premise_labeler --source judge

  # a model's considerato_predicted -> <results-dir>/sillogismo_labels.jsonl
  python -m metrics.premise_labeler --source llm --results-dir results/<model>

The endpoint is LABELER_BASE_URL (default http://localhost:8000/v1) and the
served model LABELER_MODEL. The study runs three labellers and merges them
with metrics.consensus_labels (see pipeline/run_panel_labeling.py).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
SENTENZE_DIR = ROOT / "sentenze_md_filtered"
RESULTS_DIR = ROOT / "results"

PREMISES = ["maggiore", "minore", "conclusione"]

DEFAULT_BASE_URL = os.environ.get("LABELER_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("LABELER_MODEL", "openai/gpt-oss-20b")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 8192))
# gpt-oss is a reasoning model; keep its chain-of-thought short for a labelling
# task so the token budget reaches the JSON answer (finish=length otherwise).
# Set to empty ("") for labellers that don't accept the param (nemotron, qwen).
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low")
# For think-tag labellers (nemotron, qwen): ask the chat template to skip the
# reasoning block entirely so the token budget goes to the JSON answer.
# Value is a JSON dict passed as chat_template_kwargs, e.g. '{"enable_thinking": false}'.
CHAT_TEMPLATE_KWARGS = os.environ.get("LABELER_CHAT_TEMPLATE_KWARGS", "")
# Considerati range up to ~360 sentences; chunk so the prompt fits the 16k
# context window (facts + rubric + numbered sentences).
CHUNK_SIZE = int(os.environ.get("LABELER_CHUNK_SIZE", 35))
# Extra request body merged last, for labellers reached over a gateway whose
# knobs are neither harmony's reasoning_effort nor a chat-template kwarg
# (OpenRouter: {"reasoning": {...}, "usage": {"include": true}}). Drivers set
# this module attribute directly, as they do REASONING_EFFORT.
EXTRA_BODY: dict = {}
# Optional callback invoked after every successful call with
# (model, filename, chunk_idx, usage) so a driver can meter spend. Kept as a
# module attribute rather than a parameter so the panel drivers stay unchanged.
USAGE_SINK = None
# Optional zero-arg callback invoked before every request. Raise StopRequested
# from it to stop the run without losing already completed concurrent results.
BEFORE_CALL = None


class StopRequested(Exception):
    """Cooperative stop raised before a remote request."""


# ── Sentence splitting ──────────────
_ABBREV_TOKENS = frozenset([
    "art", "artt", "n", "nn", "sez", "ss", "cfr", "v", "vs", "pag", "p",
    "fig", "tab", "ecc", "etc", "op", "cit", "vol", "es", "id", "ivi",
    "ibid", "loc", "dott", "avv", "prof", "sig", "ed", "comm",
    "cass", "trib", "corr", "cost", "sent", "ord", "dec", "dep",
    "r", "d", "l", "dlgs", "dpr", "dm", "c", "cp", "cpp", "cc",
    "c.p", "c.p.p", "c.c", "d.lgs", "d.p.r",
])
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀÈÌÒÙ])")
# Tolerate markdown decorations ("## **P.Q.M.**") emitted by the OCR path.
PQM_RE = re.compile(r"\n[ \t#>*_]*P\.?\s*Q\.?\s*M\.?|\n[ \t#>*_]*PER\s+QUESTI\s+MOTIVI", re.I)

# Section-heading recognition. The corpus mixes born-digital extractions
# (plain "CONSIDERATO IN DIRITTO" lines) with OCR/markdown renderings
# ("## **CONSIDERATO IN DIRITTO**", nbsp, OCR splits like "FAT TO") and
# ordinanza-style merged sections. This pattern recognises all of them except
# one file with no recognisable headings at all (raw OCR text).
_CONSIDERATO_HEAD_RE = re.compile(
    r"^[ \t#>*_]*(?:RITENUTO\s+IN\s+FAT\s?TO\s+E\s+)?"
    r"CONSIDERATO\s+IN\s+(?:FAT\s?TO\s+E\s+(?:IN\s+)?)?DIRIT\s?TO[ \t#>*_.]*$",
    re.I | re.M,
)
# Merged / alternative headings, tried in order when no CONSIDERATO exists.
# For merged sections the extracted body includes the facts — the best
# available approximation of the court's reasoning for those ordinanze.
_FALLBACK_HEAD_RES = [
    re.compile(r"^[ \t#>*_]*RITENUTO\s+IN\s+FAT\s?TO\s+E\s+(?:IN\s+)?DIRIT\s?TO[ \t#>*_.]*$", re.I | re.M),
    re.compile(r"^[ \t#>*_]*MOTIVI\s+DELLA\s+DECISIONE[ \t#>*_.]*$", re.I | re.M),
    re.compile(r"^[ \t#>*_]*OSSERVA(?:\s+IN\s+DIRIT\s?TO)?[ \t#>*_.]*$", re.I | re.M),
]
_RITENUTO_HEAD_RE = re.compile(
    r"^[ \t#>*_]*(?:RITENUTO|RILEVATO)\s+IN\s+FAT\s?TO[ \t#>*_.]*$", re.I | re.M,
)


def _is_abbreviation_boundary(text: str, match_start: int) -> bool:
    before = text[:match_start].rstrip()
    last_word = re.split(r"\s+", before)[-1].rstrip(".").lower()
    return last_word in _ABBREV_TOKENS


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    text = re.sub(r"\n{2,}", "\n\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        candidates = list(_SPLIT_RE.finditer(para))
        if not candidates:
            if para:
                sentences.append(para)
            continue
        prev = 0
        for m in candidates:
            if _is_abbreviation_boundary(para, m.start()):
                continue
            chunk = para[prev:m.start()].strip()
            if chunk:
                sentences.append(chunk)
            prev = m.end()
        tail = para[prev:].strip()
        if tail:
            sentences.append(tail)
    return [s for s in sentences if len(s) > 8]


def _read_sentenza(filename: str) -> str:
    path = SENTENZE_DIR / filename
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def extract_considerato(filename: str) -> str:
    text = _read_sentenza(filename)
    m = _CONSIDERATO_HEAD_RE.search(text)
    if not m:
        for pat in _FALLBACK_HEAD_RES:
            m = pat.search(text)
            if m:
                break
    if not m:
        return ""
    after = text[m.end():]
    pqm = PQM_RE.search(after)
    body = after[: pqm.start()] if pqm else after
    return body.strip()


def full_case_facts(filename: str) -> str:
    """The complete facts section (Ritenuto in fatto), as the M5 judge sees it.

    Where no facts heading is recognised (two of the 91 evaluated cases),
    the full judgment text is used.
    """
    text = _read_sentenza(filename)
    m = _RITENUTO_HEAD_RE.search(text)
    if not m:
        return text
    after = text[m.end():]
    c = _CONSIDERATO_HEAD_RE.search(after)
    if c:
        after = after[: c.start()]
    return after.strip()


def extract_case_facts(filename: str) -> str:
    """Facts excerpt used as context for premise labelling (first 3,000 characters)."""
    text = _read_sentenza(filename)
    m = _RITENUTO_HEAD_RE.search(text)
    if not m:
        return text[:1500]
    after = text[m.end():]
    c = _CONSIDERATO_HEAD_RE.search(after)
    if c:
        after = after[: c.start()]
    return after.strip()[:3000]


# ── Prompt ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
Sei un esperto di diritto penale italiano e di teoria del ragionamento giudiziale \
(sillogismo giudiziale, Taruffo, Iacoviello). Analizzi il "considerato in diritto" di \
sentenze della Corte di Cassazione, frase per frase, assegnando a ciascuna frase la sua \
funzione nel sillogismo. Sii preciso e conservativo."""

_RUBRIC = """\
Le tre premesse del sillogismo giudiziale. Assegna a OGNI frase UNA o PIÙ premesse:

1. "maggiore" — la regola di diritto in astratto: richiamo di norme (articoli c.p./c.p.p.),
   principi, precedenti e orientamenti giurisprudenziali, massime.
   Es.: "L'art. 319 c.p. punisce il pubblico ufficiale che riceve denaro per omettere un atto."

2. "minore" — i fatti del caso qualificati in diritto: identificazione della questione
   giuridica sollevata dal motivo, inquadramento dei fatti concreti nella fattispecie.
   Es.: "Il motivo censura la violazione dell'art. 319 c.p. per erronea qualificazione del fatto."

3. "conclusione" — applicazione della norma ai fatti ed esito: sussunzione, gestione delle
   eccezioni difensive, sintesi argomentativa, verifica di coerenza, enunciazione dell'esito.
   Es.: "Nel caso di specie la condotta integra la corruzione susseguente; il ricorso va rigettato."

Se una frase è puramente formale/stilistica e non svolge nessuna funzione, assegna []."""

_USER_TEMPLATE = """\
{rubric}

---
Contesto del caso (fatti rilevanti):
<CASO>
{case_context}
</CASO>

---
Frasi del considerato da etichettare (numerate):
<FRASI>
{numbered}
</FRASI>

---
Restituisci SOLO un oggetto JSON valido: una voce per ogni indice di frase, con la lista
delle premesse assegnate (valori ammessi: "maggiore", "minore", "conclusione"; oppure []).
Esempio:
{{"0": ["maggiore"], "1": ["minore"], "2": ["maggiore", "conclusione"], "3": []}}"""


def _build_user_prompt(sentences: list[str], case_context: str) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return _USER_TEMPLATE.format(rubric=_RUBRIC, case_context=case_context[:3000], numbered=numbered)


def _call_labeler(client: OpenAI, model: str, sentences: list[str], case_context: str,
                  retries: int = 3, filename: str = "", chunk_idx: int = 0) -> dict[str, list[str]]:
    prompt = _build_user_prompt(sentences, case_context)
    # Only send params the serving model understands: reasoning_effort is a
    # gpt-oss/harmony concept; chat_template_kwargs covers qwen-style
    # enable_thinking toggles. Both empty -> no extra_body at all.
    extra_body: dict = {}
    if REASONING_EFFORT:
        extra_body["reasoning_effort"] = REASONING_EFFORT
    if CHAT_TEMPLATE_KWARGS:
        extra_body["chat_template_kwargs"] = json.loads(CHAT_TEMPLATE_KWARGS)
    extra_body.update(EXTRA_BODY)
    for attempt in range(retries):
        # Check every attempt, outside the retry handler: retries are paid calls
        # too, and a cooperative budget stop must not be swallowed.
        if BEFORE_CALL is not None:
            BEFORE_CALL()
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=MAX_TOKENS, timeout=180,
                messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                extra_body=extra_body or None,
            )
            if USAGE_SINK is not None and getattr(resp, "usage", None) is not None:
                USAGE_SINK(model, filename, chunk_idx, resp.usage)
            content = resp.choices[0].message.content
            if not content:
                raise ValueError(f"null content (finish={resp.choices[0].finish_reason})")
            # Think-tag labellers emit <think>…</think> before the answer when
            # thinking can't be disabled; the trace may itself contain braces,
            # so strip it before hunting for the JSON object.
            content = re.sub(r"<think>[\s\S]*?</think>", "", content)
            m = re.search(r"\{[\s\S]+\}", content)
            if not m:
                raise ValueError("no JSON object in response")
            return json.loads(m.group())
        except Exception as e:  # noqa: BLE001 — labeller is best-effort with fallback
            if attempt == retries - 1:
                print(f"  [labeler] failed after {retries}: {e}")
                return {str(i): [] for i in range(len(sentences))}
            time.sleep(2 ** attempt)
    return {}


def label_considerato(client: OpenAI, model: str, sentences: list[str],
                      facts: str, filename: str = "") -> dict[str, list[str]]:
    """Label all sentences, chunking to fit the context window; merge with
    global sentence indices."""
    raw_all: dict[str, list[str]] = {}
    for start in range(0, len(sentences), CHUNK_SIZE):
        chunk = sentences[start:start + CHUNK_SIZE]
        raw = _call_labeler(client, model, chunk, facts,
                            filename=filename, chunk_idx=start)
        for k, v in (raw or {}).items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(chunk):
                raw_all[str(start + idx)] = v
    return raw_all


def _to_record(filename: str, sentences: list[str], raw: dict, source: str) -> dict:
    by_premise: dict[str, list[int]] = {p: [] for p in PREMISES}
    premise_labels: list[list[str]] = []
    for i in range(len(sentences)):
        tags = raw.get(str(i)) or []
        valid = sorted({t for t in tags if t in PREMISES}, key=PREMISES.index)
        premise_labels.append(valid)
        for p in valid:
            by_premise[p].append(i)
    return {"filename": filename, "source": source, "sentences": sentences,
            "premise_labels": premise_labels, "by_premise": by_premise}


def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["filename"])
                except Exception:  # noqa: BLE001
                    pass
    return done


def _judge_cases() -> list[tuple[str, str, str]]:
    """(filename, considerato, case_facts) for every corpus .md with a considerato."""
    out = []
    for p in sorted(SENTENZE_DIR.glob("*.md")):
        fn = p.name
        cons = extract_considerato(fn)
        if cons:
            out.append((fn, cons, extract_case_facts(fn)))
    return out


def _llm_cases(results_dir: Path) -> list[tuple[str, str, str]]:
    """One case per filename. Multi-seed result files
    carry 5 records per case; labels are keyed by filename downstream
    (m2/m3), so keep only the base pass (sample_idx 0 / seed 42)."""
    out = []
    seen: set[str] = set()
    src = results_dir / "inference_results.jsonl"
    for line in src.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("sample_idx", 0) != 0:
            continue
        if r["filename"] in seen:
            continue
        cons = (r.get("considerato_predicted") or "").strip()
        if cons:
            seen.add(r["filename"])
            out.append((r["filename"], cons, extract_case_facts(r["filename"])))
    return out


def run(source: str, results_dir: Path | None, base_url: str, model: str,
        limit: int | None, resume: bool, workers: int,
        out: Path | None = None) -> None:
    if source == "judge":
        out_path = out or (RESULTS_DIR / "sillogismo_judge.jsonl")
        cases = _judge_cases()
    else:
        if results_dir is None:
            raise SystemExit("--source llm requires --results-dir")
        out_path = out or (results_dir / "sillogismo_labels.jsonl")
        cases = _llm_cases(results_dir)

    if limit:
        cases = cases[:limit]
    done = _load_done(out_path) if resume else set()
    todo = [c for c in cases if c[0] not in done]
    print(f"[premise_labeler] source={source} model={model} cases={len(cases)} "
          f"todo={len(todo)} (resume skipped {len(cases) - len(todo)})")

    client = OpenAI(base_url=base_url, api_key=os.environ.get("LABELER_API_KEY", "EMPTY"))
    mode = "a" if resume else "w"

    def _work(case):
        fn, cons, facts = case
        sents = split_sentences(cons)
        if not sents:
            return _to_record(fn, [], {}, source)
        raw = label_considerato(client, model, sents, facts, filename=fn)
        return _to_record(fn, sents, raw, source)

    n = 0
    stop: StopRequested | None = None
    with out_path.open(mode, encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_work, c): c for c in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except StopRequested as exc:
                stop = stop or exc
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 20 == 0:
                print(f"  labelled {n}/{len(todo)}")
    print(f"[premise_labeler] wrote {n} records -> {out_path}")
    if stop is not None:
        raise stop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["judge", "llm"], required=True)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output path (panel labelling writes per-labeller files).")
    args = ap.parse_args()
    run(args.source, args.results_dir, args.base_url, args.model, args.limit,
        args.resume, args.workers, out=args.out)


if __name__ == "__main__":
    main()
