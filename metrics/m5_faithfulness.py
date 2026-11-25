"""M5 — Affidabilità / faithfulness (gate).

Validates: *"the LLM invents nothing"* — the considerato's citations are real
and relevant, and its factual claims are grounded in the case record.

Each judge of the three-model panel (pipeline/run_panel_m5.py, merged by
metrics/consensus_m5.py) inspects a model's considerato against the facts
section of its case (RITENUTO IN FATTO) on two axes:

  * citations — cited norms/precedents that are wrong, irrelevant or fabricated
  * facts     — factual assertions about the case not supported by the record

We compute groundedness from the judge's COUNTS (more defensible than its
holistic float, which we keep only as a cross-check):

    groundedness = 1 - (citations_unsupported + facts_ungrounded)
                       / max(1, citations_total + facts_total)

CLI:
  python m5_faithfulness.py --results-dir results/gpt-oss-20b
    Reads:  <results-dir>/inference_results.jsonl + the .md corpus (facts)
    Writes: <results-dir>/m5_faithfulness.jsonl  (per-case)
            <results-dir>/m5_faithfulness.json   (aggregate)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI

from metrics.premise_labeler import StopRequested, full_case_facts

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BASE_URL = os.environ.get("LABELER_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("M5_JUDGE_MODEL", os.environ.get("LABELER_MODEL", "openai/gpt-oss-20b"))
MAX_TOKENS = int(os.environ.get("M5_MAX_TOKENS", 4096))
# gpt-oss reasons before answering; keep CoT short so the budget reaches the JSON.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low")
# qwen-style enable_thinking toggles for non-harmony judges (JSON string, e.g.
# '{"enable_thinking": false}'); empty -> not sent. Panel drivers set these
# module attributes directly, mirroring premise_labeler.
CHAT_TEMPLATE_KWARGS = os.environ.get("LABELER_CHAT_TEMPLATE_KWARGS", "")
# Extra request body merged last, and an optional usage callback — same
# contract as premise_labeler (see its docstrings): lets a gateway-backed
# driver pass provider-specific knobs and meter spend without forking prompts.
EXTRA_BODY: dict = {}
USAGE_SINK = None
BEFORE_CALL = None

_SYSTEM_PROMPT = """\
Sei un giurista revisore. Verifichi l'AFFIDABILITÀ di un "considerato in diritto" \
prodotto da un sistema automatico, confrontandolo con i fatti del caso. NON valuti la \
qualità giuridica né l'esito: verifichi solo se il testo INVENTA qualcosa. Sii rigoroso \
e conservativo: segnala come problematico solo ciò che è chiaramente errato o non \
sostenuto dai fatti."""

_USER_TEMPLATE = """\
Fatti del caso (RITENUTO IN FATTO):
<FATTI>
{facts}
</FATTI>

Considerato in diritto da verificare:
<TESTO>
{considerato}
</TESTO>

Verifica due dimensioni:
1) CITAZIONI: ogni norma (es. art. 319 c.p.) o precedente (es. Cass. n. .../...) citato.
   Conta quante citazioni totali ci sono e quante sono ERRATE, irrilevanti o INVENTATE
   (numero di sentenza implausibile/inesistente, norma non pertinente).
2) FATTI: ogni affermazione su FATTI specifici del caso. Conta quante affermazioni di
   fatto verificabili ci sono e quante NON sono sostenute dai FATTI sopra (fatti inventati).

Restituisci SOLO un oggetto JSON valido:
{{"citations_total": int, "citations_unsupported": int,
  "facts_total": int, "facts_ungrounded": int,
  "groundedness": float (0..1, stima complessiva),
  "notes": "una frase sulle criticità principali, o vuota"}}"""


# The judge reads the complete facts section and the complete Considerato.
# These caps are a context guard: of the 91 evaluated cases, one has a longer
# facts section (76,030 characters) and is cut at the cap.
FACTS_CHARS = int(os.environ.get("M5_FACTS_CHARS", "40000"))
CONSIDERATO_CHARS = int(os.environ.get("M5_CONS_CHARS", "60000"))


def _call_judge(client: OpenAI, model: str, considerato: str, facts: str,
                retries: int = 3, filename: str = "") -> dict | None:
    facts, considerato = facts or "", considerato or ""
    if len(facts) > FACTS_CHARS or len(considerato) > CONSIDERATO_CHARS:
        print(f"  [m5-judge] {filename}: input above the context guard "
              f"(facts {len(facts)}, considerato {len(considerato)} chars), truncated")
    prompt = _USER_TEMPLATE.format(facts=facts[:FACTS_CHARS],
                                   considerato=considerato[:CONSIDERATO_CHARS])
    # Only send params the serving model understands (same rule as
    # premise_labeler): both empty -> no extra_body at all.
    extra_body: dict = {}
    if REASONING_EFFORT:
        extra_body["reasoning_effort"] = REASONING_EFFORT
    if CHAT_TEMPLATE_KWARGS:
        extra_body["chat_template_kwargs"] = json.loads(CHAT_TEMPLATE_KWARGS)
    extra_body.update(EXTRA_BODY)
    for attempt in range(retries):
        if BEFORE_CALL is not None:  # see premise_labeler._call_labeler
            BEFORE_CALL()
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=MAX_TOKENS, timeout=180,
                messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                extra_body=extra_body or None,
            )
            if USAGE_SINK is not None and getattr(resp, "usage", None) is not None:
                USAGE_SINK(model, filename, 0, resp.usage)
            content = resp.choices[0].message.content
            if not content:
                if resp.choices[0].finish_reason == "length":
                    print("  [m5-judge] completion truncated before JSON; "
                          "increase M5_MAX_TOKENS before retrying")
                    return None
                raise ValueError(f"null content (finish={resp.choices[0].finish_reason})")
            # think-tag judges may emit a trace (with braces) before the JSON
            content = re.sub(r"<think>[\s\S]*?</think>|\[THINK\][\s\S]*?\[/THINK\]", "", content)
            m = re.search(r"\{[\s\S]+\}", content)
            if not m:
                if resp.choices[0].finish_reason == "length":
                    print("  [m5-judge] completion truncated before JSON; "
                          "increase M5_MAX_TOKENS before retrying")
                    return None
                raise ValueError("no JSON in response")
            result = json.loads(m.group())
            for total_key, bad_key in (
                ("citations_total", "citations_unsupported"),
                ("facts_total", "facts_ungrounded"),
            ):
                total = int(result.get(total_key, 0) or 0)
                bad = int(result.get(bad_key, 0) or 0)
                if total < 0 or bad < 0 or bad > total:
                    raise ValueError(
                        f"invalid counts: {bad_key}={bad}, {total_key}={total}"
                    )
            return result
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  [m5-judge] failed after {retries}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def _groundedness(d: dict) -> float:
    ct = max(0, int(d.get("citations_total", 0) or 0))
    cu = max(0, int(d.get("citations_unsupported", 0) or 0))
    ft = max(0, int(d.get("facts_total", 0) or 0))
    fu = max(0, int(d.get("facts_ungrounded", 0) or 0))
    denom = max(1, ct + ft)
    return 1.0 - min(cu + fu, denom) / denom


def _read_jsonl(path: Path) -> list[dict]:
    """Tolerant reader: skips blank/malformed lines (e.g. a half-written last
    line left by an interrupted run)."""
    if not path.exists():
        return []
    out: list[dict] = []
    for l in path.read_text(encoding="utf-8").split("\n"):
        l = l.strip()
        if not l:
            continue
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return out


def score_model(results_dir: Path, base_url: str = DEFAULT_BASE_URL,
                model: str = DEFAULT_MODEL, workers: int = 12,
                limit: int | None = None, resume: bool = False,
                out: Path | None = None) -> dict:
    # 5-seed files carry 5 records per case; M5 (like M2/M3) is defined on
    # the base pass — keep sample_idx 0 / first occurrence per filename.
    rows = _read_jsonl(results_dir / "inference_results.jsonl")
    cases: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in rows:
        if r.get("sample_idx", 0) != 0 or r["filename"] in seen:
            continue
        cons = (r.get("considerato_predicted") or "").strip()
        if cons:
            seen.add(r["filename"])
            cases.append((r["filename"], cons))
    if limit:
        cases = cases[:limit]

    panel_mode = out is not None
    out_jsonl = out if panel_mode else results_dir / "m5_faithfulness.jsonl"
    existing: list[dict] = []
    if resume:
        existing = _read_jsonl(out_jsonl)
        done = {r["filename"] for r in existing if r.get("filename")}
        before = len(cases)
        cases = [c for c in cases if c[0] not in done]
        if done:
            print(f"  M5 resume: {len(done)} già fatti, {len(cases)}/{before} da fare")

    client = OpenAI(base_url=base_url, api_key=os.environ.get("LABELER_API_KEY", "EMPTY"))

    def _work(case):
        fn, cons = case
        d = _call_judge(client, model, cons, full_case_facts(fn), filename=fn)
        if d is None:
            return None
        return {"filename": fn,
                "groundedness": _groundedness(d),
                "judge_groundedness": float(d.get("groundedness", float("nan")) or float("nan")),
                "citations_total": int(d.get("citations_total", 0) or 0),
                "citations_unsupported": int(d.get("citations_unsupported", 0) or 0),
                "facts_total": int(d.get("facts_total", 0) or 0),
                "facts_ungrounded": int(d.get("facts_ungrounded", 0) or 0),
                "notes": (d.get("notes") or "")[:300]}

    new_rows: list[dict] = []
    stop: StopRequested | None = None
    mode = "a" if resume else "w"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open(mode, encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_work, c) for c in cases]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except StopRequested as exc:
                stop = stop or exc
                continue
            if r is not None:
                new_rows.append(r)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            if i % 20 == 0:
                print(f"  judged {i}/{len(cases)}")

    per_case = existing + new_rows  # resume merges prior results
    if stop is not None:
        raise stop

    g = [r["groundedness"] for r in per_case]
    agg = {
        "n_cases": len(per_case),
        "judge_model": model,
        "groundedness": float(np.mean(g)) if g else float("nan"),
        "hallucination_rate": float(1 - np.mean(g)) if g else float("nan"),
        "n_with_any_unsupported": sum(1 for r in per_case
                                      if r["citations_unsupported"] + r["facts_ungrounded"] > 0),
        "citations_total": sum(r["citations_total"] for r in per_case),
        "citations_unsupported": sum(r["citations_unsupported"] for r in per_case),
        "facts_total": sum(r["facts_total"] for r in per_case),
        "facts_ungrounded": sum(r["facts_ungrounded"] for r in per_case),
    }
    if not panel_mode:  # panel runs: consensus_m5 writes the canonical aggregate
        (results_dir / "m5_faithfulness.json").write_text(
            json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M5 faithfulness (n={agg['n_cases']}, judge={model}): "
          f"groundedness={agg['groundedness']:.3f}  "
          f"hallucination_rate={agg['hallucination_rate']:.3f}  "
          f"cases_with_issue={agg['n_with_any_unsupported']}")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip cases already judged")
    ap.add_argument("--out", type=Path, default=None,
                    help="per-judge output path (panel mode; metrics.consensus_m5 writes the merged file)")
    args = ap.parse_args()
    score_model(args.results_dir, base_url=args.base_url, model=args.model,
                workers=args.workers, limit=args.limit, resume=args.resume,
                out=args.out)


if __name__ == "__main__":
    main()
