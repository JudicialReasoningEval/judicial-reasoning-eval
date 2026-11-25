"""Few-shot leakage probe.

For every seed-42 generation, classify each precedent the model cites by what
the model could and could not see:

  * in the case's own RITENUTO (model input)          -> visible / legitimate
  * in the k=2 few-shot exemplars but NOT the input   -> LEAKAGE-SUSPECT
  * in the withheld gold CONSIDERATO only             -> convergent (court cited it too)
  * in none of the above                              -> external / fabricated

The leakage-suspect rate asks whether M1 (citation correctness) and M5
(faithfulness) are inflated by the model echoing authorities it
saw in its own prompt exemplars, rather than reasoning about the case?

Precedent key = the Cassazione sentence number (3-6 digits after "n."), kept
only when it is dated (a nearby 4-digit year) or in an explicit citation
context (Sez./Cass./S.U./sentenza/ordinanza). Same extractor applied to output,
exemplars, gold, and input, so any bias is shared.

Caveat: exemplars are compared at full text; the prompt truncates them to the
per-model budget, so E is an UPPER bound and the leakage rate reported here is
conservative (if anything it over-attributes to leakage).

Run: python -m analysis.fewshot_leakage
Writes results/_fewshot_leakage.json.
"""
from __future__ import annotations
import json, re, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FILTERED = ROOT / "sentenze_md_filtered"

_CONSID_SPLIT = re.compile(r"\n\s*CONSIDERATO\s+IN\s+DIRITTO", re.I)
_PQM_SPLIT = re.compile(r"\n\s*P\.?\s*Q\.?\s*M\.?", re.I)
_NUM = re.compile(r"n\.?\s*(\d{3,6})\b", re.I)
_YEAR = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")
_CTX = re.compile(r"(sez|cass|s\.?\s*u|sentenza|ordinanza|ord\.)", re.I)


def precedents(text: str) -> set[str]:
    """Set of precedent sentence-numbers cited in `text`."""
    if not text:
        return set()
    out: set[str] = set()
    for m in _NUM.finditer(text):
        num = m.group(1)
        head = text[max(0, m.start() - 30): m.start()]
        tail = text[m.end(): m.end() + 30]
        ym = _YEAR.search(tail)
        if ym or _CTX.search(head):
            out.add(num)
    return out


def split_judgment(md: str) -> tuple[str, str]:
    """Return (ritenuto_part, considerato_plus_pqm) from a filtered .md.

    The facts part is exactly the model input (pipeline/case_sections.py);
    a judgment without a separable facts section falls back to a plain split.
    """
    from pipeline.case_sections import facts_and_withheld

    split = facts_and_withheld(md)
    if split is not None:
        return split
    parts = _CONSID_SPLIT.split(md, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return md, ""            # no section structure: treat all as input


def load_case(filename: str) -> tuple[set[str], set[str], set[str]]:
    """Return (ritenuto_precs, gold_considerato_precs, whole_precs) for a case."""
    p = FILTERED / filename
    if not p.exists():
        return set(), set(), set()
    md = p.read_text(encoding="utf-8", errors="ignore")
    rit, con = split_judgment(md)
    return precedents(rit), precedents(con), precedents(md)


def analyse_model(model_dir: Path) -> dict | None:
    jf = model_dir / "inference_results.jsonl"
    if not jf.exists():
        return None
    case_cache: dict[str, tuple[set[str], set[str], set[str]]] = {}
    ex_cache: dict[str, set[str]] = {}          # exemplar filename -> precedents (whole doc)

    def case_precs(fn):
        if fn not in case_cache:
            case_cache[fn] = load_case(fn)
        return case_cache[fn]

    def exemplar_precs(fn):
        if fn not in ex_cache:
            _, _, whole = load_case(fn)
            ex_cache[fn] = whole
        return ex_cache[fn]

    n_cases = 0
    tot = visible = leak = convergent = external = 0
    cases_with_leak = 0
    leaked_numbers: dict[str, int] = {}

    for line in jf.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("seed") != 42 or r.get("sample_idx", 0) != 0:
            continue
        out_text = (r.get("considerato_predicted") or "") + "\n" + (r.get("pqm_predicted") or "")
        P = precedents(out_text)
        if not P and not r.get("considerato_predicted"):
            continue
        n_cases += 1
        rit, gold, _ = case_precs(r["filename"])
        E: set[str] = set()
        for ex in r.get("fewshot_used", []):
            E |= exemplar_precs(ex["filename"])
        case_leak = 0
        for p in P:
            tot += 1
            if p in rit:
                visible += 1
            elif p in E:
                leak += 1
                case_leak += 1
                leaked_numbers[p] = leaked_numbers.get(p, 0) + 1
            elif p in gold:
                convergent += 1
            else:
                external += 1
        if case_leak:
            cases_with_leak += 1

    if tot == 0:
        return {"model": model_dir.name, "n_cases": n_cases, "precedents_cited": 0}
    return {
        "model": model_dir.name,
        "n_cases": n_cases,
        "precedents_cited": tot,
        "per_output": round(tot / max(1, n_cases), 2),
        "visible_in_ritenuto": visible,
        "leakage_suspect": leak,
        "convergent_with_gold": convergent,
        "external_or_fabricated": external,
        "leakage_rate": round(leak / tot, 4),
        "external_rate": round(external / tot, 4),
        "cases_with_any_leak": cases_with_leak,
        "top_leaked_numbers": sorted(leaked_numbers.items(), key=lambda kv: -kv[1])[:8],
    }


def main():
    rows = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        res = analyse_model(d)
        if res and res.get("precedents_cited", 0) >= 0:
            rows.append(res)
    rows.sort(key=lambda r: -r.get("precedents_cited", 0))
    out = {"per_model": rows}
    # pooled over models with meaningful output (>=50 precedents cited)
    big = [r for r in rows if r.get("precedents_cited", 0) >= 50]
    if big:
        T = sum(r["precedents_cited"] for r in big)
        out["pooled"] = {
            "models": [r["model"] for r in big],
            "precedents_cited": T,
            "leakage_rate": round(sum(r["leakage_suspect"] for r in big) / T, 4),
            "external_rate": round(sum(r["external_or_fabricated"] for r in big) / T, 4),
            "visible_rate": round(sum(r["visible_in_ritenuto"] for r in big) / T, 4),
            "convergent_rate": round(sum(r["convergent_with_gold"] for r in big) / T, 4),
        }
    (RESULTS / "_fewshot_leakage.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"models analysed: {len(rows)}")
    print(f"{'model':26} {'preds':>6} {'/out':>5} {'leak%':>6} {'ext%':>6} {'vis%':>6}")
    for r in rows:
        if r.get("precedents_cited", 0) == 0:
            print(f"{r['model']:26} {0:>6}"); continue
        print(f"{r['model']:26} {r['precedents_cited']:>6} {r['per_output']:>5} "
              f"{r['leakage_rate']*100:>5.1f}% {r['external_rate']*100:>5.1f}% "
              f"{r['visible_in_ritenuto']/r['precedents_cited']*100:>5.1f}%")
    if "pooled" in out:
        p = out["pooled"]
        print(f"\nPOOLED ({len(big)} models, {p['precedents_cited']} precedents): "
              f"leakage {p['leakage_rate']*100:.1f}%, external {p['external_rate']*100:.1f}%, "
              f"visible {p['visible_rate']*100:.1f}%, convergent {p['convergent_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
