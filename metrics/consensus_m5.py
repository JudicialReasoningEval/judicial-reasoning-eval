"""M5 panel consensus — majority-vote faithfulness gate.

Merges the per-judge M5 outputs in `results/panel_m5/<model>.m5.<tag>.jsonl`
(produced by pipeline/run_panel_m5.py) into the canonical per-model
`m5_faithfulness.jsonl` / `m5_faithfulness.json`, mirroring
consensus_labels.py for the premise labels:

  * per case, per check (citations / facts): each judge casts a binary
    "has unsupported items" vote; the consensus flag needs a strict
    majority (2 of 3);
  * the per-case consensus groundedness is the MEDIAN of the judges'
    count-based groundedness (the scalar analogue of majority voting);
  * inter-judge agreement is Fleiss kappa per check, per model and pooled,
    written to `results/_panel_m5_agreement.json`.

An existing canonical file is backed up to `.pre_panel.bak`
before being overwritten (once — an existing backup is never clobbered).

Usage:
    python -m metrics.consensus_m5
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics.consensus_labels import _fleiss_kappa

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PANEL_DIR = RESULTS_DIR / "panel_m5"

CHECKS = ("citations", "facts")


def _load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["filename"]] = rec
    return out


def _discover() -> tuple[list[str], dict[str, dict[str, Path]]]:
    """Return (tags, groups): groups maps model-dir name -> {judge tag: path}."""
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    tags: set[str] = set()
    for p in sorted(PANEL_DIR.glob("*.jsonl")):
        stem = p.name[: -len(".jsonl")]
        if ".m5." not in stem:
            continue
        model, tag = stem.split(".m5.", 1)
        groups[model][tag] = p
        tags.add(tag)
    return sorted(tags), dict(groups)


def _flag(rec: dict, check: str) -> int:
    key = "citations_unsupported" if check == "citations" else "facts_ungrounded"
    return 1 if int(rec.get(key, 0) or 0) > 0 else 0


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".pre_panel.bak")
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)


def main() -> None:
    tags, groups = _discover()
    if len(tags) < 3:
        raise SystemExit(f"[m5-consensus] need 3 judge tags, found {tags}")
    threshold = len(tags) // 2 + 1
    print(f"[m5-consensus] judges: {tags} (majority = {threshold}/{len(tags)})")

    agreement: dict[str, dict] = {}
    pooled_rows: dict[str, list[list[int]]] = {c: [] for c in CHECKS}

    for model, paths in sorted(groups.items()):
        if set(paths) != set(tags):
            print(f"[m5-consensus] {model}: missing judges {set(tags) - set(paths)} — skipped")
            continue
        per_judge = {t: _load(paths[t]) for t in tags}
        common = sorted(set.intersection(*(set(v) for v in per_judge.values())))
        n_union = len(set.union(*(set(v) for v in per_judge.values())))
        if n_union != len(common):
            print(f"[m5-consensus] {model}: {n_union - len(common)} cases lack full panel — dropped")

        rows: dict[str, list[list[int]]] = {c: [] for c in CHECKS}
        per_case: list[dict] = []
        for fn in common:
            recs = {t: per_judge[t][fn] for t in tags}
            flags = {}
            for c in CHECKS:
                votes = sum(_flag(recs[t], c) for t in tags)
                flags[c] = votes >= threshold
                rows[c].append([len(tags) - votes, votes])
            g = float(np.median([recs[t]["groundedness"] for t in tags]))
            per_case.append({
                "filename": fn,
                "groundedness": g,
                "flag_citations": flags["citations"],
                "flag_facts": flags["facts"],
                "flagged": flags["citations"] or flags["facts"],
                "judge_groundedness": {t: recs[t]["groundedness"] for t in tags},
                # kept for compare_models/report compatibility: consensus counts
                # are the medians of the judges' counts per axis
                "citations_total": int(np.median([recs[t]["citations_total"] for t in tags])),
                "citations_unsupported": int(np.median([recs[t]["citations_unsupported"] for t in tags])),
                "facts_total": int(np.median([recs[t]["facts_total"] for t in tags])),
                "facts_ungrounded": int(np.median([recs[t]["facts_ungrounded"] for t in tags])),
            })
        for c in CHECKS:
            pooled_rows[c].extend(rows[c])

        model_dir = RESULTS_DIR / model
        out_jsonl = model_dir / "m5_faithfulness.jsonl"
        out_json = model_dir / "m5_faithfulness.json"
        _backup(out_jsonl)
        _backup(out_json)
        with out_jsonl.open("w", encoding="utf-8") as fh:
            for r in per_case:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        g = [r["groundedness"] for r in per_case]
        kappas = {c: round(_fleiss_kappa(rows[c]), 4) for c in CHECKS}
        agg = {
            "n_cases": len(per_case),
            "judge_model": f"panel-{len(tags)}-majority({', '.join(tags)})",
            "groundedness": float(np.mean(g)) if g else float("nan"),
            "hallucination_rate": float(1 - np.mean(g)) if g else float("nan"),
            "n_with_any_unsupported": sum(1 for r in per_case if r["flagged"]),
            "n_flag_citations": sum(1 for r in per_case if r["flag_citations"]),
            "n_flag_facts": sum(1 for r in per_case if r["flag_facts"]),
            "per_judge_groundedness": {
                t: float(np.mean([r["judge_groundedness"][t] for r in per_case]))
                for t in tags} if per_case else {},
            "fleiss_kappa": kappas,
        }
        out_json.write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        agreement[model] = {"n_cases": len(per_case), "fleiss_kappa": kappas,
                            "n_flagged": agg["n_with_any_unsupported"]}
        print(f"[m5-consensus] {model}: n={len(per_case)} "
              f"groundedness={agg['groundedness']:.3f} flagged={agg['n_with_any_unsupported']} "
              f"kappa(cit)={kappas['citations']} kappa(facts)={kappas['facts']}")

    payload = {
        "judges": tags,
        "majority_threshold": threshold,
        "per_model": agreement,
        "pooled": {
            "n_cases": len(pooled_rows[CHECKS[0]]),
            "fleiss_kappa": {c: round(_fleiss_kappa(pooled_rows[c]), 4) for c in CHECKS},
        },
    }
    out = RESULTS_DIR / "_panel_m5_agreement.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[m5-consensus] pooled kappa: {payload['pooled']['fleiss_kappa']} -> {out}")


if __name__ == "__main__":
    main()
