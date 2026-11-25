"""Merge panel premise labels by majority vote + Fleiss kappa.

Reads the per-labeller files produced by pipeline/run_panel_labeling.py:

    results/panel_labels/sillogismo_judge.<tag>.jsonl
    results/panel_labels/<model>.sillogismo_labels.<tag>.jsonl

and writes, for each group, the consensus file at the CANONICAL path consumed
by m2_structural / m3_semantic (an existing file is backed up
to *.pre_panel.bak once):

    results/sillogismo_judge.jsonl
    results/<model>/sillogismo_labels.jsonl

Consensus rule: a premise is assigned to a sentence iff >= ceil(n/2)+ (strict
majority, i.e. 2 of 3) of the labellers assigned it. Sentences keep the
original text/order (all labellers split identically — verified per case).

Agreement: Fleiss kappa per premise (binary assigned/not-assigned, one item
per sentence) per group + pooled, written to results/_panel_agreement.json.

Usage:
    python -m metrics.consensus_labels                # all groups, all tags found
    python -m metrics.consensus_labels --dry-run      # report only, write nothing
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PANEL_DIR = RESULTS_DIR / "panel_labels"

PREMISES = ["maggiore", "minore", "conclusione"]


def _load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["filename"]] = rec
    return out


def _discover() -> tuple[list[str], dict[str, dict[str, Path]]]:
    """Return (tags, groups) where groups maps group-name -> {tag: path}.

    Group 'judge' is the corpus side; every other group is a model dir name.
    """
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    tags: set[str] = set()
    for p in sorted(PANEL_DIR.glob("*.jsonl")):
        stem = p.name[: -len(".jsonl")]
        if stem.startswith("sillogismo_judge."):
            tag = stem[len("sillogismo_judge."):]
            groups["judge"][tag] = p
        elif ".sillogismo_labels." in stem:
            model, tag = stem.split(".sillogismo_labels.", 1)
            groups[model][tag] = p
        else:
            continue
        tags.add(tag)
    return sorted(tags), dict(groups)


def _fleiss_kappa(rows: list[list[int]], n_categories: int = 2) -> float:
    """Fleiss kappa. rows[i][c] = number of raters assigning item i to category c.
    Every row must sum to the same rater count n."""
    if not rows:
        return float("nan")
    n = sum(rows[0])
    N = len(rows)
    p_cat = [sum(r[c] for r in rows) / (N * n) for c in range(n_categories)]
    P_bar = sum(
        (sum(c * c for c in r) - n) / (n * (n - 1)) for r in rows
    ) / N
    P_e = sum(p * p for p in p_cat)
    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def _consensus_record(recs: list[dict], threshold: int) -> dict | None:
    """Majority-vote record from the same case labelled by each panel member."""
    base = recs[0]
    n_sent = len(base["sentences"])
    if any(len(r["sentences"]) != n_sent for r in recs[1:]):
        return None  # split mismatch — should not happen (same splitter/text)
    premise_labels: list[list[str]] = []
    by_premise: dict[str, list[int]] = {p: [] for p in PREMISES}
    for i in range(n_sent):
        assigned = []
        for p in PREMISES:
            votes = sum(1 for r in recs if p in (r["premise_labels"][i] or []))
            if votes >= threshold:
                assigned.append(p)
        premise_labels.append(assigned)
        for p in assigned:
            by_premise[p].append(i)
    return {"filename": base["filename"], "source": base["source"],
            "sentences": base["sentences"], "premise_labels": premise_labels,
            "by_premise": by_premise}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tags, groups = _discover()
    if len(tags) < 2:
        raise SystemExit(f"need >=2 labeller tags for a panel, found {tags}")
    threshold = len(tags) // 2 + 1
    print(f"[consensus] tags={tags} majority threshold={threshold}/{len(tags)}")

    agreement: dict[str, dict] = {}
    pooled_rows: dict[str, list[list[int]]] = {p: [] for p in PREMISES}

    for group, tag_paths in sorted(groups.items()):
        missing = [t for t in tags if t not in tag_paths]
        if missing:
            print(f"[consensus] {group}: SKIP (missing labellers: {missing})")
            continue
        per_tag = {t: _load(p) for t, p in tag_paths.items()}
        common = set.intersection(*(set(d) for d in per_tag.values()))

        out_records: list[dict] = []
        rows: dict[str, list[list[int]]] = {p: [] for p in PREMISES}
        skipped = 0
        for fn in sorted(common):
            recs = [per_tag[t][fn] for t in tags]
            cons = _consensus_record(recs, threshold)
            if cons is None:
                skipped += 1
                continue
            out_records.append(cons)
            for i in range(len(cons["sentences"])):
                for p in PREMISES:
                    votes = sum(1 for r in recs if p in (r["premise_labels"][i] or []))
                    rows[p].append([len(tags) - votes, votes])
                    pooled_rows[p].append([len(tags) - votes, votes])

        kappas = {p: round(_fleiss_kappa(rows[p]), 4) for p in PREMISES}
        n_sent = len(rows[PREMISES[0]])
        agreement[group] = {"cases": len(out_records), "sentences": n_sent,
                            "split_mismatches": skipped, "fleiss_kappa": kappas}
        print(f"[consensus] {group}: {len(out_records)} cases, {n_sent} sentences, "
              f"kappa={kappas}" + (f", {skipped} split-mismatch SKIPPED" if skipped else ""))

        if args.dry_run:
            continue

        if group == "judge":
            target = RESULTS_DIR / "sillogismo_judge.jsonl"
        else:
            target = RESULTS_DIR / group / "sillogismo_labels.jsonl"
            target.parent.mkdir(exist_ok=True)
        if target.exists() and not target.with_suffix(target.suffix + ".pre_panel.bak").exists():
            shutil.copy2(target, target.with_suffix(target.suffix + ".pre_panel.bak"))
        with target.open("w", encoding="utf-8") as fh:
            for rec in out_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    agreement["_pooled"] = {
        "sentences": len(pooled_rows[PREMISES[0]]),
        "fleiss_kappa": {p: round(_fleiss_kappa(pooled_rows[p]), 4) for p in PREMISES},
        "labelers": tags,
        "majority_threshold": threshold,
    }
    if not args.dry_run:
        out = RESULTS_DIR / "_panel_agreement.json"
        out.write_text(json.dumps(agreement, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[consensus] wrote {out}")
    else:
        print(json.dumps(agreement.get("_pooled", {}), indent=2))


if __name__ == "__main__":
    main()
