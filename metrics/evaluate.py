"""Evaluation orchestrator — the full 5-metric system for one model.

Computes the organic metric set comparing the LLM `considerato in diritto`
against the judge:

  M1  Correttezza normativa     (article-citation F1, weighted)   — composite
  M2  Fedeltà strutturale       (JSD + presence-F1)               — composite
  M3  Allineamento semantico    (soft-cosine F1, Qwen3)           — composite
  M4  Accordo dispositivo       (deterministic + Cohen's kappa)   — GATE
  M5  Affidabilità              (groundedness / hallucination)    — GATE

Quality composite (gate-passing models only):
  Q = w1·M1 + w2·M2 + w3·M3        (default w1=w2=w3=1/3)

M1/M2/M4 are fully offline. M3 needs the Qwen3-Embedding-8B model (use
--skip-m3 for a fast offline pass, then run with M3 once the model is local).
M5 is read from `m5_faithfulness.json` if present (produced separately by
m5_faithfulness.py, which needs a served model); otherwise reported as null.

CLI:
  python evaluate.py --results-dir results/<model>
  python evaluate.py --results-dir results/<model> --skip-m3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import m2_structural
from metrics import m3_semantic
from metrics import m4_dispositive
from pipeline.baselines import ARTICLE_LABELS
from metrics.stats_utils import multi_label_eval

ROOT = Path(__file__).resolve().parent.parent

# Gate thresholds (committed 2026-07-11, justified in the paper):
# M4: Landis-Koch lower bound of "moderate" agreement; M5: deployment bar
# (at most 1 unsupported assertion in 10), set independently of the field.
GATE_M4_KAPPA = 0.40
GATE_M5_GROUNDEDNESS = 0.90


def _m1(results_dir: Path) -> dict | None:
    scored = results_dir / "inference_results_scored.jsonl"
    if not scored.exists():
        return None
    res = multi_label_eval(scored, ARTICLE_LABELS, bootstrap=True)
    if res.get("n", 0) == 0:
        return None
    return {
        "n_cases": res["n"],
        "weighted_f1": res["weighted"]["f1"],
        "macro_f1": res["macro"]["f1"],
        "micro_f1": res["micro"]["f1"],
        "subset_acc": res["subset_acc"],
        "jaccard_mean": res["jaccard_mean"],
        "per_class_f1": {lbl: res["per_class"][lbl]["f1"] for lbl in ARTICLE_LABELS},
        "bootstrap": res.get("bootstrap"),
    }


def _m5(results_dir: Path) -> dict | None:
    path = results_dir / "m5_faithfulness.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def evaluate(results_dir: Path, skip_m3: bool = False,
             weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)) -> dict:
    m1 = _m1(results_dir)
    # M2 needs the direct-3 premise labels (premise_labeler, GPU-served model).
    # Like M5, report null when they haven't been produced yet.
    if (results_dir / "sillogismo_labels.jsonl").exists():
        m2 = m2_structural.score_model(results_dir)
    else:
        m2 = None
    if skip_m3:
        # Don't clobber a previously-computed M3: reuse it from metrics.json if present.
        prev = results_dir / "metrics.json"
        m3 = json.loads(prev.read_text()).get("M3_semantic") if prev.exists() else None
    else:
        m3 = m3_semantic.score_model(results_dir)
    m4 = m4_dispositive.score_model(results_dir)
    m5 = _m5(results_dir)

    # Quality composite (only defined when M1, M2, M3 are all present).
    composite = None
    if m1 and m2 and m3:
        w1, w2, w3 = weights
        composite = w1 * m1["weighted_f1"] + w2 * m2["m2"]["mean"] + w3 * m3["m3"]["mean"]

    # Gates (report values + provisional pass/fail).
    m4_kappa = m4.get("cohen_kappa") if m4 else None
    m5_grounded = m5.get("groundedness") if m5 else None
    gates = {
        "m4_dispositive": {
            "kappa": m4_kappa,
            "threshold": GATE_M4_KAPPA,
            "pass": (m4_kappa is not None and m4_kappa >= GATE_M4_KAPPA),
        },
        "m5_faithfulness": {
            "groundedness": m5_grounded,
            "threshold": GATE_M5_GROUNDEDNESS,
            "pass": (m5_grounded is not None and m5_grounded >= GATE_M5_GROUNDEDNESS)
                    if m5 is not None else None,
        },
    }

    payload = {
        "model": results_dir.name,
        "weights": {"m1": weights[0], "m2": weights[1], "m3": weights[2]},
        "M1_normative": m1,
        "M2_structural": m2,
        "M3_semantic": m3,
        "M4_dispositive": m4,
        "M5_faithfulness": m5,
        "quality_composite": composite,
        "gates": gates,
    }
    out = results_dir / "metrics.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                   encoding="utf-8")

    print()
    print(f"=== {results_dir.name} ===")
    if m1:
        print(f"M1 normative   weighted-F1 = {m1['weighted_f1']:.3f}  (macro {m1['macro_f1']:.3f})")
    if m2:
        print(f"M2 structural  = {m2['m2']['mean']:.3f}")
    else:
        print("M2 structural  = (no premise labels yet)")
    if m3:
        print(f"M3 semantic    = {m3['m3']['mean']:.3f}")
    else:
        print("M3 semantic    = (skipped)")
    if m4:
        print(f"M4 dispositive GATE kappa = {m4['cohen_kappa']:.3f}  -> pass={gates['m4_dispositive']['pass']}")
    print(f"M5 faithfulness GATE = {m5_grounded if m5_grounded is not None else '(not computed)'}")
    if composite is not None:
        print(f"Quality composite Q = {composite:.3f}")
    print(f"Wrote {out}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--skip-m3", action="store_true",
                        help="skip M3 (no Qwen3 embedding model required)")
    parser.add_argument("--weights", type=float, nargs=3, default=(1 / 3, 1 / 3, 1 / 3),
                        metavar=("W1", "W2", "W3"), help="composite weights for M1 M2 M3")
    args = parser.parse_args()
    evaluate(args.results_dir, skip_m3=args.skip_m3, weights=tuple(args.weights))


if __name__ == "__main__":
    main()
