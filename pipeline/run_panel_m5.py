"""M5 panel judging driver (majority-vote faithfulness gate).

M5 uses the same 3-model majority panel as the premise labels. This driver mirrors
run_panel_labeling.py: for EACH judge it rents ONE vast.ai instance
(SSH-exec serving via CassazioneVastManager.launch()), judges the seed-42
generation of every results/<model>/ against the case facts, then destroys
the instance — one rental per judge.

    results/panel_m5/<model>.m5.<judge_tag>.jsonl   (per-judge, resume-safe)

Afterwards run `python -m metrics.consensus_m5` to majority-merge into the
canonical m5_faithfulness.{jsonl,json} and
compute inter-judge Fleiss kappa.

Usage:
    python -m pipeline.run_panel_m5                 # all 3 judges
    python -m pipeline.run_panel_m5 --judges qwen3.6-27b
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    from cassazione_manager import CassazioneVastManager  # noqa: E402
    from cost_tracker import VastCostTracker  # noqa: E402
except ImportError:                                        # pragma: no cover
    # GPU provisioning is deliberately outside this repository: the panel ran on
    # rented instances from a commodity marketplace, and that plumbing is
    # account-specific. The driver still imports, so its phases, prompts,
    # budgets and resume logic remain auditable. To re-run a phase, serve the
    # judge model on any OpenAI-compatible endpoint and call the metrics
    # modules directly (see README).
    class _ProvisioningUnavailable:
        _WHY = (
            "GPU provisioning is not included in this repository. Serve the judge "
            "model yourself on an OpenAI-compatible endpoint and drive it with "
            "the metrics modules; see README."
        )

        def __init__(self, *_args, **_kwargs):
            raise SystemExit(self._WHY)

    class CassazioneVastManager(_ProvisioningUnavailable):  # type: ignore[no-redef]
        pass

    class VastCostTracker(_ProvisioningUnavailable):        # type: ignore[no-redef]
        pass

from pipeline.run_panel_labeling import (  # noqa: E402
    BALANCE_FLOOR_USD, LABELERS, LAUNCH_MARGIN_USD,
    _balance, _model_dirs,
)
import metrics.m5_faithfulness as m5  # noqa: E402

# Complete facts plus a complete Considerato need a longer window than
# premise labelling (32,768).
SERVE_MAX_LEN = 65536

RESULTS_DIR = ROOT / "results"
PANEL_M5_DIR = RESULTS_DIR / "panel_m5"


def _run_judge(tag: str, cfg: dict, dirs: list[Path]) -> None:
    bal = _balance()
    need = BALANCE_FLOOR_USD + LAUNCH_MARGIN_USD
    if bal < need:
        raise SystemExit(
            f"[m5-panel] balance ${bal:.2f} < floor ${BALANCE_FLOOR_USD:.0f} + "
            f"margin ${LAUNCH_MARGIN_USD:.0f} — refusing to launch {tag}."
        )
    print(f"[m5-panel] === judge {tag} === balance ${bal:.2f}")

    m5.REASONING_EFFORT = cfg["reasoning_effort"]
    m5.CHAT_TEMPLATE_KWARGS = cfg["chat_template_kwargs"]

    manager = CassazioneVastManager(
        model=cfg["vllm_model"],
        num_gpus=cfg["num_gpus"],
        max_model_len=SERVE_MAX_LEN,
        extra_vllm_args=cfg["extra_vllm_args"],
        vllm_image=cfg["vllm_image"],
    )
    with VastCostTracker(manager, model_label=f"m5-panel-{tag}", step="m5-judging"):
        try:
            base_url, bearer = manager.launch()
            os.environ["LABELER_API_KEY"] = bearer

            t0 = time.time()
            for d in dirs:
                print(f"[m5-panel] {tag} <- {d.name}")
                m5.score_model(d, base_url=base_url, model=cfg["vllm_model"],
                               workers=16, resume=True,
                               out=PANEL_M5_DIR / f"{d.name}.m5.{tag}.jsonl")
            print(f"[m5-panel] {tag} done in {(time.time() - t0) / 60:.1f} min")
        finally:
            manager.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judges", nargs="*", default=list(LABELERS),
                    help="Subset of panel judges to run (default: all).")
    args = ap.parse_args()

    PANEL_M5_DIR.mkdir(exist_ok=True)
    dirs = _model_dirs()
    print(f"[m5-panel] model dirs to judge: {[d.name for d in dirs]}")

    for tag in args.judges:
        if tag not in LABELERS:
            raise SystemExit(f"unknown judge {tag!r}; choose from {list(LABELERS)}")
        _run_judge(tag, LABELERS[tag], dirs)

    print("[m5-panel] all judges complete.")


if __name__ == "__main__":
    main()
