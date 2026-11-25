"""Panel premise-labelling driver.

A three-LLM panel labels every considerato sentence into the 3 syllogism premises; the
majority vote becomes the silver-standard production label and inter-labeller
Fleiss kappa quantifies convergence.

For EACH panel member this script rents ONE vast.ai instance (online vLLM
serving via CassazioneVastManager.launch()), then drives ALL labelling work
through it before destroying it — one rental per labeller:

    judge side : 181 corpus considerati -> results/panel_labels/sillogismo_judge.<tag>.jsonl
    llm side   : every results/<model>/ with inference_results.jsonl
                 (seed-42 / sample_idx-0 records only)
                 -> results/panel_labels/<model>.sillogismo_labels.<tag>.jsonl

Resume-safe at every level: per-labeller output files are appended with
--resume semantics (done filenames skipped), so a crashed run relaunches the
instance and continues where it left off.

Budget guard: refuses to launch a new instance if the vast.ai balance is
below BALANCE_FLOOR_USD (+ small margin).

Usage:
    python -m pipeline.run_panel_labeling            # all 3 labellers
    python -m pipeline.run_panel_labeling --labelers gpt-oss-120b
"""
from __future__ import annotations

import argparse
import json
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

import metrics.premise_labeler as pl  # noqa: E402

RESULTS_DIR = ROOT / "results"
PANEL_DIR = RESULTS_DIR / "panel_labels"

BALANCE_FLOOR_USD = float(os.environ.get("VAST_BALANCE_FLOOR", "50"))
# Require this much headroom above the floor before starting a labeller
# (a full labelling pass is estimated at $2-6).
LAUNCH_MARGIN_USD = float(os.environ.get("VAST_LAUNCH_MARGIN", "6"))

# ── Panel definition ────────────────────────────────────────────────────────
# Each labeller: served model, GPUs, vLLM extra args, and how to suppress /
# bound the reasoning trace for a labelling task.
LABELERS: dict[str, dict] = {
    "gpt-oss-120b": {
        "vllm_model": "openai/gpt-oss-120b",
        "num_gpus": 1,
        "extra_vllm_args": "",
        "vllm_image": None,                      # proven default (v0.21)
        "reasoning_effort": "low",               # harmony knob
        "chat_template_kwargs": "",
    },
    "nemotron-3-super-120b": {
        "vllm_model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "num_gpus": 1,
        # Same hybrid-Mamba constraints as the inference profile: fp8 KV and
        # max_num_seqs far below the default 1024 (Mamba SSM cache blocks).
        "extra_vllm_args": "--kv-cache-dtype fp8 --max-num-seqs 64",
        "vllm_image": None,                      # inference ran on the default image
        "reasoning_effort": "",
        # Honoured if the chat template supports it; harmless extra var if not
        # (the client also strips <think>…</think> defensively).
        "chat_template_kwargs": '{"enable_thinking": false}',
    },
    "qwen3.6-27b": {
        "vllm_model": "nvidia/Qwen3.6-27B-NVFP4",
        "num_gpus": 1,
        "extra_vllm_args": "--kv-cache-dtype fp8",
        # Needs the v0.24 image: the v0.21 image's vLLM cannot load this NVFP4
        # qwen3_5 checkpoint (weight-loader shape mismatch).
        "vllm_image": "vastai/vllm:v0.24.0-cuda-12.9",
        "reasoning_effort": "",
        "chat_template_kwargs": '{"enable_thinking": false}',
    },
}

# Serving context: labelling prompts are ~4-6k tokens (rubric + facts + 35
# numbered sentences); 32k leaves generous room and keeps KV small -> more
# concurrent sequences on one card.
SERVE_MAX_LEN = 32768


def _balance() -> float:
    try:
        from vast_manager import get_vast_api_key
        from vastai_sdk import VastAI
    except ImportError as exc:                              # pragma: no cover
        raise SystemExit(
            "GPU provisioning is not included in this repository. Serve the judge "
            "model yourself on an OpenAI-compatible endpoint and drive it with "
            "the metrics modules; see README.") from exc

    client = VastAI(api_key=get_vast_api_key())
    raw = client.show_user()
    data = json.loads(raw) if isinstance(raw, str) else raw
    credit = data.get("credit")
    return float(credit) if credit is not None else float("nan")


def _model_dirs() -> list[Path]:
    """Every results/<model>/ with a non-empty inference_results.jsonl."""
    dirs = []
    for d in sorted(RESULTS_DIR.iterdir()):
        f = d / "inference_results.jsonl"
        if d.is_dir() and f.exists() and f.stat().st_size > 0:
            dirs.append(d)
    return dirs


def _run_labeller(tag: str, cfg: dict, dirs: list[Path]) -> None:
    bal = _balance()
    need = BALANCE_FLOOR_USD + LAUNCH_MARGIN_USD
    if bal < need:
        raise SystemExit(
            f"[panel] balance ${bal:.2f} < floor ${BALANCE_FLOOR_USD:.0f} + "
            f"margin ${LAUNCH_MARGIN_USD:.0f} — refusing to launch {tag}."
        )
    print(f"[panel] === labeller {tag} === balance ${bal:.2f}")

    # Point the (already imported) premise_labeler module at this labeller.
    pl.REASONING_EFFORT = cfg["reasoning_effort"]
    pl.CHAT_TEMPLATE_KWARGS = cfg["chat_template_kwargs"]

    manager = CassazioneVastManager(
        model=cfg["vllm_model"],
        num_gpus=cfg["num_gpus"],
        max_model_len=SERVE_MAX_LEN,
        extra_vllm_args=cfg["extra_vllm_args"],
        vllm_image=cfg["vllm_image"],
    )
    with VastCostTracker(manager, model_label=f"panel-{tag}", step="labeling"):
        try:
            base_url, bearer = manager.launch()
            os.environ["LABELER_API_KEY"] = bearer

            t0 = time.time()
            # Judge side (181 corpus considerati).
            pl.run("judge", None, base_url, cfg["vllm_model"],
                   limit=None, resume=True, workers=16,
                   out=PANEL_DIR / f"sillogismo_judge.{tag}.jsonl")

            # LLM side, one file per judged model.
            for d in dirs:
                pl.run("llm", d, base_url, cfg["vllm_model"],
                       limit=None, resume=True, workers=16,
                       out=PANEL_DIR / f"{d.name}.sillogismo_labels.{tag}.jsonl")
            print(f"[panel] {tag} done in {(time.time() - t0) / 60:.1f} min")
        finally:
            manager.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labelers", nargs="*", default=list(LABELERS),
                    help="Subset of panel members to run (default: all).")
    args = ap.parse_args()

    PANEL_DIR.mkdir(exist_ok=True)
    dirs = _model_dirs()
    print(f"[panel] model dirs to label: {[d.name for d in dirs]}")

    for tag in args.labelers:
        if tag not in LABELERS:
            raise SystemExit(f"unknown labeller {tag!r}; choose from {list(LABELERS)}")
        _run_labeller(tag, LABELERS[tag], dirs)

    print("[panel] all labellers complete.")


if __name__ == "__main__":
    main()
