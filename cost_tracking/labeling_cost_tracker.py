from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PRICING = Path(__file__).resolve().parent / "openrouter_pricing.json"
DEFAULT_LEDGER = Path(__file__).resolve().parent.parent / "results" / "_labeling_costs.jsonl"
DEFAULT_BUDGET_USD = float(os.environ.get("OPENROUTER_BUDGET_CEILING", "50"))

_LOCK = threading.Lock()
_PRICING_CACHE: dict | None = None
_UNKNOWN_MODELS_WARNED: set[str] = set()


class LabelingBudgetExceededError(RuntimeError):
    pass


def _load_pricing(path: Path | str = DEFAULT_PRICING) -> dict:
    global _PRICING_CACHE
    if _PRICING_CACHE is None:
        p = Path(path)
        if not p.exists():
            _PRICING_CACHE = {}
        else:
            with open(p) as f:
                data = json.load(f)
            _PRICING_CACHE = {k: v for k, v in data.items() if not k.startswith("_")}
    return _PRICING_CACHE


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int, pricing: dict) -> float | None:
    entry = pricing.get(model)
    if entry is None:
        if model not in _UNKNOWN_MODELS_WARNED:
            print(f"  [labeling-cost] WARNING: no pricing entry for {model!r}; cost logged as null")
            _UNKNOWN_MODELS_WARNED.add(model)
        return None
    return (prompt_tokens / 1_000_000.0) * entry["input"] + (completion_tokens / 1_000_000.0) * entry["output"]


def cumulative_spend(ledger_path: Path | str = DEFAULT_LEDGER) -> float:
    p = Path(ledger_path)
    if not p.exists():
        return 0.0
    total = 0.0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cost = json.loads(line).get("cost_usd")
                if cost is not None:
                    total += float(cost)
            except json.JSONDecodeError:
                continue
    return total


def log_labeler_call(
    *,
    labeller_model: str,
    model_label: str,
    filename: str,
    source: str,
    prompt_tokens: int,
    completion_tokens: int,
    chunk_idx: int,
    ledger_path: Path | str = DEFAULT_LEDGER,
    pricing_path: Path | str = DEFAULT_PRICING,
    budget_ceiling: float | None = DEFAULT_BUDGET_USD,
) -> dict:
    """
    Append a record of one OpenRouter labeller API call to the labeling-cost ledger.

    Raises LabelingBudgetExceededError if cumulative spend has crossed the ceiling.
    Thread-safe: uses a process-level lock around the file append.
    """
    pricing = _load_pricing(pricing_path)
    cost = _compute_cost(labeller_model, prompt_tokens, completion_tokens, pricing)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "labeller_model": labeller_model,
        "model_label": model_label,
        "filename": filename,
        "source": source,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost,
        "chunk_idx": chunk_idx,
    }

    ledger_p = Path(ledger_path)
    ledger_p.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with open(ledger_p, "a") as f:
            f.write(json.dumps(record) + "\n")

        if budget_ceiling is not None:
            total = cumulative_spend(ledger_p)
            if total >= budget_ceiling:
                raise LabelingBudgetExceededError(
                    f"OpenRouter cumulative spend ${total:.2f} ≥ ceiling ${budget_ceiling:.2f}. "
                    f"Raise OPENROUTER_BUDGET_CEILING or rotate the ledger at {ledger_p}."
                )

    return record
