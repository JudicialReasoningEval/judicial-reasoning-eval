"""Frontier-judge validation of the open-weight panel.

Provides the reference-panel figures for the premise labels and for M5, by
re-running the panel's OWN prompts against three frontier judges reached over
OpenRouter, on a stratified subset of the test split:

    A (premise) : the court's considerato
                  -> results/frontier_labels/sillogismo_judge.<tag>.jsonl
    C (M5)      : each evaluated model's considerato_predicted
                  -> results/frontier_m5/<model>.m5.<tag>.jsonl

Methodological parity is the point: this driver imports the prompts, sentence
splitter and chunking from metrics.premise_labeler / metrics.m5_faithfulness
unchanged, so the only difference between panel and frontier labels is the model
behind the endpoint. In particular CHUNK_SIZE stays at the panel's 35 even
though frontier context windows are ~1M — larger chunks would be cheaper but
would confound the comparison with context granularity.

Cost control (this runs against a hard OpenRouter budget ceiling):
  * every call is metered from the response's own `usage` and appended to
    results/_labeling_costs.jsonl via cost_tracking.labeling_cost_tracker;
  * --budget sets a ceiling on THIS run's spend, checked after every call; the
    run aborts (resume-safe) as soon as it is crossed;
  * --dry-run prices the whole plan from measured per-unit costs and spends nothing;
  * --reasoning-max-tokens bounds thinking tokens, which measurement showed to
    be the dominant cost term (one model emitted 6.2k reasoning tokens on a
    single 34-sentence labelling chunk, eight times claude-sonnet-5's 815).

Usage:
    python -m pipeline.run_frontier_validation --dry-run
    python -m pipeline.run_frontier_validation --task subset
    python -m pipeline.run_frontier_validation --task A --budget <USD>
    python -m pipeline.run_frontier_validation --task C --models qwen3.5-397b-a17b,... --budget <USD>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

import metrics.m5_faithfulness as m5  # noqa: E402
import metrics.premise_labeler as pl  # noqa: E402
from cost_tracking.labeling_cost_tracker import (  # noqa: E402
    cumulative_spend,
    log_labeler_call,
)

RESULTS = ROOT / "results"
FRONTIER_LABELS = RESULTS / "frontier_labels"
FRONTIER_M5 = RESULTS / "frontier_m5"
SPLITS = ROOT / "splits"
LEDGER = RESULTS / "_labeling_costs.jsonl"
FRONTIER_M5_MAX_TOKENS = 8192

BASE_URL = "https://openrouter.ai/api/v1"

# ── The frontier reference panel ─────────────────────────────────────────────
# Three judges so the reference standard is itself a strict-majority panel,
# matching the open-weight panel's 2-of-3 rule.
#
# Reasoning stays ENABLED for all three, but the knob that
# actually bounds it differs per provider AND per task. All of the below is
# measured on one real case, not assumed:
#   * Anthropic honours `reasoning.max_tokens` as a true thinking budget on the
#     premise task (589 thinking tokens, $0.0378/chunk). `effort: "low"` instead
#     returns ZERO thinking tokens there, i.e. reasoning off.
#   * Gemini largely IGNORES `reasoning.max_tokens` (4,877 -> 4,437, only 9% off)
#     so it needs `effort`; but on M5, `effort: "low"` drops it to zero thinking
#     tokens, so M5 leaves it uncapped.
#   * On M5 a LOW `max_tokens` cap BACKFIRES for Anthropic: 1,024 yielded 1,567
#     thinking + 3,989 answer tokens = $0.064, vs $0.029 without a reasoning cap.
#     Provider reasoning is therefore uncapped, while an 8,192-token completion
#     ceiling prevents runaway output.
#   * GLM on M5 is cheaper uncapped than at effort:low ($0.0074 vs $0.0114) AND
#     reasons more (2,256 vs 1,820 tokens) — so uncapped wins on both counts.
# Net: task A caps reasoning where that works; task C leaves provider reasoning
# uncapped for all three and uses the shared 8,192-token completion ceiling.
# $/unit for all three judges: A $0.0944/chunk, C $0.0598/(case x model).
JUDGES: dict[str, dict] = {
    "claude-sonnet-5": {
        "model": "anthropic/claude-sonnet-5",
        "reasoning": {"A": {"enabled": True, "max_tokens": 1024},
                      "C": {"enabled": True}},
    },
    "gemini-3.6-flash": {
        "model": "google/gemini-3.6-flash",   # $1.50/$7.50 vs 3.5-flash's $1.50/$9.00
        "reasoning": {"A": {"enabled": True, "effort": "low"},
                      "C": {"enabled": True}},
    },
    "glm-5.2": {
        "model": "z-ai/glm-5.2",
        "reasoning": {"A": {"enabled": True, "effort": "low"},
                      "C": {"enabled": True}},
    },
}

# Coarse strata for the subset. The raw `stratum` field of test_100.json has 19
# distinct article combinations over the 96 test cases whose court side is
# panel-labelled, which cannot be sampled proportionally at n=20; these buckets
# collapse it to 7, keeping the 318<->319 requalification stratum (the
# diagnostically richest) as its own cell.
def coarse_stratum(gt_articles: list[str]) -> str:
    s = set(gt_articles)
    if {"318", "319"} <= s:
        return "318+319"
    if "318" in s:
        return "318"
    if "319" in s:
        return "319"
    if "321" in s and "322" in s:
        return "321+322"
    if "321" in s:
        return "321"
    if "322" in s:
        return "322"
    return "other"


def _panel_labelled() -> set[str]:
    """Cases the open-weight panel actually labelled on the judge side — the
    subset must be drawn from these or there is nothing to agree with."""
    p = RESULTS / "panel_labels" / "sillogismo_judge.gpt-oss-120b.jsonl"
    return {json.loads(l)["filename"]
            for l in p.read_text(encoding="utf-8").split("\n") if l.strip()}


def build_subset(n: int, seed: int = 42) -> dict:
    """Deterministic stratified subset, largest-remainder allocation."""
    import numpy as np

    test = json.loads((SPLITS / "test_100.json").read_text(encoding="utf-8"))
    have = _panel_labelled()
    elig = [c for c in test["cases"] if c["filename"] in have]

    buckets: dict[str, list[dict]] = {}
    for c in elig:
        buckets.setdefault(coarse_stratum(c["gt_articles"]), []).append(c)
    for k in buckets:
        buckets[k].sort(key=lambda c: c["filename"])

    # Largest remainder: floor the proportional shares, then hand the leftover
    # seats to the largest fractional parts.
    total = len(elig)
    exact = {k: len(v) / total * n for k, v in buckets.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    for k in sorted(buckets, key=lambda k: (-(exact[k] - alloc[k]), k)):
        if sum(alloc.values()) >= n:
            break
        if alloc[k] < len(buckets[k]):
            alloc[k] += 1

    rng = np.random.default_rng(seed)
    picked: list[dict] = []
    for k in sorted(buckets):
        pool = buckets[k]
        take = min(alloc[k], len(pool))
        idx = rng.choice(len(pool), size=take, replace=False) if take else []
        for i in sorted(int(x) for x in idx):
            picked.append({**pool[i], "coarse_stratum": k})
    picked.sort(key=lambda c: c["filename"])

    return {
        "split": "frontier_validation",
        "seed": seed,
        "rng": "numpy.random.default_rng",
        "size": len(picked),
        "drawn_from": "splits/test_100.json ∩ panel-labelled judge side",
        "eligible": total,
        "allocation": {k: alloc[k] for k in sorted(alloc)},
        "cases": picked,
    }


def subset_path(n: int) -> Path:
    return SPLITS / f"frontier_validation_{n}.json"


def load_subset(n: int) -> list[dict]:
    p = subset_path(n)
    if not p.exists():
        raise SystemExit(f"{p} missing — run --task subset first")
    return json.loads(p.read_text(encoding="utf-8"))["cases"]


# ── Spend metering ───────────────────────────────────────────────────────────
class BudgetStop(pl.StopRequested):
    """Budget abort.

    Both shared request hooks run outside their retry handlers, and their
    runners catch this cooperative stop at the future boundary. Already paid,
    completed responses can therefore flush before the run exits.
    """


class Meter:
    """Ledger-backed spend guard, shared across every judge and model in a run.

    Prefers OpenRouter's own `usage.cost` (the authoritative charged amount) over
    our static price table, which cannot know about reasoning-token billing or
    provider-side discounts.
    """

    def __init__(self, budget: float, task: str):
        self.budget = budget
        self.task = task
        self.spent = 0.0
        self.calls = 0
        self.reasoning = 0
        self.per_tag: dict[str, float] = {}
        self._lock = threading.Lock()
        self.tag = ""  # set by the driver before each judge's pass

    def sink(self, model: str, filename: str, chunk_idx: int, usage) -> None:
        cost = getattr(usage, "cost", None)
        det = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(det, "reasoning_tokens", None) if det else None
        with self._lock:
            log_labeler_call(
                labeller_model=model,
                model_label=f"frontier-{self.task}-{self.tag}",
                filename=filename,
                source=self.task,
                prompt_tokens=int(usage.prompt_tokens or 0),
                completion_tokens=int(usage.completion_tokens or 0),
                chunk_idx=int(chunk_idx),
                ledger_path=LEDGER,
                budget_ceiling=None,  # this run's own ceiling is enforced here
            )
            if cost is not None:
                self.spent += float(cost)
                self.per_tag[self.tag] = self.per_tag.get(self.tag, 0.0) + float(cost)
            self.calls += 1
            if reasoning:
                self.reasoning += int(reasoning)
            # The response is already paid for. Let the caller parse and persist
            # it; BEFORE_CALL blocks queued work after this total reaches budget.

    def check(self) -> None:
        """Pre-call circuit breaker. Needed because premise_labeler.run submits
        EVERY case to its ThreadPoolExecutor up front: raising from the usage
        sink stops the result loop, but the executor's shutdown(wait=True) would
        still drain — and bill — every queued case. Refusing before the request
        is what actually caps spend."""
        with self._lock:
            if self.spent >= self.budget:
                raise BudgetStop(
                    f"[frontier] budget ${self.budget:.2f} reached "
                    f"(${self.spent:.4f} charged, {self.calls} calls) — "
                    f"refusing further requests. Re-run with a raised --budget."
                )

    def report(self) -> None:
        print(f"[frontier] task {self.task}: {self.calls} calls, ${self.spent:.4f} charged, "
              f"{self.reasoning} reasoning tokens")
        for t, c in sorted(self.per_tag.items()):
            print(f"    {t:20s} ${c:.4f}")


def _authenticate() -> None:
    """premise_labeler.run() and m5_faithfulness.score_model() each construct
    their OWN OpenAI client from LABELER_API_KEY (defaulting to the literal
    "EMPTY"), so handing this driver the key is not enough — it has to land in
    that variable. Getting this wrong is silent: every
    request 401s, and _call_labeler's fallback then writes a full set of records
    with EMPTY labels, which `resume` afterwards treats as already done.
    """
    key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_KEY not set — export it before a live run")
    os.environ["LABELER_API_KEY"] = key


def _preflight(estimated_cost: float) -> None:
    """Check the KEY's spend cap, not just the account credit balance.

    These are two DIFFERENT limits and the key's is the binding one: an
    OpenRouter key carries its own total-spend cap, so a run can 403 with
    "Key limit exceeded (total limit)" while the account still shows plenty of
    credit — topping up credits does not lift it. In the logs a key-cap 403 is
    indistinguishable from a generic auth failure, so the cap is checked up
    front, for free, before any spend.
    """
    import urllib.request

    key = os.environ["LABELER_API_KEY"]
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/key", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as fh:
            d = json.load(fh)["data"]
    except Exception as e:  # noqa: BLE001 — advisory; never block on the check itself
        print(f"[frontier] preflight: could not read key limits ({e}); continuing")
        return

    limit, used = d.get("limit"), d.get("usage") or 0.0
    remaining = d.get("limit_remaining")
    print(f"[frontier] preflight: key usage ${used:.2f}"
          + (f" of ${limit:.2f} cap, ${remaining:.2f} left" if limit is not None
             else " (no key cap set)"))
    if remaining is not None and remaining < estimated_cost:
        raise SystemExit(
            f"[frontier] ABORT before spending: this run needs about "
            f"${estimated_cost:.2f} but the KEY has only ${remaining:.2f} left of "
            f"its ${limit:.2f} total cap. Account credits are a SEPARATE pool and "
            f"topping them up does NOT lift this. Raise the key's cap in the "
            f"OpenRouter dashboard (Settings -> Keys), then re-run — it resumes."
        )


def _assert_labels_present(path: Path, judge: str) -> None:
    """Guard against the empty-label failure mode above: a pass that produced
    records but essentially no labels is a broken run, not a finding."""
    if not path.exists():
        return
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
    n_sent = sum(len(r["sentences"]) for r in recs)
    n_lab = sum(1 for r in recs for lab in r["premise_labels"] if lab)
    frac = n_lab / n_sent if n_sent else 0.0
    print(f"    integrity: {n_lab}/{n_sent} sentences labelled ({frac:.1%})")
    if n_sent and frac < 0.20:
        raise SystemExit(
            f"[frontier] ABORT: {judge} labelled only {frac:.1%} of sentences in "
            f"{path.name}. That is the API-failure signature, not a result. "
            f"DELETE {path} before retrying — `resume` would otherwise skip these "
            f"cases and bake the empty labels into the agreement numbers."
        )


def _extra_body(tag: str, task: str, reasoning_max_tokens: int | None) -> dict:
    """Per-judge, per-task reasoning config (see JUDGES), with a CLI override.
    `usage.include` asks OpenRouter to return the actual charged cost so the
    meter is exact rather than modelled from a price table."""
    reasoning = dict(JUDGES[tag]["reasoning"][task])
    if reasoning_max_tokens:
        reasoning.pop("effort", None)
        reasoning["max_tokens"] = reasoning_max_tokens
    return {"reasoning": reasoning, "usage": {"include": True}}


# ── Task A: premise labelling of the court's considerato ─────────────────────
def run_task_a(n: int, budget: float, reasoning_max_tokens: int | None,
               workers: int, judges: list[str]) -> None:
    _authenticate()
    _preflight(budget)
    FRONTIER_LABELS.mkdir(parents=True, exist_ok=True)
    cases = load_subset(n)
    keep = {c["filename"] for c in cases}
    # ONE meter for the whole task: a per-judge meter would let the run spend
    # len(judges) x --budget in total.
    meter = Meter(budget, "A")

    # premise_labeler.run() walks the whole corpus; restrict it to the subset.
    orig = pl._judge_cases  # noqa: SLF001
    pl._judge_cases = lambda: [c for c in orig() if c[0] in keep]  # noqa: SLF001
    try:
        for tag in judges:
            model = JUDGES[tag]["model"]
            out = FRONTIER_LABELS / f"sillogismo_judge.{tag}.jsonl"
            meter.tag = tag
            # Point the shared labeller at this judge. REASONING_EFFORT /
            # CHAT_TEMPLATE_KWARGS are vLLM-side knobs and must be off here.
            pl.REASONING_EFFORT = ""
            pl.CHAT_TEMPLATE_KWARGS = ""
            pl.EXTRA_BODY = _extra_body(tag, "A", reasoning_max_tokens)
            pl.USAGE_SINK = meter.sink
            pl.BEFORE_CALL = meter.check
            print(f"[frontier] A {tag} ({model}) reasoning={pl.EXTRA_BODY['reasoning']} "
                  f"-> {out.name}")
            pl.run("judge", None, BASE_URL, model, limit=None, resume=True,
                   workers=workers, out=out)
            _assert_labels_present(out, tag)
    except BudgetStop as e:
        print(f"  {e}")
    finally:
        pl._judge_cases = orig  # noqa: SLF001
        pl.USAGE_SINK = None
        pl.BEFORE_CALL = None
        meter.report()


# ── Task C: M5 groundedness verdicts on each model's output ──────────────────
def _judgeable_m5_cases(results_dir: Path, keep: set[str]) -> set[str]:
    """Seed-42 cases with a non-empty generated considerato."""
    out: set[str] = set()
    for r in m5._read_jsonl(results_dir / "inference_results.jsonl"):  # noqa: SLF001
        fn = r.get("filename")
        if (r.get("sample_idx", 0) == 0 and fn in keep and fn not in out
                and (r.get("considerato_predicted") or "").strip()):
            out.add(fn)
    return out


def _assert_m5_complete(path: Path, expected: set[str], judge: str, model: str) -> None:
    have = {r.get("filename") for r in m5._read_jsonl(path) if r.get("filename")}  # noqa: SLF001
    missing = sorted(expected - have)
    if missing:
        raise SystemExit(
            f"[frontier] ABORT: {model} × {judge} saved {len(have & expected)}/"
            f"{len(expected)} expected M5 verdicts. Missing: {missing}. "
            "Do not advance to another judge/model until the completion setting "
            "is fixed; the run will resume these cases."
        )


def run_task_c(n: int, budget: float, reasoning_max_tokens: int | None,
               workers: int, judges: list[str], models: list[str],
               selected_cases: list[str] | None = None,
               out_dir: Path = FRONTIER_M5,
               max_completion_tokens: int = FRONTIER_M5_MAX_TOKENS) -> None:
    _authenticate()
    _preflight(budget)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_subset(n)
    if selected_cases:
        requested = set(selected_cases)
        available = {c["filename"] for c in cases}
        unknown = sorted(requested - available)
        if unknown:
            raise SystemExit(f"--cases contains filenames outside the n={n} subset: {unknown}")
        cases = [c for c in cases if c["filename"] in requested]
    keep = {c["filename"] for c in cases}
    if not keep:
        raise SystemExit("task C selected zero cases")
    ledger_task = "C" if out_dir.resolve() == FRONTIER_M5.resolve() else "C-smoke"
    meter = Meter(budget, ledger_task)  # one meter for the whole judge x model grid

    # score_model() reads every case in the model dir; restrict to the subset by
    # filtering its reader. Resume files must pass through unfiltered.
    orig = m5._read_jsonl  # noqa: SLF001
    orig_max_tokens = m5.MAX_TOKENS
    m5._read_jsonl = lambda p, _o=orig: [  # noqa: SLF001
        r for r in _o(p)
        if p.name != "inference_results.jsonl" or r.get("filename") in keep
    ]
    m5.MAX_TOKENS = max_completion_tokens
    m5.USAGE_SINK = meter.sink
    m5.BEFORE_CALL = meter.check
    # MODEL is the outer loop and judge the inner one, deliberately. The M5
    # consensus needs a strict 2-of-3 majority, so a model judged by only one or
    # two of the three contributes NOTHING to the agreement analysis. Iterating
    # judge-outer means a budget stop mid-way leaves every model partially
    # judged and the whole run unusable; model-outer means it leaves a prefix of
    # COMPLETE models plus one partial, which is immediately analysable. With an
    # uncertain cost estimate this is what makes exhaustion degrade gracefully.
    try:
        for name in models:
            rdir = RESULTS / name
            if not (rdir / "inference_results.jsonl").exists():
                print(f"[frontier] SKIP {name} (no inference_results.jsonl)")
                continue
            expected = _judgeable_m5_cases(rdir, keep)
            if not expected:
                print(f"[frontier] SKIP {name} (no judgeable selected cases)")
                continue
            print(f"[frontier] C model {name}")
            for tag in judges:
                jmodel = JUDGES[tag]["model"]
                meter.tag = tag
                m5.REASONING_EFFORT = ""
                m5.CHAT_TEMPLATE_KWARGS = ""
                m5.EXTRA_BODY = _extra_body(tag, "C", reasoning_max_tokens)
                out = out_dir / f"{name}.m5.{tag}.jsonl"
                print(f"  {tag} reasoning={m5.EXTRA_BODY['reasoning']} "
                      f"max_tokens={m5.MAX_TOKENS} -> {out.name}")
                m5.score_model(rdir, base_url=BASE_URL, model=jmodel,
                               workers=workers, resume=True, out=out)
                _assert_m5_complete(out, expected, tag, name)
    except BudgetStop as e:
        print(f"  {e}")
    finally:
        m5._read_jsonl = orig  # noqa: SLF001
        m5.MAX_TOKENS = orig_max_tokens
        m5.USAGE_SINK = None
        m5.BEFORE_CALL = None
        meter.report()


# ── Dry-run pricing ──────────────────────────────────────────────────────────
def dry_run(n: int, models: list[str], reasoning_max_tokens: int | None,
            measured: dict) -> None:
    """Price the plan from measured per-unit costs, not from guesses.
    `measured` carries $/unit for A (per chunk) and C (per case-model).
    """
    cases = load_subset(n) if subset_path(n).exists() else build_subset(n)["cases"]
    keep = {c["filename"] for c in cases}

    chunks = 0
    for fn in keep:
        sents = pl.split_sentences(pl.extract_considerato(fn))
        chunks += max(1, -(-len(sents) // pl.CHUNK_SIZE))
    a_cost = chunks * measured["A_per_chunk"]

    units = 0
    for name in models:
        f = RESULTS / name / "inference_results.jsonl"
        if not f.exists():
            continue
        seen = set()
        for line in f.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("sample_idx", 0) != 0 or r["filename"] in seen:
                continue
            if r["filename"] in keep and (r.get("considerato_predicted") or "").strip():
                seen.add(r["filename"])
                units += 1
    c_cost = units * measured["C_per_unit"]

    print(f"[dry-run] n={len(keep)} cases, reasoning_max_tokens={reasoning_max_tokens}")
    print(f"  A  {chunks:4d} chunks x ${measured['A_per_chunk']:.4f} = ${a_cost:6.2f}")
    print(f"  C  {units:4d} units  x ${measured['C_per_unit']:.4f} = ${c_cost:6.2f}"
          f"   ({len(models)} models)")
    print(f"  TOTAL ${a_cost + c_cost:.2f}")
    print(f"  ledger to date: ${cumulative_spend(LEDGER):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["subset", "A", "C"], default=None)
    ap.add_argument("--n", type=int, default=20, help="subset size (default 20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=None,
                    help="hard ceiling on THIS run's charged spend, in USD")
    ap.add_argument("--reasoning-max-tokens", type=int, default=None,
                    help="cap thinking tokens per call (reasoning stays enabled)")
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--models", default=None,
                    help="comma-separated results/<model> dirs for task C")
    ap.add_argument("--cases", default=None,
                    help="comma-separated subset filenames for a targeted task-C smoke run")
    ap.add_argument("--out-dir", type=Path, default=FRONTIER_M5,
                    help="task-C output directory (use a separate directory for smoke runs)")
    ap.add_argument("--m5-max-tokens", type=int, default=FRONTIER_M5_MAX_TOKENS,
                    help="task-C completion ceiling (default 8192; verified and used in production)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--a-per-chunk", type=float, default=0.0,
                    help="measured $/chunk for task A, all judges (dry-run)")
    ap.add_argument("--c-per-unit", type=float, default=0.0,
                    help="measured $/(case x model) for task C, all judges (dry-run)")
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    for j in judges:
        if j not in JUDGES:
            raise SystemExit(f"unknown judge {j!r}; choose from {list(JUDGES)}")

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = sorted(d.name for d in RESULTS.iterdir()
                        if d.is_dir() and (d / "inference_results.jsonl").exists()
                        and (d / "inference_results.jsonl").stat().st_size > 0)
    selected_cases = ([c.strip() for c in args.cases.split(",") if c.strip()]
                      if args.cases else None)

    if args.task == "subset" or (args.task is None and not args.dry_run):
        sub = build_subset(args.n, args.seed)
        p = subset_path(args.n)
        p.write_text(json.dumps(sub, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[frontier] wrote {p} — {sub['size']} cases from {sub['eligible']} eligible")
        print(f"  allocation: {sub['allocation']}")
        for c in sub["cases"]:
            print(f"    {c['coarse_stratum']:8s} {c['filename']}")
        if args.task == "subset":
            return

    if args.dry_run:
        dry_run(args.n, models, args.reasoning_max_tokens,
                {"A_per_chunk": args.a_per_chunk, "C_per_unit": args.c_per_unit})
        return

    if args.budget is None:
        raise SystemExit("--budget is required for a live run (spend ceiling)")

    if args.task == "A":
        run_task_a(args.n, args.budget, args.reasoning_max_tokens, args.workers, judges)
    elif args.task == "C":
        run_task_c(args.n, args.budget, args.reasoning_max_tokens, args.workers,
                   judges, models, selected_cases=selected_cases,
                   out_dir=args.out_dir,
                   max_completion_tokens=args.m5_max_tokens)


if __name__ == "__main__":
    main()
