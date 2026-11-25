"""
Run a model as an offline judge on the evaluated test cases.

For each test case the model receives the court header and the RITENUTO IN
FATTO section, with k=2 article-matched few-shot exemplars drawn from the
held-out pool, and is asked to produce BOTH the "CONSIDERATO IN DIRITTO"
reasoning and the "P.Q.M." dispositive.

Protocol (see pipeline/case_sections.py):
  * Test judgments whose facts cannot be separated from the reasoning are set
    aside: 91 of the 100 test judgments are evaluated.
  * Exemplars without a separable reasoning section are dropped from the pool
    (79 of 81 remain), and a companion judgment of the same proceedings is
    never offered as an exemplar for its test case.
  * Every prompt of every pass is checked before the model is loaded for
    passages of the test court's own reasoning; a failed check aborts the run.
  * Five generations per case (seeds 42-46) with --n-samples 5.

Per-case output schema (inference_results.jsonl):
  filename, judge_reasoning_trace, considerato_predicted, pqm_predicted,
  pqm_ground_truth, raw_dual, finish_reason, seed, sample_idx,
  prompt_language, gt_articles,
  fewshot_used (list of {filename, gt_articles, fewshot_path, truncated})

The run directory also receives run_manifest.json (the rule applied to every
judgment, the cases set aside, the exemplars dropped and the companion pairs)
and fewshot_used.jsonl.

Usage:
    python -m pipeline.inference_judge --profile gpt-oss-20b --dry-run
    python -m pipeline.inference_judge --profile gpt-oss-20b --n-samples 5
    python -m pipeline.inference_judge --profile gpt-oss-20b --n-samples 5 --k 0   # zero-shot control

Env vars:
    MODEL_PROFILE   Profile key from model_backends.PROFILES (or pass --profile)
    VLLM_MODEL      HuggingFace model id or local path  (default: from profile)
    VLLM_TP         Tensor parallel size (# GPUs)        (default: from profile)
    VLLM_GPU_MEM    GPU memory utilisation fraction      (default: from profile)
    VLLM_MAX_LEN    Max model context length (tokens)    (default: from profile)
    MAX_NEW_TOKENS  Max tokens to generate               (default: from profile)
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pipeline.case_sections import (
    COMPANION_THRESHOLD,
    PromptCheckFailed,
    check_prompts,
    prepare_cases,
)
from pipeline.cassazione_split import INPUT_DIR, load_split
from pipeline.fewshot_selector import pick_exemplars
from pipeline.model_backends import (
    PROFILES,
    _get_tokenizer,
    build_developer_instructions,
    get_backend,
    load_done,
    truncate_exemplar,
)

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_FILE = ROOT / "articoli.json"

# Per-call budget reserved for everything that isn't an exemplar (test-case
# input + instructions + headers + safety slack). Used to compute the
# per-exemplar token budget when truncating few-shots.
_NON_EXEMPLAR_BUDGET = 4000


# ── Few-shot wiring ────────────────────────────────────────────────────────────

def _per_exemplar_budget(profile: dict, k: int) -> int:
    """Token budget for each exemplar inside the developer instructions."""
    if k <= 0:
        return 0
    budget = profile["max_len"] - profile["max_new_tokens"] - _NON_EXEMPLAR_BUDGET
    return max(500, budget // k)


def _exemplar_log_entry(ex: dict) -> dict:
    return {
        "filename": ex["filename"],
        "gt_articles": ex.get("gt_articles", []),
        "fewshot_path": ex.get("fewshot_path", ""),
        "truncated": ex.get("truncated", False),
    }


def _build_per_case_prompt_data(
    test_case: dict,
    pool: list,
    companions: dict,
    articles: dict,
    profile: dict,
    language: str,
    rng: np.random.Generator,
    k: int = 2,
) -> tuple[str, list]:
    """Return (developer_instructions_text, fewshot_used_records).

    Companion judgments of the test case are removed from the candidates
    before the article-matched selection.
    """
    banned = companions.get(test_case["filename"], {})
    candidates = [p for p in pool if p["filename"] not in banned] if banned else pool
    raw_exemplars = pick_exemplars(test_case, pool=candidates, k=k, rng=rng)
    per_budget = _per_exemplar_budget(profile, k)
    trimmed = [truncate_exemplar(ex, max_tokens=per_budget, profile=profile) for ex in raw_exemplars]
    instructions = build_developer_instructions(articles, language=language, exemplars=trimmed)
    return instructions, [_exemplar_log_entry(ex) for ex in trimmed]


# ── Local generation: one engine for all passes ────────────────────────────────

_ENGINE: dict = {}


def _run_local(sentenze, dev_instructions_per_case, backend, profile, max_new_tok, seed):
    """In-process vLLM. Returns a list of parsed dicts in input order.

    * Harmony models (GPT-OSS): token-id prompts from openai_harmony, so that
      reasoning effort HIGH is kept in the system content.
    * Every other model: the system/user conversation through llm.chat(), i.e.
      the model's own chat template, with the profile's chat_template_kwargs.

    The engine is built once and reused across seeds.
    """
    # FlashInfer's sampling-kernel JIT rejects some consumer GPUs (SM120); vLLM's
    # PyTorch sampler draws from the same distribution.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    model_path = os.environ.get("VLLM_MODEL", profile["vllm_model"])
    tp = int(os.environ.get("VLLM_TP", str(profile["tp"])))
    gpu_mem = float(os.environ.get("VLLM_GPU_MEM", str(profile["gpu_mem"])))
    max_len = int(os.environ.get("VLLM_MAX_LEN", str(profile["max_len"])))
    extra = {k: profile[k] for k in ("kv_cache_dtype", "max_num_seqs", "tokenizer_mode") if profile.get(k)}
    if profile.get("enforce_eager"):
        extra["enforce_eager"] = True
    if profile.get("limit_mm_per_prompt"):
        extra["limit_mm_per_prompt"] = profile["limit_mm_per_prompt"]
    key = (model_path, tp, gpu_mem, max_len, json.dumps(extra, sort_keys=True))
    if key not in _ENGINE:
        _ENGINE.clear()
        _ENGINE[key] = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=gpu_mem,
                           max_model_len=max_len, tensor_parallel_size=tp, **extra)
    llm = _ENGINE[key]

    if profile["format"] == "harmony":
        stop_ids = backend.stop_token_ids()
        sampling = SamplingParams(max_tokens=max_new_tok, temperature=1.0, seed=seed,
                                  **({"stop_token_ids": stop_ids} if stop_ids else {}))
        prompts = [{"prompt_token_ids": backend.build_inference_prompt(s["input"], dev_instructions=dev)}
                   for s, dev in zip(sentenze, dev_instructions_per_case)]
        print(f"Running inference (seed {seed}, {len(prompts)} token-id prompts)…")
        outputs = llm.generate(prompts, sampling_params=sampling)
    else:
        conversations = [[{"role": "system", "content": dev}, {"role": "user", "content": s["input"]}]
                         for s, dev in zip(sentenze, dev_instructions_per_case)]
        sampling = [SamplingParams(temperature=1.0, max_tokens=max_new_tok, seed=seed) for _ in conversations]
        chat_kwargs = ({"chat_template_kwargs": profile["chat_template_kwargs"]}
                       if profile.get("chat_template_kwargs") else {})
        print(f"Running inference (seed {seed}, {len(conversations)} chat conversations)…")
        outputs = llm.chat(conversations, sampling, use_tqdm=True, **chat_kwargs)
    return [backend.parse_output(o) for o in outputs]


# ── Run manifest ───────────────────────────────────────────────────────────────

def _git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT, check=False).stdout.strip()
    except OSError:
        return None
    return out or None


def _print_report(prep: dict) -> None:
    report = prep["report"]
    rules = report["test_rules"]
    print("── Test cases ──")
    for rule in ("plain-heading", "decorated-heading", "no-separable-facts"):
        print(f"  {rule:26s} {sum(r == rule for r in rules.values()):3d}")
    for f in report["excluded_test"]:
        print(f"    set aside: {f}")
    print("── Few-shot pool ──")
    pool_rules = report["pool_rules"]
    for rule in sorted(set(pool_rules.values())):
        print(f"  {rule:26s} {sum(r == rule for r in pool_rules.values()):3d}")
    for f in report["dropped_pool"]:
        print(f"    dropped: {f}  ({pool_rules[f]})")
    n_pairs = sum(len(v) for v in prep["companions"].values())
    print(f"── Companion judgments kept out of the exemplars: {n_pairs} pairs over "
          f"{len(prep['companions'])} test cases")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile",     default=os.environ.get("MODEL_PROFILE"),
                        help="Profile key from pipeline/model_backends.PROFILES")
    parser.add_argument("--limit",       type=int,  default=0,    help="First N evaluated cases only (smoke tests)")
    parser.add_argument("--resume",      action="store_true",      help="Continue an interrupted run in the same directory")
    parser.add_argument("--results-dir", type=Path, default=None,  help="Output directory (default: results/<profile>/)")
    parser.add_argument("--seed",        type=int,  default=42,    help="Base sampling seed")
    parser.add_argument("--n-samples",   type=int,  default=1,     help="Generations per case (seeds = seed, seed+1, …, seed+n-1)")
    parser.add_argument("--skip-existing-seeds", action="store_true",
                        help="Drop passes whose seed already has records in the output file. "
                             "sample_idx keeps its canonical position, so the (seed, sample_idx) "
                             "exemplar RNG matches a from-scratch run.")
    parser.add_argument("--prompt-language", choices=["it", "en"], default="it",
                        help="System-prompt language. 'it' is the reported setting; 'en' is a parity translation.")
    parser.add_argument("--k",           type=int,  default=2,
                        help="Few-shot exemplars per case (default 2; 0 = zero-shot control)")
    parser.add_argument("--companion-threshold", type=float, default=COMPANION_THRESHOLD)
    parser.add_argument("--dry-run",     action="store_true",
                        help="Build and check every prompt; load no model and write nothing")
    args = parser.parse_args()

    if args.profile:
        if args.profile not in PROFILES:
            sys.exit(f"unknown profile {args.profile!r}; choose from {sorted(PROFILES)}")
        os.environ["MODEL_PROFILE"] = args.profile
    if not ARTICLES_FILE.exists():
        sys.exit(f"articoli.json not found at {ARTICLES_FILE}")
    if not INPUT_DIR.exists():
        sys.exit(f"Input directory not found: {INPUT_DIR} (see README: fetch the corpus first)")

    articles = json.loads(ARTICLES_FILE.read_text())
    prep = prepare_cases(args.companion_threshold)
    _print_report(prep)

    default_dev = build_developer_instructions(articles, language=args.prompt_language)
    backend, profile, label = get_backend(developer_instructions=default_dev)

    # Exemplars are truncated to a token budget with the model's own tokenizer;
    # without it the budget falls back to a character heuristic and the prompts
    # differ from the reported runs.
    tokenizer_ok = _get_tokenizer(profile) is not None
    print(f"Exemplar truncation: {'model tokenizer' if tokenizer_ok else 'CHARACTER HEURISTIC (tokenizer failed to load)'}")
    if not tokenizer_ok and not args.dry_run and args.k > 0:
        sys.exit("The model tokenizer did not load, so exemplar truncation would not match the "
                 "reported protocol; aborting.")

    results_dir = (args.results_dir or ROOT / "results" / label).resolve()
    output_file = results_dir / "inference_results.jsonl"
    fewshot_log = results_dir / "fewshot_used.jsonl"
    if not args.dry_run and output_file.exists() and not args.resume and not args.skip_existing_seeds:
        sys.exit(f"{results_dir} already holds outputs; pass --resume to continue, or pick a new --results-dir")

    max_new_tok = int(os.environ.get("MAX_NEW_TOKENS", str(profile["max_new_tokens"])))

    excluded = set(prep["report"]["excluded_test"])
    test_cases = [c for c in load_split("test") if c["filename"] not in excluded]
    pool = prep["pool"]
    by_filename = {c["filename"]: c for c in test_cases}

    done = load_done(output_file) if (args.resume and not args.dry_run) else set()
    files = [c["filename"] for c in test_cases if c["filename"] not in done]
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("No files to process.")
        return

    seeds = [args.seed + i for i in range(max(1, args.n_samples))]

    print(f"Model profile : {label}  ({profile['vllm_model']})")
    print(f"Prompt lang   : {args.prompt_language}")
    print(f"Few-shot      : {'none (zero-shot control)' if args.k == 0 else f'k={args.k}, article-matched (pool={len(pool)})'}")
    print(f"Results dir   : {results_dir}")
    print(f"Processing    : {len(files)} test cases × {len(seeds)} seed(s) → {output_file}")

    sentenze: list[dict] = []
    for fname in files:
        parsed = dict(prep["parsed"][fname])
        parsed["filename"] = fname
        parsed["gt_articles"] = by_filename[fname]["gt_articles"]
        parsed["stratum"] = by_filename[fname]["stratum"]
        sentenze.append(parsed)

    # Build prompt data for every pass up front. Exemplar selection is
    # deterministic per pass (rng seeded with (seed, sample_idx)).
    passes: list[tuple[int, int, list[str], list[list[dict]]]] = []
    for sample_idx, seed in enumerate(seeds):
        rng = np.random.default_rng((seed, sample_idx))
        dev_instructions_per_case: list[str] = []
        fewshot_per_case: list[list[dict]] = []
        for sentenza in tqdm(sentenze, desc=f"building prompts (seed={seed})"):
            dev, fs_log = _build_per_case_prompt_data(
                test_case=by_filename[sentenza["filename"]],
                pool=pool,
                companions=prep["companions"],
                articles=articles,
                profile=profile,
                language=args.prompt_language,
                rng=rng,
                k=args.k,
            )
            dev_instructions_per_case.append(dev)
            fewshot_per_case.append(fs_log)
        passes.append((sample_idx, seed, dev_instructions_per_case, fewshot_per_case))

    # Check every prompt of every pass before any model is loaded.
    total_fail = 0
    for sample_idx, seed, dev_list, _fs in passes:
        failures, warnings = check_prompts(sentenze, dev_list, prep["passages"])
        total_fail += len(failures)
        print(f"[prompt-check] seed {seed}: {len(sentenze)} prompts, {len(failures)} failures, "
              f"{len(warnings)} single-passage warnings")
        for line in failures + warnings:
            print(f"    {line}")
    if total_fail:
        raise PromptCheckFailed("[prompt-check] aborting before generation")

    if args.dry_run:
        print(f"DRY RUN PASSED: {len(sentenze)} cases × {len(seeds)} seeds. Nothing was written.")
        return

    if args.skip_existing_seeds and output_file.exists():
        existing_seeds = set()
        for line in output_file.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line:
                try:
                    existing_seeds.add(json.loads(line).get("seed", 42))
                except json.JSONDecodeError:
                    continue
        skipped = [s for _, s, _, _ in passes if s in existing_seeds]
        passes = [p for p in passes if p[1] not in existing_seeds]
        if skipped:
            print(f"--skip-existing-seeds: seeds {sorted(skipped)} already present, "
                  f"running {[p[1] for p in passes] or 'nothing'}")
        if not passes:
            print("All requested seeds already present — nothing to do.")
            return

    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git_head(),
        "profile": label, "n_samples": args.n_samples, "seed": args.seed, "k": args.k,
        "prompt_language": args.prompt_language,
        "companion_threshold": args.companion_threshold,
        **prep["report"],
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")

    out_fh = output_file.open("a")
    fs_fh = fewshot_log.open("a")

    def _write_pass(parsed_outputs, sample_idx, seed, fewshot_per_case):
        for parsed_out, sentenza, fs_log in zip(parsed_outputs, sentenze, fewshot_per_case):
            record = {
                "filename":              sentenza["filename"],
                "judge_reasoning_trace": parsed_out["judge_reasoning_trace"],
                "considerato_predicted": parsed_out["considerato_predicted"],
                "pqm_predicted":         parsed_out["pqm_predicted"],
                "pqm_ground_truth":      sentenza["pqm_ground_truth"],
                "raw_dual":              parsed_out["raw_dual"],
                "finish_reason":         parsed_out["finish_reason"],
                "seed":                  seed,
                "sample_idx":            sample_idx,
                "prompt_language":       args.prompt_language,
                "gt_articles":           sentenza["gt_articles"],
                "fewshot_used":          fs_log,
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fs_fh.write(json.dumps({
                "filename": sentenza["filename"],
                "seed": seed,
                "sample_idx": sample_idx,
                "exemplars": fs_log,
            }, ensure_ascii=False) + "\n")
        out_fh.flush()
        fs_fh.flush()

    try:
        for sample_idx, seed, dev_instructions_per_case, fewshot_per_case in passes:
            print(f"\n--- pass {sample_idx + 1}/{len(seeds)}  (seed={seed}) ---")
            parsed_outputs = _run_local(
                sentenze, dev_instructions_per_case, backend, profile, max_new_tok, seed
            )
            _write_pass(parsed_outputs, sample_idx, seed, fewshot_per_case)
    finally:
        out_fh.close()
        fs_fh.close()

    print(f"\nDone. {len(passes)} pass(es) × {len(sentenze)} cases written to {output_file}")


if __name__ == "__main__":
    main()
