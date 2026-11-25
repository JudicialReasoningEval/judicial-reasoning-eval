"""
Model backend registry and abstraction layer for the Cassazione inference pipeline.

Defines:
  PROFILES                   — supported model configurations keyed by label
  build_developer_instructions() — shared prompt utility (dual output + few-shot)
  truncate_exemplar()        — per-profile tokenizer-aware truncation of few-shot blocks
  parse_dual_output()        — split a model response into considerato + P.Q.M.
  load_done()                — shared JSONL resume helper
  ModelBackend               — abstract base class
  ThinkTagBackend            — for standard open-weight models (DeepSeek-R1, Qwen, …)
  HarmonyBackend             — for gpt-oss-20b via openai_harmony
  get_backend()              — factory that resolves label → (backend, profile, label)

Profile labels are used as output directory names under results/.
"""

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

# ── Model profiles ─────────────────────────────────────────────────────────────

PROFILES: dict[str, dict] = {
    "gpt-oss-20b": {
        "vllm_model":     "openai/gpt-oss-20b",
        "format":         "harmony",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             1,
        "params_total":   "20B",
        "params_active":  "3.6B",
        "quant":          "mxfp4-native",
        "license":        "Apache-2.0",
        "release_date":   "2025-08",
        "tier":           "small",
        "runs_local":     True,
    },
    "qwen3.5-9b": {
        "vllm_model":     "Qwen/Qwen3.5-9B",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             1,
        "params_total":   "9B",
        "params_active":  "9B",
        "quant":          "bf16",
        "license":        "Apache-2.0",
        "release_date":   "2026-03",
        "tier":           "small",
        "runs_local":     True,
    },
    "minicpm5-1b": {
        "vllm_model":     "openbmb/MiniCPM5-1B",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 16384,
        "gpu_mem":        0.85,
        "tp":             1,
        "params_total":   "1B",
        "params_active":  "1B",
        "quant":          "bf16",
        "license":        "apache-2.0",
        "release_date":   "2026",
        "tier":           "small",
        "runs_local":     True,
        "rationale":      "Tiny 1B model — fast local smoke test of the full pipeline (not a serious judge)",
    },
    "gpt-oss-120b": {
        "vllm_model":     "openai/gpt-oss-120b",
        "format":         "harmony",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             1,
        "params_total":   "117B",
        "params_active":  "5.1B",
        "quant":          "mxfp4-native",
        "license":        "Apache-2.0",
        "release_date":   "2025-08",
        "tier":           "mid",
        "runs_local":     False,
        "rationale":      "Same-architecture scaling pair with gpt-oss-20b (isolates size effect)",
    },
    "nemotron-3-super-120b": {
        "vllm_model":     "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             1,
        "kv_cache_dtype": "fp8",
        "max_num_seqs":   64,
        "params_total":   "120.6B",
        "params_active":  "12.7B",
        "quant":          "nvfp4",
        "license":        "nvidia-open-model-license",
        "release_date":   "2026-03",
        "tier":           "mid",
        "runs_local":     False,
        "rationale":      "NVIDIA's hybrid Mamba-Transformer reasoning model; pairs with gpt-oss-120b for reasoning-vs-general at ~120B",
    },
    "minimax-m2.7-nvfp4": {
        "vllm_model":     "nvidia/MiniMax-M2.7-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             2,
        "params_total":   "230B",
        "params_active":  "16B",
        "quant":          "nvfp4",
        "license":        "modified-MIT (non-commercial)",
        "release_date":   "2026-03",
        "tier":           "large",
        "runs_local":     False,
        "rationale":      "Headline 200B-class target; fits on two 96GB PRO 6000 at NVFP4",
    },
    # "minimax-m2.5-nvfp4": {
    #     "vllm_model":     "nvidia/MiniMax-M2.5-NVFP4",
    #     "format":         "think-tag",
    #     "max_len":        205000,
    #     "max_new_tokens": 65536,
    #     "gpu_mem":        0.85,
    #     "tp":             2,
    #     "params_total":   "230B",
    #     "params_active":  "10B",
    #     "quant":          "NVFP4",
    #     "license":        "modified-MIT (non-commercial)",
    #     "release_date":   "2025-12",
    #     "tier":           "large",
    #     "runs_local":     False,
    #     "rationale":      "Pre-M2.7 baseline (wider NVFP4 tooling); 230B → needs 2x PRO 6000",
    # },
    "deepseek-v4-flash": {
        # NVFP4 checkpoint (168.3 GB) instead of the native FP4+FP8 release: the
        # native one requires DeepGEMM FP8 block-scaled kernels that are SM100-only,
        # while the NVFP4 MoE path (FlashInfer FP4 GEMM) works on SM120 as of
        # vLLM v0.24.0 ("DeepSeek-V4 now enabled on SM120") — hence the per-profile
        # image override below. kv_cache_dtype=fp8 is mandatory (fp8_ds_mla asserts
        # an fp8 KV layout). max_len capped at 131072 (not the model's 1M): weights
        # leave ~15 GiB for KV+indexer on 2×96GB, and Think-Max (needs ≥393216) is
        # not used — thinking=high matches the other reasoning profiles.
        "vllm_model":     "nvidia/DeepSeek-V4-Flash-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.9,
        "tp":             2,
        "kv_cache_dtype": "fp8",
        "enforce_eager":  True,
        "tokenizer_mode": "deepseek_v4",
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
        "vllm_image":     "vastai/vllm:v0.24.0-cuda-12.9",
        # vLLM v0.24.0 bug (github.com/vllm-project/vllm#47436): support_deep_gemm()
        # wrongly reports True on SM120 (RTX PRO 6000 is consumer Blackwell, not the
        # SM100 data-center parts DeepGEMM's SF-layout transform actually supports),
        # so weight loading dies with "Unknown SF transformation". Confirmed
        # workaround: force the Cutlass FP8 block-scaled kernel instead.
        "remote_env":     {"VLLM_USE_DEEP_GEMM": "0"},
        "params_total":   "284B",
        "params_active":  "13B",
        "quant":          "nvfp4",
        "license":        "MIT",
        "release_date":   "2026-04",
        "tier":           "large",
        "runs_local":     False,
        "rationale":      "Large 2026 reasoning model (profile kept; not among the reported models)",
    },
    "qwen3.6-35b-a3b": {
        "vllm_model":     "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             1,
        "kv_cache_dtype": "fp8",
        "vllm_image":     "vastai/vllm:v0.24.0-cuda-12.9",
        "params_total":   "35B",
        "params_active":  "3B",
        "quant":          "nvfp4",
        "license":        "Apache-2.0",
        "release_date":   "2026-04",
        "tier":           "small",
        "runs_local":     False,
        "rationale":      "Small end of the Qwen3.5/3.6 MoE scaling triple with qwen3.5-122b-a10b / qwen3.5-397b-a17b",
    },
    "qwen3.6-27b": {
        # Dense 27B (qwen3_5 architecture, same family as qwen3.5-9b which
        # runs locally). Official ModelOpt NVFP4 (~15 GB) — trivial
        # single-GPU fit. Added primarily as a premise-labelling panel member
        # (3-model consensus with gpt-oss-120b / nemotron-3-super-120b); also
        # usable as a judged model.
        "vllm_model":     "nvidia/Qwen3.6-27B-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             1,
        "kv_cache_dtype": "fp8",
        "vllm_image":     "vastai/vllm:v0.24.0-cuda-12.9",
        "params_total":   "27B",
        "params_active":  "27B",
        "quant":          "nvfp4",
        "license":        "Apache-2.0",
        "release_date":   "2026-04",
        "tier":           "small",
        "runs_local":     False,
        "rationale":      "Dense Qwen for the labelling panel + dense-vs-MoE contrast within the Qwen family",
    },
    "qwen3.5-122b-a10b": {
        # tp=1 OOM'd in practice (92.0/94.97 GiB allocated, only 0.77 GiB free)
        # against an 83.5 GB checkpoint — actual usage during profiling runs
        # *above* the raw file size, not below it (activation memory +
        # dequant scratch), invalidating the lighter tp=1 estimate. tp=2
        # (2x96x0.85=163.2 GiB budget) is comfortably safe.
        "vllm_model":     "nvidia/Qwen3.5-122B-A10B-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             2,
        "kv_cache_dtype": "fp8",
        "vllm_image":     "vastai/vllm:v0.24.0-cuda-12.9",
        "params_total":   "122B",
        "params_active":  "10B",
        "quant":          "nvfp4",
        "license":        "Apache-2.0",
        "release_date":   "2026-02",
        "tier":           "mid",
        "runs_local":     False,
        "rationale":      "Mid point of the Qwen3.5/3.6 MoE scaling triple",
    },
    "qwen3.5-397b-a17b": {
        # 251.2 GB NVFP4 checkpoint (measured via HF tree API). Originally
        # tp=3 (244.8 GiB budget, ~26 GiB "headroom"); bumped to tp=4 after
        # qwen3.5-122b-a10b's tp=1 OOM'd despite a similar-looking margin —
        # real usage during profiling runs *above* the raw file size, not
        # below it. tp=4 (326.4 GiB budget) leaves ~75 GiB of real headroom.
        "vllm_model":     "nvidia/Qwen3.5-397B-A17B-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             4,
        "kv_cache_dtype": "fp8",
        "vllm_image":     "vastai/vllm:v0.24.0-cuda-12.9",
        "params_total":   "397B",
        "params_active":  "17B",
        "quant":          "nvfp4",
        "license":        "Apache-2.0",
        "release_date":   "2026-02",
        "tier":           "large",
        "runs_local":     False,
        "rationale":      "Large end of the Qwen3.5/3.6 MoE scaling triple",
    },
    "nemotron-3-nano-30b-a3b": {
        # Same nemotron_h hybrid Mamba+MoE+Attention architecture as
        # nemotron-3-super-120b, which needed max_num_seqs capped well below
        # vLLM's default 1024 (Mamba SSM cache blocks are a separate, much
        # smaller pool than the KV cache). Preemptively capped here too.
        "vllm_model":     "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             1,
        "kv_cache_dtype": "fp8",
        "max_num_seqs":   64,
        "vllm_image":     "vastai/vllm:v0.24.0-cuda-12.9",
        "params_total":   "30B",
        "params_active":  "3B",
        "quant":          "nvfp4",
        "license":        "nvidia-open-model-license",
        "release_date":   "2025-12",
        "tier":           "small",
        "runs_local":     False,
        "rationale":      "Small end of the Nemotron hybrid-Mamba scaling triple with nemotron-3-super-120b / nemotron-3-ultra-550b",
    },
    "nemotron-3-ultra-550b-a55b": {
        # 352.3 GB NVFP4 checkpoint (measured via HF tree API). tp=5 (408 GiB
        # budget) would be the tightest fit with real headroom, but vast.ai
        # currently has ZERO 5/6/7-GPU RTX PRO 6000 bundles listed at all
        # (checked live: only 1/2/3/4/8 GPU counts exist, and 3 has a single
        # $3.65/hr offer) — tp=4 doesn't clear the memory budget (326.4 GiB
        # < 352.3 GiB checkpoint), so tp=8 is the only available option that
        # both fits in memory and actually has offers on the marketplace.
        # max_num_seqs lowered vs the Super profile (64) since this is ~4.5x
        # bigger — same Mamba-cache risk as nemotron-3-super-120b,
        # with more margin on the registry's largest model.
        "vllm_model":     "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             8,
        "kv_cache_dtype": "fp8",
        "max_num_seqs":   32,
        "vllm_image":     "vastai/vllm:v0.24.0-cuda-12.9",
        "disk_gb":        480,
        "params_total":   "550B",
        "params_active":  "55B",
        "quant":          "nvfp4",
        "license":        "nvidia-open-model-license",
        "release_date":   "2026-06",
        "tier":           "large",
        "runs_local":     False,
        "rationale":      "Large end of the Nemotron hybrid-Mamba scaling triple; largest model in the registry",
    },
    "glm-4.7": {
        # 229.9 GB NVFP4 checkpoint (measured via HF tree API). Originally
        # tp=3 (244.8 GiB budget, ~45 GiB "headroom"); bumped to tp=4 after
        # qwen3.5-122b-a10b's tp=1 OOM'd despite a similar-looking margin —
        # real usage during profiling runs *above* the raw file size, not
        # below it. tp=4 (326.4 GiB budget) leaves ~96 GiB of real headroom.
        # Uses the established glm4_moe architecture (GLM-4.5/4.6 lineage,
        # long-supported in vLLM) — deliberately NOT GLM-4.7-Flash, whose
        # glm4_moe_lite architecture only exists on vLLM/transformers main
        # branch (no released version supports it — the same "needs nightly
        # vLLM" trap deepseek-v4-flash hit on SM120, confirmed via a live
        # vLLM GitHub issue).
        #
        # kv_cache_dtype deliberately left unset (default "auto", not "fp8"):
        # requesting fp8 KV cache crashes glm4_moe's weight loading with
        # TypeError: KVCacheScaleParameter.weight_loader() takes 2
        # positional arguments but 3 were given — vllm/model_executor/
        # models/glm4_moe.py:526 passes a shard_id into a loader that
        # doesn't accept one, a genuine vLLM v0.24.0 bug in this model's
        # KV-cache-scale-parameter loading path, not something fixable from
        # a profile config. Plenty of memory headroom at tp=4 without it.
        "vllm_model":     "nvidia/GLM-4.7-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             4,
        "vllm_image":     "vastai/vllm:v0.24.0-cuda-12.9",
        "params_total":   "355B",
        "params_active":  "32B",
        "quant":          "nvfp4",
        "license":        "MIT",
        "release_date":   "2026-02",
        "tier":           "large",
        "runs_local":     False,
        "rationale":      "New vendor family (Zhipu/Z.ai)",
    },
    "mistral-medium-3.5-128b": {
        # 95.2 GB NVFP4 checkpoint; tp=2 (2x96x0.85=163.2 GiB budget) — tp=1's
        # 81.6 GiB budget doesn't clear the loaded-weight estimate. Dense (not
        # MoE) — new architecture shape for the registry, not just a new
        # vendor. Needs tokenizer_mode=mistral: Mistral ships its own
        # mistral_common encoding pipeline, not a Jinja apply_chat_template
        # (same category of special-casing as DeepSeek V4's tokenizer_mode=
        # deepseek_v4). Its reasoning trace is wrapped in [THINK]...[/THINK]
        # (not <think>...</think>) — _THINK_CLOSE_RE and the no-match fallback
        # in ThinkTagBackend.parse_output were both generalized for this.
        # Also vision-capable (Pixtral encoder) — vLLM's MultiModalBudget
        # unconditionally builds a dummy image input to size the encoder
        # budget at engine-construction time, which crashes with "Mismatch in
        # image token count" unless limit_mm_per_prompt is set explicitly,
        # even though we never send an image (known vLLM issue class, e.g.
        # vllm-project/vllm#8382).
        "vllm_model":     "nvidia/Mistral-Medium-3.5-128B-NVFP4",
        "format":         "think-tag",
        "max_len":        131072,
        "max_new_tokens": 65536,
        "gpu_mem":        0.85,
        "tp":             2,
        "kv_cache_dtype": "fp8",
        "tokenizer_mode": "mistral",
        "chat_template_kwargs": {"reasoning_effort": "high"},
        "limit_mm_per_prompt": {"image": 0},
        "vllm_image":     "vastai/vllm:v0.24.0-cuda-12.9",
        "params_total":   "128B",
        "params_active":  "128B",
        "quant":          "nvfp4",
        "license":        "modified-MIT",
        "release_date":   "2026-04",
        "tier":           "mid",
        "runs_local":     False,
        "rationale":      "New vendor family (Mistral); dense architecture, tests diversity beyond the registry's MoE-heavy lineup",
    },
}

# Appended after a truncated reasoning trace to force a P.Q.M. continuation.
FORCED_PQM_SUFFIX = "\n</think>\n\nP.Q.M.\n\n"

# Short system instruction for importance-scoring forced prompts.
_FORCED_SYSTEM_INSTRUCTION = (
    "Sei un giudice della Corte di Cassazione italiana, Sezione Penale, "
    "specializzato in reati contro la pubblica amministrazione.\n\n"
    "Esamina il ricorso e i fatti così come ritenuti in fatto dai giudici "
    "di merito, e pronuncia il dispositivo della sentenza nella forma "
    "\"P.Q.M.\" (Per Questi Motivi).\n\n"
    "La tua risposta deve iniziare direttamente con \"P.Q.M.\" senza preamboli."
)


# ── Shared utilities ───────────────────────────────────────────────────────────

def _format_articles(articles: dict) -> str:
    parts = [f"Art. {num} — {art['title']}\n{art['text']}" for num, art in articles.items()]
    return "\n\n".join(parts)


_FEWSHOT_HEADER_IT = (
    "Di seguito sono riportati alcuni esempi tratti da sentenze realmente "
    "decise dalla stessa Sezione su casi affini per articoli contestati. "
    "Imita la struttura, il livello di dettaglio e lo stile motivazionale. "
    "Non riprodurre gli esempi letteralmente."
)
_FEWSHOT_HEADER_EN = (
    "Below are excerpts from real decisions of the same Section on cases "
    "involving the same statutory cluster. Imitate the structure, level of "
    "detail and motivational style. Do not reproduce the examples verbatim."
)


def _format_exemplar(idx: int, ex: dict, language: str) -> str:
    if language == "en":
        return (
            f"=== EXAMPLE {idx} ===\n"
            f"RITENUTO IN FATTO + heading:\n{ex['input']}\n\n"
            f"CONSIDERATO IN DIRITTO:\n{ex['considerato']}\n\n"
            f"P.Q.M.:\n{ex['pqm']}\n"
            f"=== END EXAMPLE {idx} ===\n"
        )
    return (
        f"=== ESEMPIO {idx} ===\n"
        f"RITENUTO IN FATTO + intestazione:\n{ex['input']}\n\n"
        f"CONSIDERATO IN DIRITTO:\n{ex['considerato']}\n\n"
        f"P.Q.M.:\n{ex['pqm']}\n"
        f"=== FINE ESEMPIO {idx} ===\n"
    )


def build_developer_instructions(
    articles: dict,
    language: str = "it",
    exemplars: "list[dict] | None" = None,
) -> str:
    """Full developer/system instructions for the inference judge step.

    Prompt:
      - the model must produce BOTH `CONSIDERATO IN DIRITTO` and `P.Q.M.`.
      - optional few-shot exemplars are interpolated between the articles
        block and the closing instruction. Each exemplar must already be
        truncated to fit the per-profile budget (see `truncate_exemplar`).

    Language parity: the Italian arm is canonical, the English arm is a parity
    translation.
    """
    articles_block = _format_articles(articles)
    exemplars = exemplars or []

    if language == "en":
        intro = (
            "You are a judge of the Italian Supreme Court of Cassation (Corte di "
            "Cassazione), Criminal Section, specialized in crimes against the "
            "public administration.\n\n"
            "Examine the appeal and the facts as established by the lower-court "
            "judges, and produce the full motivation of the judgment.\n\n"
            "Your response must contain EXACTLY two sections, each introduced by "
            "the literal heading shown below:\n"
            "  CONSIDERATO IN DIRITTO:  (full legal reasoning — articles cited, "
            "doctrine, subsumption of facts under norms, treatment of objections)\n"
            "  P.Q.M.:                 (the operative part, including rulings on "
            "procedural costs and any accessory rulings)\n\n"
            "Do not include any other section, preamble or commentary outside "
            "these two headings.\n\n"
            "These are the criminal-code articles to consider:\n\n"
            f"{articles_block}\n"
        )
    else:
        intro = (
            "Sei un giudice della Corte di Cassazione italiana, Sezione Penale, "
            "specializzato in reati contro la pubblica amministrazione.\n\n"
            "Esamina il ricorso e i fatti così come ritenuti in fatto dai giudici "
            "di merito, e produci la motivazione completa della sentenza.\n\n"
            "La risposta deve contenere ESATTAMENTE due sezioni, ciascuna "
            "intestata letteralmente come segue:\n"
            "  CONSIDERATO IN DIRITTO:  (motivazione giuridica completa — "
            "articoli citati, dottrina, sussunzione dei fatti alle norme, "
            "trattazione delle eccezioni)\n"
            "  P.Q.M.:                 (il dispositivo, comprensivo di tutte le "
            "statuizioni sulle spese processuali e ogni altra statuizione "
            "accessoria)\n\n"
            "Non includere altre sezioni, preamboli o commenti al di fuori di "
            "queste due intestazioni.\n\n"
            "Questi sono gli articoli del codice penale da considerare:\n\n"
            f"{articles_block}\n"
        )

    if not exemplars:
        return intro

    header = _FEWSHOT_HEADER_EN if language == "en" else _FEWSHOT_HEADER_IT
    blocks = [
        _format_exemplar(i + 1, ex, language) for i, ex in enumerate(exemplars)
    ]
    closing = (
        "\nProduce ora la motivazione per il caso che ti sarà fornito, "
        "rispettando le due intestazioni indicate sopra.\n"
        if language == "it"
        else "\nNow produce the motivation for the case that follows, "
        "respecting the two headings indicated above.\n"
    )
    return intro + "\n" + header + "\n\n" + "\n".join(blocks) + closing


# ── Few-shot exemplar truncation ───────────────────────────────────────────────

_TOKENIZER_CACHE: dict[str, "object"] = {}


def _get_tokenizer(profile: dict):
    """Cached HF tokenizer loader. Returns None if transformers is missing or
    the model is gated and we can't load it locally — callers should fall back
    to a character-count heuristic."""
    model_id = profile["vllm_model"]
    if model_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_id]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        tok = None
    _TOKENIZER_CACHE[model_id] = tok
    return tok


def _count_tokens(text: str, profile: dict) -> int:
    tok = _get_tokenizer(profile)
    if tok is None:
        # Italian heuristic: ~3.6 chars per token for Latin script + legal vocab.
        return max(1, len(text) // 4)
    return len(tok.encode(text, add_special_tokens=False))


def truncate_exemplar(
    exemplar: dict,
    max_tokens: int,
    profile: dict,
) -> dict:
    """Return an exemplar trimmed to fit `max_tokens` for the given profile.

    Strategy: if the full exemplar fits, return it unchanged. Otherwise
    truncate the longest of `input` / `considerato` (P.Q.M. is short, always
    kept whole) by character-length, keeping the head and appending a
    truncation marker. A `truncated` flag is set on the returned dict so
    callers can log it.
    """
    parts = {
        "input": exemplar.get("input", ""),
        "considerato": exemplar.get("considerato", ""),
        "pqm": exemplar.get("pqm", ""),
    }
    total_tokens = _count_tokens(parts["input"] + parts["considerato"] + parts["pqm"], profile)
    if total_tokens <= max_tokens:
        return dict(exemplar, truncated=False)

    # The P.Q.M. is short — keep it whole. Split the budget between input
    # and considerato proportionally to their current sizes, but with a
    # floor so each is at least 500 tokens.
    pqm_tokens = _count_tokens(parts["pqm"], profile)
    remaining = max_tokens - pqm_tokens
    if remaining < 1000:
        # Tiny budget — keep only the considerato head + the P.Q.M.
        cons = _truncate_to_tokens(parts["considerato"], max(200, remaining - 100), profile)
        return dict(
            exemplar,
            input="[RITENUTO IN FATTO omesso per ragioni di lunghezza]",
            considerato=cons + "\n[…]",
            pqm=parts["pqm"],
            truncated=True,
        )

    input_tokens = _count_tokens(parts["input"], profile)
    cons_tokens = _count_tokens(parts["considerato"], profile)
    if input_tokens + cons_tokens == 0:
        return dict(exemplar, truncated=False)
    input_share = min(int(remaining * input_tokens / (input_tokens + cons_tokens)), input_tokens)
    cons_share = remaining - input_share

    new_input = _truncate_to_tokens(parts["input"], input_share, profile)
    new_cons = _truncate_to_tokens(parts["considerato"], cons_share, profile)
    if new_input != parts["input"]:
        new_input += "\n[…]"
    if new_cons != parts["considerato"]:
        new_cons += "\n[…]"
    return dict(
        exemplar,
        input=new_input,
        considerato=new_cons,
        pqm=parts["pqm"],
        truncated=True,
    )


def _truncate_to_tokens(text: str, max_tokens: int, profile: dict) -> str:
    if max_tokens <= 0:
        return ""
    tok = _get_tokenizer(profile)
    if tok is None:
        # Character heuristic; trim at last paragraph break inside the limit.
        cap = max_tokens * 4
        if len(text) <= cap:
            return text
        cut = text.rfind("\n\n", 0, cap)
        return text[: cut if cut > 0 else cap]
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)


# ── Dual-output response parser ────────────────────────────────────────────────

_CONSIDERATO_HEADING_RE = re.compile(
    r"\b(?:CONSIDERATO\s+IN\s+DIRITTO|CONSIDERATO IN DIRITTO)\s*[:\-]?\s*",
    re.IGNORECASE,
)
_PQM_HEADING_RE = re.compile(
    r"\b(?:P\.\s*Q\.\s*M\.?|PQM|PER\s+QUESTI\s+MOTIVI)\s*[:\-]?\s*",
    re.IGNORECASE,
)


def parse_dual_output(text: str) -> dict:
    """Split a model response into considerato_predicted and pqm_predicted.

    Tolerates `P.Q.M.` / `PQM` / `PER QUESTI MOTIVI`, optional colons,
    and a leading `CONSIDERATO IN DIRITTO:` heading. If no `P.Q.M.`
    heading is found we put everything in considerato_predicted and leave
    pqm_predicted empty (the regex-based article extractor still hits on
    citations inside the considerato).

    Returns `{considerato_predicted, pqm_predicted, raw_dual}` where
    `raw_dual` is True iff both headings were detected.
    """
    if not text:
        return {"considerato_predicted": "", "pqm_predicted": "", "raw_dual": False}

    cm = _CONSIDERATO_HEADING_RE.search(text)
    pm = _PQM_HEADING_RE.search(text)

    if cm and pm and pm.start() > cm.start():
        considerato = text[cm.end(): pm.start()].strip()
        pqm = text[pm.start():].strip()
        return {"considerato_predicted": considerato, "pqm_predicted": pqm, "raw_dual": True}

    if pm and not cm:
        # No considerato heading — everything before the P.Q.M. is the reasoning.
        considerato = text[: pm.start()].strip()
        pqm = text[pm.start():].strip()
        return {"considerato_predicted": considerato, "pqm_predicted": pqm, "raw_dual": False}

    if cm and not pm:
        considerato = text[cm.end():].strip()
        return {"considerato_predicted": considerato, "pqm_predicted": "", "raw_dual": False}

    # No headings at all: assume the whole response is the considerato.
    return {"considerato_predicted": text.strip(), "pqm_predicted": "", "raw_dual": False}


def load_done(path: Path) -> set[str]:
    """Return the set of filenames already written to a JSONL output file."""
    import json
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            done.add(json.loads(line.strip())["filename"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return done


# ── Backend abstract base class ────────────────────────────────────────────────

class ModelBackend(ABC):
    """Interface shared by all model backends in the pipeline."""

    @abstractmethod
    def build_inference_prompt(
        self,
        user_message: str,
        dev_instructions: "str | None" = None,
    ) -> "str | list[int]":
        """
        Build a prompt for the judge inference step.
        Returns a string (think-tag) or a list of token IDs (Harmony).

        When `dev_instructions` is supplied, it overrides the cached
        developer/system instruction for this call only — used by the
        few-shot path, which builds a different prompt per case.
        """

    @abstractmethod
    def parse_output(self, output) -> dict:
        """
        Parse a vLLM RequestOutput into:
          {"judge_reasoning_trace": str,
           "considerato_predicted": str,
           "pqm_predicted":         str,
           "raw_dual":              bool,   # True iff both CONSIDERATO and P.Q.M. headings were found
           "finish_reason":         str}
        """

    @abstractmethod
    def build_forced_prompt(
        self, case_input: str, trace_prefix: str, articles_block: str
    ) -> "str | list[int]":
        """
        Build a forced-P.Q.M. prompt for importance scoring.
        articles_block is the formatted penal-code text (may be "" for Harmony
        since developer_instructions already embeds it).
        """

    def extract_logprob_dict(self, output) -> dict[int, float]:
        """Return {token_id: logprob} for the first generated token (vLLM shared)."""
        token_logprobs = output.outputs[0].logprobs
        if not token_logprobs:
            return {}
        return {tid: lp.logprob for tid, lp in token_logprobs[0].items()}

    @abstractmethod
    def stop_token_ids(self) -> "list[int] | None":
        """Token IDs at which generation stops, or None to use the model default."""


# ── Think-tag backend ──────────────────────────────────────────────────────────

# Mistral's reasoning parser wraps the trace in [THINK]...[/THINK] instead of
# the <think>...</think> convention every other profile uses.
_THINK_CLOSE_RE = re.compile(r"</think>|\[/THINK\]", re.IGNORECASE)


class ThinkTagBackend(ModelBackend):
    """
    Backend for standard open-weight reasoning models (DeepSeek-R1, Qwen, Llama, …).
    Prompts use <|system|> / <|user|> / <|assistant|> roles and <think>…</think>
    for the chain-of-thought trace.
    """

    def __init__(self, developer_instructions: str = "") -> None:
        self._system = developer_instructions or _FORCED_SYSTEM_INSTRUCTION

    def build_inference_prompt(
        self,
        user_message: str,
        dev_instructions: "str | None" = None,
    ) -> str:
        system = dev_instructions if dev_instructions is not None else self._system
        return (
            f"<|system|>\n{system}\n"
            f"<|user|>\n{user_message}\n"
            f"<|assistant|>\n<think>\n"
        )

    def parse_output(self, output) -> dict:
        text          = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason
        m = _THINK_CLOSE_RE.search(text)
        if m:
            reasoning_trace = text[: m.start()].strip()
            body            = text[m.end() :].strip()
        else:
            # No closing think-tag: the model answered directly (reasoning
            # off, or a non-reasoning profile) — the whole text IS the answer,
            # not a discarded reasoning trace.
            reasoning_trace = ""
            body            = text.strip()
        dual = parse_dual_output(body)
        return {
            "judge_reasoning_trace": reasoning_trace,
            "considerato_predicted": dual["considerato_predicted"],
            "pqm_predicted":         dual["pqm_predicted"],
            "raw_dual":              dual["raw_dual"],
            "finish_reason":         finish_reason,
        }

    def build_forced_prompt(
        self, case_input: str, trace_prefix: str, articles_block: str
    ) -> str:
        system = _FORCED_SYSTEM_INSTRUCTION
        if articles_block:
            system += f"\n\nArticoli rilevanti:\n{articles_block}"
        return (
            f"<|system|>\n{system}\n"
            f"<|user|>\n{case_input}\n"
            f"<|assistant|>\n<think>\n{trace_prefix}"
            + FORCED_PQM_SUFFIX
        )

    def stop_token_ids(self) -> None:
        return None


# ── Harmony backend ────────────────────────────────────────────────────────────

class HarmonyBackend(ModelBackend):
    """
    Backend for gpt-oss-20b via the openai_harmony multi-channel protocol.
    Encodes prompts to token IDs; parses the 'analysis' and 'final' output channels.
    Requires: pip install openai_harmony
    """

    def __init__(self, developer_instructions: str) -> None:
        from openai_harmony import (
            load_harmony_encoding,
            HarmonyEncodingName,
            Role,
            Message,
            Conversation,
            SystemContent,
            DeveloperContent,
            ReasoningEffort,
            TextContent,
        )
        self._Role             = Role
        self._Message          = Message
        self._Conversation     = Conversation
        self._SystemContent    = SystemContent
        self._DeveloperContent = DeveloperContent
        self._ReasoningEffort  = ReasoningEffort
        self._TextContent      = TextContent
        self._enc              = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self._stop_ids         = self._enc.stop_tokens_for_assistant_actions()
        self._developer_instructions = developer_instructions

    def _render(
        self,
        user_message: str,
        partial_assistant: str = "",
        dev_instructions: "str | None" = None,
    ) -> list[int]:
        Role = self._Role
        instructions = dev_instructions if dev_instructions is not None else self._developer_instructions
        convo = self._Conversation.from_messages([
            self._Message.from_role_and_content(
                Role.SYSTEM,
                self._SystemContent.new().with_reasoning_effort(self._ReasoningEffort.HIGH),
            ),
            self._Message.from_role_and_content(
                Role.DEVELOPER,
                self._DeveloperContent.new().with_instructions(instructions),
            ),
            self._Message.from_role_and_content(Role.USER, user_message),
        ])
        base = self._enc.render_conversation_for_completion(convo, Role.ASSISTANT)
        if partial_assistant:
            base = base + self._enc.encode(partial_assistant, disallowed_special=())
        return base

    def build_inference_prompt(
        self,
        user_message: str,
        dev_instructions: "str | None" = None,
    ) -> list[int]:
        return self._render(user_message, dev_instructions=dev_instructions)

    def parse_output(self, output) -> dict:
        token_ids     = output.outputs[0].token_ids
        finish_reason = output.outputs[0].finish_reason
        messages      = self._enc.parse_messages_from_completion_tokens(
            token_ids, role=self._Role.ASSISTANT, strict=False
        )
        reasoning_trace = ""
        body            = ""
        for msg in messages:
            text = "".join(
                c.text for c in msg.content if isinstance(c, self._TextContent)
            )
            if msg.channel == "analysis":
                reasoning_trace = text
            elif msg.channel == "final":
                body = text
        dual = parse_dual_output(body)
        return {
            "judge_reasoning_trace": reasoning_trace,
            "considerato_predicted": dual["considerato_predicted"],
            "pqm_predicted":         dual["pqm_predicted"],
            "raw_dual":              dual["raw_dual"],
            "finish_reason":         finish_reason,
        }

    def build_forced_prompt(
        self, case_input: str, trace_prefix: str, articles_block: str
    ) -> list[int]:
        # articles_block is already embedded in developer_instructions for Harmony.
        return self._render(case_input, partial_assistant=trace_prefix + FORCED_PQM_SUFFIX)

    def stop_token_ids(self) -> list[int]:
        return self._stop_ids


# ── Factory ────────────────────────────────────────────────────────────────────

def get_backend(
    label: "str | None" = None,
    developer_instructions: str = "",
) -> "tuple[ModelBackend, dict, str]":
    """
    Resolve a backend and profile by label.

    Label resolution order:
      1. explicit ``label`` argument
      2. MODEL_PROFILE env var
      3. auto-detect from VLLM_MODEL env var (matched against profile vllm_model values)
      4. first entry in PROFILES

    Returns (backend, profile_dict, resolved_label).
    """
    if label is None:
        label = os.environ.get("MODEL_PROFILE", "").strip() or None

    if label is None:
        vllm_model = os.environ.get("VLLM_MODEL", "")
        for lbl, prof in PROFILES.items():
            if prof["vllm_model"] == vllm_model:
                label = lbl
                break

    if label is None or label not in PROFILES:
        label = next(iter(PROFILES))

    profile = PROFILES[label]
    fmt     = profile["format"]

    backend: ModelBackend
    if fmt == "harmony":
        backend = HarmonyBackend(developer_instructions)
    else:
        backend = ThinkTagBackend(developer_instructions)

    return backend, profile, label


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Registered model profiles:")
    for lbl, prof in PROFILES.items():
        print(f"  {lbl:20s}  format={prof['format']:10s}  model={prof['vllm_model']}")
