"""
OCR benchmark for Italian legal documents — supports multiple vision models.

Strategy: use digital PDFs (where full text is known via pymupdf4llm) as ground
truth. For each sampled PDF, render pages to PNG (simulating a scanned document),
run OCR via vLLM offline inference, and compare against the known text.

Metrics reported per file and in aggregate:
  - CER  (character error rate)
  - WER  (word error rate)
  - Similarity (difflib ratio, 0-1)
  - Citation recall: fraction of art. XXX c.p. references correctly recovered
  - Latency: seconds per page, tokens per second

Usage:
    python benchmark_ocr.py [--n N] [--dpi DPI] [--out FILE] [--seed N]
                            [--benchmark-dir DIR]

    --n              number of PDFs to sample (default: 100)
    --dpi            render resolution        (default: 200)
    --out            JSON file for results    (default: benchmark_results.json)
    --seed           random seed              (default: 42)
    --benchmark-dir  output directory         (default: benchmark)

Env vars:
    VLLM_MODEL     HuggingFace model ID or local path (default: Qwen/Qwen3.5-9B)
    VLLM_PROFILE   Override auto-detected profile: qwen | glm | deepseek_ocr | deepseek_ocr2
    VLLM_TP        Tensor-parallel size (default: 1)
    VLLM_GPU_MEM   GPU memory utilisation fraction (default: 0.9)
    VLLM_MAX_LEN   Maximum model context length in tokens (default: 16384)
"""

import argparse
import base64
import csv
import io
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable


import fitz
import pymupdf4llm
from PIL import Image as PILImage
from tqdm import tqdm
from vllm import LLM, SamplingParams

try:
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor as _NGramProc
    _NGRAM_AVAILABLE = True
except ImportError:
    _NGramProc = None
    _NGRAM_AVAILABLE = False

SOURCE_DIR = Path("sentenze")
MD_DIR     = Path("sentenze_md")

MODEL_PATH      = os.environ.get("VLLM_MODEL",   "Qwen/Qwen3.5-9B")
VLLM_PROFILE    = os.environ.get("VLLM_PROFILE", "")
TENSOR_PARALLEL = int(os.environ.get("VLLM_TP",      "1"))
GPU_MEMORY_UTIL = float(os.environ.get("VLLM_GPU_MEM", "0.9"))
MAX_MODEL_LEN   = int(os.environ.get("VLLM_MAX_LEN",  "32768"))

OCR_MAX_TOKENS = 8192

SYSTEM_PROMPT = (
    "Convert the following scanned Italian legal document page to markdown.\n"
    "Return only the markdown with no explanation text. "
    "Do not include delimiters like ```markdown or ```html.\n\n"
    "RULES:\n"
    "  - Transcribe all text exactly, including headers, footers, and margin notes.\n"
    "  - Preserve paragraph structure and section numbering.\n"
    "  - Preserve article citations exactly (e.g. art. 319 c.p., art. 321 c.p.).\n"
    "  - Prefer using ☐ and ☑ for check boxes."
)

# Matches "art. 318 c.p.", "art. 319-bis c.p.", "art. 321 cod. pen.", etc.
CITATION_RE = re.compile(
    r"art(?:icolo)?\.?\s*"
    r"(?:3(?:1[89]|2[0-3])(?:[\s\-](?:bis|ter|quater))?)"
    r"[\s,]*"
    r"(?:c\.?\s*p\.?|cod\.?\s*pen\.?|codice\s+penale)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _levenshtein(a, b) -> int:
    """Standard edit distance (Wagner-Fischer)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref = ref.strip()
    if not ref:
        return 0.0 if not hyp.strip() else 1.0
    return min(_levenshtein(ref, hyp.strip()) / len(ref), 1.0)


def wer(ref: str, hyp: str) -> float:
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return min(_levenshtein(ref_words, hyp_words) / len(ref_words), 1.0)


def similarity(ref: str, hyp: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, ref.strip(), hyp.strip()).ratio()


def citation_recall(ref: str, hyp: str) -> tuple[float, list[str], list[str]]:
    def normalise(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    ref_hits = [normalise(m.group()) for m in CITATION_RE.finditer(ref)]
    hyp_hits = [normalise(m.group()) for m in CITATION_RE.finditer(hyp)]
    if not ref_hits:
        return 1.0, [], []
    found = sum(1 for c in ref_hits if c in hyp_hits)
    return found / len(ref_hits), ref_hits, hyp_hits


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def page_to_pil(page: fitz.Page, dpi: int) -> PILImage.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    return page.get_pixmap(matrix=mat).pil_image()


def pil_to_base64(img: PILImage.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def page_to_base64(page: fitz.Page, dpi: int) -> str:
    return pil_to_base64(page_to_pil(page, dpi))


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    name: str
    use_chat: bool
    build_requests: Callable[[list[PILImage.Image]], list]
    sampling_params: Callable[[], SamplingParams]
    postprocess: Callable[[str], str]
    llm_extra_kwargs: dict = field(default_factory=dict)
    chat_template_kwargs: dict = field(default_factory=dict)


def _qwen_profile() -> ModelProfile:
    def build(imgs: list[PILImage.Image]) -> list[list[dict]]:
        return [
            [{"role": "user", "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{pil_to_base64(img)}"
                }},
            ]}]
            for img in imgs
        ]

    def sp() -> SamplingParams:
        return SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20,
            presence_penalty=1.5, max_tokens=OCR_MAX_TOKENS,
        )

    def post(text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return text

    return ModelProfile(
        name="qwen",
        use_chat=True,
        build_requests=build,
        sampling_params=sp,
        postprocess=post,
        chat_template_kwargs={"enable_thinking": False},
    )


def _glm_profile() -> ModelProfile:
    def build(imgs: list[PILImage.Image]) -> list[list[dict]]:
        return [
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{pil_to_base64(img)}"
                }},
                {"type": "text", "text": "Text Recognition:"},
            ]}]
            for img in imgs
        ]

    def sp() -> SamplingParams:
        return SamplingParams(max_tokens=OCR_MAX_TOKENS)

    return ModelProfile(
        name="glm",
        use_chat=True,
        build_requests=build,
        sampling_params=sp,
        postprocess=lambda text: text,
    )


def _deepseek_ocr_profile() -> ModelProfile:
    def build(imgs: list[PILImage.Image]) -> list[dict]:
        return [
            {"prompt": "<image>\nFree OCR.", "multi_modal_data": {"image": img}}
            for img in imgs
        ]

    def sp() -> SamplingParams:
        kwargs: dict = dict(
            temperature=0.0,
            max_tokens=OCR_MAX_TOKENS,
            skip_special_tokens=False,
        )
        if _NGRAM_AVAILABLE:
            try:
                kwargs["extra_args"] = dict(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822},
                )
            except Exception:
                pass
        return SamplingParams(**kwargs)

    extra: dict = dict(enable_prefix_caching=False, mm_processor_cache_gb=0, max_model_len=8192)
    if _NGRAM_AVAILABLE:
        try:
            extra["logits_processors"] = [_NGramProc]
        except Exception:
            pass

    return ModelProfile(
        name="deepseek_ocr",
        use_chat=False,
        build_requests=build,
        sampling_params=sp,
        postprocess=lambda text: text,
        llm_extra_kwargs=extra,
    )


def _deepseek_ocr2_profile() -> ModelProfile:
    def build(imgs: list[PILImage.Image]) -> list[dict]:
        return [
            {
                "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
                "multi_modal_data": {"image": img},
            }
            for img in imgs
        ]

    def sp() -> SamplingParams:
        return SamplingParams(
            temperature=0.0,
            max_tokens=OCR_MAX_TOKENS,
            skip_special_tokens=False,
        )

    return ModelProfile(
        name="deepseek_ocr2",
        use_chat=False,
        build_requests=build,
        sampling_params=sp,
        postprocess=lambda text: text,
        llm_extra_kwargs={"trust_remote_code": True, "max_model_len": 8192},
    )


# Pattern → profile factory. Matched in order; first match wins.
_PROFILE_REGISTRY: list[tuple[str, Callable[[], ModelProfile]]] = [
    ("glm",            _glm_profile),
    ("deepseek-ocr-2", _deepseek_ocr2_profile),  # must come before deepseek-ocr
    ("deepseek-ocr",   _deepseek_ocr_profile),
    ("qwen",           _qwen_profile),            # default / fallback
]


def detect_profile(model_path: str, override: str = "") -> ModelProfile:
    key = (override or model_path).lower()
    for pattern, factory in _PROFILE_REGISTRY:
        if pattern in key:
            return factory()
    return _qwen_profile()


PROFILE: ModelProfile = detect_profile(MODEL_PATH, VLLM_PROFILE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def init_model(profile: ModelProfile | None = None) -> LLM:
    if profile is None:
        profile = PROFILE
    base = dict(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )
    base.update(profile.llm_extra_kwargs)
    return LLM(**base)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def pick_digital_pdfs(n: int) -> list[Path]:
    image_based_log = MD_DIR / "image_based.txt"
    image_based: set[str] = set()
    if image_based_log.exists():
        image_based = set(image_based_log.read_text(encoding="utf-8").splitlines())

    thin: set[str] = set()
    for md in MD_DIR.glob("*.md"):
        if len(md.read_text(encoding="utf-8", errors="replace").strip()) < 500:
            thin.add(md.stem + ".pdf")

    candidates = [
        p for p in SOURCE_DIR.glob("*.clean.pdf")
        if p.name not in image_based and p.name not in thin
    ]

    if len(candidates) < n:
        print(f"[warn] only {len(candidates)} digital PDFs available; using all.", file=sys.stderr)
        return candidates

    return random.sample(candidates, n)


# ---------------------------------------------------------------------------
# Per-file benchmark
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    page_idx: int
    cer: float
    wer: float
    similarity: float
    citation_recall: float
    ref_citations: list[str]
    hyp_citations: list[str]
    elapsed_s: float
    out_tokens: int
    truncated: bool


@dataclass
class FileResult:
    filename: str
    n_pages: int
    pages: list[PageResult]
    mean_cer: float
    mean_wer: float
    mean_similarity: float
    mean_citation_recall: float
    total_elapsed_s: float
    tokens_per_second: float


# vLLM error substrings that indicate a prompt exceeded the model's context window.
# Covers both older vLLM wording and the format introduced in v0.19.
_CTX_ERRS = (
    "longer than the maximum model length",  # pre-v0.19
    "maximum context length is",             # v0.19+
)


def _is_ctx_err(exc: Exception) -> bool:
    s = str(exc)
    return any(msg in s for msg in _CTX_ERRS)


def _run_inference(
    llm: LLM,
    profile: ModelProfile,
    requests: list,
    sp: SamplingParams,
    pil_images: list | None = None,
):
    """Run inference, falling back to per-page retry on context-length errors.

    pil_images must be provided when batching multiple pages so the fallback
    can rebuild each request fresh — reusing stale request objects after a
    failed batch causes vLLM's multimodal cache to raise AssertionError.

    Returns a list parallel to `requests` where None marks a page that was
    individually too long and had to be skipped.
    """
    def _call(reqs):
        if profile.use_chat:
            return llm.chat(
                messages=reqs,
                sampling_params=sp,
                chat_template_kwargs=profile.chat_template_kwargs or None,
            )
        return llm.generate(prompts=reqs, sampling_params=sp)

    try:
        return _call(requests)
    except Exception as batch_err:
        if not _is_ctx_err(batch_err):
            raise
        print(f"\n    [warn] batch context-length error ({batch_err}); retrying page by page …", file=sys.stderr)
        outputs = []
        imgs = pil_images if pil_images is not None else [None] * len(requests)
        for img, req in zip(imgs, requests):
            # vLLM v0.19 bug: after a failed batch, the encoder cache marks the
            # image hashes as "in-flight" but never commits the features. Re-submitting
            # the same image data (same hash) then hits an AssertionError inside
            # get_and_update_item. Work around this by forcing a new hash via a
            # JPEG round-trip (lossy, quality=99, so visually identical).
            if img is not None:
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=99)
                buf.seek(0)
                rehashed = PILImage.open(buf)
                rehashed.load()
                fresh = profile.build_requests([rehashed])[0]
            else:
                fresh = req
            try:
                outputs.append(_call([fresh])[0])
            except Exception as page_err:
                if not _is_ctx_err(page_err):
                    raise
                outputs.append(None)  # this page is too long even individually
        return outputs


def benchmark_file(
    llm: LLM,
    pdf_path: Path,
    dpi: int,
    save_dir: Path | None = None,
    profile: ModelProfile | None = None,
) -> FileResult:
    if profile is None:
        profile = PROFILE

    sp = profile.sampling_params()

    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)

    # Ground truth per page
    page_refs = [pymupdf4llm.to_markdown(doc, pages=[i]) for i in range(n_pages)]

    # Render all pages to PIL images, then build model-specific requests
    pil_images = [page_to_pil(doc[i], dpi) for i in range(n_pages)]
    doc.close()

    requests = profile.build_requests(pil_images)

    t0 = time.perf_counter()
    outputs = _run_inference(llm, profile, requests, sp, pil_images=pil_images)
    total_elapsed = time.perf_counter() - t0

    page_results: list[PageResult] = []
    for i, out in enumerate(outputs):
        if out is None:
            # Page exceeded context limit even individually — record as worst-case
            hyp = ""
            out_tokens = 0
            truncated = False
            if save_dir is not None:
                pdf_dir = save_dir / pdf_path.stem
                pdf_dir.mkdir(exist_ok=True)
                (pdf_dir / f"page_{i:03d}_ref.md").write_text(page_refs[i], encoding="utf-8")
                (pdf_dir / f"page_{i:03d}_hyp.md").write_text(
                    "[SKIPPED: prompt exceeded model context length]", encoding="utf-8"
                )
            ref = page_refs[i]
            c_recall, ref_cits, hyp_cits = citation_recall(ref, hyp)
            page_results.append(PageResult(
                page_idx=i,
                cer=cer(ref, hyp),
                wer=wer(ref, hyp),
                similarity=similarity(ref, hyp),
                citation_recall=c_recall,
                ref_citations=ref_cits,
                hyp_citations=hyp_cits,
                elapsed_s=0.0,
                out_tokens=0,
                truncated=False,
            ))
            continue

        hyp = profile.postprocess(out.outputs[0].text.strip())

        if save_dir is not None:
            pdf_dir = save_dir / pdf_path.stem
            pdf_dir.mkdir(exist_ok=True)
            (pdf_dir / f"page_{i:03d}_ref.md").write_text(page_refs[i], encoding="utf-8")
            (pdf_dir / f"page_{i:03d}_hyp.md").write_text(hyp, encoding="utf-8")

        out_tokens = len(out.outputs[0].token_ids)
        truncated  = out_tokens >= OCR_MAX_TOKENS
        ref        = page_refs[i]
        c_recall, ref_cits, hyp_cits = citation_recall(ref, hyp)

        page_results.append(PageResult(
            page_idx=i,
            cer=cer(ref, hyp),
            wer=wer(ref, hyp),
            similarity=similarity(ref, hyp),
            citation_recall=c_recall,
            ref_citations=ref_cits,
            hyp_citations=hyp_cits,
            elapsed_s=0.0,   # filled below
            out_tokens=out_tokens,
            truncated=truncated,
        ))

    # Distribute total elapsed proportionally to token counts
    total_tokens = sum(p.out_tokens for p in page_results)
    for p in page_results:
        share = p.out_tokens / total_tokens if total_tokens > 0 else 1 / n_pages
        p.elapsed_s = round(total_elapsed * share, 3)

    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return FileResult(
        filename=pdf_path.name,
        n_pages=n_pages,
        pages=page_results,
        mean_cer=mean([p.cer for p in page_results]),
        mean_wer=mean([p.wer for p in page_results]),
        mean_similarity=mean([p.similarity for p in page_results]),
        mean_citation_recall=mean([p.citation_recall for p in page_results]),
        total_elapsed_s=round(total_elapsed, 2),
        tokens_per_second=round(total_tokens / total_elapsed, 1) if total_elapsed > 0 else 0.0,
    )



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",             type=int, default=100,                     help="PDFs to sample")
    parser.add_argument("--dpi",           type=int, default=200,                    help="render DPI")
    parser.add_argument("--out",           type=str, default="benchmark_results.json", help="output JSON filename")
    parser.add_argument("--seed",          type=int, default=42,                     help="random seed")
    parser.add_argument("--benchmark-dir", type=str, default="benchmark",            help="directory for JSON and markdown outputs")
    args = parser.parse_args()

    random.seed(args.seed)

    benchmark_dir = Path(args.benchmark_dir)
    benchmark_dir.mkdir(exist_ok=True)

    pdfs = pick_digital_pdfs(args.n)
    if not pdfs:
        print("No digital PDFs found to benchmark.", file=sys.stderr)
        sys.exit(1)

    # Load existing results so we can skip already-processed files
    out_path = benchmark_dir / Path(args.out).name
    existing_file_dicts: list[dict] = []
    if out_path.exists():
        try:
            existing_file_dicts = json.loads(out_path.read_text(encoding="utf-8")).get("files", [])
        except Exception:
            pass
    already_done: set[str] = {d["filename"] for d in existing_file_dicts}

    pdfs = [p for p in pdfs if p.name not in already_done]
    if already_done:
        print(f"Skipping {len(already_done)} already-benchmarked file(s).")
    if not pdfs:
        print("All sampled PDFs already benchmarked — nothing to do.")
        sys.exit(0)

    print(f"Loading model {MODEL_PATH} (profile={PROFILE.name}, tp={TENSOR_PARALLEL}) …")
    llm = init_model(profile=PROFILE)
    print(f"Model ready. Benchmarking {len(pdfs)} PDFs at {args.dpi} DPI.\n")

    new_file_results: list[FileResult] = []

    for pdf_path in tqdm(pdfs, desc="Benchmarking", ncols=120):
        tqdm.write(f"  {pdf_path.name}")
        try:
            result = benchmark_file(llm, pdf_path, args.dpi, save_dir=benchmark_dir, profile=PROFILE)
            new_file_results.append(result)
            tqdm.write(
                f"    CER={result.mean_cer:.3f}  WER={result.mean_wer:.3f}  "
                f"sim={result.mean_similarity:.3f}  cit_recall={result.mean_citation_recall:.3f}  "
                f"{result.tokens_per_second:.0f} tok/s"
            )
        except Exception as e:
            tqdm.write(f"    [error] {e}", file=sys.stderr)

    if not new_file_results:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    all_file_dicts = existing_file_dicts + [asdict(r) for r in new_file_results]

    def agg_mean(attr):
        vals = [d[attr] for d in all_file_dicts]
        return round(sum(vals) / len(vals), 4)

    summary = {
        "model": MODEL_PATH,
        "dpi": args.dpi,
        "n_files": len(all_file_dicts),
        "total_pages": sum(d["n_pages"] for d in all_file_dicts),
        "aggregate": {
            "mean_cer":               agg_mean("mean_cer"),
            "mean_wer":               agg_mean("mean_wer"),
            "mean_similarity":        agg_mean("mean_similarity"),
            "mean_citation_recall":   agg_mean("mean_citation_recall"),
            "mean_tokens_per_second": agg_mean("tokens_per_second"),
            "total_elapsed_s":        round(sum(d["total_elapsed_s"] for d in all_file_dicts), 2),
        },
        "files": all_file_dicts,
    }

    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    agg = summary["aggregate"]
    csv_path = out_path.with_suffix(".csv")
    _CSV_FIELDS = ["Document", "Pages", "CER", "WER", "Similarity", "Citation Recall", "Tok/s"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for d in all_file_dicts:
            writer.writerow({
                "Document":        d["filename"].removesuffix(".pdf"),
                "Pages":           d["n_pages"],
                "CER":             round(d["mean_cer"], 4),
                "WER":             round(d["mean_wer"], 4),
                "Similarity":      round(d["mean_similarity"], 4),
                "Citation Recall": round(d["mean_citation_recall"], 4),
                "Tok/s":           round(d["tokens_per_second"], 1),
            })
        writer.writerow({
            "Document":        "AGGREGATE",
            "Pages":           summary["total_pages"],
            "CER":             round(agg["mean_cer"], 4),
            "WER":             round(agg["mean_wer"], 4),
            "Similarity":      round(agg["mean_similarity"], 4),
            "Citation Recall": round(agg["mean_citation_recall"], 4),
            "Tok/s":           round(agg["mean_tokens_per_second"], 1),
        })

    print(f"\n=== Benchmark summary ({len(all_file_dicts)} files, {summary['total_pages']} pages) ===")
    print(f"  CER              : {agg['mean_cer']:.3f}")
    print(f"  WER              : {agg['mean_wer']:.3f}")
    print(f"  Similarity       : {agg['mean_similarity']:.3f}")
    print(f"  Citation recall  : {agg['mean_citation_recall']:.3f}")
    print(f"  Throughput       : {agg['mean_tokens_per_second']:.0f} tok/s")
    print(f"  Total time       : {agg['total_elapsed_s']:.1f}s")
    print(f"\nFull results saved to {out_path}")
    print(f"Metrics table saved to {csv_path}")


if __name__ == "__main__":
    main()
