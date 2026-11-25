"""
Convert PDFs to Markdown.

Phase 1 — text extraction via pymupdf4llm (CPU, no model required).
Phase 2 — OCR via vLLM for PDFs whose Markdown is thin (< THIN_THRESHOLD chars).

vLLM is imported lazily; the GPU model is loaded only when OCR targets exist.

Usage:
    python convert_sentenze.py [--limit N] [--dry-run] [--skip-ocr]

Flags:
    --limit N     Process at most N PDFs in Phase 1 (0 = all).
    --dry-run     Print planned actions; write nothing, load no model.
    --skip-ocr    Run Phase 1 only; skip vLLM entirely.

Env vars:
    VLLM_MODEL    HuggingFace model id or local path  (default: zai-org/GLM-OCR)
    VLLM_TP       tensor parallel size (# GPUs)        (default: 1)
    VLLM_GPU_MEM  GPU memory utilisation fraction      (default: 0.9)
    VLLM_MAX_LEN  max model context length (tokens)    (default: 32768)
"""

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import fitz
import pymupdf4llm
from tqdm import tqdm

SOURCE_DIR     = Path("sentenze")
MD_DIR         = Path("sentenze_md")
THIN_THRESHOLD = 1000
OCR_MAX_TOKENS = 8192

MODEL_PATH      = os.environ.get("VLLM_MODEL",    "zai-org/GLM-OCR")
TENSOR_PARALLEL = int(os.environ.get("VLLM_TP",       "1"))
GPU_MEMORY_UTIL = float(os.environ.get("VLLM_GPU_MEM", "0.9"))
MAX_MODEL_LEN   = int(os.environ.get("VLLM_MAX_LEN",   "32768"))

# Assigned lazily by _ensure_vllm(); used by ocr_pdf() via SAMPLING_PARAMS.
SAMPLING_PARAMS = None
LLM = None


# ── vLLM lazy import ──────────────────────────────────────────────────────────

def _ensure_vllm():
    global SAMPLING_PARAMS, LLM
    if SAMPLING_PARAMS is not None:
        return
    os.environ.setdefault("TQDM_DISABLE", "1")
    from vllm import LLM as _LLM, SamplingParams as _SP
    LLM = _LLM
    SAMPLING_PARAMS = _SP(max_tokens=OCR_MAX_TOKENS)


# ── Phase 2 helpers (OCR) ─────────────────────────────────────────────────────

def init_model():
    _ensure_vllm()
    return LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )


def page_to_base64(page: fitz.Page) -> str:
    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)
    buf = io.BytesIO(pix.tobytes("png"))
    return base64.b64encode(buf.read()).decode("utf-8")


def _make_conversation(b64: str) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Text Recognition:"},
    ]}]


def collect_ocr_targets() -> list[tuple[Path, Path]]:
    """Return (md_path, pdf_path) pairs whose Markdown content is below THIN_THRESHOLD."""
    targets: list[tuple[Path, Path]] = []

    image_based_log = MD_DIR / "image_based.txt"
    listed: set[str] = set()
    if image_based_log.exists():
        for name in image_based_log.read_text(encoding="utf-8").splitlines():
            name = name.strip()
            if not name:
                continue
            listed.add(name)
            pdf_path = SOURCE_DIR / name
            md_path  = MD_DIR / Path(name).with_suffix(".md").name
            if pdf_path.exists():
                targets.append((md_path, pdf_path))

    for md_path in sorted(MD_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) >= THIN_THRESHOLD:
            continue
        pdf_name = md_path.stem + ".pdf"
        if pdf_name in listed:
            continue
        pdf_path = SOURCE_DIR / pdf_name
        if pdf_path.exists():
            targets.append((md_path, pdf_path))

    return targets


def ocr_pdf(llm, md_path: Path, pdf_path: Path) -> bool:
    existing_header = ""
    if md_path.exists():
        existing_header = md_path.read_text(encoding="utf-8", errors="replace").strip()

    doc = fitz.open(str(pdf_path))
    conversations = [_make_conversation(page_to_base64(page)) for page in doc]
    doc.close()

    outputs = llm.chat(messages=conversations, sampling_params=SAMPLING_PARAMS)

    page_texts: list[str] = []
    for i, out in enumerate(outputs):
        text = out.outputs[0].text.strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        if text:
            page_texts.append(text)
        else:
            print(f"    [warn] page {i} returned empty for {pdf_path.name}", file=sys.stderr)

    if not page_texts:
        print(f"  [skip] no OCR output for {pdf_path.name}", file=sys.stderr)
        return False

    body = "\n\n---\n\n".join(page_texts)
    content = (existing_header + "\n\n" + body).strip() if existing_header else body
    md_path.write_text(content, encoding="utf-8")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown (Phase 1) and OCR thin results (Phase 2)."
    )
    parser.add_argument("--limit",    type=int, default=0,
                        help="max PDFs to process in Phase 1 (0 = all)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="print actions without writing files or loading model")
    parser.add_argument("--skip-ocr", action="store_true",
                        help="run Phase 1 only; skip vLLM OCR entirely")
    args = parser.parse_args()

    MD_DIR.mkdir(exist_ok=True)

    # ── Phase 1: pymupdf4llm extraction ───────────────────────────────────────
    pdfs = sorted(SOURCE_DIR.glob("*.clean.pdf"))
    total = len(pdfs)

    pending: list[Path] = []
    for pdf in pdfs:
        md_path = MD_DIR / pdf.with_suffix(".md").name
        rich_already = (
            md_path.exists()
            and len(md_path.read_text(encoding="utf-8", errors="replace").strip()) >= THIN_THRESHOLD
        )
        if not rich_already:
            pending.append(pdf)

    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"Phase 1: {len(pending)} PDF(s) to process (of {total} total).")

    image_based: list[str] = []
    done = skipped_phase1 = 0

    for i, pdf in enumerate(pending, 1):
        md_path = MD_DIR / pdf.with_suffix(".md").name
        tag = f"[{i}/{len(pending)}]"

        if args.dry_run:
            print(f"{tag} [dry-run] would extract {pdf.name}")
            continue

        text = pymupdf4llm.to_markdown(str(pdf))
        if not text.strip():
            image_based.append(pdf.name)
            print(f"{tag} IMAGE-BASED (no text): {pdf.name}")
        else:
            md_path.write_text(text, encoding="utf-8")
            done += 1
            print(f"{tag} done {pdf.name}")

    if not args.dry_run:
        if image_based:
            (MD_DIR / "image_based.txt").write_text("\n".join(image_based), encoding="utf-8")
        print(f"\nPhase 1 done: {done} extracted, {len(image_based)} image-based.")
    else:
        print(f"\nPhase 1 dry-run complete ({len(pending)} would be processed).")

    if args.skip_ocr:
        print("Skipping OCR phase (--skip-ocr).")
        return

    # ── Phase 2: vLLM OCR ─────────────────────────────────────────────────────
    targets = collect_ocr_targets()
    if not targets:
        print("\nPhase 2: no OCR targets — all done.")
        return

    print(f"\nPhase 2: {len(targets)} file(s) to OCR.")

    if args.dry_run:
        for md_path, pdf_path in targets:
            print(f"  [dry-run] would OCR {pdf_path.name}")
        return

    print(f"Loading model {MODEL_PATH} (tp={TENSOR_PARALLEL}) …")
    llm = init_model()
    print("Model ready.\n")

    ok = fail = 0
    for md_path, pdf_path in tqdm(targets, desc="OCR", ncols=120):
        try:
            success = ocr_pdf(llm, md_path, pdf_path)
        except Exception as e:
            print(f"  [error] {pdf_path.name}: {e}", file=sys.stderr)
            success = False

        if success:
            ok += 1
            tqdm.write(f"  OK   {pdf_path.name}")
        else:
            fail += 1
            tqdm.write(f"  FAIL {pdf_path.name}")

    print(f"\nPhase 2 done: {ok} succeeded, {fail} failed.")


if __name__ == "__main__":
    main()
