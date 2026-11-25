"""Rebuild the evaluation corpus from the Court's public repository.

Downloads every judgment listed in ``data/corpus_manifest.json`` from
italgiure.giustizia.it and verifies it against the published SHA-256. Nothing is
redistributed by this artifact: the documents come from the Court, and the
checksums prove they are the ones the study used.

    python tools/fetch_corpus.py                  # download + verify PDFs
    python tools/fetch_corpus.py --verify-markdown # also check the converted text

**Conversion needs a GPU.** 149 of the 181 judgments are image-scanned: a text
extractor recovers only their ~130-character header line, and the corpus was
built by OCR-ing them with a vision model. Converting is therefore delegated to
``corpus/convert_sentenze.py``, which runs both phases (text extraction, then
GLM-OCR via vLLM for anything thin). Run it after this script:

    python -m corpus.convert_sentenze          # writes sentenze_md/
    python -m corpus.filter_sentenze           # writes sentenze_md_filtered/
    python tools/fetch_corpus.py --verify-markdown

``--verify-markdown`` compares each converted file against ``sha256_markdown``
from the manifest and **fails** on any mismatch. Do not skip it: OCR is not
bit-reproducible across model versions and decoding settings, so a rebuilt
corpus can differ from the one the reported numbers were computed on, and the
metrics would silently be computed on different text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import requests
import urllib3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; cassazione-artifact/1.0)",
    "Referer": "https://www.italgiure.giustizia.it/sncass/",
}
MANIFEST = Path(__file__).resolve().parent.parent / "data" / "corpus_manifest.json"

# The Court's host serves a self-signed certificate. Integrity is established by
# the published SHA-256 below, not by TLS, so the warning is suppressed rather
# than printed once per document.
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# Below this many characters a converted document is a header, not a judgment.
# Matches THIN_THRESHOLD in corpus/convert_sentenze.py.
THIN_THRESHOLD = 1000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(records: list[dict], pdf_dir: Path, delay: float) -> tuple[int, int, int]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    ok = mismatched = failed = 0
    total = len(records)

    for i, rec in enumerate(records, start=1):
        stem = rec["filename"][: -len(".clean.md")]
        pdf = pdf_dir / f"{stem}.clean.pdf"

        if not pdf.exists():
            try:
                resp = session.get(rec["url_pdf"], timeout=60, verify=False)
                resp.raise_for_status()
                pdf.write_bytes(resp.content)
            except Exception as exc:
                print(f"[{i:3d}/{total}] FAIL {stem}: {type(exc).__name__}: {exc}")
                failed += 1
                continue
            time.sleep(delay)

        if rec["sha256_pdf"] and _sha256(pdf) != rec["sha256_pdf"]:
            print(f"[{i:3d}/{total}] PDF CHECKSUM MISMATCH {stem}")
            mismatched += 1
            continue
        ok += 1

    return ok, mismatched, failed


def verify_markdown(records: list[dict], md_dir: Path) -> tuple[int, list[str], list[str]]:
    """Check converted text against the manifest. Returns (ok, thin, differing)."""
    ok, thin, differing = 0, [], []
    for rec in records:
        path = md_dir / rec["filename"]
        if not path.exists():
            differing.append(f"{rec['filename']} (absent)")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if _sha256(path) == rec["sha256_markdown"]:
            ok += 1
        elif len(text) < THIN_THRESHOLD:
            thin.append(rec["filename"])
        else:
            differing.append(rec["filename"])
    return ok, thin, differing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--pdf-dir", type=Path, default=Path("sentenze"))
    ap.add_argument("--md-dir", type=Path, default=Path("sentenze_md_filtered"))
    ap.add_argument("--verify-markdown", action="store_true",
                    help="compare converted text against the manifest checksums")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between requests; the Court's server is shared")
    args = ap.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))["judgments"]

    if args.verify_markdown:
        ok, thin, differing = verify_markdown(records, args.md_dir)
        print(f"markdown verified {ok}/{len(records)}")
        if thin:
            print(f"\n{len(thin)} file(s) are below {THIN_THRESHOLD} characters — these are "
                  f"image-scanned judgments whose OCR pass has not run. Run:\n"
                  f"    python -m corpus.convert_sentenze\n"
                  f"    python -m corpus.filter_sentenze")
            for name in thin[:5]:
                print(f"    thin: {name}")
            if len(thin) > 5:
                print(f"    ... and {len(thin) - 5} more")
        if differing:
            print(f"\n{len(differing)} file(s) differ from the corpus the reported numbers "
                  f"were computed on. OCR is not bit-reproducible across model versions and "
                  f"decoding settings; treat any metric computed on these as not comparable "
                  f"to the published figures.")
            for name in differing[:5]:
                print(f"    differs: {name}")
            if len(differing) > 5:
                print(f"    ... and {len(differing) - 5} more")
        sys.exit(1 if (thin or differing) else 0)

    ok, mismatched, failed = download(records, args.pdf_dir, args.delay)
    print(f"\nPDFs verified {ok}/{len(records)}  mismatched {mismatched}  failed {failed}")
    print("\nNext: convert them (needs a GPU for the OCR phase — 149 of the 181 "
          "judgments are image-scanned):\n"
          "    python -m corpus.convert_sentenze\n"
          "    python -m corpus.filter_sentenze\n"
          "    python tools/fetch_corpus.py --verify-markdown")
    if mismatched or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
