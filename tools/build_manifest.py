"""Build the corpus manifest that stands in for the judgments themselves.

The judgments are public but are published unredacted, naming the parties, their
dates and places of birth and the offences charged. This artifact therefore does
not redistribute them. It ships instead the exact identity of every judgment in
the corpus, together with the URL it was retrieved from and a checksum, so that
a replicator can fetch the same documents from the Court's own repository and
verify byte-for-byte that they obtained what the study used.

Emitted for each judgment: the canonical filename, the SOLR document id, the
section/year/number that identify it in the Court's register, the target
articles matched, the retrieval URL, and SHA-256 of both the source PDF and the
extracted Markdown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from pathlib import Path

PDF_BASE = "https://www.italgiure.giustizia.it/xway/application/nif/clean/hc.dll"
SEARCH_UI = "https://www.italgiure.giustizia.it/sncass/"

# snpen@s60@a2025@n00305@tS.clean.md
NAME_RE = re.compile(
    r"^(?P<kind>sn(?:pen|civ))@s(?P<sezione>\w+)@a(?P<anno>\d{4})"
    r"@n(?P<numero>\d+)@t(?P<tipo>\w+)\.clean\.md$"
)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pdf_url(kind: str, stem: str, meta_path: str | None) -> str:
    # The Court's endpoint keys on the dated path recorded in the SOLR metadata;
    # fall back to the bare filename when that record is unavailable.
    raw = meta_path or f"{stem}.clean.pdf"
    if ".clean." not in raw:
        raw = raw.replace(".pdf", ".clean.pdf")
    return f"{PDF_BASE}?verbo=attach&db={kind}&id={urllib.parse.quote(raw)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True, help="sentenze_md_filtered/")
    ap.add_argument("--pdfs", type=Path, help="sentenze/ (for source checksums)")
    ap.add_argument("--metadata", type=Path, help="SOLR metadata.json")
    ap.add_argument("--splits", type=Path, help="splits/ directory")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    by_name: dict[str, dict] = {}
    if args.metadata and args.metadata.exists():
        for doc in json.loads(args.metadata.read_text(encoding="utf-8")):
            raw = doc.get("filename")
            raw = raw[0] if isinstance(raw, list) else raw
            if not raw:
                continue
            stem = Path(raw).name.replace(".clean.pdf", "").replace(".pdf", "")
            by_name[stem] = {"path": raw, "articles": doc.get("articles", []),
                             "id": doc.get("id")}

    split_of: dict[str, str] = {}
    if args.splits:
        for label, fname in (("test", "test_100.json"), ("fewshot_pool", "fewshot_pool_81.json")):
            path = args.splits / fname
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for case in data.get("cases", []):
                split_of[case["filename"]] = label

    records = []
    for md in sorted(args.corpus.glob("*.md")):
        m = NAME_RE.match(md.name)
        if not m:
            continue
        stem = md.name[: -len(".clean.md")]
        meta = by_name.get(stem, {})
        pdf = args.pdfs / f"{stem}.clean.pdf" if args.pdfs else None
        records.append({
            "filename": md.name,
            "solr_id": meta.get("id"),
            "kind": m["kind"],
            "sezione": m["sezione"],
            "anno": int(m["anno"]),
            "numero": int(m["numero"]),
            "tipo": m["tipo"],
            "articles_matched": meta.get("articles", []),
            "split": split_of.get(md.name),
            "url_pdf": _pdf_url(m["kind"], stem, meta.get("path")),
            "sha256_pdf": _sha256(pdf) if pdf else None,
            "sha256_markdown": _sha256(md),
        })

    out = {
        "description": (
            "Identity, provenance and checksums of the 181 judgments of the study "
            "corpus. The judgment texts are deliberately not redistributed: "
            "they are published unredacted by the Court and contain personal and "
            "criminal-offence data. Retrieve them from the URLs below and verify "
            "against the checksums."
        ),
        "source": SEARCH_UI,
        "retrieved": "2026-05",
        "n_judgments": len(records),
        "judgments": records,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with_pdf = sum(1 for r in records if r["sha256_pdf"])
    print(f"judgments        : {len(records)}")
    print(f"  with PDF hash  : {with_pdf}")
    print(f"  with SOLR id   : {sum(1 for r in records if r['solr_id'])}")
    print(f"  split assigned : {sum(1 for r in records if r['split'])}")


if __name__ == "__main__":
    main()
