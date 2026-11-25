import argparse
import json
import time
import urllib.parse
import warnings
from pathlib import Path

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

SEARCH_URL = "https://www.italgiure.giustizia.it/sncass/isapi/hc.dll/sn.solr/sn-collection/select"
PDF_BASE = "https://www.italgiure.giustizia.it/xway/application/nif/clean/hc.dll"
ARTICLES = ["318", "319", "320", "321", "322", "323"]
ROWS = 100
DELAY = 1.0
RETRIES = 3
OUTPUT_DIR = Path("sentenze")

FIELDS = "id,filename,szdec,kind,ssz,tipoprov,numcard,numdec,numdep,datdep,ecli,anno,datdec,presidente,relatore"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.italgiure.giustizia.it/sncass/",
    "Origin": "https://www.italgiure.giustizia.it",
}


def build_query(article: str) -> str:
    return (
        f'("({article})"~8 OR "{article}"~8)'
        f' AND (kind:"snpen" OR kind:"snciv")'
        f" AND anno:[2025 TO *]"
    )


def search(session: requests.Session, article: str, start: int) -> dict:
    q = build_query(article)
    data = {
        "start": start,
        "rows": ROWS,
        "q": q,
        "wt": "json",
        "indent": "off",
        "sort": "datdec desc,numdec desc",
        "fl": FIELDS,
    }
    for attempt in range(RETRIES):
        try:
            r = session.post(
                SEARCH_URL,
                params={"app.query": "snpen"},
                data=data,
                timeout=30,
                verify=False,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{RETRIES} after {wait}s ({e})")
            time.sleep(wait)


def iter_docs(session: requests.Session, article: str):
    start = 0
    num_found = None
    while True:
        payload = search(session, article, start)
        resp = payload.get("response", {})
        if num_found is None:
            num_found = resp.get("numFound", 0)
            print(f"[{article}] {num_found} results found")
        docs = resp.get("docs", [])
        if not docs:
            break
        yield from docs
        start += len(docs)
        if start >= num_found:
            break
        time.sleep(DELAY)


def canonical_filename(doc: dict) -> str:
    raw = doc["filename"][0] if isinstance(doc.get("filename"), list) else doc.get("filename", "")
    name = Path(raw).name
    if ".clean." not in name:
        name = name.replace(".pdf", ".clean.pdf")
    return name


def pdf_url(doc: dict) -> str:
    raw = doc["filename"][0] if isinstance(doc.get("filename"), list) else doc.get("filename", "")
    if ".clean." not in raw:
        raw = raw.replace(".pdf", ".clean.pdf")
    return f"{PDF_BASE}?verbo=attach&db={doc['kind']}&id={urllib.parse.quote(raw)}"


def download_pdf(session: requests.Session, doc: dict, dest_dir: Path, dry_run: bool) -> Path | None:
    fname = canonical_filename(doc)
    dest = dest_dir / fname
    if dest.exists():
        return dest
    if dry_run:
        return dest
    url = pdf_url(doc)
    for attempt in range(RETRIES):
        try:
            r = session.get(url, timeout=60, stream=True, verify=False)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return dest
        except Exception as e:
            if attempt == RETRIES - 1:
                print(f"  ERROR downloading {fname}: {e}")
                return None
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, no downloads")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    all_metadata: dict[str, dict] = {}
    counts: dict[str, dict] = {}

    for article in ARTICLES:
        counts[article] = {"found": 0, "new": 0, "dup": 0, "error": 0}
        for doc in iter_docs(session, article):
            counts[article]["found"] += 1
            fname = canonical_filename(doc)

            if fname not in all_metadata:
                result = download_pdf(session, doc, OUTPUT_DIR, args.dry_run)
                if result is None:
                    counts[article]["error"] += 1
                else:
                    counts[article]["new"] += 1
                sanitized = {k: v for k, v in doc.items() if k != "ocr"}
                all_metadata[fname] = {**sanitized, "articles": [], "local_file": fname}
                tag = "dry" if args.dry_run else "new"
            else:
                counts[article]["dup"] += 1
                tag = "dup"

            all_metadata[fname]["articles"].append(article)
            n = counts[article]["found"]
            print(f"  [{article}] #{n} {fname} ({tag})")
            time.sleep(DELAY)

    if not args.dry_run:
        meta_path = OUTPUT_DIR / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(list(all_metadata.values()), f, ensure_ascii=False, indent=2)
        print(f"\nMetadata saved to {meta_path}")

    print("\nSummary:")
    print(f"{'Article':<10} {'Found':>7} {'New':>7} {'Dup':>7} {'Errors':>7}")
    print("-" * 44)
    for art, c in counts.items():
        print(f"  {art:<8} {c['found']:>7} {c['new']:>7} {c['dup']:>7} {c['error']:>7}")
    print(f"\nTotal unique PDFs: {len(all_metadata)}")


if __name__ == "__main__":
    main()
