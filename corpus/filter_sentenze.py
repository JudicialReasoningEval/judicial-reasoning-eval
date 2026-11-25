import json
import re
import shutil
import time
import warnings
from pathlib import Path

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

SEARCH_URL = "https://www.italgiure.giustizia.it/sncass/isapi/hc.dll/sn.solr/sn-collection/select"
ARTICLES = ["318", "319", "320", "321", "322", "323"]
ROWS = 100
DELAY = 1.0
RETRIES = 3

FIELDS = "id,filename,rnc-art,rnc-gen,kind,ssz,anno,numdec,numdep"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.italgiure.giustizia.it/sncass/",
    "Origin": "https://www.italgiure.giustizia.it",
}

SOURCE_DIR = Path("sentenze_md")
FILTERED_DIR = Path("sentenze_md_filtered")
REPORT_PATH = SOURCE_DIR / "classification_report.json"
METADATA_PATH = SOURCE_DIR / "metadata.json"

# rnc-art entries are zero-padded 4-digit article numbers, e.g. "0318 00"
# rnc-gen == "CP" means Codice Penale
# 0323 excluded: art. 323 c.p. is abuso d'ufficio (not in articoli.json).
# 323-bis/323-ter always co-occur with 318-322 in genuine corruption decisions.
TARGET_ART_PREFIXES = {"0318", "0319", "0320", "0321", "0322"}

# Section code in filename: @s60@ = sesta sezione penale.
# ssz field from Solr is unreliable (returns "0" for all); parse filename instead.
ACCEPTED_SECTIONS = {"60"}

# Pattern: explicit penal-code citation like "art. 318 c.p." or "art. 319-ter cod. pen."
CP_CITATION_RE = re.compile(
    r"art(?:icolo)?\.?\s*"
    r"(?:319[\s\-](?:bis|ter|quater)|3(?:1[89]|2[0-3]))"
    r"[\s,]*"
    r"(?:c\.?\s*p\.?|cod\.?\s*pen\.?|codice\s+penale)",
    re.IGNORECASE,
)

# Patterns derived from articoli.json titles
TITLE_KEYWORD_RES = [re.compile(p, re.IGNORECASE) for p in [
    r"corruzione\s+per\s+l[‘’’`]esercizio\s+della\s+funzione",
    r"corruzione\s+per\s+(?:un\s+)?atto\s+contrario",
    r"corruzione\s+per\s+atti\s+giudiziari",
    r"induzione\s+indebita\s+a\s+dare",
    r"corruzione\s+di\s+persona\s+incaricata",
    r"istigazione\s+alla\s+corruzione",
    r"pene\s+per\s+il\s+corruttore",
]]


def build_query(article: str) -> str:
    return (
        f'("({article})"~8 OR "{article}"~8)'
        f' AND (kind:"snpen" OR kind:"snciv")'
        f" AND anno:[2025 TO *]"
    )


def solr_search(session: requests.Session, article: str, start: int) -> dict:
    data = {
        "start": start,
        "rows": ROWS,
        "q": build_query(article),
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
            print(f"  retry {attempt+1}/{RETRIES} after {wait}s ({e})")
            time.sleep(wait)


def canonical_filename(doc: dict) -> str:
    raw = doc["filename"][0] if isinstance(doc.get("filename"), list) else doc.get("filename", "")
    name = Path(raw).name
    if ".clean." not in name:
        name = name.replace(".pdf", ".clean.pdf")
    return name


def section_code(doc: dict) -> str:
    """Extract section code from filename, e.g. '60' for sesta penale."""
    raw = doc["filename"][0] if isinstance(doc.get("filename"), list) else doc.get("filename", "")
    m = re.search(r"@s(\d+)@", Path(raw).name)
    return m.group(1) if m else ""


def doc_cites_target_articles(doc: dict) -> tuple[bool, str]:
    """Return (True, summary) if doc's rnc-art shows an art. 318-322 citation.

    For sesta penale decisions we trust the presence of the article number in
    rnc-art regardless of rnc-gen, because:
    - The Solr index sometimes omits or misaligns the CP genre tag for these arts.
    - kind==snpen + section==60 is already a strong filter against false positives.
    We still prefer CP-confirmed pairs as the primary signal.
    """
    rnc_art = doc.get("rnc-art", [])
    rnc_gen = doc.get("rnc-gen", [])

    # Primary: pairwise check — art at position i is explicitly tagged CP
    cp_matched = [a for a, g in zip(rnc_art, rnc_gen) if a[:4] in TARGET_ART_PREFIXES and g == "CP"]
    if cp_matched:
        return True, f"rnc: CP+art {cp_matched}"

    # Fallback: article present in rnc-art even without confirmed CP pairing
    # (handles rnc-gen length mismatch or missing genre tag in Solr index)
    any_matched = [a for a in rnc_art if a[:4] in TARGET_ART_PREFIXES]
    if any_matched:
        return True, f"rnc: art (no CP pair) {any_matched}"

    return False, ""


def fetch_metadata(session: requests.Session) -> dict[str, dict]:
    """Re-run all article queries and collect szdec per canonical filename."""
    metadata: dict[str, dict] = {}
    for article in ARTICLES:
        start = 0
        num_found = None
        while True:
            payload = solr_search(session, article, start)
            resp = payload.get("response", {})
            if num_found is None:
                num_found = resp.get("numFound", 0)
                print(f"[{article}] {num_found} results")
            docs = resp.get("docs", [])
            if not docs:
                break
            for doc in docs:
                fname = canonical_filename(doc)
                if fname not in metadata:
                    metadata[fname] = {**doc, "articles": []}
                metadata[fname]["articles"].append(article)
            start += len(docs)
            if start >= num_found:
                break
            time.sleep(DELAY)
    return metadata


def text_is_corruption(text: str) -> tuple[bool, str]:
    if CP_CITATION_RE.search(text):
        return True, "regex_cp_citation"
    for pat in TITLE_KEYWORD_RES:
        if pat.search(text):
            return True, "regex_title_keyword"
    return False, ""


def classify(fname: str, text: str, metadata: dict[str, dict]) -> dict:
    # Step A: empty / image-based
    if len(text.strip()) < 200:
        return {"decision": "discard", "reason": "empty", "signal": ""}

    # Step B: rnc-art + rnc-gen from SOLR (primary signal)
    md_fname = fname.replace(".md", ".pdf")
    doc = metadata.get(md_fname)
    if doc is not None:
        if doc.get("kind") != "snpen":
            return {"decision": "discard", "reason": "not_penal", "signal": doc.get("kind", "")}
        sec = section_code(doc)
        if sec not in ACCEPTED_SECTIONS:
            return {"decision": "discard", "reason": "not_sesta_penale", "signal": f"section={sec}"}
        found, signal = doc_cites_target_articles(doc)
        if found:
            return {"decision": "keep", "reason": "rnc", "signal": signal}
        else:
            return {"decision": "discard", "reason": "rnc", "signal": signal or "no target arts"}

    # Step C: regex fallback (file not in SOLR results at all)
    found, reason = text_is_corruption(text)
    return {"decision": "keep" if found else "discard", "reason": reason or "regex_no_match", "signal": ""}


def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: {SOURCE_DIR} does not exist. Run corpus/convert_sentenze.py first.")
        return

    FILTERED_DIR.mkdir(exist_ok=True)

    # Fetch SOLR metadata
    print("=== Fetching SOLR metadata ===")
    session = requests.Session()
    session.headers.update(HEADERS)
    metadata = fetch_metadata(session)

    # Save enriched metadata
    METADATA_PATH.write_text(
        json.dumps(list(metadata.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Metadata saved to {METADATA_PATH} ({len(metadata)} docs)\n")

    # Classify each markdown file
    print("=== Classifying markdown files ===")
    md_files = sorted(p for p in SOURCE_DIR.glob("*.md") if p.name != "image_based.txt")

    report = []
    counts = {"keep": 0, "discard": 0, "no_solr": 0}

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8", errors="replace")
        result = classify(md_path.name, text, metadata)

        if result["reason"] in ("regex_cp_citation", "regex_title_keyword", "regex_no_match"):
            counts["no_solr"] += 1

        report.append({"file": md_path.name, **result})

        if result["decision"] == "keep":
            counts["keep"] += 1
            shutil.copy2(md_path, FILTERED_DIR / md_path.name)
            print(f"  KEEP    {md_path.name}  [{result['reason']}] {result['signal']}")
        else:
            counts["discard"] += 1
            reason_tag = result["reason"]
            if reason_tag == "empty":
                print(f"  EMPTY   {md_path.name}")
            else:
                print(f"  DISCARD {md_path.name}  [{reason_tag}]")

    # Save report
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(md_files)
    print(f"""
=== Summary ===
Total files:     {total}
KEEP:            {counts['keep']}  (copied to {FILTERED_DIR}/)
DISCARD:         {counts['discard']}
  of which SOLR fallback (regex used): {counts['no_solr']}

Report saved to {REPORT_PATH}
""")


if __name__ == "__main__":
    main()
