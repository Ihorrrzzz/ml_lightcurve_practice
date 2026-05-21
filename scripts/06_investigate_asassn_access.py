from __future__ import annotations

import html
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urljoin

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "asassn_v"

CATALOGS = {
    "RRAB": RAW_DIR / "asassn_rrab_catalog.csv",
    "DCEP": RAW_DIR / "asassn_dcep_catalog.csv",
    "EW": RAW_DIR / "asassn_ew_catalog.csv",
}

KNOWN_OBJECT_URL = "https://asas-sn.osu.edu/variables/885aae98-102b-5794-a60a-e2c0b55ec984"
SKYPATROL_DOCUMENTATION_URL = "https://asas-sn.ifa.hawaii.edu/documentation"
DEBUG_OBJECT_HTML = RAW_DIR / "debug_object_page.html"
DEBUG_SKYPATROL_HTML = RAW_DIR / "debug_skypatrol_documentation.html"

SEARCH_TERMS = ("csv", "download", "phot", "light", "curve", "flux", "mag", "api", "json", "ajax", "plot")
REQUEST_HEADERS = {
    "User-Agent": "ml-lightcurve-practice/0.1 investigation script",
    "Accept": "text/html,application/json,text/csv,*/*;q=0.8",
}


class LinkAndScriptParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links: list[tuple[str, str]] = []
        self.scripts: list[str] = []
        self._in_script = False
        self._script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value for key, value in attrs if value is not None}
        if tag == "a" and "href" in attrs_dict:
            self.links.append(("a", urljoin(self.base_url, attrs_dict["href"])))
        if tag == "link" and "href" in attrs_dict:
            self.links.append(("link", urljoin(self.base_url, attrs_dict["href"])))
        if tag == "script":
            if "src" in attrs_dict:
                self.links.append(("script", urljoin(self.base_url, attrs_dict["src"])))
            self._in_script = True
            self._script_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script:
            script_text = "".join(self._script_parts).strip()
            if script_text:
                self.scripts.append(script_text)
            self._in_script = False
            self._script_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_parts.append(data)


def snippet(text: str, max_len: int = 220) -> str:
    clean = re.sub(r"\s+", " ", html.unescape(str(text))).strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def has_search_term(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in SEARCH_TERMS)


def fetch_url(session: requests.Session, url: str, timeout: int = 25) -> requests.Response | None:
    try:
        response = session.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        return response
    except requests.RequestException as exc:
        print(f"  ERROR fetching {url}: {exc}")
        return None


def load_catalogs() -> dict[str, pd.DataFrame]:
    catalogs = {}
    print("Catalog inputs")
    for class_name, path in CATALOGS.items():
        if not path.exists():
            raise SystemExit(f"Missing catalog: {path}")
        df = pd.read_csv(path)
        catalogs[class_name] = df
        print(f"- {class_name}: {path}")
        print(f"  shape: {df.shape}")
        print(f"  columns: {', '.join(df.columns)}")
    print()
    return catalogs


def select_sample(catalogs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    samples = []
    wanted_columns = ["source_id", "asassn_name", "raj2000", "dej2000", "variable_type", "class_probability"]

    print("Selected investigation sample")
    for class_name, df in catalogs.items():
        available = [col for col in wanted_columns if col in df.columns]
        sub = df.loc[:, available].copy()
        sub["class_probability"] = pd.to_numeric(sub["class_probability"], errors="coerce")
        high_conf = sub[sub["class_probability"] >= 0.95]
        chosen = high_conf if len(high_conf) >= 2 else sub
        chosen = chosen.sort_values("class_probability", ascending=False, na_position="last").head(2)
        chosen.insert(0, "catalog", class_name)
        samples.append(chosen)

        print(f"- {class_name}:")
        for _, row in chosen.iterrows():
            print(
                "  "
                f"source_id={row.get('source_id')}, "
                f"asassn_name={row.get('asassn_name')}, "
                f"raj2000={row.get('raj2000')}, "
                f"dej2000={row.get('dej2000')}, "
                f"variable_type={row.get('variable_type')}, "
                f"class_probability={row.get('class_probability')}"
            )

    print()
    return pd.concat(samples, ignore_index=True)


def clean_source_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def print_candidate_links(title: str, links: Iterable[tuple[str, str]], max_items: int = 40) -> list[str]:
    candidates = []
    for tag, url in links:
        if has_search_term(url):
            candidates.append(f"{tag}: {url}")

    print(title)
    if not candidates:
        print("- no candidate links found")
        return []

    for item in candidates[:max_items]:
        print(f"- {item}")
    if len(candidates) > max_items:
        print(f"- ... {len(candidates) - max_items} more omitted")
    return candidates


def print_candidate_strings(title: str, text: str, max_items: int = 30) -> list[str]:
    lines = []
    for term in SEARCH_TERMS:
        for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
            start = max(0, match.start() - 90)
            end = min(len(text), match.end() + 140)
            lines.append(snippet(text[start:end]))

    deduped = []
    seen = set()
    for line in lines:
        if line and line not in seen:
            deduped.append(line)
            seen.add(line)

    print(title)
    if not deduped:
        print("- no matching strings found")
        return []

    for line in deduped[:max_items]:
        print(f"- {line}")
    if len(deduped) > max_items:
        print(f"- ... {len(deduped) - max_items} more omitted")
    return deduped


def inspect_embedded_data(html_text: str, scripts: list[str]) -> None:
    print("Embedded-data scan")

    json_like_blocks = re.findall(
        r"<script[^>]+type=[\"']application/(?:json|ld\+json)[\"'][^>]*>(.*?)</script>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if json_like_blocks:
        print(f"- found {len(json_like_blocks)} application/json-like script block(s)")
        for block in json_like_blocks[:3]:
            block = html.unescape(block.strip())
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    print(f"  JSON keys: {list(parsed.keys())[:20]}")
                else:
                    print(f"  JSON type: {type(parsed).__name__}")
            except json.JSONDecodeError:
                print(f"  non-parseable JSON-like block: {snippet(block)}")
    else:
        print("- no application/json script blocks found")

    likely_scripts = [script for script in scripts if has_search_term(script)]
    if likely_scripts:
        print(f"- found {len(likely_scripts)} inline script(s) with candidate terms")
        for script in likely_scripts[:5]:
            print(f"  {snippet(script, max_len=320)}")
    else:
        print("- no inline scripts with light-curve/download/API terms")

    coordinate_terms = ("hjd", "jd", "mjd", "mag", "flux", "photometry")
    possible_arrays = []
    for term in coordinate_terms:
        if re.search(rf"\b{term}\b", html_text, flags=re.IGNORECASE):
            possible_arrays.append(term)
    if possible_arrays:
        print(f"- page contains data-like terms: {', '.join(sorted(set(possible_arrays)))}")
    else:
        print("- no obvious embedded time/magnitude array terms")
    print()


def inspect_object_page(session: requests.Session) -> None:
    print("Known object page fetch")
    print(f"- URL: {KNOWN_OBJECT_URL}")
    response = fetch_url(session, KNOWN_OBJECT_URL)
    if response is None:
        print()
        return

    DEBUG_OBJECT_HTML.write_text(response.text, encoding="utf-8")
    print(f"- status: {response.status_code}")
    print(f"- content-type: {response.headers.get('content-type', '')}")
    print(f"- saved: {DEBUG_OBJECT_HTML.relative_to(PROJECT_ROOT)}")

    parser = LinkAndScriptParser(KNOWN_OBJECT_URL)
    parser.feed(response.text)
    print_candidate_links("Candidate object-page links/scripts", parser.links)
    print_candidate_strings("Candidate object-page strings", response.text)
    inspect_embedded_data(response.text, parser.scripts)


def inspect_skypatrol_docs(session: requests.Session) -> None:
    print("Sky Patrol documentation fetch")
    print(f"- URL: {SKYPATROL_DOCUMENTATION_URL}")
    response = fetch_url(session, SKYPATROL_DOCUMENTATION_URL)
    if response is None:
        print()
        return

    DEBUG_SKYPATROL_HTML.write_text(response.text, encoding="utf-8")
    print(f"- status: {response.status_code}")
    print(f"- final URL: {response.url}")
    print(f"- content-type: {response.headers.get('content-type', '')}")
    print(f"- saved: {DEBUG_SKYPATROL_HTML.relative_to(PROJECT_ROOT)}")

    parser = LinkAndScriptParser(response.url)
    parser.feed(response.text)
    print_candidate_links("Candidate documentation links/scripts", parser.links)
    print_candidate_strings("Candidate documentation strings", response.text)

    package_candidates = sorted(
        set(re.findall(r"(?:pip install|import)\s+([A-Za-z0-9_.-]*asas[A-Za-z0-9_.-]*)", response.text, flags=re.IGNORECASE))
    )
    endpoint_candidates = sorted(set(re.findall(r"https?://[^\"'<>\\\s]+(?:api|light|phot|curve|csv|json)[^\"'<>\\\s]*", response.text, flags=re.IGNORECASE)))

    print("Documentation API/package hints")
    if package_candidates:
        for package in package_candidates:
            print(f"- package/import hint: {package}")
    else:
        print("- no package/import hint matching '*asas*' found on this page")

    if endpoint_candidates:
        for endpoint in endpoint_candidates[:20]:
            print(f"- endpoint-like URL: {endpoint}")
    else:
        print("- no endpoint-like absolute URL found on this page")
    print()


def probe_candidate_endpoints(session: requests.Session) -> list[str]:
    print("Small known-object endpoint probe")
    uuid = KNOWN_OBJECT_URL.rstrip("/").split("/")[-1]
    candidates = [
        f"{KNOWN_OBJECT_URL}.json",
        f"{KNOWN_OBJECT_URL}.csv",
        f"{KNOWN_OBJECT_URL}/download",
        f"{KNOWN_OBJECT_URL}/data",
        f"{KNOWN_OBJECT_URL}/photometry",
        f"{KNOWN_OBJECT_URL}/light_curve",
        f"{KNOWN_OBJECT_URL}/light_curve.csv",
        f"https://asas-sn.osu.edu/variables/{uuid}/download",
    ]

    usable = []
    for url in candidates:
        time.sleep(0.5)
        response = fetch_url(session, url, timeout=20)
        if response is None:
            continue
        content_type = response.headers.get("content-type", "")
        body_start = snippet(response.text[:500])
        looks_usable = (
            response.status_code == 200
            and (
                "json" in content_type.lower()
                or "csv" in content_type.lower()
                or re.search(r"\b(time|hjd|jd|mag|flux)\b", response.text[:2000], flags=re.IGNORECASE)
            )
            and "html" not in content_type.lower()
        )
        print(f"- {url}")
        print(f"  status={response.status_code}, content-type={content_type}, bytes={len(response.content)}")
        print(f"  first text: {body_start}")
        if looks_usable:
            usable.append(url)

    if usable:
        print("Usable-looking endpoint(s) found:")
        for url in usable:
            print(f"- {url}")
    else:
        print("No direct CSV/JSON light-curve endpoint found by these simple probes.")
    print()
    return usable


def probe_catalog_source_endpoints(session: requests.Session, sample: pd.DataFrame) -> list[str]:
    print("Catalog source_id endpoint probe")
    print("- probing first selected object per class, using source_id from local catalog")

    usable = []
    probe_rows = sample.groupby("catalog", sort=False).head(1)
    for _, row in probe_rows.iterrows():
        source_id = clean_source_id(row["source_id"])
        for suffix in (".csv", ".json"):
            url = f"https://asas-sn.osu.edu/variables/{quote(source_id, safe='')}{suffix}"
            time.sleep(0.5)
            response = fetch_url(session, url, timeout=20)
            if response is None:
                continue
            content_type = response.headers.get("content-type", "")
            body_start = snippet(response.text[:320])
            looks_usable = (
                response.status_code == 200
                and ("csv" in content_type.lower() or "json" in content_type.lower())
                and re.search(r"\b(hjd|mag|flux)\b", response.text[:1500], flags=re.IGNORECASE)
            )
            print(f"- catalog={row['catalog']}, source_id={source_id}, url={url}")
            print(f"  status={response.status_code}, content-type={content_type}, bytes={len(response.content)}")
            print(f"  first text: {body_start}")
            if looks_usable:
                usable.append(url)

    if usable:
        print("Usable catalog source_id endpoint(s) found:")
        for url in usable:
            print(f"- {url}")
    else:
        print("No usable source_id CSV/JSON endpoint found in this small probe.")
    print()
    return usable


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    catalogs = load_catalogs()
    sample = select_sample(catalogs)

    session = requests.Session()
    inspect_object_page(session)
    time.sleep(0.75)
    inspect_skypatrol_docs(session)
    known_usable = probe_candidate_endpoints(session)
    catalog_usable = probe_catalog_source_endpoints(session, sample)
    usable = catalog_usable or known_usable

    print("Investigation conclusion")
    if usable:
        print("- Found usable-looking direct CSV/JSON light-curve endpoints.")
        if catalog_usable:
            print("- The local catalog source_id appears to map directly to /variables/{source_id}.csv and /variables/{source_id}.json.")
            print("- Recommended next step: implement a downloader that loops over selected source_id values and reads:")
            print("  https://asas-sn.osu.edu/variables/{source_id}.csv")
            print("  or:")
            print("  https://asas-sn.osu.edu/variables/{source_id}.json")
        else:
            print("- A known object UUID endpoint worked, but source_id mapping was not confirmed.")
            print(f"- Recommended next step: inspect {usable[0]} before implementing the downloader.")
    else:
        print("- Did not find a confirmed direct CSV/JSON light-curve endpoint from the variable object page probes.")
        print("- If the Sky Patrol documentation mentions a Python package or API, use that documented access path next.")
        print("- Otherwise, the next implementation step is to inspect the saved HTML/debug strings and any documented package/API examples before writing the full downloader.")


if __name__ == "__main__":
    main()
