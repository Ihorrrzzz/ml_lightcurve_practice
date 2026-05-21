from __future__ import annotations

from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
import time
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests


OGLE_LMC_CLASSES = {
    "rrlyr": "https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/rrlyr/phot/I/",
    "cep": "https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/phot/I/",
    "ecl": "https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/ecl/phot/I/",
}
NORMALIZED_COLUMNS = ["object_id", "label", "time", "mag", "mag_err", "band"]
REQUEST_TIMEOUT = (10, 120)
REQUEST_HEADERS = {"User-Agent": "ml_lightcurve_practice/0.1"}


class _LinkParser(HTMLParser):
    """Minimal href extractor for simple Apache-style directory listings."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def list_ogle_dat_files(url: str) -> list[str]:
    """Return absolute URLs for ``.dat`` files linked from an OGLE directory."""
    html = _get_text(url)

    parser = _LinkParser()
    parser.feed(html)

    dat_urls: list[str] = []
    seen: set[str] = set()
    for href in parser.links:
        href_path = href.split("?", 1)[0].split("#", 1)[0]
        if not href_path.lower().endswith(".dat"):
            continue
        file_url = urljoin(url, href)
        if file_url not in seen:
            dat_urls.append(file_url)
            seen.add(file_url)

    return dat_urls


def download_ogle_lightcurve(file_url: str, label: str) -> pd.DataFrame:
    """Download one OGLE I-band photometry file and normalize it."""
    text = _get_text(file_url)

    photometry = pd.read_csv(
        StringIO(text),
        sep=r"\s+",
        header=None,
        names=["time", "mag", "mag_err"],
        usecols=[0, 1, 2],
        comment="#",
    )
    for column in ("time", "mag", "mag_err"):
        photometry[column] = pd.to_numeric(photometry[column], errors="coerce")
    photometry = photometry.dropna(subset=["time", "mag", "mag_err"]).copy()

    filename = Path(unquote(urlparse(file_url).path)).name
    object_id = filename.removesuffix(".dat")

    photometry["object_id"] = object_id
    photometry["label"] = label
    photometry["band"] = "I"

    return photometry.loc[:, NORMALIZED_COLUMNS]


def _get_text(url: str, max_attempts: int = 3) -> str:
    """Fetch text from a public OGLE URL with a small retry loop."""
    last_error: requests.RequestException | None = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(2.0 * (attempt + 1))

    assert last_error is not None
    raise last_error


def build_ogle_lmc_sample(
    n_per_class: int = 30,
    output_path: str | Path = "data/processed/real_lightcurves.csv",
) -> pd.DataFrame:
    """Build a small reproducible OGLE-IV LMC I-band sample.

    This downloads only the first ``n_per_class`` light curves for each selected
    class. It is an MVP sample for testing the pipeline, not the full OGLE
    catalog.
    """
    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive.")

    parts: list[pd.DataFrame] = []
    for label, url in OGLE_LMC_CLASSES.items():
        file_urls = list_ogle_dat_files(url)
        for file_url in file_urls[:n_per_class]:
            parts.append(download_ogle_lightcurve(file_url, label=label))

    if not parts:
        raise RuntimeError("No OGLE light curves were downloaded.")

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.loc[:, NORMALIZED_COLUMNS].sort_values(["object_id", "time"]).reset_index(drop=True)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)

    return sample
