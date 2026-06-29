"""Robust scraper for Tamil Nadu Agriculture - Farmers Welfare schemes."""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from requests import Response, Session
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import AppConfig, load_config


LOGGER = logging.getLogger(__name__)
APPROVED_HOSTS = {"tn.gov.in", "www.tn.gov.in"}
DEPARTMENT = "Agriculture - Farmers Welfare Department"
SCHEME_FIELDS = [
    "scheme_id",
    "scheme_name",
    "department",
    "category",
    "description",
    "objective",
    "benefits",
    "eligibility",
    "documents_required",
    "application_process",
    "contact_information",
    "scheme_detail_url",
    "source_list_url",
    "scraped_at",
    "raw_text",
]
DETAIL_HINTS = {"scheme", "benefit", "subsidy", "agriculture", "farmer", "dept_id", "scheme_id"}
UNWANTED_LINK_TEXT = {
    "accessibility menu",
    "skip to main content",
    "screen reader access",
    "increase font size",
    "decrease font size",
    "default font size",
    "high contrast",
    "normal contrast",
    "english",
    "tamil",
    "home",
}
HOMEPAGE_MARKERS = {
    "tourism",
    "documents",
    "press release",
    "forms",
    "visitor count",
    "chief minister",
}
UNWANTED_SELECTORS = [
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    ".breadcrumb",
    ".breadcrumbs",
    ".menu",
    ".navbar",
    ".footer",
    ".header",
    ".accessibility",
    "#accessibility",
]


def create_session() -> Session:
    """Create a persistent session with transparent browser-like headers."""

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "TN-Agriculture-Schemes-RAG/1.0 "
                "(educational retrieval app; contact: local developer)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9,ta;q=0.8",
            "Referer": "https://www.tn.gov.in/schemes.php",
        }
    )
    return session


def _retry_for(config: AppConfig):
    return retry(
        reraise=True,
        stop=stop_after_attempt(config.request_retries),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
                requests.exceptions.SSLError,
            )
        ),
    )


def fetch_page(session: Session, url: str, config: AppConfig) -> Response:
    """Fetch a page with retries and HTTP error handling."""

    @_retry_for(config)
    def _fetch() -> Response:
        response = session.get(url, timeout=config.request_timeout, allow_redirects=True)
        if response.status_code >= 500:
            response.raise_for_status()
        if response.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTP {response.status_code} while fetching {url}",
                response=response,
            )
        if not response.text.strip():
            raise ValueError(f"Empty HTML returned for {url}")
        return response

    return _fetch()


def is_unwanted_redirect(response: Response, requested_url: str) -> bool:
    """Detect redirects from the scheme list to the generic Tamil Nadu homepage."""

    final = _canonical_url(response.url)
    requested = _canonical_url(requested_url)
    homepage = "https://www.tn.gov.in"
    return final != requested and final.rstrip("/") in {homepage, "https://tn.gov.in"}


def is_homepage_content(html: str) -> bool:
    """Detect common homepage text so it is never indexed as scheme data."""

    soup = BeautifulSoup(html, "lxml")
    text = clean_text(soup.get_text(" "))
    lowered = text.lower()
    marker_count = sum(1 for marker in HOMEPAGE_MARKERS if marker in lowered)
    has_scheme_signal = "agriculture" in lowered and "scheme" in lowered
    return marker_count >= 3 and not has_scheme_signal


def clean_text(text: str | None) -> str:
    """Normalize repeated whitespace while preserving Tamil and English Unicode."""

    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def extract_scheme_links(html: str, base_url: str) -> list[dict[str, str]]:
    """Find probable scheme links using tables, lists, headings, cards, and anchors."""

    soup = BeautifulSoup(html, "lxml")
    _clean_soup(soup)
    candidates: list[dict[str, str]] = []

    content_nodes = soup.select("main, article, section, .content, .maincontent, table, ul, ol")
    if not content_nodes:
        content_nodes = [soup.body or soup]

    for node in content_nodes:
        for anchor in node.find_all("a", href=True):
            name = clean_text(anchor.get_text(" "))
            href = urljoin(base_url, anchor["href"])
            if not name or _is_unwanted_navigation_text(name) or not _is_approved_url(href):
                continue
            lowered = f"{name} {href}".lower()
            if "scheme" in lowered or any(hint in lowered for hint in DETAIL_HINTS):
                candidates.append({"scheme_name": name, "scheme_detail_url": href})

    if not candidates:
        for row in soup.select("tr"):
            cells = [clean_text(cell.get_text(" ")) for cell in row.find_all(["td", "th"])]
            joined = " ".join(cells)
            link = row.find("a", href=True)
            if joined and "scheme" in joined.lower():
                candidates.append(
                    {
                        "scheme_name": clean_text(link.get_text(" ")) if link else joined,
                        "scheme_detail_url": urljoin(base_url, link["href"]) if link else base_url,
                    }
                )

    if not candidates:
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "li"]):
            text = clean_text(heading.get_text(" "))
            if text and "scheme" in text.lower():
                candidates.append({"scheme_name": text, "scheme_detail_url": base_url})

    return _dedupe_link_candidates(candidates)


def extract_scheme_details(
    html: str,
    list_url: str,
    detail_url: str,
    fallback_name: str,
    scraped_at: str,
) -> dict[str, Any] | None:
    """Extract a normalized scheme record from a list or detail page."""

    soup = BeautifulSoup(html, "lxml")
    _clean_soup(soup)
    title = _extract_title(soup)
    if not title or _is_unwanted_navigation_text(title):
        title = fallback_name
    raw_text = clean_text(soup.get_text(" "))
    if not title or not raw_text or is_homepage_content(html):
        return None

    sections = _extract_labelled_sections(soup)
    record = {
        "scheme_id": _extract_scheme_id(detail_url),
        "scheme_name": title,
        "department": DEPARTMENT,
        "category": sections.get("category", ""),
        "description": sections.get("description", _first_sentences(raw_text)),
        "objective": sections.get("objective", ""),
        "benefits": sections.get("benefits", ""),
        "eligibility": sections.get("eligibility", ""),
        "documents_required": sections.get("documents_required", ""),
        "application_process": sections.get("application_process", ""),
        "contact_information": sections.get("contact_information", ""),
        "scheme_detail_url": detail_url,
        "source_list_url": list_url,
        "scraped_at": scraped_at,
        "raw_text": raw_text,
    }
    normalized = normalize_scheme(record)
    return normalized or None


def normalize_scheme(record: dict[str, Any]) -> dict[str, Any]:
    """Ensure every scheme field exists and text fields are cleaned."""

    normalized: dict[str, Any] = {}
    for field in SCHEME_FIELDS:
        value = record.get(field, "")
        normalized[field] = clean_text(str(value)) if value is not None else ""
    if not normalized["scheme_name"] or len(normalized["raw_text"]) < 25:
        return {}
    return normalized


def deduplicate_schemes(schemes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by normalized name and canonical source URL."""

    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for scheme in schemes:
        key = (
            re.sub(r"\W+", "", scheme.get("scheme_name", "").lower()),
            _canonical_url(scheme.get("scheme_detail_url") or scheme.get("source_list_url", "")),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        unique.append(scheme)
    return unique


def _csv_safe(value: Any) -> str:
    """Neutralize spreadsheet (CSV) formula injection.

    A value beginning with =, +, -, @, tab, or carriage return can be executed
    as a formula by Excel/LibreOffice. Prefix such values with a single quote so
    the cell is treated as text while staying human-readable. JSON keeps the
    exact source value; only the CSV export is sanitized.
    """

    text = "" if value is None else str(value)
    if text and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text


def save_schemes(schemes: list[dict[str, Any]], config: AppConfig) -> None:
    """Persist JSON and CSV copies without writing empty data."""

    if not schemes:
        raise ValueError("Refusing to overwrite existing files with empty scheme data.")

    config.data_directory.mkdir(parents=True, exist_ok=True)
    json_tmp = config.schemes_json_path.with_suffix(".json.tmp")
    csv_tmp = config.schemes_csv_path.with_suffix(".csv.tmp")

    json_tmp.write_text(
        json.dumps(schemes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEME_FIELDS)
        writer.writeheader()
        for scheme in schemes:
            writer.writerow({field: _csv_safe(scheme.get(field, "")) for field in SCHEME_FIELDS})

    json_tmp.replace(config.schemes_json_path)
    csv_tmp.replace(config.schemes_csv_path)


def load_existing_schemes(config: AppConfig | None = None) -> list[dict[str, Any]]:
    """Load previously saved schemes when available."""

    config = config or load_config()
    if not config.schemes_json_path.exists():
        return []
    try:
        return json.loads(config.schemes_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        LOGGER.exception("Failed to load existing scheme data.")
        return []


def scrape_all_schemes(config: AppConfig | None = None, save: bool = True) -> dict[str, Any]:
    """Scrape the source page and detail pages, preserving old data on failure."""

    config = config or load_config()
    scraped_at = datetime.now(timezone.utc).isoformat()
    warnings: list[str] = []
    errors: list[str] = []
    schemes: list[dict[str, Any]] = []

    session = create_session()
    try:
        fetch_page(session, config.schemes_landing_url, config)
        response = fetch_page(session, config.source_url, config)

        if is_unwanted_redirect(response, config.source_url):
            warning = "The target scheme page redirected to the Tamil Nadu homepage."
            warnings.append(warning)
            raise RuntimeError(warning)

        if is_homepage_content(response.text):
            warning = "The target page appears to contain homepage content, not scheme data."
            warnings.append(warning)
            raise RuntimeError(warning)

        links = extract_scheme_links(response.text, response.url)
        if not links:
            warnings.append("No detail links found; attempting to parse the list page itself.")
            page_scheme = extract_scheme_details(
                response.text,
                config.source_url,
                response.url,
                "Tamil Nadu Agriculture Schemes",
                scraped_at,
            )
            if page_scheme:
                schemes.append(page_scheme)

        for link in links:
            detail_url = link["scheme_detail_url"]
            if not _is_approved_url(detail_url):
                warnings.append(f"Skipped unapproved detail URL: {detail_url}")
                continue
            try:
                time.sleep(config.request_delay_seconds)
                detail_response = fetch_page(session, detail_url, config)
                if is_homepage_content(detail_response.text):
                    warnings.append(f"Skipped homepage-like detail content: {detail_url}")
                    continue
                scheme = extract_scheme_details(
                    detail_response.text,
                    config.source_url,
                    detail_response.url,
                    link["scheme_name"],
                    scraped_at,
                )
                if scheme:
                    schemes.append(scheme)
            except Exception as exc:  # noqa: BLE001 - scraper should continue per link
                LOGGER.exception("Failed to scrape detail page %s", detail_url)
                warnings.append(f"Could not read detail page for {link['scheme_name']}: {exc}")

        schemes = deduplicate_schemes(schemes)
        if not schemes:
            raise RuntimeError("No valid schemes were extracted.")

        if save:
            save_schemes(schemes, config)

        return {
            "success": True,
            "schemes": schemes,
            "scheme_count": len(schemes),
            "scraped_at": scraped_at,
            "warnings": warnings,
            "errors": errors,
        }
    except Exception as exc:  # noqa: BLE001 - return structured UI-safe result
        LOGGER.exception("Scraping failed.")
        errors.append(str(exc))
        existing = load_existing_schemes(config)
        if existing:
            warnings.append("Using the last valid local dataset because refresh failed.")
        return {
            "success": False,
            "schemes": existing,
            "scheme_count": len(existing),
            "scraped_at": existing[0].get("scraped_at", "") if existing else "",
            "warnings": warnings,
            "errors": errors,
        }


def schemes_to_dataframe(schemes: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a stable dataframe for Streamlit browsing."""

    return pd.DataFrame(schemes, columns=SCHEME_FIELDS)


def _clean_soup(soup: BeautifulSoup | Tag) -> BeautifulSoup | Tag:
    for selector in UNWANTED_SELECTORS:
        for node in soup.select(selector):
            node.decompose()
    return soup


def _extract_title(soup: BeautifulSoup) -> str:
    candidates = []
    for selector in ["h1", "h2", "h3", ".page-title", ".title", "caption"]:
        candidates.extend(clean_text(node.get_text(" ")) for node in soup.select(selector))
    for candidate in candidates:
        lowered = candidate.lower()
        if (
            candidate
            and "government of tamil nadu" not in lowered
            and not _is_unwanted_navigation_text(candidate)
            and len(candidate) < 220
        ):
            return candidate
    return ""


def _is_unwanted_navigation_text(text: str) -> bool:
    lowered = clean_text(text).lower()
    return lowered in UNWANTED_LINK_TEXT or any(
        marker in lowered for marker in ("accessibility", "skip to", "screen reader")
    )


def _extract_labelled_sections(soup: BeautifulSoup) -> dict[str, str]:
    text = clean_text(soup.get_text("\n"))
    mapping = {
        "objective": ["objective", "objectives", "aim"],
        "benefits": ["benefit", "benefits", "assistance", "subsidy"],
        "eligibility": ["eligibility", "eligible"],
        "documents_required": ["documents required", "documents", "certificate"],
        "application_process": ["how to apply", "application", "procedure", "apply"],
        "contact_information": ["contact", "office", "address", "phone", "email"],
        "category": ["category"],
        "description": ["description", "details"],
    }
    sections: dict[str, str] = {}

    for key, labels in mapping.items():
        for label in labels:
            pattern = re.compile(
                rf"{re.escape(label)}\s*:?\s*(.+?)(?=\n[A-Za-z][A-Za-z /()_-]{{2,40}}\s*:|\Z)",
                re.IGNORECASE | re.DOTALL,
            )
            match = pattern.search(text)
            if match:
                sections[key] = clean_text(match.group(1))
                break

    for row in soup.select("tr"):
        cells = [clean_text(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
        if len(cells) >= 2:
            label = cells[0].lower()
            value = clean_text(" ".join(cells[1:]))
            for key, labels in mapping.items():
                if key not in sections and any(item in label for item in labels):
                    sections[key] = value
    return sections


def _first_sentences(text: str, limit: int = 550) -> str:
    return text[:limit].strip()


def _extract_scheme_id(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for key in ("scheme_id", "id", "sid"):
        if key in query and query[key]:
            return query[key][0]
    slug = Path(parsed.path).stem
    return slug if slug not in {"", "scheme_list"} else ""


def _is_approved_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in APPROVED_HOSTS or host.endswith(".tn.gov.in")


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    clean = parsed._replace(fragment="", query=parsed.query.rstrip("&"))
    return urlunparse(clean).rstrip("/")


def _dedupe_link_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for candidate in candidates:
        name = clean_text(candidate.get("scheme_name", ""))
        url = _canonical_url(candidate.get("scheme_detail_url", ""))
        key = (name.lower(), url)
        if not name or not url or key in seen:
            continue
        seen.add(key)
        unique.append({"scheme_name": name, "scheme_detail_url": url})
    return unique
