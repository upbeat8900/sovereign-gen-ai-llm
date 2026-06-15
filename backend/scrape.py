import re
from typing import Callable, Optional
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - exercised only when optional browser support is absent.
    sync_playwright = None


MAX_SCRAPE_BYTES = 2 * 1024 * 1024
SCRAPE_TIMEOUT_SEC = 20
PLAYWRIGHT_TIMEOUT_MS = SCRAPE_TIMEOUT_SEC * 1000
MAX_CRAWL_PAGES = 25
MIN_RENDERED_TEXT_CHARS = 200


def _validate_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("A valid http(s) URL is required for scraping")
    return url.strip()


def _normalize_link(base_url: str, href: str, root_netloc: str) -> Optional[str]:
    if not href or href.startswith(("mailto:", "tel:", "javascript:")):
        return None
    absolute = urldefrag(urljoin(base_url, href))[0]
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != root_netloc:
        return None
    return absolute


def _soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html or "", "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "canvas", "template"]):
        element.decompose()
    return soup


def _extract_links(html: str, base_url: str, root_netloc: str) -> list[str]:
    links: list[str] = []
    for anchor in _soup(html).find_all("a", href=True):
        normalized = _normalize_link(base_url, str(anchor.get("href") or ""), root_netloc)
        if normalized and normalized not in links:
            links.append(normalized)
    return links


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tag_text(tag: Tag) -> str:
    return _clean_text(tag.get_text(" ", strip=True))


def _looks_like_spa_shell(html: str, extracted: str) -> bool:
    soup = BeautifulSoup(html or "", "html.parser")
    scripts = [str(script.get("src") or "") for script in soup.find_all("script")]
    has_script_bundle = any(src.endswith(".js") or "/assets/" in src or "/static/" in src for src in scripts)
    if not has_script_bundle:
        return False

    body_text = _clean_text(soup.body.get_text(" ", strip=True)) if soup.body else ""
    mount = (
        soup.find(id="root")
        or soup.find(id="app")
        or soup.find(attrs={"data-reactroot": True})
        or soup.find(attrs={"data-v-app": True})
    )
    mount_text = _clean_text(mount.get_text(" ", strip=True)) if mount else ""

    if mount and len(mount_text) < 80 and len(body_text) < MIN_RENDERED_TEXT_CHARS:
        return True
    return len(extracted or "") < 80 and len(body_text) < MIN_RENDERED_TEXT_CHARS and bool(scripts)


def _render_page_html(url: str, user_agent: str) -> str:
    if sync_playwright is None:
        raise RuntimeError(
            "Playwright is not installed. Install backend requirements and run `python -m playwright install chromium`."
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context(user_agent=user_agent)
            try:
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT_MS)
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                try:
                    page.wait_for_function(
                        """() => {
                            const root = document.querySelector('#root, #app, main, article') || document.body;
                            const text = root && (root.innerText || root.textContent || '').trim();
                            return (text && text.length > 200) || document.querySelector('h1, h2, p, li, table');
                        }""",
                        timeout=5000,
                    )
                except Exception:
                    pass
                return page.content()
            finally:
                context.close()
        finally:
            browser.close()


def _extract_readable_structure(html: str, url: str) -> tuple[str, dict]:
    soup = _soup(html)
    title = _clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    main = soup.find("main") or soup.find("article") or soup.body or soup
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")

    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "table"], recursive=True):
        text = _tag_text(element)
        if not text:
            continue
        name = element.name.lower()
        if name in {"h1", "h2", "h3", "h4"}:
            level = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}[name]
            lines.extend([f"{level} {text}", ""])
        elif name == "li":
            lines.append(f"- {text}")
        elif name == "blockquote":
            lines.extend([f"> {text}", ""])
        elif name == "table":
            rows = []
            for row in element.find_all("tr"):
                cells = [_tag_text(cell) for cell in row.find_all(["th", "td"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                lines.extend(["Table:", *rows, ""])
        else:
            lines.extend([text, ""])

    readable = "\n".join(lines).strip()
    if not readable:
        readable = _clean_text(soup.get_text(" ", strip=True))

    headings = [_tag_text(tag) for tag in soup.find_all(["h1", "h2", "h3"]) if _tag_text(tag)]
    return readable, {"title": title, "headings": headings[:30], "url": url}


def scrape_website_content(
    url: str,
    query: Optional[str] = None,
    depth: int = 1,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:

    normalized_url = _validate_url(url)
    crawl_depth = max(1, min(3, int(depth or 1)))
    root_netloc = urlparse(normalized_url).netloc
    headers = {"User-Agent": "SovereignGenAI/1.0 (+local research scrape)"}
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(normalized_url, 1)]
    page_results: list[dict] = []
    rendered_pages = 0
    render_errors: list[str] = []

    with httpx.Client(timeout=SCRAPE_TIMEOUT_SEC, follow_redirects=True) as client:
        while queue and len(visited) < MAX_CRAWL_PAGES:
            current_url, current_depth = queue.pop(0)
            if current_url in visited:
                continue
            visited.add(current_url)
            if progress_callback:
                progress_callback(
                    {
                        "phase": "scraping",
                        "status": "analyzing",
                        "current_url": current_url,
                        "current_depth": current_depth,
                        "pages_scraped": len(page_results),
                        "queued_pages": len(queue),
                        "extracted_chars": sum(item["extracted_chars"] for item in page_results),
                        "rendered_pages": rendered_pages,
                    }
                )
            response = client.get(current_url, headers=headers)
            response.raise_for_status()
            raw = response.content[:MAX_SCRAPE_BYTES]
            html = raw.decode(response.encoding or "utf-8", errors="replace")
            extracted, structure = _extract_readable_structure(html, current_url)
            used_rendered_dom = False
            if _looks_like_spa_shell(html, extracted):
                try:
                    rendered_html = _render_page_html(current_url, headers["User-Agent"])
                    rendered_extracted, rendered_structure = _extract_readable_structure(rendered_html, current_url)
                    if len(rendered_extracted) > len(extracted):
                        html = rendered_html
                        extracted = rendered_extracted
                        structure = rendered_structure
                        used_rendered_dom = True
                        rendered_pages += 1
                except Exception as exc:
                    render_errors.append(f"{current_url}: {exc}")
            page_results.append(
                {
                    "url": current_url,
                    "depth": current_depth,
                    "extracted": extracted,
                    "extracted_chars": len(extracted),
                    "structure": structure,
                    "rendered": used_rendered_dom,
                }
            )
            if progress_callback:
                progress_callback(
                    {
                        "phase": "scraping",
                        "status": "extracted",
                        "current_url": current_url,
                        "current_depth": current_depth,
                        "pages_scraped": len(page_results),
                        "queued_pages": len(queue),
                        "last_page_chars": len(extracted),
                        "extracted_chars": sum(item["extracted_chars"] for item in page_results),
                        "rendered_pages": rendered_pages,
                        "used_rendered_dom": used_rendered_dom,
                    }
                )
            if current_depth < crawl_depth:
                for link in _extract_links(html, current_url, root_netloc):
                    if link not in visited and all(item[0] != link for item in queue):
                        queue.append((link, current_depth + 1))

    combined = "\n\n".join(
        f"Source: {item['url']} (level {item['depth']})\n{item['extracted']}"
        for item in page_results
        if item["extracted"]
    )
    summary = _focus_extract(combined, query)
    full_content = combined.strip() or "No readable content extracted."
    return {
        "url": normalized_url,
        "query": (query or "").strip() or None,
        "depth": crawl_depth,
        "pages_scraped": len(page_results),
        "rendered_pages": rendered_pages,
        "render_errors": render_errors,
        "page_urls": [item["url"] for item in page_results],
        "structure": [item["structure"] for item in page_results],
        "extracted_chars": sum(item["extracted_chars"] for item in page_results),
        "summary": summary,
        "content_markdown": full_content,
        "raw_excerpt": full_content[:4000],
    }


def _focus_extract(text: str, query: Optional[str]) -> str:
    text = (text or "").strip()
    if not text:
        return "No readable content extracted."
    if not query or not query.strip():
        return text[:3000]
    needle = query.strip().lower()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    matched = [part for part in paragraphs if needle in part.lower()]
    if matched:
        return "\n\n".join(matched)[:3000]
    return text[:3000]
