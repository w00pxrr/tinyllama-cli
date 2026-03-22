#!/usr/bin/env python3
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


def should_search_web(user_input: str) -> bool:
    text = user_input.lower().strip()
    if text.startswith("/web "):
        return True

    hints = (
        "latest",
        "today",
        "current",
        "currently",
        "recent",
        "news",
        "search the web",
        "look it up",
        "online",
        "on the web",
        "what happened",
        "stock price",
        "weather",
    )
    return any(hint in text for hint in hints)


def normalize_query(user_input: str) -> str:
    text = user_input.strip()
    if text.lower().startswith("/web "):
        return text[5:].strip()
    return text


def _strip_tags(value: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", value)
    return html.unescape(re.sub(r"\s+", " ", no_tags)).strip()


def _decode_duckduckgo_url(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/l/?uddg="):
        encoded = url.split("uddg=", 1)[1].split("&", 1)[0]
        return unquote(encoded)
    return url


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url


def _parse_results(page: str, limit: int) -> list[WebResult]:
    matches = list(
        re.finditer(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            page,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    results: list[WebResult] = []
    for idx, match in enumerate(matches):
        if len(results) >= limit:
            break
        title = _strip_tags(match.group(2))
        url = _decode_duckduckgo_url(html.unescape(match.group(1)))
        window = page[match.end() : matches[idx + 1].start() if idx + 1 < len(matches) else match.end() + 1600]
        snippet_match = re.search(
            r'(?:result__snippet|result-snippet)[^>]*>(.*?)</',
            window,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippet = _strip_tags(snippet_match.group(1)) if snippet_match else ""
        if title and url.startswith(("http://", "https://")):
            results.append(WebResult(title=title, url=url, snippet=snippet))
    return results


def search_web(query: str, limit: int = 5) -> list[WebResult]:
    url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=15) as resp:
        page = resp.read().decode("utf-8", errors="replace")
    return _parse_results(page, limit=limit)


def format_web_context(results: Iterable[WebResult]) -> str:
    lines = []
    for idx, result in enumerate(results, start=1):
        line = f"[{idx}] {result.title} - {_domain(result.url)}"
        if result.snippet:
            line += f" :: {result.snippet}"
        line += f" (URL: {result.url})"
        lines.append(line)
    return "\n".join(lines)
