"""
Free, keyless research sources for the forecasting bot.

Everything here is best-effort: every function catches its own errors and
returns an empty result, so a dead source never blocks a forecast.

Sources:
- Google News RSS search (dated headlines)
- GDELT DOC 2.0 API (dated headlines, max 1 request / 5 s)
- Wikipedia API (intro extract of the best-matching article)
- URLs named in the question's resolution criteria / fine print / background
  (plain-text page extract; FRED series fetched as keyless CSV)
- Prediction markets: Polymarket gamma API, Manifold API, Kalshi public search
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

logger = logging.getLogger(__name__)

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 metaculus-forecast-bot"
)
TIMEOUT = httpx.Timeout(20.0, connect=10.0)

_STOP = set(
    """a an the of in on at to for by with from and or not no will would be is are was were
    this that these those it its as than then before after between during until what which
    who whom how when where why any all each more most less least over under above below
    into out up down yes resolve resolves resolved question date dates value values number
    per according reported report reports official least greater higher lower 2025 2026 2027
    2028 2029 2030 january february march april may june july august september october
    november december""".split()
)


def _words(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9][a-z0-9\-\.]+", (text or "").lower())
        if w not in _STOP and len(w) > 2
    }


def relevance(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": UA}, timeout=TIMEOUT, follow_redirects=True
    )


def _fmt_date(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d") if dt else "date?"


def _parse_iso(s) -> datetime | None:
    if s is None:
        return None
    try:
        if isinstance(s, (int, float)):
            # Manifold uses ms epoch
            return datetime.fromtimestamp(s / 1000 if s > 1e11 else s, tz=timezone.utc)
        s = str(s).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


# --------------------------------------------------------------------- news


async def google_news(query: str, max_items: int = 8, days: int | None = None) -> list[dict]:
    q = query + (f" when:{days}d" if days else "")
    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote_plus(q)
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        async with _client() as c:
            r = await c.get(url)
            r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            src = (it.findtext("source") or "").strip()
            pub = it.findtext("pubDate")
            dt = None
            try:
                dt = parsedate_to_datetime(pub) if pub else None
            except Exception:
                pass
            items.append({"title": title, "source": src, "date": dt, "via": "GoogleNews"})
        items.sort(key=lambda x: x["date"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return items[:max_items]
    except Exception as e:
        logger.warning(f"Google News failed for '{query}': {e}")
        return []


_gdelt_lock: asyncio.Lock | None = None
_gdelt_last = 0.0


async def gdelt(query: str, max_items: int = 8, timespan: str = "30d") -> list[dict]:
    global _gdelt_lock, _gdelt_last
    if _gdelt_lock is None:
        _gdelt_lock = asyncio.Lock()
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 2][:6]
    if not words:
        return []
    q = " ".join(words)
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?query="
        + urllib.parse.quote(q)
        + f"&mode=artlist&maxrecords={max_items}&format=json&timespan={timespan}&sort=datedesc&sourcelang=english"
    )
    try:
        async with _gdelt_lock:
            wait = 5.5 - (time.time() - _gdelt_last)
            if wait > 0:
                await asyncio.sleep(wait)
            async with _client() as c:
                r = await c.get(url)
            _gdelt_last = time.time()
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return []
        arts = json.loads(r.text).get("articles", [])
        out = []
        for a in arts:
            dt = None
            try:
                dt = datetime.strptime(a.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except Exception:
                pass
            out.append({"title": a.get("title", ""), "source": a.get("domain", ""), "date": dt, "via": "GDELT"})
        return out
    except Exception as e:
        logger.warning(f"GDELT failed for '{query}': {e}")
        return []


async def wikipedia(query: str, max_chars: int = 1500) -> str:
    try:
        async with _client() as c:
            r = await c.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
            )
            hits = r.json().get("query", {}).get("search", [])
            if not hits:
                return ""
            title = hits[0]["title"]
            r = await c.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "prop": "extracts|revisions", "exintro": 1, "explaintext": 1,
                    "titles": title, "format": "json", "rvprop": "timestamp",
                },
            )
            pages = r.json().get("query", {}).get("pages", {})
            for p in pages.values():
                ext = (p.get("extract") or "").strip()
                rev = (p.get("revisions") or [{}])[0].get("timestamp", "")[:10]
                if ext:
                    return f"Wikipedia '{title}' (last edited {rev}): {ext[:max_chars]}"
        return ""
    except Exception as e:
        logger.warning(f"Wikipedia failed for '{query}': {e}")
        return ""


# ---------------------------------------------------------- resolution URLs

URL_RE = re.compile(r"https?://[^\s\)\]\}\"'<>]+")


def extract_urls(*texts: str) -> list[str]:
    seen, out = set(), []
    for t in texts:
        for u in URL_RE.findall(t or ""):
            u = u.rstrip(".,;:")
            if "metaculus.com" in u or u in seen:
                continue
            seen.add(u)
            out.append(u)
    return out


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|form)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>|</h\d>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s*\n+", "\n", raw)
    return raw.strip()


async def fred_series(series_id: str) -> str:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        async with _client() as c:
            r = await c.get(url)
            r.raise_for_status()
        rows = [l.split(",") for l in r.text.strip().splitlines()[1:]]
        rows = [(d, v) for d, v in rows if v not in (".", "")]
        if not rows:
            return ""
        last = rows[-12:]
        vals = [float(v) for _, v in rows[-260:]]
        txt = ", ".join(f"{d}: {v}" for d, v in last)
        return (
            f"FRED {series_id} (fetched today): latest {rows[-1][0]} = {rows[-1][1]}; "
            f"last obs: {txt}; min/max of last {len(vals)} obs: {min(vals)} / {max(vals)}"
        )
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
        return ""


async def fetch_url_text(url: str, max_chars: int = 3500) -> str:
    m = re.search(r"fred\.stlouisfed\.org/(?:series|graph)/(?:\?id=)?([A-Za-z0-9_]+)", url)
    if m:
        return await fred_series(m.group(1))
    try:
        async with _client() as c:
            r = await c.get(url)
        if r.status_code >= 400:
            return ""
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype and "json" not in ctype:
            return ""
        text = _html_to_text(r.text) if "html" in ctype else r.text
        return f"Page {url} (fetched today): {text[:max_chars]}"
    except Exception as e:
        logger.warning(f"Fetch failed for {url}: {e}")
        return ""


# ------------------------------------------------------- prediction markets


def _rank(rows: list[dict], limit: int, per_event: int = 2) -> list[dict]:
    """Most relevant first; within ties prefer informative (non-extreme) and liquid
    markets; at most `per_event` outcomes from one multi-outcome event."""
    def score(m):
        informative = 0.1 if 0.02 <= m.get("yes", 0.5) <= 0.98 else 0.0
        return (round(m["rel"], 2) + informative, m.get("yes", 0), m["volume"])
    rows = sorted(rows, key=score, reverse=True)
    out, per = [], {}
    for m in rows:
        k = m.get("event", m["title"])
        if per.get(k, 0) >= per_event:
            continue
        per[k] = per.get(k, 0) + 1
        out.append(m)
        if len(out) >= limit:
            break
    return out


async def polymarket(query: str, question_text: str, limit: int = 3) -> list[dict]:
    try:
        async with _client() as c:
            r = await c.get(
                "https://gamma-api.polymarket.com/public-search",
                params={"q": query, "limit_per_type": 8, "events_status": "active"},
            )
            data = r.json()
        out = []
        for ev in data.get("events", []) or []:
            if ev.get("closed"):
                continue
            for m in ev.get("markets", []) or []:
                if m.get("closed") or not m.get("active", True):
                    continue
                try:
                    outcomes = json.loads(m.get("outcomes") or "[]")
                    prices = json.loads(m.get("outcomePrices") or "[]")
                except Exception:
                    continue
                if not prices:
                    continue
                title = m.get("question") or ev.get("title", "")
                price_txt = ", ".join(f"{o} {float(p)*100:.0f}%" for o, p in zip(outcomes, prices))
                out.append({
                    "platform": "Polymarket",
                    "title": title,
                    "price": price_txt,
                    "close": _parse_iso(m.get("endDate") or ev.get("endDate")),
                    "volume": float(m.get("volume") or 0),
                    "rel": relevance(title + " " + ev.get("title", ""), question_text),
                    "event": "pm:" + str(ev.get("id")),
                    "yes": float(prices[0]),
                    "url": f"https://polymarket.com/event/{ev.get('slug','')}",
                })
        return _rank(out, limit)
    except Exception as e:
        logger.warning(f"Polymarket failed for '{query}': {e}")
        return []


async def manifold(query: str, question_text: str, limit: int = 3) -> list[dict]:
    try:
        async with _client() as c:
            r = await c.get(
                "https://api.manifold.markets/v0/search-markets",
                params={"term": query, "limit": 10, "filter": "open", "sort": "score"},
            )
            data = r.json()
        out = []
        for m in data or []:
            if m.get("isResolved") or m.get("probability") is None:
                continue
            title = m.get("question", "")
            out.append({
                "platform": "Manifold (play money)",
                "title": title,
                "price": f"Yes {m['probability']*100:.0f}%",
                "close": _parse_iso(m.get("closeTime")),
                "volume": float(m.get("volume") or 0),
                "rel": relevance(title, question_text),
                "event": "mf:" + str(m.get("id")),
                "yes": float(m["probability"]),
                "url": m.get("url", ""),
            })
        return _rank(out, limit)
    except Exception as e:
        logger.warning(f"Manifold failed for '{query}': {e}")
        return []


async def kalshi(query: str, question_text: str, limit: int = 3) -> list[dict]:
    try:
        async with _client() as c:
            r = await c.get(
                "https://api.elections.kalshi.com/v1/search/series",
                params={"query": query, "page_size": 5},
            )
            data = r.json()
        out = []
        for s in data.get("current_page", []) or []:
            for m in s.get("markets", []) or []:
                if m.get("result"):
                    continue
                title = m.get("title") or f"{s.get('event_title','')} {m.get('yes_subtitle','')}"
                lp = m.get("last_price")
                if lp is None:
                    continue
                out.append({
                    "platform": "Kalshi",
                    "title": f"{title} [{m.get('yes_subtitle','')}]",
                    "price": f"Yes {lp}% (bid {m.get('yes_bid')}/ask {m.get('yes_ask')})",
                    "close": _parse_iso(m.get("close_ts")),
                    "volume": float(s.get("total_volume") or 0),
                    "rel": relevance(title + " " + s.get("event_title", ""), question_text),
                    "event": "ks:" + str(s.get("event_ticker")),
                    "yes": float(lp) / 100,
                    "url": f"https://kalshi.com/markets/{s.get('series_ticker','').lower()}",
                })
        return _rank(out, limit)
    except Exception as e:
        logger.warning(f"Kalshi failed for '{query}': {e}")
        return []


async def market_anchors(query: str, question_text: str, min_rel: float = 0.3) -> str:
    results = await asyncio.gather(
        polymarket(query, question_text), manifold(query, question_text), kalshi(query, question_text),
        return_exceptions=True,
    )
    rows = []
    for res in results:
        if isinstance(res, BaseException):
            continue
        rows.extend(m for m in res if m["rel"] >= min_rel)
    rows = _rank(rows, 6)
    if not rows:
        return "Prediction markets: no matching market found on Polymarket/Manifold/Kalshi."
    lines = [
        "Possibly related prediction markets (auto-matched by keywords; may NOT match this "
        "question's exact criteria or dates; thin/play-money markets are weak evidence):"
    ]
    for m in rows:
        vol = f", volume ~{m['volume']:,.0f}" if m["volume"] else ""
        lines.append(
            f"- {m['platform']}: \"{m['title'][:140]}\" -> {m['price']}; closes {_fmt_date(m['close'])}{vol}"
        )
    return "\n".join(lines)


def format_headlines(items: list[dict], max_items: int = 14) -> str:
    seen, lines = set(), []
    items = sorted(items, key=lambda x: x["date"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    for it in items:
        key = re.sub(r"\W+", "", it["title"].lower())[:60]
        if not it["title"] or key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{_fmt_date(it['date'])}] {it['title']} ({it['source'] or it['via']})")
        if len(lines) >= max_items:
            break
    return "\n".join(lines)


def heuristic_query(question_text: str, n_words: int = 8) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.']*", question_text) if w.lower() not in _STOP]
    return " ".join(words[:n_words])
