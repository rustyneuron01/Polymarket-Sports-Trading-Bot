"""
Polymarket CLOB client: order books, market discovery. See BOT_SPEC.
NHL game markets are discovered via Gamma API (tag_id=899); CLOB get_markets() does not tag them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

from config import (
    CHAIN_ID,
    CLOB_HOST,
    GAMMA_API_URL,
    POLYMARKET_FUNDER,
    POLYMARKET_PRIVATE_KEY,
)
from py_clob_clients import ClobClient


def get_clob_client_authenticated():
    """
    Return a ClobClient with signing (for placing orders). Use when PAPER_TRADING=false.
    Returns None if POLYMARKET_PRIVATE_KEY is not set.
    """
    if not (POLYMARKET_PRIVATE_KEY or "").strip():
        return None
    try:
        client = ClobClient(
            CLOB_HOST,
            key=POLYMARKET_PRIVATE_KEY.strip(),
            chain_id=CHAIN_ID,
            signature_type=0,  # EOA (MetaMask / direct private key)
            funder=POLYMARKET_FUNDER.strip() or None,
        )
        client.set_api_creds(client.create_or_derive_api_creds())
        return client
    except Exception as e:
        print(f"Authenticated CLOB client error: {e}")
        return None


@dataclass
class TwoSidedMarket:
    condition_id: str
    question: str
    token_id_home: str
    token_id_away: str
    outcome_home: str
    outcome_away: str
    liquidity_home: float = 0.0  # sum of bid size + ask size at best
    liquidity_away: float = 0.0
    spread_home: float = 0.0
    spread_away: float = 0.0


def get_clob_client() -> ClobClient:
    return ClobClient(host=CLOB_HOST, chain_id=CHAIN_ID)


def _normalize_book(book: Any) -> dict[str, Any]:
    """Convert CLOB response (dict or OrderBookSummary) to dict with bids/asks as list of {price, size}."""
    if hasattr(book, "bids"):
        bids, asks = book.bids or [], book.asks or []
    else:
        bids = book.get("bids") or []
        asks = book.get("asks") or []
    def row(r: Any) -> dict:
        if hasattr(r, "price"):
            return {"price": float(r.price), "size": float(r.size)}
        return {"price": float(r.get("price", 0)), "size": float(r.get("size", 0))}
    bids = [row(b) for b in bids]
    asks = [row(a) for a in asks]
    best_bid = float(bids[0]["price"]) if bids else None
    best_ask = float(asks[0]["price"]) if asks else None
    return {"best_bid": best_bid, "best_ask": best_ask, "bids": bids, "asks": asks}


def get_order_book(client: ClobClient, token_id: str, silent_404: bool = False) -> dict[str, Any]:
    """Single token order book; returns best_bid, best_ask, bids, asks. Resolved markets often return 404."""
    try:
        book = client.get_order_book(token_id)
    except Exception as e:
        err_str = str(e).lower()
        is_404 = "404" in err_str or "no orderbook" in err_str
        if not (silent_404 and is_404):
            print(f"Order book error {token_id[:20]}...: {e}")
        return {"best_bid": None, "best_ask": None, "bids": [], "asks": []}
    return _normalize_book(book)


def get_order_books(client: ClobClient, market: TwoSidedMarket) -> tuple[dict, dict]:
    """Order books for both tokens (fetched from Polymarket CLOB)."""
    home_book = get_order_book(client, market.token_id_home)
    away_book = get_order_book(client, market.token_id_away)
    return home_book, away_book


# CLOB prices-history: prefer full market history (interval "max"), then filter to date range.
# If that fails or returns empty, try 1-week window (some markets don't support 1M/ALL long range).
# Then fall back to chunked startTs/endTs (API rejects long ranges).
# See https://docs.polymarket.com/api-reference/markets/get-prices-history and GH #216
PRICES_HISTORY_MAX_INTERVAL_SEC = 3600  # fallback: 1h per request when using startTs/endTs
PRICES_HISTORY_ONE_WEEK_SEC = 7 * 24 * 3600  # 1-week window when max/1M not supported


def get_prices_history(
    token_id: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    interval: str = "1m",
    fidelity: Optional[int] = None,
    full_history: bool = False,
) -> list[dict]:
    """
    Fetch full historical price series for a token (all prices in range) from Polymarket CLOB.
    We always request the full time range; you can then correlate price changes with events.

    - full_history=True: try interval "all" first (finest granularity the API has), then "max".
    - Otherwise: try "max" then 1m with startTs/endTs.
    - fidelity: API requires minimum 10 for interval "1m"; default 10. Use 10 for requests that use 1m.
    Returns list of {"t": unix_ts_sec, "p": price} for the entire range.
    """
    fid = fidelity if fidelity is not None else 10  # API min for 1m is 10

    def _request(
        s: Optional[int], e: Optional[int], use_interval: Optional[str] = None
    ) -> tuple[list[dict], Optional[str]]:
        p: dict[str, Any] = {"market": token_id, "fidelity": fid}
        if use_interval is not None:
            p["interval"] = use_interval
        else:
            p["interval"] = interval
        if s is not None:
            p["startTs"] = s
        if e is not None:
            p["endTs"] = e
        r = requests.get(f"{CLOB_HOST}/prices-history", params=p, timeout=60)
        if r.status_code == 400:
            try:
                body = r.json()
                return [], body.get("error") or body.get("message") or r.text
            except Exception:
                return [], r.text[:200] or "400 Bad Request"
        r.raise_for_status()
        data = r.json()
        return list(data.get("history") or []), None

    try:
        # 1) If full_history: try "all" first (maximum density), then "max"
        if full_history:
            for try_interval in ("all", "max"):
                history, err = _request(None, None, use_interval=try_interval)
                if err is None and history:
                    if start_ts is not None and end_ts is not None:
                        history = [h for h in history if start_ts <= int(h.get("t", 0)) <= end_ts]
                    if history:
                        return sorted(history, key=lambda h: int(h.get("t", 0)))
        # 2) Try full market history (interval "max") then filter to range if needed
        history, err = _request(None, None, use_interval="max")
        if err is None and history:
            if start_ts is not None and end_ts is not None:
                history = [h for h in history if start_ts <= int(h.get("t", 0)) <= end_ts]
            return sorted(history, key=lambda h: int(h.get("t", 0)))
        if start_ts is None or end_ts is None or end_ts <= start_ts:
            return history if err is None else []

        # 2) Some markets don't support 1M/ALL (max) price history; try 1-week window
        end_1w = min(end_ts, start_ts + PRICES_HISTORY_ONE_WEEK_SEC)
        history, err = _request(start_ts, end_1w, use_interval=None)
        if err is None and history:
            history = [h for h in history if start_ts <= int(h.get("t", 0)) <= end_ts]
            if history:
                return sorted(history, key=lambda h: int(h.get("t", 0)))

        # 3) Fallback: use startTs/endTs (chunked if range too long)
        span = end_ts - start_ts
        if span <= PRICES_HISTORY_MAX_INTERVAL_SEC:
            history, err = _request(start_ts, end_ts, use_interval=None)
            if err:
                print(f"prices-history 400 {token_id[:20]}...: {err}")
                return []
            return history

        all_history: list[dict] = []
        t = start_ts
        while t < end_ts:
            chunk_end = min(t + PRICES_HISTORY_MAX_INTERVAL_SEC, end_ts)
            history, err = _request(t, chunk_end, use_interval=None)
            if err:
                print(f"prices-history 400 {token_id[:20]}...: {err}")
                break
            all_history.extend(history)
            t = chunk_end
        by_t: dict[int, dict] = {int(h.get("t", 0)): h for h in all_history}
        return [by_t[k] for k in sorted(by_t.keys())]
    except requests.RequestException as e:
        print(f"prices-history error {token_id[:20]}...: {e}")
        return []


# CLOB prices-history returns a single price "p" per tick (no bid/ask). Using that as both bid and ask
# makes backtest round-trips always break-even (buy at p, sell at p). Apply a synthetic half-spread
# so buy = p + half, sell = p - half (round-trip loses the spread), unless caller has real bid/ask.
PRICES_HISTORY_HALF_SPREAD = 0.01  # 1% half-spread: ask = p*(1+0.01), bid = p*(1-0.01)


def get_prices_history_for_market(
    token_id_home: str,
    token_id_away: str,
    start_ts: int,
    end_ts: int,
    interval: str = "1m",
    synthetic_spread: float = PRICES_HISTORY_HALF_SPREAD,
    full_history: bool = False,
    fidelity: Optional[int] = None,
) -> list[dict]:
    """
    Fetch full price series for both tokens over [start_ts, end_ts] and merge by timestamp.
    full_history=True: request finest granularity (interval "all" then "max") for maximum points per game.
    fidelity: minutes between samples (default 1 = up to 60/hour; use 1 for real-time-like density).
    CLOB API returns one price "p" per tick; we use ask = p*(1+half_spread), bid = p*(1-half_spread)
    so backtests don't show fake $0 profit. Returns list of { timestamp (ISO), bid_home, ask_home, bid_away, ask_away }.
    """
    home_hist = get_prices_history(
        token_id_home, start_ts=start_ts, end_ts=end_ts, interval=interval,
        full_history=full_history, fidelity=fidelity,
    )
    away_hist = get_prices_history(
        token_id_away, start_ts=start_ts, end_ts=end_ts, interval=interval,
        full_history=full_history, fidelity=fidelity,
    )
    by_ts: dict[int, dict[str, Any]] = {}
    for h in home_hist:
        t = int(h.get("t", 0))
        p = float(h.get("p", 0)) if h.get("p") is not None else None
        if p is not None and synthetic_spread != 0:
            bid_h = max(0.01, p * (1 - synthetic_spread))
            ask_h = min(0.99, p * (1 + synthetic_spread))
        else:
            bid_h = ask_h = p
        if t not in by_ts:
            by_ts[t] = {"t": t, "bid_home": bid_h, "ask_home": ask_h, "bid_away": None, "ask_away": None}
        else:
            by_ts[t]["bid_home"] = bid_h
            by_ts[t]["ask_home"] = ask_h
    for h in away_hist:
        t = int(h.get("t", 0))
        p = float(h.get("p", 0)) if h.get("p") is not None else None
        if p is not None and synthetic_spread != 0:
            bid_a = max(0.01, p * (1 - synthetic_spread))
            ask_a = min(0.99, p * (1 + synthetic_spread))
        else:
            bid_a = ask_a = p
        if t not in by_ts:
            by_ts[t] = {"t": t, "bid_home": None, "ask_home": None, "bid_away": bid_a, "ask_away": ask_a}
        else:
            by_ts[t]["bid_away"] = bid_a
            by_ts[t]["ask_away"] = ask_a
    out = []
    for t in sorted(by_ts.keys()):
        row = by_ts[t]
        ts_iso = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        out.append({
            "timestamp": ts_iso,
            "bid_home": row.get("bid_home"),
            "ask_home": row.get("ask_home"),
            "bid_away": row.get("bid_away"),
            "ask_away": row.get("ask_away"),
        })
    return out


def fetch_token_prices_from_polymarket(client: ClobClient, market: TwoSidedMarket) -> dict:
    """
    Fetch current token prices from Polymarket (no assumed prices).
    Returns dict with ask_home, bid_home, ask_away, bid_away for use in strategy/backtest.
    Always use this (or get_order_books) when checking token prices — do not assume.
    """
    home_book, away_book = get_order_books(client, market)
    return {
        "ask_home": home_book.get("best_ask"),
        "bid_home": home_book.get("best_bid"),
        "ask_away": away_book.get("best_ask"),
        "bid_away": away_book.get("best_bid"),
    }


def liquidity_and_spread(book: dict) -> tuple[float, float]:
    """Rough liquidity (best bid size + best ask size) and spread."""
    bids, asks = book.get("bids") or [], book.get("asks") or []
    best_bid = float(bids[0]["price"]) if bids else None
    best_ask = float(asks[0]["price"]) if asks else None
    size_bid = float(bids[0].get("size", 0)) if bids else 0
    size_ask = float(asks[0].get("size", 0)) if asks else 0
    liq = size_bid + size_ask
    spread = (best_ask - best_bid) if (best_bid and best_ask) else 0
    return liq, spread


# Gamma API: pagination max; NHL tag_id and series slug are resolved dynamically from GET /sports and GET /series/{id}
GAMMA_PAGE_SIZE = 250  # max per request; we paginate to fetch all
_nhl_gamma_params_cache: tuple[str | None, str | None] | None = None  # (tag_id, series_slug)
_nhl_series_events_cache: list[dict] | None = None


def _get_nhl_gamma_params() -> tuple[str | None, str | None]:
    """
    Resolve NHL tag_id and series slug from Gamma API (GET /sports → sport nhl, then GET /series/{id}).
    Returns (tag_id, series_slug) or (None, None) on failure. Result is cached.
    """
    global _nhl_gamma_params_cache, _nhl_series_events_cache
    if _nhl_gamma_params_cache is not None:
        return _nhl_gamma_params_cache
    try:
        r = requests.get(f"{GAMMA_API_URL}/sports", timeout=15)
        r.raise_for_status()
        sports = r.json() if isinstance(r.json(), list) else []
        nhl = next((s for s in sports if (s.get("sport") or "").strip().lower() == "nhl"), None)
        if not nhl:
            return (None, None)
        tags_str = (nhl.get("tags") or "").strip()
        tag_ids = [t.strip() for t in tags_str.split(",") if t.strip()]
        tag_id = next((t for t in tag_ids if t != "1"), tag_ids[0] if tag_ids else None)
        series_id = (nhl.get("series") or "").strip()
        if not series_id or not tag_id:
            _nhl_gamma_params_cache = (tag_id or "", "")
            return _nhl_gamma_params_cache
        r2 = requests.get(f"{GAMMA_API_URL}/series/{series_id}", timeout=15)
        if r2.status_code != 200:
            _nhl_gamma_params_cache = (tag_id, "")
            return _nhl_gamma_params_cache
        series_data = r2.json()
        if isinstance(series_data, dict):
            series_slug = (series_data.get("slug") or "").strip() or (series_data.get("ticker") or "").strip()
            events = series_data.get("events") or []
            if events:
                _nhl_series_events_cache = events
        else:
            series_slug = ""
        _nhl_gamma_params_cache = (tag_id, series_slug)
        return _nhl_gamma_params_cache
    except Exception:
        _nhl_gamma_params_cache = (None, None)
        return _nhl_gamma_params_cache

# ESPN team abbrev -> Gamma slug team codes (Gamma uses different codes: cal not cgy, las not vgk, etc.)
# Some markets use nickname (jets, sabres); include both so slug lookup finds them.
ESPN_TO_GAMMA_CODES: dict[str, list[str]] = {
    "CGY": ["cal", "cgy"],
    "VGK": ["las", "veg"],
    "VEG": ["las", "veg"],
    "MTL": ["mon"],
    "WSH": ["wsh", "was"],
    "LA": ["la", "lak"],
    "LAK": ["la", "lak"],
    "SJ": ["sj"],
    "SJS": ["sj"],
    "UTAH": ["utah", "uta"],
    "UTA": ["utah", "uta"],
    "NJ": ["nj"],
    "NJD": ["nj"],
    "TB": ["tb"],
    "TBL": ["tb"],
    "BUF": ["buf", "sabres"],
    "WPG": ["wpg", "jets"],
}


def _gamma_codes(abbrev: str) -> list[str]:
    """ESPN abbrev -> list of Gamma slug team codes to try."""
    a = (abbrev or "").strip().upper()
    if a in ESPN_TO_GAMMA_CODES:
        return ESPN_TO_GAMMA_CODES[a]
    return [(abbrev or "").strip().lower()]


def _candidate_slugs(date_slug: str, home_abbrev: str, away_abbrev: str) -> list[str]:
    """Build candidate Gamma slugs (nhl-away-home-date) for matching."""
    home_codes = _gamma_codes(home_abbrev)
    away_codes = _gamma_codes(away_abbrev)
    out = []
    for a in away_codes:
        for h in home_codes:
            if a and h:
                out.append(f"nhl-{a}-{h}-{date_slug}".lower())
    return out if out else [f"nhl-{(away_abbrev or '').strip().lower()}-{(home_abbrev or '').strip().lower()}-{date_slug}"]


def get_market_ids_for_game(
    date_yyyymmdd: str, home_abbrev: str, away_abbrev: str
) -> Optional[tuple[str, str, str]]:
    """
    Resolve Polymarket condition_id and CLOB token IDs for an NHL game.
    Returns (condition_id, token_id_home, token_id_away) or None. Use this to fetch all info (e.g. current
    order book) when building game records.
    """
    if len(date_yyyymmdd) == 8:
        date_slug = f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:8]}"
    else:
        date_slug = date_yyyymmdd
    if not (home_abbrev or "").strip() or not (away_abbrev or "").strip():
        return None
    candidates = set(_candidate_slugs(date_slug, home_abbrev, away_abbrev))
    # Gamma sometimes uses next-day date in slug (e.g. nhl-buf-wpg-2025-12-06 for game on 2025-12-05)
    try:
        dt = datetime.strptime(date_yyyymmdd[:10].replace("-", ""), "%Y%m%d")
        next_dt = dt + timedelta(days=1)
        next_slug = next_dt.strftime("%Y-%m-%d")
        candidates |= set(_candidate_slugs(next_slug, home_abbrev, away_abbrev))
    except Exception:
        pass

    def search(events: list) -> Optional[tuple[str, str, str]]:
        for ev in events:
            slug = (ev.get("slug") or "").strip().lower()
            if slug not in candidates:
                continue
            markets = ev.get("markets") or []
            moneyline = next((m for m in markets if m.get("sportsMarketType") == "moneyline"), None)
            if not moneyline:
                continue
            cond = moneyline.get("conditionId") or moneyline.get("condition_id") or ""
            if not cond:
                continue
            try:
                clob_ids = moneyline.get("clobTokenIds")
                if isinstance(clob_ids, str):
                    clob_ids = json.loads(clob_ids)
                # Gamma NHL slug is nhl-away-home-date; outcomes are [away, home]. So clob_ids[0]=away, [1]=home.
                if clob_ids and len(clob_ids) >= 2:
                    return (cond, str(clob_ids[1]), str(clob_ids[0]))  # token_id_home, token_id_away
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            return (cond, "", "")
        return None

    # 0) Season games live under NHL series (slug from Gamma /sports + /series/{id}); tag_id list is dominated by Champion/props
    _get_nhl_gamma_params()
    try:
        events0 = _nhl_series_events_cache or []
        if events0:
            result = search(events0)
            if result:
                return result
            # Series events omit full markets; fetch by slug or by id to get moneyline
            for ev in events0:
                slug = (ev.get("slug") or "").strip().lower()
                if slug not in candidates:
                    continue
                full_ev = None
                try:
                    r0b = requests.get(
                        f"{GAMMA_API_URL}/events/slug/{slug}",
                        timeout=15,
                    )
                    if r0b.status_code == 200:
                        full_ev = r0b.json()
                except Exception:
                    pass
                if not full_ev and ev.get("id"):
                    try:
                        r0c = requests.get(
                            f"{GAMMA_API_URL}/events/{ev['id']}",
                            timeout=15,
                        )
                        if r0c.status_code == 200:
                            full_ev = r0c.json()
                    except Exception:
                        pass
                if full_ev:
                    full = [full_ev] if isinstance(full_ev, dict) else (full_ev if isinstance(full_ev, list) else [])
                    result = search(full)
                    if result:
                        return result
    except Exception:
        pass

    # 1) Open events – paginate until no more pages
    tag_id, _ = _get_nhl_gamma_params()
    if tag_id:
        try:
            offset = 0
            while True:
                r = requests.get(
                    f"{GAMMA_API_URL}/events",
                    params={"tag_id": tag_id, "closed": "false", "limit": GAMMA_PAGE_SIZE, "offset": offset},
                    timeout=15,
                )
                r.raise_for_status()
                events = r.json() if isinstance(r.json(), list) else []
                result = search(events)
                if result:
                    return result
                if len(events) < GAMMA_PAGE_SIZE:
                    break
                offset += GAMMA_PAGE_SIZE
        except Exception as e:
            print(f"Gamma API error (condition_id lookup): {e}")
    # 2) Closed events – paginate until we find the event or run out of pages
    if tag_id:
        try:
            offset = 0
            while True:
                r2 = requests.get(
                    f"{GAMMA_API_URL}/events",
                    params={"tag_id": tag_id, "closed": "true", "limit": GAMMA_PAGE_SIZE, "offset": offset},
                    timeout=15,
                )
                r2.raise_for_status()
                events2 = r2.json() if isinstance(r2.json(), list) else []
                result = search(events2)
                if result:
                    return result
                if len(events2) < GAMMA_PAGE_SIZE:
                    break
                offset += GAMMA_PAGE_SIZE
        except Exception:
            pass
    # 3) Fetch by slug: GET /events/slug/{slug} returns single event (dict); GET /events?slug= often returns []
    for cand in list(candidates):
        try:
            r3 = requests.get(
                f"{GAMMA_API_URL}/events/slug/{cand}",
                timeout=15,
            )
            if r3.status_code == 404:
                continue
            r3.raise_for_status()
            data = r3.json()
            events3 = [data] if isinstance(data, dict) else (data if isinstance(data, list) else [])
            result = search(events3)
            if result:
                return result
        except Exception:
            pass
    return None


def get_condition_id_for_game(date_yyyymmdd: str, home_abbrev: str, away_abbrev: str) -> Optional[str]:
    """Return condition_id only (for backward compatibility). Use get_market_ids_for_game to get token IDs too."""
    result = get_market_ids_for_game(date_yyyymmdd, home_abbrev, away_abbrev)
    return result[0] if result else None


def _discover_nhl_markets_gamma(
    client: ClobClient,
    game_date_yyyymmdd: str | None = None,
) -> list[TwoSidedMarket]:
    """
    Discover NHL single-game (moneyline) markets via Gamma API.
    Events have slug like nhl-det-ott-2026-02-26; we take the market with sportsMarketType=="moneyline".
    If game_date_yyyymmdd is set (e.g. "2026-02-27"), only include events whose slug ends with that date.
    Paginates to fetch all events (not just first 250).
    """
    tag_id, _ = _get_nhl_gamma_params()
    if not tag_id:
        return []
    results = []
    all_events: list[dict] = []
    try:
        offset = 0
        while True:
            r = requests.get(
                f"{GAMMA_API_URL}/events",
                params={
                    "tag_id": tag_id,
                    "active": "true",
                    "closed": "false",
                    "limit": GAMMA_PAGE_SIZE,
                    "offset": offset,
                },
                timeout=15,
            )
            r.raise_for_status()
            events = r.json() if isinstance(r.json(), list) else []
            all_events.extend(events)
            if len(events) < GAMMA_PAGE_SIZE:
                break
            offset += GAMMA_PAGE_SIZE
    except Exception as e:
        print(f"Gamma API error: {e}")
        return []
    for ev in all_events:
        slug = (ev.get("slug") or "").strip()
        # Game events: nhl-xxx-xxx-YYYY-MM-DD (e.g. nhl-det-ott-2026-02-26)
        if not slug.startswith("nhl-") or "-20" not in slug:
            continue
        if game_date_yyyymmdd and not slug.endswith(game_date_yyyymmdd):
            continue
        if ev.get("closed"):
            continue
        markets = ev.get("markets") or []
        moneyline = next((m for m in markets if m.get("sportsMarketType") == "moneyline"), None)
        if not moneyline:
            continue
        try:
            outcomes = moneyline.get("outcomes")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            clob_ids = moneyline.get("clobTokenIds")
            if isinstance(clob_ids, str):
                clob_ids = json.loads(clob_ids)
        except (json.JSONDecodeError, TypeError):
            continue
        if not outcomes or not clob_ids or len(outcomes) < 2 or len(clob_ids) < 2:
            continue
        cond = moneyline.get("conditionId") or moneyline.get("condition_id") or ""
        question = moneyline.get("question") or ev.get("title") or ""
        # Gamma NHL outcomes are [away, home]; clob_ids match. We want token_id_home = home outcome, token_id_away = away.
        tid_home, tid_away = str(clob_ids[1]), str(clob_ids[0])
        out0, out1 = str(outcomes[0]), str(outcomes[1])
        home_book = get_order_book(client, tid_home)
        away_book = get_order_book(client, tid_away)
        liq0, sp0 = liquidity_and_spread(home_book)
        liq1, sp1 = liquidity_and_spread(away_book)
        results.append(TwoSidedMarket(
            condition_id=cond,
            question=question,
            token_id_home=tid_home,
            token_id_away=tid_away,
            outcome_home=out1,
            outcome_away=out0,
            liquidity_home=liq0,
            liquidity_away=liq1,
            spread_home=sp0,
            spread_away=sp1,
        ))
    return results


def discover_nhl_markets(
    client: ClobClient,
    limit: int = 200,
    game_date_yyyymmdd: str | None = None,
) -> list[TwoSidedMarket]:
    """
    Discover NHL two-outcome (moneyline) markets. Uses Gamma API first (tag 899);
    falls back to CLOB get_markets() if Gamma returns none.
    If game_date_yyyymmdd is set, only events for that date (from slug) are included.
    """
    results = _discover_nhl_markets_gamma(client, game_date_yyyymmdd=game_date_yyyymmdd)
    if results:
        return _pick_best_market_per_question(results)
    try:
        payload = client.get_markets()
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(data, list):
            data = []
        for m in data[:limit]:
            tags = m.get("tags") or []
            if not any("nhl" in str(t).lower() for t in tags):
                continue
            tokens = m.get("tokens") or []
            if len(tokens) < 2:
                continue
            t0, t1 = tokens[0], tokens[1]
            tid0 = t0.get("token_id") if isinstance(t0, dict) else None
            tid1 = t1.get("token_id") if isinstance(t1, dict) else None
            if not tid0 or not tid1:
                continue
            home_book = get_order_book(client, tid0)
            away_book = get_order_book(client, tid1)
            liq0, sp0 = liquidity_and_spread(home_book)
            liq1, sp1 = liquidity_and_spread(away_book)
            results.append(TwoSidedMarket(
                condition_id=m.get("conditionId") or m.get("condition_id", ""),
                question=m.get("question", ""),
                token_id_home=tid0,
                token_id_away=tid1,
                outcome_home=(t0.get("outcome") or "Home") if isinstance(t0, dict) else "Home",
                outcome_away=(t1.get("outcome") or "Away") if isinstance(t1, dict) else "Away",
                liquidity_home=liq0,
                liquidity_away=liq1,
                spread_home=sp0,
                spread_away=sp1,
            ))
        return _pick_best_market_per_question(results)
    except Exception as e:
        print(f"CLOB get_markets error: {e}")
        return []


def _pick_best_market_per_question(markets: list[TwoSidedMarket]) -> list[TwoSidedMarket]:
    """When multiple markets match, pick one with highest total liquidity (or tightest spread)."""
    by_q: dict[str, list[TwoSidedMarket]] = {}
    for m in markets:
        key = m.question[:80]
        by_q.setdefault(key, []).append(m)
    out = []
    for group in by_q.values():
        best = max(group, key=lambda x: x.liquidity_home + x.liquidity_away)
        out.append(best)
    return out
