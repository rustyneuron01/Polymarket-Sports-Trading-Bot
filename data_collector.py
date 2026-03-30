"""
Polymarket data collector: poll order books for NHL games; save snapshots (real bid/ask).
Default 60s poll; use --fast (5s) during live games to capture full-time price updates for real-time replay.
Usage: python -m polymarket_nhl_bot.data_collector
       python -m polymarket_nhl_bot.data_collector --fast
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from config import COLLECTOR_OUTPUT_DIR, COLLECTOR_POLL_INTERVAL_SEC, COLLECTOR_FAST_POLL_SEC
from polymarket_client import discover_nhl_markets, get_clob_client, get_order_books

# Use all CPUs for parallel I/O (order book fetches)
MAX_WORKERS = min(32, (os.cpu_count() or 4) * 4)


def _fetch_one(market) -> dict:
    """Fetch order book for one market (own client per thread for thread safety)."""
    client = get_clob_client()
    book_home, book_away = get_order_books(client, market)
    return {
        "condition_id": market.condition_id,
        "question": market.question,
        "token_id_home": market.token_id_home,
        "token_id_away": market.token_id_away,
        "bid_home": book_home.get("best_bid"),
        "ask_home": book_home.get("best_ask"),
        "bid_away": book_away.get("best_bid"),
        "ask_away": book_away.get("best_ask"),
    }


def collect_snapshot() -> tuple[str, list[dict]]:
    """
    Fetch token prices from Polymarket for all NHL markets in parallel (use all CPUs).
    Returns (poll_timestamp_iso, list of rows). One timestamp for the whole poll so time series are correct.
    """
    poll_ts = datetime.utcnow().isoformat() + "Z"
    today_yyyymmdd = datetime.utcnow().strftime("%Y-%m-%d")
    client = get_clob_client()
    markets = discover_nhl_markets(client, game_date_yyyymmdd=today_yyyymmdd)
    out = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, m): m for m in markets}
        for fut in as_completed(futures):
            try:
                row = fut.result()
                row["timestamp"] = poll_ts
                out.append(row)
            except Exception as e:
                m = futures[fut]
                print(f"  skip {m.condition_id[:16]}...: {e}")
    return poll_ts, out


def run_collector(interval_sec: float | None = None) -> None:
    """Loop: poll every interval_sec (or COLLECTOR_POLL_INTERVAL_SEC), append one batch per poll."""
    sec = interval_sec if interval_sec is not None else COLLECTOR_POLL_INTERVAL_SEC
    while True:
        try:
            poll_ts, rows = collect_snapshot()
            today = datetime.utcnow().strftime("%Y-%m-%d")
            path = COLLECTOR_OUTPUT_DIR / f"snapshots_{today}.jsonl"
            with open(path, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"Poll {poll_ts} | {len(rows)} markets -> {path.name} (+{len(rows)} rows)")
        except Exception as e:
            print(f"Collector error: {e}")
        time.sleep(sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poll Polymarket order books; append snapshots for game records.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"Poll every {COLLECTOR_FAST_POLL_SEC}s (real-time-like) so you get many more price points per game for full-time replay.",
    )
    parser.add_argument("--interval", type=float, default=None, help="Poll interval in seconds (default: 60, or 5 with --fast)")
    args = parser.parse_args()
    interval = args.interval
    if interval is None and args.fast:
        interval = COLLECTOR_FAST_POLL_SEC
    if interval is not None:
        print(f"Polling every {interval}s" + (" (fast mode for full-time capture)" if args.fast else "") + "\n")
    run_collector(interval_sec=interval)
