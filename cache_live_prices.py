"""
Cache Polymarket prices every 1 second for games that have started (real-time data for testing).

When an NHL game is in progress (ESPN status = "in"), this script fetches order books for that
game's market every 1 second and appends rows to a JSONL file. You get ~3600 price points per
hour per game instead of ~300 from the API, so you can test your model on full-time-like data.

Usage:
  # Start before games begin; when a game goes live, recording starts automatically (1s poll)
  python cache_live_prices.py

  # Custom interval (e.g. 2 seconds)
  python cache_live_prices.py --interval 2

  # Output directory (default: data/polymarket_snapshots/live_1s)
  python cache_live_prices.py --out-dir data/my_1s_cache

Output: one file per day, e.g. snapshots_1s_2026-02-28.jsonl. Each line is JSON:
  {"condition_id": "...", "timestamp": "2026-02-28T01:23:45.123Z", "bid_home": 0.55, "ask_home": 0.57, ...}
Same format as data_collector so you can use it when building game records with --snapshots-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from config import COLLECTOR_OUTPUT_DIR
from espn_client import get_scoreboard, event_to_game_info, get_live_game_state_from_event
from polymarket_client import get_clob_client, get_order_books, discover_nhl_markets

DEFAULT_OUT_DIR = COLLECTOR_OUTPUT_DIR / "live_1s"
MAX_WORKERS = 8


def _team_in_question(team_name: str, question: str) -> bool:
    """True if question contains full team name or team nickname (last word)."""
    if not team_name or not question:
        return False
    t = team_name.lower().strip()
    q = question.lower()
    if t in q:
        return True
    parts = t.split()
    if parts:
        nickname = parts[-1]  # e.g. Rangers, Penguins
        if nickname in q:
            return True
    return False


def _match_markets_to_events(markets, events):
    """Yield (market, event) for each market that matches an event by team names."""
    for m in markets:
        q = (m.question or "").lower()
        for ev in events:
            info = event_to_game_info(ev)
            if not info:
                continue
            home = (info.get("home_team_name") or "").strip()
            away = (info.get("away_team_name") or "").strip()
            if home and away and _team_in_question(home, q) and _team_in_question(away, q):
                yield m, ev
                break


def get_in_progress_markets(client, today_yyyymmdd: str):
    """Return list of (market, event) for games currently in progress.
    today_yyyymmdd: YYYY-MM-DD for discover_nhl_markets; ESPN get_scoreboard gets YYYYMMDD internally.
    """
    # ESPN scoreboard expects dates=YYYYMMDD
    dates_param = today_yyyymmdd.replace("-", "")
    events = get_scoreboard(dates_param)
    live = [ev for ev in events if get_live_game_state_from_event(ev) is not None]
    if not live:
        return []
    markets = discover_nhl_markets(client, game_date_yyyymmdd=today_yyyymmdd)
    return list(_match_markets_to_events(markets, live))


def fetch_one(market):
    """Fetch order book for one market."""
    client = get_clob_client()
    book_home, book_away = get_order_books(client, market)
    return {
        "condition_id": market.condition_id,
        "bid_home": book_home.get("best_bid"),
        "ask_home": book_home.get("best_ask"),
        "bid_away": book_away.get("best_bid"),
        "ask_away": book_away.get("best_ask"),
    }


def run_recorder(interval_sec: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = get_clob_client()
    last_log = 0
    total_rows = 0

    print(f"Live 1s price cache: polling every {interval_sec}s when games are in progress.")
    print(f"Output: {out_dir}")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            today_yyyymmdd = datetime.utcnow().strftime("%Y-%m-%d")
            matches = get_in_progress_markets(client, today_yyyymmdd)

            if matches:
                poll_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                path = out_dir / f"snapshots_1s_{today_yyyymmdd}.jsonl"

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                    futures = [pool.submit(fetch_one, m) for m, _ in matches]
                    rows = []
                    for fut in as_completed(futures):
                        try:
                            row = fut.result()
                            row["timestamp"] = poll_ts
                            rows.append(row)
                        except Exception:
                            pass
                with open(path, "a") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                total_rows += len(rows)

                if time.time() - last_log >= 10:
                    print(f"  {poll_ts}  {len(matches)} live game(s) -> {path.name}  (total rows today: {total_rows})")
                    last_log = time.time()
            else:
                if time.time() - last_log >= 60:
                    print(f"  No games in progress. Waiting... (next check in {interval_sec}s)")
                    last_log = time.time()

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache Polymarket prices every 1s for in-progress NHL games (for real-time testing)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between fetches (default: 1)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=f"Output directory for JSONL files (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    if not out_dir.is_absolute():
        out_dir = BASE / out_dir

    run_recorder(args.interval, out_dir)


if __name__ == "__main__":
    main()
