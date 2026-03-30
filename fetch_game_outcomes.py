"""
Backfill past NHL game outcomes from ESPN for training and analysis.
Writes one JSON object per game to data/game_outcomes.jsonl.

Usage:
  python -m polymarket_nhl_bot.fetch_game_outcomes --from 2025-10-01 --to 2026-02-26
  python -m polymarket_nhl_bot.fetch_game_outcomes --from 2025-10-01 --to 2026-02-26 --output data/my_outcomes.jsonl
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from espn_client import get_completed_games_with_scores_for_date


def _parse_date(s: str) -> str:
    """Parse YYYY-MM-DD or YYYYMMDD -> YYYYMMDD."""
    s = s.strip().replace("-", "")
    if len(s) == 8:
        return s
    raise ValueError(f"Expected YYYY-MM-DD or YYYYMMDD, got {s!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill NHL game outcomes from ESPN to JSONL.")
    parser.add_argument("--from", dest="from_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: data/game_outcomes.jsonl)",
    )
    args = parser.parse_args()

    try:
        start_yyyymmdd = _parse_date(args.from_date)
        end_yyyymmdd = _parse_date(args.to_date)
    except ValueError as e:
        print(e)
        raise SystemExit(1)

    start_dt = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    end_dt = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    if start_dt > end_dt:
        print("--from must be <= --to")
        raise SystemExit(1)

    out_path = Path(args.output) if args.output else Path(__file__).resolve().parent / "data" / "game_outcomes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        games = get_completed_games_with_scores_for_date(date_str)
        with open(out_path, "a") as f:
            for g in games:
                f.write(json.dumps(g) + "\n")
                total += 1
        if games:
            print(f"{date_str}: {len(games)} games -> {out_path.name}")
        current += timedelta(days=1)

    print(f"Done. Total games written: {total} -> {out_path}")


if __name__ == "__main__":
    main()
