"""
Validate game record JSONs: check token_price_series structure and values.
Reports which files have valid price data, empty (re-fetch candidate), or malformed.

Usage (from repo root or with PYTHONPATH):
  python -m polymarket_nhl_bot.validate_game_records
  python -m polymarket_nhl_bot.validate_game_records --dir data/game_records
Or from polymarket_nhl_bot:  python validate_game_records.py --dir data/game_records
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

REQUIRED_KEYS = {"timestamp", "bid_home", "ask_home", "bid_away", "ask_away"}

# Price timestamps: [game_date - 1 day, game_date + 2 days] UTC. Games are evening local so pre-game prices can be previous day UTC.
PRICE_WINDOW_DAYS_BEFORE = 1
PRICE_WINDOW_DAYS_AFTER = 2


def _valid_price(v) -> bool:
    if v is None:
        return True
    try:
        f = float(v)
        return 0 <= f <= 1
    except (TypeError, ValueError):
        return False


def _parse_ts(s: str) -> datetime | None:
    """Parse ISO timestamp (e.g. 2025-12-29T03:15:00Z) to naive UTC datetime."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None


# Canonical format for training: ISO UTC with seconds (no fractional), e.g. 2025-12-29T03:15:00Z
UTC_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def normalize_timestamp_to_utc_iso(ts: str) -> str:
    """
    Parse a timestamp string and return canonical ISO UTC (YYYY-MM-DDTHH:MM:SSZ).
    If unparseable, returns the original string so callers don't lose data.
    """
    if not ts or not isinstance(ts, str):
        return ts or ""
    dt = _parse_ts(ts)
    if dt is None:
        return ts.strip()
    return dt.strftime(UTC_ISO_FORMAT)


def normalize_price_series_timestamps(series: list[dict]) -> list[dict]:
    """
    Normalize every row's 'timestamp' to canonical ISO UTC and sort by timestamp.
    Affects training: consistent format and chronological order. Returns a new list (does not mutate input).
    """
    out = []
    for row in series:
        if not isinstance(row, dict):
            out.append(row)
            continue
        r = dict(row)
        if "timestamp" in r:
            r["timestamp"] = normalize_timestamp_to_utc_iso(r.get("timestamp") or "")
        out.append(r)
    out.sort(key=lambda r: (r.get("timestamp") or ""))
    return out


def _last_scoring_event(events: list) -> dict | None:
    """Return the last event with scoring_play=True, or None. Events assumed sorted by (period, clock)."""
    for ev in reversed(events):
        if ev.get("scoring_play") is True:
            return ev
    return None


def validate_record(rec: dict, path: Path) -> tuple[str, str]:
    """
    Returns (status, message).
    status: "ok" | "empty_no_condition" | "empty_has_condition" | "malformed"
    """
    condition_id = (rec.get("condition_id") or "").strip()
    series = rec.get("token_price_series")
    if not isinstance(series, list):
        return "malformed", "token_price_series is not a list"

    if len(series) == 0:
        if condition_id:
            return "empty_has_condition", "0 price rows but condition_id present (re-fetch candidate)"
        return "empty_no_condition", "0 price rows, no condition_id (expected)"

    errors = []
    for i, row in enumerate(series):
        if not isinstance(row, dict):
            errors.append(f"row {i} not a dict")
            continue
        missing = REQUIRED_KEYS - set(row.keys())
        if missing:
            errors.append(f"row {i} missing keys: {missing}")
        for k in ["bid_home", "ask_home", "bid_away", "ask_away"]:
            if k in row and not _valid_price(row.get(k)):
                errors.append(f"row {i} {k}={row.get(k)!r} invalid (expect 0-1 or null)")
        if "timestamp" in row and not isinstance(row.get("timestamp"), str):
            errors.append(f"row {i} timestamp not string")

    # Ensure price timestamps fall within game window (date - 1 day through date + 2 days UTC)
    rec_date = (rec.get("date") or "").strip()
    if not errors and series and rec_date and len(rec_date) >= 10:
        try:
            game_date = datetime.strptime(rec_date[:10].replace("-", ""), "%Y%m%d")
            window_start = game_date - timedelta(days=PRICE_WINDOW_DAYS_BEFORE)
            window_end = game_date + timedelta(days=PRICE_WINDOW_DAYS_AFTER + 1)  # exclusive
            for i, row in enumerate(series):
                ts = row.get("timestamp")
                if not ts:
                    continue
                dt = _parse_ts(ts)
                if dt is None:
                    continue
                if dt < window_start or dt >= window_end:
                    errors.append(f"row {i} timestamp {ts!r} outside game window [{window_start.date()}..{window_end.date()})")
                    break
        except (ValueError, TypeError):
            pass

    if errors:
        return "malformed", "; ".join(errors[:3]) + (" ..." if len(errors) > 3 else "")
    return "ok", f"{len(series)} price rows"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate game record JSON price data.")
    parser.add_argument(
        "--dir",
        default=None,
        help="Game records directory (default: data/game_records)",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print summary and re-fetch list")
    parser.add_argument("--sync-check", action="store_true", help="Check outcome vs end-of-series prices and events vs final score")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = Path(args.dir) if args.dir else base / "data" / "game_records"
    if not out_dir.exists():
        print(f"Directory not found: {out_dir}")
        raise SystemExit(1)

    results = {"ok": [], "empty_no_condition": [], "empty_has_condition": [], "malformed": []}
    for path in sorted(out_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            results["malformed"].append((path.name, f"read error: {e}"))
            continue
        status, msg = validate_record(rec, path)
        results[status].append((path.name, msg))

    total = sum(len(v) for v in results.values())
    print(f"Validated {total} game records in {out_dir}")
    print(f"  ok (valid price data):     {len(results['ok'])}")
    print(f"  empty, no condition_id:    {len(results['empty_no_condition'])}")
    print(f"  empty, has condition_id:   {len(results['empty_has_condition'])} (re-fetch with --refresh-prices)")
    print(f"  malformed:                 {len(results['malformed'])}")

    if not args.quiet:
        for name, msg in results["malformed"]:
            print(f"  MALFORMED {name}: {msg}")
        for name, msg in results["empty_has_condition"][:20]:
            print(f"  RE-FETCH  {name}: {msg}")
        if len(results["empty_has_condition"]) > 20:
            print(f"  ... and {len(results['empty_has_condition']) - 20} more (re-fetch candidates)")

    if results["empty_has_condition"]:
        print("\nTo re-fetch prices for games that have condition_id but 0 price rows, run:")
        print("  python -m polymarket_nhl_bot.build_game_records --from YYYY-MM-DD --to YYYY-MM-DD --refresh-prices")

    if args.sync_check and results["ok"]:
        _run_sync_check(out_dir, results["ok"], args.quiet)


def _last_prices(series: list) -> tuple[float | None, float | None]:
    """From the last 20 rows, get the last non-null bid_home and bid_away (or ask)."""
    home_p, away_p = None, None
    for row in reversed(series[-20:] if len(series) >= 20 else series):
        if row.get("bid_home") is not None:
            home_p = float(row["bid_home"])
        if row.get("ask_home") is not None and home_p is None:
            home_p = float(row["ask_home"])
        if row.get("bid_away") is not None:
            away_p = float(row["bid_away"])
        if row.get("ask_away") is not None and away_p is None:
            away_p = float(row["ask_away"])
        if home_p is not None and away_p is not None:
            break
    return home_p, away_p


def _run_sync_check(out_dir: Path, ok_list: list[tuple[str, str]], quiet: bool) -> None:
    """Check outcome vs end-of-series prices and events vs final score."""
    synced = []  # resolution 1/0 and matches outcome
    no_resolution = []  # has prices but end not near 1/0
    mismatch = []  # winner token not high at end
    score_mismatch = []
    for (fname, _) in ok_list:
        path = out_dir / fname
        try:
            with open(path) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        series = rec.get("token_price_series") or []
        if not series:
            continue
        home_won = rec.get("home_won", False)
        home_score = rec.get("home_score", 0)
        away_score = rec.get("away_score", 0)
        events = rec.get("events") or []

        # Events: last scoring event (goal) score should match final (period 5 = OT/shootout has different semantics)
        # ESPN may list non-scoring events (e.g. penalty) after the last goal, so use last scoring event.
        if events:
            last_goal = _last_scoring_event(events)
            if last_goal is not None and last_goal.get("period", 0) < 5:
                if last_goal.get("home_score") != home_score or last_goal.get("away_score") != away_score:
                    score_mismatch.append((fname, f"last goal {last_goal.get('home_score')}-{last_goal.get('away_score')} (period {last_goal.get('period')}) != final {home_score}-{away_score}"))
            elif last_goal is None and events:
                last = events[-1]
                if last.get("period", 0) < 5 and (last.get("home_score") != home_score or last.get("away_score") != away_score):
                    score_mismatch.append((fname, f"last event {last.get('home_score')}-{last.get('away_score')} (period {last.get('period')}) != final {home_score}-{away_score}"))

        home_p, away_p = _last_prices(series)
        if home_p is None and away_p is None:
            no_resolution.append((fname, "no non-null price at end"))
            continue
        # Use 0.8/0.2 as threshold for "resolved"
        home_high = (home_p or 0) >= 0.8
        away_low = (away_p or 1) <= 0.2
        away_high = (away_p or 0) >= 0.8
        home_low = (home_p or 1) <= 0.2
        resolved = (home_high and away_low) or (away_high and home_low)
        if not resolved:
            no_resolution.append((fname, f"end prices home={home_p} away={away_p}"))
            continue
        if home_won and not (home_high and away_low):
            mismatch.append((fname, f"home_won=true but end home={home_p} away={away_p}"))
        elif not home_won and not (away_high and home_low):
            mismatch.append((fname, f"home_won=false but end home={home_p} away={away_p}"))
        else:
            synced.append(fname)

    print("\n--- Sync check (outcome vs end-of-series prices) ---")
    print(f"  Synced (resolution matches outcome): {len(synced)}")
    print(f"  No resolution at end (prices not 1/0): {len(no_resolution)}")
    print(f"  Mismatch (winner token not high):     {len(mismatch)}")
    print(f"  Event score != final score:          {len(score_mismatch)}")
    if not quiet:
        for name, msg in mismatch[:15]:
            print(f"  MISMATCH {name}: {msg}")
        if len(mismatch) > 15:
            print(f"  ... and {len(mismatch) - 15} more")
        for name, msg in score_mismatch[:10]:
            print(f"  SCORE    {name}: {msg}")
        if len(no_resolution) > 0 and len(no_resolution) <= 5:
            for name, msg in no_resolution:
                print(f"  NO_RES   {name}: {msg}")


if __name__ == "__main__":
    main()
