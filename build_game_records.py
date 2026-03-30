"""
Build one combined record per game and fetch all info in one place: outcome, score, linescores,
in-game events (goals, penalties, injuries), and Polymarket token price series. Uses ESPN for
outcomes/events, Gamma (paginated) for condition_id and token IDs. Price series: (1) snapshot
files if present, (2) Polymarket CLOB prices-history API to backfill historical prices per token,
(3) current CLOB as fallback (often 404 for resolved markets).

Usage:
  python -m polymarket_nhl_bot.build_game_records --from 2026-02-01 --to 2026-02-26
  python -m polymarket_nhl_bot.build_game_records --from 2026-02-01 --to 2026-02-26 --output-dir data/game_records

Output: one JSON file per game under output_dir (or one JSONL file). Schema: DOCS.md §3.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests

from config import COLLECTOR_OUTPUT_DIR
from espn_client import get_completed_games_rich_for_date, get_game_events
from polymarket_client import (
    get_clob_client,
    get_market_ids_for_game,
    get_order_book,
    get_prices_history_for_market,
)
from validate_game_records import normalize_price_series_timestamps, validate_record, _last_scoring_event


def _team_id_to_abbrev() -> dict[str, str]:
    """ESPN team id -> abbreviation."""
    try:
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams?limit=50",
            timeout=10,
        )
        r.raise_for_status()
        teams = r.json()["sports"][0]["leagues"][0]["teams"]
        return {t["team"]["id"]: t["team"]["abbreviation"] for t in teams}
    except Exception:
        return {}


def _load_snapshots(snap_dir: Path) -> dict[str, list[dict]]:
    """Load snapshot JSONL; return { condition_id: [rows sorted by timestamp] }."""
    by_game: dict[str, list[dict]] = defaultdict(list)
    if not snap_dir.exists():
        return dict(by_game)
    for p in sorted(snap_dir.glob("*.jsonl")):
        if p.name.startswith("README"):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = row.get("condition_id")
                if not cid:
                    continue
                by_game[cid].append({
                    "timestamp": row.get("timestamp", ""),
                    "bid_home": _f(row.get("bid_home")),
                    "ask_home": _f(row.get("ask_home")),
                    "bid_away": _f(row.get("bid_away")),
                    "ask_away": _f(row.get("ask_away")),
                })
    for cid in by_game:
        by_game[cid].sort(key=lambda x: x["timestamp"])
    return dict(by_game)


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_date(s: str) -> str:
    """YYYY-MM-DD or YYYYMMDD -> YYYYMMDD."""
    s = s.strip().replace("-", "")
    if len(s) == 8:
        return s
    raise ValueError(f"Expected YYYY-MM-DD or YYYYMMDD, got {s!r}")


def _annotate_event_feed_incomplete(out_dir: Path) -> None:
    """Add event_feed_incomplete=true to existing records where last goal != final (ESPN feed incomplete)."""
    annotated = 0
    for path in sorted(out_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {path.name}: {e}")
            continue
        if rec.get("event_feed_incomplete"):
            continue
        events = rec.get("events") or []
        if not events:
            continue
        last_goal = _last_scoring_event(events)
        if last_goal is None or last_goal.get("period", 0) >= 5:
            continue
        if last_goal.get("home_score") == rec.get("home_score") and last_goal.get("away_score") == rec.get("away_score"):
            continue
        rec["event_feed_incomplete"] = True
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        annotated += 1
        print(f"  annotated {path.name} (last goal {last_goal.get('home_score')}-{last_goal.get('away_score')} != final {rec.get('home_score')}-{rec.get('away_score')})")
    print(f"Annotated {annotated} record(s) with event_feed_incomplete.")


def _remove_event_feed_incomplete(out_dir: Path) -> None:
    """Delete game record JSONs where event feed is incomplete (last goal != final). Use for a clean training set."""
    removed = 0
    for path in sorted(out_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("event_feed_incomplete"):
            path.unlink()
            removed += 1
            print(f"  removed {path.name} (event_feed_incomplete)")
            continue
        events = rec.get("events") or []
        if not events:
            continue
        last_goal = _last_scoring_event(events)
        if last_goal is None or last_goal.get("period", 0) >= 5:
            continue
        if last_goal.get("home_score") != rec.get("home_score") or last_goal.get("away_score") != rec.get("away_score"):
            path.unlink()
            removed += 1
            print(f"  removed {path.name} (last goal {last_goal.get('home_score')}-{last_goal.get('away_score')} != final {rec.get('home_score')}-{rec.get('away_score')})")
    print(f"Removed {removed} record(s) with incomplete event feed.")


def _run_fix_timestamps(out_dir: Path) -> None:
    """Normalize token_price_series timestamps to ISO UTC and re-sort in all existing JSON files."""
    fixed = 0
    for path in sorted(out_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {path.name}: {e}")
            continue
        series = rec.get("token_price_series")
        if not isinstance(series, list) or len(series) == 0:
            continue
        new_series = normalize_price_series_timestamps(series)
        rec["token_price_series"] = new_series
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)
        fixed += 1
    print(f"Fixed timestamps in {fixed} game records under {out_dir}")


def _record_needs_fix(out_path: Path, accept_event_score_mismatch: bool = False) -> tuple[bool, str | None]:
    """
    Return (needs_rebuild, reason). reason is None if file missing (will be built) or if valid.
    When accept_event_score_mismatch=True, "last goal != final" is not treated as needing fix
    (ESPN feed is often incomplete and won't change, so we avoid rewriting the same file every run).
    """
    if not out_path.exists():
        return True, None  # new file
    try:
        with open(out_path) as f:
            rec = json.load(f)
    except (json.JSONDecodeError, OSError):
        return True, "unreadable or malformed"
    status, msg = validate_record(rec, out_path)
    if status != "ok":
        return True, msg
    # Last scoring event (goal) score should match final (unless accept_event_score_mismatch)
    events = rec.get("events") or []
    if events and not accept_event_score_mismatch:
        last_goal = _last_scoring_event(events)
        if last_goal is not None and last_goal.get("period", 0) < 5:
            if last_goal.get("home_score") != rec.get("home_score") or last_goal.get("away_score") != rec.get("away_score"):
                return True, "event score != final (last goal != final; ESPN feed may be incomplete)"
        elif last_goal is None:
            last = events[-1]
            if last.get("period", 0) < 5 and (last.get("home_score") != rec.get("home_score") or last.get("away_score") != rec.get("away_score")):
                return True, "event score != final"
    return False, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-game records: outcome, events, token price series for training/eval."
    )
    parser.add_argument("--from", dest="from_date", default=None, help="Start date YYYY-MM-DD (required unless --fix-timestamps)")
    parser.add_argument("--to", dest="to_date", default=None, help="End date YYYY-MM-DD (required unless --fix-timestamps)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for one JSON per game (default: data/game_records)",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write single game_records.jsonl instead of one file per game",
    )
    parser.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Re-fetch token_price_series from API/CLOB (skip snapshots) for all games with condition_id",
    )
    parser.add_argument(
        "--full-price-history",
        action="store_true",
        help="When fetching from API: request finest granularity (interval 'all' then 'max', fidelity=1) for maximum price points per game instead of ~300–400. Use for real-time-like replay.",
    )
    parser.add_argument(
        "--fix-invalid-only",
        action="store_true",
        help="Only rebuild records that are missing, have no price data (but condition_id), are malformed, or event score != final; skip valid existing files. Use with --refresh-prices to re-fetch price history for fixed games.",
    )
    parser.add_argument(
        "--fix-timestamps",
        action="store_true",
        help="Only fix existing JSON files: normalize token_price_series timestamps to ISO UTC and re-sort. Does not fetch games or prices. Use with --output-dir to target a directory.",
    )
    parser.add_argument(
        "--accept-event-score-mismatch",
        action="store_true",
        help="With --fix-invalid-only: do not rebuild when the only issue is last goal != final. Use to avoid repeatedly rewriting the same files (ESPN event feed is often incomplete and won't change).",
    )
    parser.add_argument(
        "--annotate-event-feed-incomplete",
        action="store_true",
        help="One-time: add event_feed_incomplete=true to existing records where last goal != final. Does not fetch; only reads and rewrites. Use so training can filter these.",
    )
    parser.add_argument(
        "--remove-event-feed-incomplete",
        action="store_true",
        help="Delete game record JSONs where last goal != final (ESPN feed incomplete). Use for a clean training set so the model does not see inconsistent event timelines.",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        default=None,
        help="Directory containing snapshot JSONL (e.g. data/polymarket_snapshots/live_1s for 1s-cached live prices). Default: COLLECTOR_OUTPUT_DIR.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "data" / "game_records"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fix_timestamps:
        _run_fix_timestamps(out_dir)
        return

    if args.annotate_event_feed_incomplete:
        _annotate_event_feed_incomplete(out_dir)
        return

    if args.remove_event_feed_incomplete:
        _remove_event_feed_incomplete(out_dir)
        return

    if not args.from_date or not args.to_date:
        parser.error("--from and --to are required unless using --fix-timestamps")

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

    id_to_abbrev = _team_id_to_abbrev()
    snap_dir = args.snapshots_dir
    if snap_dir is None:
        snap_dir = COLLECTOR_OUTPUT_DIR
    else:
        snap_dir = Path(snap_dir)
        if not snap_dir.is_absolute():
            snap_dir = Path(__file__).resolve().parent / snap_dir
    snapshots = _load_snapshots(snap_dir)
    print(f"Loaded {sum(len(v) for v in snapshots.values())} snapshot rows for {len(snapshots)} condition_ids (from {snap_dir})")
    if args.fix_invalid_only:
        print("Mode: --fix-invalid-only (only rebuild records that need fix; skip valid).")

    all_records = []
    skipped_valid = 0
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        games = get_completed_games_rich_for_date(date_str)
        for g in games:
            home_abbrev = id_to_abbrev.get(g["home_team_id"], g["home_team_id"])
            away_abbrev = id_to_abbrev.get(g["away_team_id"], g["away_team_id"])
            record_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            fname = f"{record_date}_{away_abbrev}_{home_abbrev}.json"
            out_path = out_dir / fname
            if args.fix_invalid_only:
                needs_fix, reason = _record_needs_fix(out_path, accept_event_score_mismatch=args.accept_event_score_mismatch)
                if not needs_fix:
                    skipped_valid += 1
                    print(f"  {record_date} {away_abbrev} @ {home_abbrev}: skip (valid)")
                    continue
                if reason:
                    print(f"  {record_date} {away_abbrev} @ {home_abbrev}: rebuild ({reason})")
            market_ids = get_market_ids_for_game(date_str, home_abbrev, away_abbrev)
            condition_id = market_ids[0] if market_ids else ""
            token_id_home = market_ids[1] if market_ids and len(market_ids) > 1 else ""
            token_id_away = market_ids[2] if market_ids and len(market_ids) > 2 else ""
            # Skip games with no Polymarket market (no condition_id) so we don't create empty records
            if not condition_id:
                print(f"  {record_date} {away_abbrev} @ {home_abbrev}: skip (no Polymarket market)")
                continue
            events = get_game_events(g["event_id"])
            price_series = []
            if condition_id and condition_id in snapshots and not args.refresh_prices:
                price_series = snapshots[condition_id]
            # When we have token IDs but no snapshot (or --refresh-prices), try Polymarket prices-history API
            # Use 48h window (game date + next day) so US evening games (e.g. 7 PM Central = 01:00 UTC next day)
            # and resolution are included; otherwise we'd cut off at midnight UTC and miss the game.
            if condition_id and not price_series and token_id_home and token_id_away:
                try:
                    day_start = datetime.strptime(date_str, "%Y%m%d")
                    day_end = day_start + timedelta(days=2)  # 48h to cover game + resolution in UTC
                    start_ts = int(day_start.timestamp())
                    end_ts = int(day_end.timestamp())
                    price_series = get_prices_history_for_market(
                        token_id_home, token_id_away, start_ts, end_ts, interval="1m",
                        full_history=getattr(args, "full_price_history", False),
                        # fidelity: API requires min 10 for 1m; do not pass 1
                    )
                except Exception:
                    pass
            # Fallback: current CLOB (resolved markets often 404)
            if condition_id and not price_series and token_id_home and token_id_away:
                try:
                    client = get_clob_client()
                    home_book = get_order_book(client, token_id_home, silent_404=True)
                    away_book = get_order_book(client, token_id_away, silent_404=True)
                    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                    row = {
                        "timestamp": ts,
                        "bid_home": home_book.get("best_bid"),
                        "ask_home": home_book.get("best_ask"),
                        "bid_away": away_book.get("best_bid"),
                        "ask_away": away_book.get("best_ask"),
                    }
                    if any(v is not None for v in [row["bid_home"], row["ask_home"], row["bid_away"], row["ask_away"]]):
                        price_series = [row]
                except Exception:
                    pass

            # Normalize timestamps to canonical ISO UTC and sort by time (for correct training alignment)
            if price_series:
                price_series = normalize_price_series_timestamps(price_series)

            # Annotate when ESPN event feed is incomplete (last goal in events != final score)
            event_feed_incomplete = False
            if events:
                last_goal = _last_scoring_event(events)
                if last_goal is not None and last_goal.get("period", 0) < 5:
                    if last_goal.get("home_score") != g["home_score"] or last_goal.get("away_score") != g["away_score"]:
                        event_feed_incomplete = True
                elif last_goal is None:
                    last = events[-1]
                    if last.get("period", 0) < 5 and (last.get("home_score") != g["home_score"] or last.get("away_score") != g["away_score"]):
                        event_feed_incomplete = True

            record = {
                "event_id": g["event_id"],
                "condition_id": condition_id or "",
                "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                "home_team_id": g["home_team_id"],
                "away_team_id": g["away_team_id"],
                "home_abbrev": home_abbrev,
                "away_abbrev": away_abbrev,
                "home_score": g["home_score"],
                "away_score": g["away_score"],
                "home_won": g["home_won"],
                "linescores_home": g["home_linescores"],
                "linescores_away": g["away_linescores"],
                "events": events,
                "token_price_series": price_series,
            }
            if event_feed_incomplete:
                record["event_feed_incomplete"] = True
            all_records.append(record)

            if args.jsonl:
                continue
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            n_events = len(events)
            n_prices = len(price_series)
            if n_prices == 0:
                extra = " (condition_id found; run data_collector on game days to get price rows)"
            else:
                extra = ""
            print(f"  {record['date']} {away_abbrev} @ {home_abbrev}: {n_events} events, {n_prices} price rows -> {fname}{extra}")
            # Warn if we just rebuilt but event score still doesn't match final (ESPN feed incomplete)
            if args.fix_invalid_only and events:
                still_bad, _ = _record_needs_fix(out_path, accept_event_score_mismatch=False)
                if still_bad:
                    print(f"    ^ still invalid after rebuild (ESPN event feed may be incomplete; use --accept-event-score-mismatch to stop rewriting)")

        current += timedelta(days=1)

    if args.jsonl:
        jsonl_path = out_dir / "game_records.jsonl"
        with open(jsonl_path, "w") as f:
            for rec in all_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(all_records)} game records -> {jsonl_path}")

    if args.fix_invalid_only and skipped_valid > 0:
        print(f"Done. Rebuilt {len(all_records)} record(s), skipped {skipped_valid} valid.")
    else:
        print(f"Done. Total games: {len(all_records)}")


if __name__ == "__main__":
    main()
