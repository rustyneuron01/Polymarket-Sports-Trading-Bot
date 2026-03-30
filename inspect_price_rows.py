"""
Explain why game record JSONs have different numbers of price rows (e.g. 109 vs 300).

Price rows come from one of three sources when the record was built:
  1. Snapshot file (data_collector): one row per poll. Count = how many times the collector
     ran that day and this market was included. Same day ≈ same count if collector ran
     the whole time; different days or different collector run length → different counts.
  2. Polymarket prices-history API: we request a 48h window with fidelity=10 (one point
     per 10 min per token). Merged series = union of home + away timestamps. Count varies
     by how many points the API returns for that market (some markets have fewer points).
  3. Fallback (current CLOB): 1 row only.

Usage:
  python inspect_price_rows.py [--dir data/game_records] [--sample 20] [--date 2026-01-15]
  python inspect_price_rows.py --file data/game_records/2026-01-15_TOR_VGK.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _parse_ts(ts: str) -> float | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        s = ts.strip().replace("Z", "+00:00")
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def inspect_record(path: Path) -> dict | None:
    try:
        with open(path) as f:
            rec = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    series = rec.get("token_price_series") or []
    if not series:
        return {
            "file": path.name,
            "date": rec.get("date", ""),
            "game": f"{rec.get('away_abbrev', '')} @ {rec.get('home_abbrev', '')}",
            "n_rows": 0,
            "span_hours": None,
            "first_ts": None,
            "last_ts": None,
            "reason": "no price data (snapshots missing or API returned nothing)",
        }
    first_ts = series[0].get("timestamp")
    last_ts = series[-1].get("timestamp")
    t0 = _parse_ts(first_ts)
    t1 = _parse_ts(last_ts)
    span_hours = (t1 - t0) / 3600.0 if (t0 is not None and t1 is not None and t1 >= t0) else None

    # Infer likely source from span and count
    n = len(series)
    if n == 1:
        reason = "1 row = fallback (current CLOB only; market likely resolved)"
    elif span_hours is not None and span_hours >= 40:
        reason = "~48h = from API (fidelity 10 min); count varies per market (merge of home+away timestamps)"
    elif span_hours is not None and span_hours <= 24:
        # Same day, same span, different counts: API returns different # of points per token per market; merge = union of timestamps
        reason = "from API or snapshot; same span different count = API points per token differ per market (merged home+away)"
    else:
        reason = "from snapshot (collector) or API; see span vs count"

    return {
        "file": path.name,
        "date": rec.get("date", ""),
        "game": f"{rec.get('away_abbrev', '')} @ {rec.get('home_abbrev', '')}",
        "n_rows": n,
        "span_hours": round(span_hours, 2) if span_hours is not None else None,
        "first_ts": first_ts[:19] if first_ts else None,
        "last_ts": last_ts[:19] if last_ts else None,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain why game records have different price row counts")
    parser.add_argument("--dir", type=str, default=None, help="Game records directory (default: data/game_records)")
    parser.add_argument("--file", type=str, default=None, help="Inspect a single JSON file")
    parser.add_argument("--sample", type=int, default=0, help="Show up to N files (0 = all)")
    parser.add_argument("--date", type=str, default=None, help="Filter by date YYYY-MM-DD")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = BASE / path
        if not path.exists():
            print(f"File not found: {path}")
            return
        rows = [path]
    else:
        rec_dir = Path(args.dir) if args.dir else BASE / "data" / "game_records"
        if not rec_dir.exists():
            print(f"Directory not found: {rec_dir}")
            return
        if args.date:
            rows = sorted(rec_dir.glob(f"{args.date}_*.json"))
        else:
            rows = sorted(rec_dir.glob("*.json"))
        if args.sample > 0:
            rows = rows[: args.sample]

    if not rows:
        print("No files found.")
        return

    print("Why different files have different price row counts\n")
    print(f"{'Date':<12} {'Game':<20} {'Rows':>6} {'Span(h)':>8}  Reason")
    print("-" * 80)
    for p in rows:
        if p.name.startswith("."):
            continue
        info = inspect_record(p)
        if not info:
            continue
        span = f"{info['span_hours']}" if info["span_hours"] is not None else "—"
        print(f"{info['date']:<12} {info['game']:<20} {info['n_rows']:>6} {span:>8}  {info['reason']}")
    print("-" * 80)
    print("""
Reasons in short:
  • From API: we request 48h with fidelity 10 min. Each token gets its own points; we MERGE by
    timestamp (union). So row count = size of that union. Different markets get different numbers
    of points from the API → e.g. 109 rows vs 166 rows for same day (same span) is normal.
  • From snapshot file (data_collector): 1 row per poll. Same day ≈ same count; different days
    or run length → different counts.
  • 1 row = fallback (current order book only).
""")


if __name__ == "__main__":
    main()
