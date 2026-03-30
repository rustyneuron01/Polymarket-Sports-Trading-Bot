"""
Check training data: stats (price rows, events), validate random sample, list available and possible data.
Usage:
  python -m polymarket_nhl_bot.check_training_data
  python check_training_data.py --sample 50 --dir data/game_records
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from validate_game_records import _last_scoring_event, validate_record


def deep_validate(rec: dict, path: Path, accept_event_score_mismatch: bool = False) -> list[str]:
    """Run validate_record plus extra checks. Return list of error messages (empty if ok).
    When accept_event_score_mismatch=True, do not fail on last goal != final (ESPN feed incomplete)."""
    status, msg = validate_record(rec, path)
    if status != "ok":
        return [f"{status}: {msg}"]

    errors = []

    # Linescores: home total should equal home_score, away total = away_score
    home_lines = rec.get("linescores_home") or []
    away_lines = rec.get("linescores_away") or []
    home_sum = sum(float(p.get("value", 0)) for p in home_lines if isinstance(p, dict))
    away_sum = sum(float(p.get("value", 0)) for p in away_lines if isinstance(p, dict))
    if abs(home_sum - rec.get("home_score", 0)) > 0.01 or abs(away_sum - rec.get("away_score", 0)) > 0.01:
        errors.append(f"linescores sum {home_sum}-{away_sum} != final {rec.get('home_score')}-{rec.get('away_score')}")

    # Last scoring event (goal) score should match final (for period < 5), unless accept_event_score_mismatch
    if not accept_event_score_mismatch:
        events = rec.get("events") or []
        if events:
            last_goal = _last_scoring_event(events)
            if last_goal and last_goal.get("period", 0) < 5:
                if last_goal.get("home_score") != rec.get("home_score") or last_goal.get("away_score") != rec.get("away_score"):
                    errors.append(f"last goal {last_goal.get('home_score')}-{last_goal.get('away_score')} != final {rec.get('home_score')}-{rec.get('away_score')}")

    # home_won consistent with score
    home_won = rec.get("home_won", False)
    home_score = rec.get("home_score", 0)
    away_score = rec.get("away_score", 0)
    if home_won and home_score <= away_score:
        errors.append(f"home_won=true but score {home_score}-{away_score}")
    if not home_won and home_score >= away_score and home_score != away_score:
        errors.append(f"home_won=false but score {home_score}-{away_score}")

    # Price series chronological
    series = rec.get("token_price_series") or []
    timestamps = [r.get("timestamp") or "" for r in series]
    if timestamps != sorted(timestamps):
        errors.append("token_price_series not sorted by timestamp")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check training data stats and validate random sample.")
    parser.add_argument("--dir", default=None, help="Game records directory (default: data/game_records)")
    parser.add_argument("--sample", type=int, default=15, help="Number of random files to deep-validate (0 = all)")
    parser.add_argument(
        "--accept-event-score-mismatch",
        action="store_true",
        help="Do not fail when last goal != final (ESPN feed incomplete). Use to match build_game_records --accept-event-score-mismatch.",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = Path(args.dir) if args.dir else base / "data" / "game_records"
    if not out_dir.exists():
        print(f"Directory not found: {out_dir}")
        raise SystemExit(1)

    files = sorted(out_dir.glob("*.json"))
    files = [p for p in files if not p.name.startswith(".")]
    if not files:
        print("No game record JSONs found.")
        raise SystemExit(1)

    # --- Stats ---
    price_counts = []
    event_counts = []
    goal_counts = []
    for p in files:
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception:
            continue
        series = rec.get("token_price_series") or []
        events = rec.get("events") or []
        goals = [e for e in events if e.get("scoring_play")]
        price_counts.append(len(series))
        event_counts.append(len(events))
        goal_counts.append(len(goals))

    n = len(price_counts)
    print("=== Training data stats ===\n")
    print(f"Games: {n}")
    for name, arr in [
        ("token_price_series rows (price snapshots) per game", price_counts),
        ("events (goals + penalties + injuries) per game", event_counts),
        ("goals only (scoring_play) per game", goal_counts),
    ]:
        if not arr:
            continue
        s = sorted(arr)
        print(f"  {name}:")
        print(f"    min={min(arr)} max={max(arr)} mean={sum(arr)/len(arr):.1f} median={s[len(s)//2]}")
        print(f"    p10={s[int(len(s)*0.10)]} p90={s[int(len(s)*0.90)]}")

    # --- Random sample deep validation ---
    random.seed(42)
    k = min(args.sample, len(files)) if args.sample > 0 else len(files)
    sample_files = random.sample(files, k) if k < len(files) else files

    print(f"\n=== Deep validation (random sample n={len(sample_files)}) ===\n")
    if args.accept_event_score_mismatch:
        print("(--accept-event-score-mismatch: ignoring last goal != final)\n")
    ok_count = 0
    for p in sorted(sample_files):
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception as e:
            print(f"  FAIL {p.name}: read error {e}")
            continue
        errs = deep_validate(rec, p, accept_event_score_mismatch=args.accept_event_score_mismatch)
        if not errs:
            note = " (event_feed_incomplete)" if rec.get("event_feed_incomplete") else ""
            print(f"  OK   {p.name} ({len(rec.get('token_price_series') or [])} prices, {len(rec.get('events') or [])} events){note}")
            ok_count += 1
        else:
            print(f"  FAIL {p.name}: {'; '.join(errs)}")
    print(f"\nSample result: {ok_count}/{len(sample_files)} passed deep validation.")

    # --- Data we have / could add ---
    print("\n=== Data in each game record (for training) ===\n")
    print("  Per game (top-level): event_id, condition_id, date, home/away_team_id, home/away_abbrev,")
    print("  home_score, away_score, home_won, linescores_home, linescores_away, events, token_price_series.")
    print("  Per event: period, clock, home_score, away_score, event_type, scoring_play, text.")
    print("  Per price row: timestamp (UTC), bid_home, ask_home, bid_away, ask_away.")
    print("\n  Other data you could add or join at training time:")
    print("  - Game start time (UTC): not stored; infer from first price timestamp or ESPN schedule API.")
    print("  - Rest days / back-to-back: fetch from ESPN team schedule by date when building features.")
    print("  - Goalie / lineup: ESPN or NHL API at prediction time; not in game record.")
    print("  - Derived: minutes_elapsed per price row (from game_start_utc), score at each timestamp (interpolate from events).")


if __name__ == "__main__":
    main()
