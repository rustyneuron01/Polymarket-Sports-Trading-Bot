"""
Build in-game training dataset: one row per (game, snapshot time).
Target = REWARD (did we profit if we buy at this point and sell in the next window?), not who wins.
Win/loss is for Elo; the trading model must predict token price change / profitability.

Output: JSONL with columns:
  - game_id, timestamp, game_elapsed_sec, period, score_home, score_away, score_gap (home - away),
  - time_remaining_sec, elo_home, elo_away (pre-game Elo at prediction time: computed chronologically, updated after each game; home/away matter),
  - ask_home, bid_home, ask_away, bid_away (at this snapshot),
  - ask_home_delta, bid_home_delta, ask_away_delta, bid_away_delta (price change since previous snapshot; event/score-aligned),
  - reward_home (1 if max bid_home in next window > ask_home + fee), reward_away (same),
  - loss_home (1 if min bid_home in next window < entry - fee; "would have lost" — model learns when to sell early), loss_away (same),
  - optional: max_bid_home_next, max_bid_away_next (for regression).
  Score/period are aligned to wall_elapsed so they reflect game state when the price was observed. score_gap = home - away; high gap -> leading team token tends to go up. Strategy: focus on score first, then time remaining; strong team trailing + time left + low price -> buy; when to sell is learned from reward/loss on past data.

Usage:
  python -m polymarket_nhl_bot.build_in_game_dataset --dir data/game_records --out data/in_game_dataset.jsonl --window-sec 600 --fee 0.02
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

# NHL: 20 min regulation period in seconds
PERIOD_SEC = 20 * 60


def _parse_clock(clock: str) -> float:
    """Clock '14:40' (min:sec left) -> seconds remaining in period."""
    if not clock or not isinstance(clock, str):
        return 0.0
    parts = clock.strip().split(":")
    if len(parts) == 2:
        try:
            return float(parts[0]) * 60 + float(parts[1])
        except ValueError:
            pass
    return 0.0


def event_to_game_second(period: int, clock: str) -> float:
    """Event (period, clock) -> game elapsed seconds. Clock = time left in period."""
    p = max(1, int(period))
    sec_left = _parse_clock(clock)
    sec_elapsed_in_period = PERIOD_SEC - sec_left
    return (p - 1) * PERIOD_SEC + max(0, sec_elapsed_in_period)


def merge_price_rows(series: list[dict]) -> list[dict]:
    """
    Price series can have separate rows for home/away. Merge so each row has all four (bid/ask home/away).
    Forward-fill nulls from previous row; use first row's timestamp for each merged row.
    """
    if not series:
        return []
    out: list[dict] = []
    last = {"bid_home": None, "ask_home": None, "bid_away": None, "ask_away": None}
    for row in series:
        ts = row.get("timestamp") or ""
        r = dict(last)
        for k in ("bid_home", "ask_home", "bid_away", "ask_away"):
            v = row.get(k)
            if v is not None:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
            if r.get(k) is not None:
                last[k] = r[k]
        r["timestamp"] = ts
        out.append(r)
    return out


def parse_ts(ts: str) -> datetime | None:
    """Parse ISO timestamp (with or without Z / +00:00, optional milliseconds)."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        # Include timezone so parsing doesn't fail (do not truncate to [:19] or we drop +00:00).
        s = ts.strip().replace("Z", "+00:00")
        if "+" in s or s.endswith("00:00"):
            return datetime.fromisoformat(s)
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


def ts_to_sec(t: datetime) -> float:
    return t.timestamp()


def build_events_game_seconds(events: list[dict]) -> list[tuple[float, int, int, int]]:
    """Return [(game_second, period, home_score, away_score), ...] sorted by game_second."""
    out = []
    for ev in events:
        p = int(ev.get("period", 1))
        clock = ev.get("clock", "")
        gs = event_to_game_second(p, clock)
        h = int(ev.get("home_score", 0))
        a = int(ev.get("away_score", 0))
        out.append((gs, p, h, a))
    out.sort(key=lambda x: x[0])
    return out


def state_at_game_second(events_gs: list[tuple[float, int, int, int]], game_elapsed_sec: float) -> tuple[int, int, int, float]:
    """Return (period, score_home, score_away, time_remaining_sec) at game_elapsed_sec."""
    period, score_home, score_away = 1, 0, 0
    for gs, p, h, a in events_gs:
        if gs <= game_elapsed_sec:
            period, score_home, score_away = p, h, a
    # Time remaining in game (regulation 60 min)
    regulation_total = 3 * PERIOD_SEC
    time_remaining = max(0.0, regulation_total - game_elapsed_sec)
    return period, score_home, score_away, time_remaining


def price_and_state_series_from_record(
    rec: dict,
    min_price_rows: int = 2,
) -> tuple[list[dict], list[dict]]:
    """
    From one game record, build (price_series, game_state_series) for backtest.
    price_series: list of { timestamp, ask_home, bid_home, ask_away, bid_away }.
    game_state_series[i]: { timestamp, score_home, score_away, home_goals, away_goals, period, time_remaining_sec, game_elapsed_sec, game_second_proxy }.
    Returns ([], []) if record has too few price rows.
    """
    series = rec.get("token_price_series") or []
    events = rec.get("events") or []
    if len(series) < min_price_rows:
        return [], []
    merged = merge_price_rows(series)
    events_gs = build_events_game_seconds(events)
    state_series = []
    for i, row in enumerate(merged):
        game_second_proxy = i * 60.0
        period, score_home, score_away, time_remaining = state_at_game_second(events_gs, game_second_proxy)
        state_series.append({
            "timestamp": row.get("timestamp") or "",
            "score_home": score_home,
            "score_away": score_away,
            "home_goals": score_home,
            "away_goals": score_away,
            "period": period,
            "time_remaining_sec": round(time_remaining, 1),
            "game_elapsed_sec": round(game_second_proxy, 1),
            "game_second_proxy": round(game_second_proxy, 1),
        })
    return merged, state_series


def build_dataset_from_record(
    rec: dict,
    window_sec: float = 600,
    fee: float = 0.02,
    min_price_rows: int = 5,
    elo_home: float | None = None,
    elo_away: float | None = None,
) -> list[dict]:
    """
    From one game record, build list of in-game snapshot rows with reward target.
    reward_home = 1 if in the next window_sec the bid_home ever >= ask_home(now) + fee (so we could sell at profit).
    elo_home, elo_away: pre-game Elo at the time of this game (home and away). Caller should compute these
    chronologically so Elo reflects only prior results. If None, fall back to 1500 (no historical file).
    """
    series = rec.get("token_price_series") or []
    events = rec.get("events") or []
    if len(series) < min_price_rows:
        return []
    merged = merge_price_rows(series)
    events_gs = build_events_game_seconds(events)
    game_id = rec.get("event_id") or rec.get("date", "") + "_" + (rec.get("home_abbrev") or "") + "_" + (rec.get("away_abbrev") or "")
    # Team strength: use pre-game Elo passed by caller (historical at game date). Home/away matter for training.
    if elo_home is None:
        elo_home = 1500.0
    if elo_away is None:
        elo_away = 1500.0
    # Events use game clock; we don't have game start timestamp. Use row index * 60s as proxy for game_second to align score/period.
    rows = []
    for i in range(len(merged)):
        row = merged[i]
        ts_str = row.get("timestamp") or ""
        t = parse_ts(ts_str)
        if not t:
            continue
        first_ts = parse_ts(merged[0].get("timestamp") or "")
        wall_elapsed = (ts_to_sec(t) - ts_to_sec(first_ts)) if first_ts else 0.0
        if wall_elapsed < 0:
            wall_elapsed = 0.0
        game_second_proxy = i * 60.0  # proxy: ~1 snapshot per minute
        period, score_home, score_away, time_remaining = state_at_game_second(events_gs, game_second_proxy)
        ask_h = row.get("ask_home")
        bid_h = row.get("bid_home")
        ask_a = row.get("ask_away")
        bid_a = row.get("bid_away")
        if ask_h is None and bid_h is None and ask_a is None and bid_a is None:
            continue

        # Max/min bid and min ask in next window (for reward, loss, and buy/sell-opportunity targets)
        max_bid_h = None
        max_bid_a = None
        min_bid_h = None
        min_bid_a = None
        min_ask_h = ask_h  # include current row in window
        min_ask_a = ask_a
        t_end = ts_to_sec(t) + window_sec
        for j in range(i + 1, len(merged)):
            rj = merged[j]
            tj = parse_ts(rj.get("timestamp") or "")
            if not tj or ts_to_sec(tj) > t_end:
                break
            if rj.get("bid_home") is not None:
                max_bid_h = max(max_bid_h, rj["bid_home"]) if max_bid_h is not None else rj["bid_home"]
                min_bid_h = min(min_bid_h, rj["bid_home"]) if min_bid_h is not None else rj["bid_home"]
            if rj.get("bid_away") is not None:
                max_bid_a = max(max_bid_a, rj["bid_away"]) if max_bid_a is not None else rj["bid_away"]
                min_bid_a = min(min_bid_a, rj["bid_away"]) if min_bid_a is not None else rj["bid_away"]
            if rj.get("ask_home") is not None:
                min_ask_h = min(min_ask_h, rj["ask_home"]) if min_ask_h is not None else rj["ask_home"]
            if rj.get("ask_away") is not None:
                min_ask_a = min(min_ask_a, rj["ask_away"]) if min_ask_a is not None else rj["ask_away"]

        entry_h = ask_h if ask_h is not None else bid_h
        entry_a = ask_a if ask_a is not None else bid_a
        threshold = fee
        reward_home = 1 if (entry_h is not None and max_bid_h is not None and max_bid_h >= entry_h + threshold) else 0
        reward_away = 1 if (entry_a is not None and max_bid_a is not None and max_bid_a >= entry_a + threshold) else 0
        # Loss = would have realized a loss in window (price dropped below entry - fee). Model learns when to sell early.
        loss_home = 1 if (entry_h is not None and min_bid_h is not None and min_bid_h < entry_h - threshold) else 0
        loss_away = 1 if (entry_a is not None and min_bid_a is not None and min_bid_a < entry_a - threshold) else 0
        # Buy/sell opportunities: buy when current price is near the LOW in window, sell when near the HIGH (predict low/high per token; 4 trades per game = buy home, sell home, buy away, sell away).
        buy_opportunity_home = 1 if (entry_h is not None and min_ask_h is not None and entry_h <= min_ask_h * 1.01) else 0
        buy_opportunity_away = 1 if (entry_a is not None and min_ask_a is not None and entry_a <= min_ask_a * 1.01) else 0
        sell_opportunity_home = 1 if (bid_h is not None and max_bid_h is not None and bid_h >= max_bid_h * 0.99) else 0
        sell_opportunity_away = 1 if (bid_a is not None and max_bid_a is not None and bid_a >= max_bid_a * 0.99) else 0

        # Price-range targets for regression: ensure "high" >= "low" + fee so the model learns a profitable range.
        # Raw max_bid in window is often < min_ask (spread); then the model predicts sell < buy and we lose every time.
        floor_h = (min_ask_h + threshold) if min_ask_h is not None else None
        floor_a = (min_ask_a + threshold) if min_ask_a is not None else None
        if floor_h is not None:
            max_bid_h = max(max_bid_h, floor_h) if max_bid_h is not None else floor_h
            max_bid_h = min(1.0, float(max_bid_h))
        if floor_a is not None:
            max_bid_a = max(max_bid_a, floor_a) if max_bid_a is not None else floor_a
            max_bid_a = min(1.0, float(max_bid_a))

        # Price deltas since previous snapshot (score/event-aligned)
        prev = merged[i - 1] if i > 0 else {}
        def _delta(key: str) -> float:
            cur = row.get(key)
            pv = prev.get(key)
            if cur is not None and pv is not None:
                try:
                    return float(cur) - float(pv)
                except (TypeError, ValueError):
                    pass
            return 0.0
        ask_home_delta = _delta("ask_home")
        bid_home_delta = _delta("bid_home")
        ask_away_delta = _delta("ask_away")
        bid_away_delta = _delta("bid_away")

        # Score and time are primary; add derived features for prediction.
        score_gap = score_home - score_away
        time_remaining_ratio = min(1.0, max(0.0, time_remaining / (3 * PERIOD_SEC)))  # 0-1 over regulation
        abs_score_gap = abs(score_gap)
        elo_advantage = elo_home - elo_away

        rows.append({
            "game_id": game_id,
            "timestamp": ts_str,
            "game_elapsed_sec": round(wall_elapsed, 1),
            "game_second_proxy": round(game_second_proxy, 1),
            "period": period,
            "score_home": score_home,
            "score_away": score_away,
            "score_gap": score_gap,
            "time_remaining_sec": round(time_remaining, 1),
            "time_remaining_ratio": round(time_remaining_ratio, 4),
            "abs_score_gap": abs_score_gap,
            "elo_advantage": elo_advantage,
            "elo_home": elo_home,
            "elo_away": elo_away,
            "ask_home": ask_h,
            "bid_home": bid_h,
            "ask_away": ask_a,
            "bid_away": bid_a,
            "ask_home_delta": ask_home_delta,
            "bid_home_delta": bid_home_delta,
            "ask_away_delta": ask_away_delta,
            "bid_away_delta": bid_away_delta,
            "reward_home": reward_home,
            "reward_away": reward_away,
            "loss_home": loss_home,
            "loss_away": loss_away,
            "buy_opportunity_home": buy_opportunity_home,
            "buy_opportunity_away": buy_opportunity_away,
            "sell_opportunity_home": sell_opportunity_home,
            "sell_opportunity_away": sell_opportunity_away,
            "min_ask_home_next": min_ask_h,
            "min_ask_away_next": min_ask_a,
            "max_bid_home_next": max_bid_h,
            "max_bid_away_next": max_bid_a,
        })
    return rows


def _parse_date_from_fname(fname: str) -> str | None:
    """Extract YYYY-MM-DD from game record filename like 2026-02-28_AWAY_HOME.json. Returns None if not parseable."""
    if not fname or not fname.endswith(".json"):
        return None
    base = fname[:-5]  # drop .json
    if len(base) >= 10 and base[4] == "-" and base[7] == "-":
        return base[:10]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build in-game dataset: target = reward (profit if buy now, sell in window).")
    parser.add_argument("--dir", default=None, help="Game records directory (default: data/game_records)")
    parser.add_argument("--out", default=None, help="Output path (default: data/in_game_dataset.jsonl)")
    parser.add_argument("--from", dest="from_date", default=None, metavar="DATE",
        help="Only include games on or after this date (YYYY-MM-DD or YYYYMMDD). If not set, use all records in --dir.")
    parser.add_argument("--to", dest="to_date", default=None, metavar="DATE",
        help="Only include games on or before this date (YYYY-MM-DD or YYYYMMDD). If not set, use all records in --dir.")
    parser.add_argument("--window-sec", type=float, default=600, help="Lookahead window in seconds (default: 600 = 10 min)")
    parser.add_argument("--fee", type=float, default=0.02, help="Fee threshold for reward (default: 0.02)")
    parser.add_argument("--skip-incomplete", action="store_true", default=True, help="Skip event_feed_incomplete (default: True)")
    parser.add_argument("--no-skip-incomplete", action="store_false", dest="skip_incomplete")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    records_dir = Path(args.dir) if args.dir else base / "data" / "game_records"
    out_path = Path(args.out) if args.out else base / "data" / "in_game_dataset.jsonl"

    def normalize_date(s: str) -> str:
        s = s.strip().replace("-", "")
        if len(s) == 8:
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    from_date = None
    to_date = None
    if args.from_date:
        try:
            from_date = normalize_date(args.from_date)
        except Exception:
            parser.error("--from must be YYYY-MM-DD or YYYYMMDD")
    if args.to_date:
        try:
            to_date = normalize_date(args.to_date)
        except Exception:
            parser.error("--to must be YYYY-MM-DD or YYYYMMDD")
    if from_date and to_date and from_date > to_date:
        parser.error("--from must be <= --to")

    if not records_dir.exists():
        print(f"Records directory not found: {records_dir}")
        raise SystemExit(1)

    # Load all records that pass filters; sort by date so we can compute Elo at prediction time (before each game).
    from collections import defaultdict
    from elo import update_elo

    records_with_meta: list[tuple[dict, str, Path]] = []  # (rec, date, path)
    n_skipped = 0
    n_date_skipped = 0
    for p in sorted(records_dir.glob("*.json")):
        if p.name.startswith("."):
            continue
        rec_date = _parse_date_from_fname(p.name)
        if rec_date is not None:
            if from_date is not None and rec_date < from_date:
                n_date_skipped += 1
                continue
            if to_date is not None and rec_date > to_date:
                n_date_skipped += 1
                continue
        try:
            with open(p) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if args.skip_incomplete and rec.get("event_feed_incomplete"):
            n_skipped += 1
            continue
        date_str = rec_date or rec.get("date", "") or "0000-00-00"
        records_with_meta.append((rec, date_str, p))

    # Chronological order: Elo at prediction time = pre-game Elo (only prior results). Same day: stable order by event_id.
    records_with_meta.sort(key=lambda x: (x[1], x[0].get("event_id", ""), str(x[2])))

    current_elo: dict[str, float] = defaultdict(lambda: 1500.0)
    game_pre_elo: dict[str, dict[str, float]] = {}  # game_key -> { elo_home, elo_away } for backtest/replay
    all_rows = []
    for rec, _date, _path in records_with_meta:
        hid = str(rec.get("home_team_id") or "")
        aid = str(rec.get("away_team_id") or "")
        elo_h = current_elo[hid]
        elo_a = current_elo[aid]
        game_key = rec.get("event_id") or f"{rec.get('date','')}_{hid}_{aid}"
        game_pre_elo[game_key] = {"elo_home": elo_h, "elo_away": elo_a}
        rows = build_dataset_from_record(
            rec, window_sec=args.window_sec, fee=args.fee, elo_home=elo_h, elo_away=elo_a
        )
        all_rows.extend(rows)
        # Update Elo after this game so next games see updated strength
        home_won = bool(rec.get("home_won", False))
        new_h, new_a = update_elo(elo_h, elo_a, home_won)
        current_elo[hid] = new_h
        current_elo[aid] = new_a

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Sidecar: pre-game Elo per game so backtest/replay use same historical Elo (not current elo_ratings.json)
    pre_elo_path = out_path.parent / "game_pre_elo.json"
    with open(pre_elo_path, "w") as f:
        json.dump(game_pre_elo, f, indent=0)

    date_info = ""
    if from_date or to_date:
        date_info = f", date filter {from_date or '…'} to {to_date or '…'} (excluded {n_date_skipped} files)"
    print(f"Wrote {len(all_rows)} in-game snapshot rows to {out_path} (skipped {n_skipped} incomplete records{date_info}).")
    print(f"Wrote pre-game Elo for {len(game_pre_elo)} games to {pre_elo_path} (Elo at prediction time, chronological).")
    print("Target: reward_home / reward_away = 1 if we can sell in the next window at profit (after fee).")
    print("Use train_in_game_model.py to train on this dataset.")


if __name__ == "__main__":
    main()
