"""
Replay a PAST game using its stored price and event data — test the model as if in real time.

Uses game_records (previous games with full token_price_series and events). Steps through
each time step in order; at each step runs the reward model on the history-so-far (prices
and game state that would have been available at that moment) and prints the signal.
So you test the model on real past data, in "real time" replay.

Usage:
  # List available game records (with price row counts)
  python replay_game.py --list
  python replay_game.py --list --date 2026-01-15

  # Replay one game (fast: no delay between steps)
  python replay_game.py 2026-01-15_TOR_VGK
  python replay_game.py --record data/game_records/2026-01-15_TOR_VGK.json

  # Replay with 0.1s delay per step so you can watch
  python replay_game.py 2026-01-15_TOR_VGK --speed 0.1

  # Replay and show only steps where signal is not HOLD
  python replay_game.py 2026-01-31_NYR_PIT --signal-only

  # Same strategy as live_test (price-range, both tokens)
  python replay_game.py 2026-02-28_AWAY_HOME --use-price-range
  python replay_game.py 2026-02-28_AWAY_HOME --use-price-range --single-token
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from build_in_game_dataset import price_and_state_series_from_record


def load_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def replay(
    record_path: Path,
    model_path: Path,
    speed_sec: float = 0,
    signal_only: bool = False,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    use_price_range: bool = False,
    use_dual: bool = True,
) -> None:
    from train_in_game_model import load_reward_model
    from in_game_strategy import reward_model_signal_fn, price_range_signal_fn, price_range_signal_fn_dual

    rec = load_record(record_path)
    if not rec:
        print(f"Failed to load record: {record_path}")
        return
    price_series, game_state_series = price_and_state_series_from_record(rec)
    if len(price_series) < 2:
        print(f"Record has too few price rows ({len(price_series)}). Need at least 2.")
        return
    # Elo at prediction time (pre-game). Prefer game_pre_elo.json from build_in_game_dataset (chronological).
    elo_h, elo_a = 1500.0, 1500.0
    pre_elo_path = record_path.parent.parent / "game_pre_elo.json"  # data/game_pre_elo.json when record in data/game_records/
    if pre_elo_path.exists():
        try:
            with open(pre_elo_path) as f:
                pre_elo = json.load(f)
            game_key = rec.get("event_id") or f"{rec.get('date','')}_{rec.get('home_team_id')}_{rec.get('away_team_id')}"
            if game_key in pre_elo:
                elo_h = pre_elo[game_key]["elo_home"]
                elo_a = pre_elo[game_key]["elo_away"]
        except Exception:
            pass
    if elo_h == 1500.0 and elo_a == 1500.0:
        try:
            from model import get_elo
            elo_h = get_elo(rec.get("home_team_id") or "")
            elo_a = get_elo(rec.get("away_team_id") or "")
        except Exception:
            pass
    for s in game_state_series:
        s["elo_home"] = elo_h
        s["elo_away"] = elo_a

    bundle = load_reward_model(model_path)
    if bundle is None:
        print(f"Failed to load model: {model_path}")
        return
    if use_price_range and "model_min_ask_home" in bundle:
        signal_fn = price_range_signal_fn_dual(bundle) if use_dual else price_range_signal_fn(bundle)
        strategy_label = "price-range (both tokens)" if use_dual else "price-range (single)"
    else:
        signal_fn = reward_model_signal_fn(bundle, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
        strategy_label = "reward (P(reward) thresholds)"
    print(f"Strategy: {strategy_label}")

    game_id = rec.get("event_id") or record_path.stem
    date = rec.get("date", "")
    home = rec.get("home_abbrev", "")
    away = rec.get("away_abbrev", "")
    final = f"{rec.get('home_score', 0)}-{rec.get('away_score', 0)}"
    print(f"Replay: {date} {away} @ {home} (final {final})  |  {len(price_series)} steps")

    # Actual min/max from full price series (so you can verify program and data)
    ah_vals = [r.get("ask_home") for r in price_series if r.get("ask_home") is not None]
    bh_vals = [r.get("bid_home") for r in price_series if r.get("bid_home") is not None]
    aa_vals = [r.get("ask_away") for r in price_series if r.get("ask_away") is not None]
    ba_vals = [r.get("bid_away") for r in price_series if r.get("bid_away") is not None]
    actual_home_low = min(ah_vals) if ah_vals else None
    actual_home_high = max(bh_vals) if bh_vals else None
    actual_away_low = min(aa_vals) if aa_vals else None
    actual_away_high = max(ba_vals) if ba_vals else None
    if actual_home_low is not None or actual_away_low is not None:
        home_str = f"Home low={actual_home_low:.3f} high={actual_home_high:.3f}" if actual_home_low is not None else "Home —"
        away_str = f"Away low={actual_away_low:.3f} high={actual_away_high:.3f}" if actual_away_low is not None else "Away —"
        print(f"Actual price range (full game):  {home_str}  |  {away_str}")
    print("---")

    trades = []
    total_profit = 0.0

    if use_dual and use_price_range and "model_min_ask_home" in bundle:
        # Dual price-range: same logic as live_test (position_home, position_away, SELL_HOME/SELL_AWAY)
        position_home, position_away = False, False
        entry_price_home, entry_price_away = None, None
        for i in range(len(price_series)):
            row = price_series[i]
            state = dict(game_state_series[i] if i < len(game_state_series) else {})
            state["_entry_price_home"] = entry_price_home
            state["_entry_price_away"] = entry_price_away
            history_so_far = price_series[: i + 1]
            signal = signal_fn(state, history_so_far, position_home, position_away)
            ts = row.get("timestamp", "")
            score = f"{state.get('score_home', 0)}-{state.get('score_away', 0)}"
            period = state.get("period", 1)
            ah, bh = row.get("ask_home"), row.get("bid_home")
            aa, ba = row.get("ask_away"), row.get("bid_away")
            price_str = f"home bid={bh} ask={ah}  away bid={ba} ask={aa}"
            if signal_only and signal == "HOLD" and not (position_home or position_away):
                if speed_sec > 0:
                    time.sleep(speed_sec)
                continue
            if signal == "BUY_HOME" and not position_home and ah is not None:
                position_home, entry_price_home = True, float(ah)
                trades.append(("BUY", "home", ah, ts))
                print(f"[{i+1}/{len(price_series)}] {ts}  {score} P{period}  |  {price_str}  ->  {signal} (buy @ {ah:.3f})")
            elif signal == "BUY_AWAY" and not position_away and aa is not None:
                position_away, entry_price_away = True, float(aa)
                trades.append(("BUY", "away", aa, ts))
                print(f"[{i+1}/{len(price_series)}] {ts}  {score} P{period}  |  {price_str}  ->  {signal} (buy @ {aa:.3f})")
            elif signal == "SELL_HOME" and position_home:
                sell_price = float(bh) if bh is not None else (entry_price_home or 0)
                profit = sell_price - (entry_price_home or 0)
                total_profit += profit
                trades.append(("SELL", "home", sell_price, ts))
                print(f"[{i+1}/{len(price_series)}] {ts}  {score} P{period}  |  {price_str}  ->  {signal} (sell @ {sell_price:.3f}  profit={profit:+.3f})")
                position_home, entry_price_home = False, None
            elif signal == "SELL_AWAY" and position_away:
                sell_price = float(ba) if ba is not None else (entry_price_away or 0)
                profit = sell_price - (entry_price_away or 0)
                total_profit += profit
                trades.append(("SELL", "away", sell_price, ts))
                print(f"[{i+1}/{len(price_series)}] {ts}  {score} P{period}  |  {price_str}  ->  {signal} (sell @ {sell_price:.3f}  profit={profit:+.3f})")
                position_away, entry_price_away = False, None
            else:
                if not signal_only or signal != "HOLD":
                    print(f"[{i+1}/{len(price_series)}] {ts}  {score} P{period}  |  {price_str}  ->  {signal}")
            if speed_sec > 0:
                time.sleep(speed_sec)
        # Close at end
        last = price_series[-1] if price_series else {}
        if position_home:
            bid = last.get("bid_home")
            sell_price = float(bid) if bid is not None else (entry_price_home or 0)
            total_profit += sell_price - (entry_price_home or 0)
            trades.append(("SELL", "home", sell_price, last.get("timestamp", "")))
            print(f"[END] Close home @ {sell_price:.3f}")
        if position_away:
            bid = last.get("bid_away")
            sell_price = float(bid) if bid is not None else (entry_price_away or 0)
            total_profit += sell_price - (entry_price_away or 0)
            trades.append(("SELL", "away", sell_price, last.get("timestamp", "")))
            print(f"[END] Close away @ {sell_price:.3f}")
    else:
        # Single position: reward model or single-token price-range (BUY_HOME/BUY_AWAY/SELL)
        position = None
        buy_price = 0.0
        for i in range(len(price_series)):
            row = price_series[i]
            state = game_state_series[i] if i < len(game_state_series) else {}
            history_so_far = price_series[: i + 1]
            signal = signal_fn(state, history_so_far, position)
            if signal_only and signal == "HOLD":
                if position is not None or (i == len(price_series) - 1 and not trades):
                    pass
                else:
                    if speed_sec > 0:
                        time.sleep(speed_sec)
                    continue
            ts = row.get("timestamp", "")
            score = f"{state.get('score_home', 0)}-{state.get('score_away', 0)}"
            period = state.get("period", 1)
            ah, bh = row.get("ask_home"), row.get("bid_home")
            aa, ba = row.get("ask_away"), row.get("bid_away")
            price_str = f"home bid={bh} ask={ah}  away bid={ba} ask={aa}"
            if signal == "BUY_HOME" and position is None and ah is not None:
                position, buy_price = "home", ah
                trades.append(("BUY", "home", ah, ts))
                print(f"[{i+1}/{len(price_series)}] {ts}  {score} P{period}  |  {price_str}  ->  {signal} (buy @ {ah:.3f})")
            elif signal == "BUY_AWAY" and position is None and aa is not None:
                position, buy_price = "away", aa
                trades.append(("BUY", "away", aa, ts))
                print(f"[{i+1}/{len(price_series)}] {ts}  {score} P{period}  |  {price_str}  ->  {signal} (buy @ {aa:.3f})")
            elif signal == "SELL" and position is not None:
                bid = bh if position == "home" else ba
                sell_price = float(bid) if bid is not None else buy_price
                total_profit += sell_price - buy_price
                trades.append(("SELL", position, sell_price, ts))
                print(f"[{i+1}/{len(price_series)}] {ts}  {score} P{period}  |  {price_str}  ->  {signal} (sell @ {sell_price:.3f}  profit={sell_price - buy_price:+.3f})")
                position = None
            else:
                if not signal_only:
                    print(f"[{i+1}/{len(price_series)}] {ts}  {score} P{period}  |  {price_str}  ->  {signal}")
            if speed_sec > 0:
                time.sleep(speed_sec)
        if position is not None and price_series:
            last = price_series[-1]
            bid = last.get("bid_home") if position == "home" else last.get("bid_away")
            sell_price = float(bid) if bid is not None else buy_price
            total_profit += sell_price - buy_price
            trades.append(("SELL", position, sell_price, last.get("timestamp", "")))
            print(f"[END] Close {position} @ {sell_price:.3f}  profit={sell_price - buy_price:+.3f}")

    # Predicted min/max (next-window) at last step, if model has price-range heads
    from train_in_game_model import predict_price_range
    from in_game_strategy import _build_feature_dict_for_reward
    if "model_min_ask_home" in bundle and price_series and game_state_series:
        last_row = price_series[-1]
        last_state = game_state_series[-1] if len(game_state_series) >= len(price_series) else (game_state_series[-1] if game_state_series else {})
        fd = _build_feature_dict_for_reward(last_state, last_row, len(price_series) - 1, price_series)
        pred_min_h, pred_max_h, pred_min_a, pred_max_a = predict_price_range(bundle, fd)
        if pred_min_h is not None:
            print("Predicted price range (next window at last step):  Home low={:.3f} high={:.3f}  |  Away low={:.3f} high={:.3f}".format(
                pred_min_h, pred_max_h if pred_max_h is not None else 0, pred_min_a if pred_min_a is not None else 0, pred_max_a if pred_max_a is not None else 0))

    print("---")
    print(f"Trades: {len([t for t in trades if t[0] == 'SELL'])} round-trips  |  Total profit (price): {total_profit:+.4f}")
    if trades:
        for t in trades:
            print(f"  {t[0]} {t[1]} @ {t[2]:.3f}  {t[3]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a past game from game_records: run model at each time step as if in real time"
    )
    parser.add_argument(
        "record",
        nargs="?",
        default=None,
        help="Game record file name (e.g. 2026-01-15_TOR_VGK) or path",
    )
    parser.add_argument("--record", dest="record_path_opt", type=str, default=None, help="Path to game record JSON")
    parser.add_argument("--list", action="store_true", help="List available game records (optionally filtered by --date)")
    parser.add_argument("--date", type=str, default=None, help="Filter list or pick record by date YYYY-MM-DD")
    parser.add_argument("--records-dir", type=str, default=None, help="Game records directory (default: data/game_records)")
    parser.add_argument("--model", type=str, default=None, help="Path to reward_model.pkl (default: data/reward_model.pkl)")
    parser.add_argument("--speed", type=float, default=0, help="Seconds to wait between steps (0 = as fast as possible)")
    parser.add_argument("--signal-only", action="store_true", help="Print only steps where signal is not HOLD")
    parser.add_argument("--buy-threshold", type=float, default=0.55)
    parser.add_argument("--sell-threshold", type=float, default=0.45)
    parser.add_argument("--use-price-range", action="store_true",
        help="Use price-range strategy (predict low/high; buy at low, sell at high). Same as live_test --use-price-range.")
    parser.add_argument("--single-token", action="store_true",
        help="With --use-price-range: trade one token per game only (else both home and away).")
    args = parser.parse_args()

    records_dir = Path(args.records_dir) if args.records_dir else BASE / "data" / "game_records"
    model_path = Path(args.model) if args.model else BASE / "data" / "reward_model.pkl"
    if not model_path.is_absolute():
        model_path = BASE / model_path

    if args.list:
        if not records_dir.exists():
            print(f"Records dir not found: {records_dir}")
            sys.exit(1)
        if args.date:
            files = sorted(records_dir.glob(f"{args.date}_*.json"))
        else:
            files = sorted(records_dir.glob("*.json"))
        if not files:
            print(f"No records found in {records_dir}" + (f" for date {args.date}" if args.date else ""))
            sys.exit(0)
        print(f"Game records ({len(files)})" + (f" for {args.date}" if args.date else "") + ":\n")
        for p in files[:80]:
            try:
                with open(p) as f:
                    rec = json.load(f)
                n = len(rec.get("token_price_series") or [])
                e = len(rec.get("events") or [])
                print(f"  {p.name}  prices={n}  events={e}")
            except Exception:
                print(f"  {p.name}  (read error)")
        if len(files) > 80:
            print(f"  ... and {len(files) - 80} more")
        print("\nReplay: python replay_game.py <name>   e.g.  python replay_game.py 2026-01-15_TOR_VGK")
        return

    record_path = None
    if args.record_path_opt:
        record_path = Path(args.record_path_opt)
        if not record_path.is_absolute():
            record_path = BASE / record_path
    elif args.record:
        r = args.record
        if "/" in r or r.endswith(".json"):
            record_path = Path(r)
        else:
            record_path = records_dir / (r if r.endswith(".json") else f"{r}.json")
    if not record_path or not record_path.exists():
        print("Specify a game record:  python replay_game.py <name>   or   --record <path>")
        print("Use --list to see available records.")
        sys.exit(1)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    use_dual = args.use_price_range and not args.single_token
    replay(
        record_path,
        model_path,
        speed_sec=args.speed,
        signal_only=args.signal_only,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        use_price_range=args.use_price_range,
        use_dual=use_dual,
    )


if __name__ == "__main__":
    main()
