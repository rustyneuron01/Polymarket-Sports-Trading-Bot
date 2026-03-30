"""
Test in-game strategy with REAL-TIME data from Polymarket and ESPN.

Fetches live order books (real bid/ask) and live game state (score, period) when a game is in progress,
runs the model and prints signals. Integrates with Polymarket for live testing during games.

Usage:
  # One-shot: fetch current prices + game state, run model, print signals
  python live_test.py
  python live_test.py --model data/reward_model.pkl

  # Use price-range strategy (predict low/high, buy at low / sell at high; both tokens by default)
  python live_test.py --use-price-range

  # Loop every 60s during a live game (keeps price history for deltas)
  python live_test.py --loop 60
  python live_test.py --use-price-range --loop 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_TRADES_LOG = BASE / "data" / "live_trades.jsonl"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from polymarket_client import get_clob_client, get_order_books, discover_nhl_markets
from espn_client import get_scoreboard, event_to_game_info, get_live_game_state_from_event

MAX_PRICE_HISTORY = 120  # keep last N snapshots per market when --loop (for deltas)


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
        nickname = parts[-1]  # e.g. Rangers, Penguins, Predators
        if nickname in q:
            return True
    return False


def _match_market_to_event(markets, events):
    """Yield (market, event) for markets that match an event by team names in question.
    Uses full name or nickname (last word) so 'Predators vs. Blue Jackets' matches
    ESPN's 'Nashville Predators' and 'Columbus Blue Jackets'."""
    for m in markets:
        q = (m.question or "").strip()
        for ev in events:
            info = event_to_game_info(ev)
            if not info:
                continue
            home = (info.get("home_team_name") or "").strip()
            away = (info.get("away_team_name") or "").strip()
            if home and away and _team_in_question(home, q) and _team_in_question(away, q):
                yield m, ev
                break


def _append_trade_log(log_path: Path, record: dict) -> None:
    """Append one trade record (one JSON object per line) to log_path."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [Warning] Could not write trade log: {e}")


def _add_elo_to_game_state(game_state: dict, event) -> None:
    """Add elo_home, elo_away to game_state from event team IDs (for price model)."""
    info = event_to_game_info(event) if event else None
    if not info:
        game_state.setdefault("elo_home", 1500.0)
        game_state.setdefault("elo_away", 1500.0)
        return
    try:
        from model import get_elo
        hid = info.get("home_team_id")
        aid = info.get("away_team_id")
        game_state["elo_home"] = float(get_elo(str(hid))) if hid else 1500.0
        game_state["elo_away"] = float(get_elo(str(aid))) if aid else 1500.0
    except Exception:
        game_state.setdefault("elo_home", 1500.0)
        game_state.setdefault("elo_away", 1500.0)


def run_once(
    model_path: Path,
    verbose: bool = True,
    use_price_range: bool = False,
    use_dual: bool = True,
    market_states: dict | None = None,
    log_path: Path | None = None,
) -> dict | None:
    """
    Fetch live prices and game state from Polymarket + ESPN; run model; print signals.
    If market_states is provided (e.g. from previous --loop), append new row and use for history/deltas.
    Returns updated market_states for next poll (keyed by condition_id).
    Multiple games in one poll are handled: each market has its own state (condition_id); no cross-game mixing.
    """
    from train_in_game_model import load_reward_model
    from in_game_strategy import reward_model_signal_fn, price_range_signal_fn, price_range_signal_fn_dual

    bundle = load_reward_model(model_path)
    if bundle is None:
        print(f"Failed to load model: {model_path}")
        return market_states

    if use_price_range and "model_min_ask_home" in bundle:
        signal_fn = price_range_signal_fn_dual(bundle) if use_dual else price_range_signal_fn(bundle)
        strategy_label = "price-range (both tokens)" if use_dual else "price-range (single)"
    else:
        signal_fn = reward_model_signal_fn(bundle, buy_threshold=0.55, sell_threshold=0.45)
        strategy_label = "reward model (P(reward) thresholds)"

    client = get_clob_client()
    today_yyyymmdd = datetime.utcnow().strftime("%Y-%m-%d")
    markets = discover_nhl_markets(client, game_date_yyyymmdd=today_yyyymmdd)
    if not markets:
        print("No NHL markets found for today.")
        return market_states

    events = get_scoreboard(today_yyyymmdd.replace("-", ""))
    matched = list(_match_market_to_event(markets, events))
    if not matched:
        print("No markets matched today's games. Showing first market only.")
        matched = [(markets[0], None)]

    if market_states is None:
        market_states = {}

    # One timestamp per poll so all games in this cycle share the same time (handles multiple games per second/poll).
    poll_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for market, event in matched:
        home_book, away_book = get_order_books(client, market)
        row = {
            "timestamp": poll_ts,
            "ask_home": home_book.get("best_ask"),
            "bid_home": home_book.get("best_bid"),
            "ask_away": away_book.get("best_ask"),
            "bid_away": away_book.get("best_bid"),
        }
        cid = getattr(market, "condition_id", None) or str(id(market))
        state = market_states.get(cid) or {
            "price_series": [],
            "position_home": False,
            "position_away": False,
            "entry_price_home": None,
            "entry_size_home": None,
            "entry_time_home": None,
            "entry_price_away": None,
            "entry_size_away": None,
            "entry_time_away": None,
        }
        price_series_so_far = state["price_series"] + [row]
        if len(price_series_so_far) > MAX_PRICE_HISTORY:
            price_series_so_far = price_series_so_far[-MAX_PRICE_HISTORY:]
        state["price_series"] = price_series_so_far

        if event and get_live_game_state_from_event(event):
            game_state = dict(get_live_game_state_from_event(event))
        else:
            game_state = {
                "score_home": 0,
                "score_away": 0,
                "period": 1,
                "time_remaining_sec": 20 * 60,
                "game_elapsed_sec": 0,
                "game_second_proxy": 0,
            }
        _add_elo_to_game_state(game_state, event)

        if use_dual and use_price_range and "model_min_ask_home" in bundle:
            signal = signal_fn(game_state, price_series_so_far, state["position_home"], state["position_away"])
            # Simulate position update for next poll (so next time we know if we're "holding")
            if signal == "BUY_HOME":
                state["position_home"] = True
            elif signal == "SELL_HOME":
                state["position_home"] = False
            elif signal == "BUY_AWAY":
                state["position_away"] = True
            elif signal == "SELL_AWAY":
                state["position_away"] = False
        else:
            position = "home" if state.get("position_home") else ("away" if state.get("position_away") else None)
            signal = signal_fn(game_state, price_series_so_far, position)
            if signal == "BUY_HOME":
                state["position_home"], state["position_away"] = True, False
            elif signal == "BUY_AWAY":
                state["position_home"], state["position_away"] = False, True
            elif signal == "SELL":
                state["position_home"], state["position_away"] = False, False
        market_states[cid] = state

        pred_min_h = pred_max_h = pred_min_a = pred_max_a = None
        if "model_min_ask_home" in bundle:
            from train_in_game_model import predict_price_range
            from in_game_strategy import _build_feature_dict_for_reward
            fd = _build_feature_dict_for_reward(game_state, row, len(price_series_so_far) - 1, price_series_so_far)
            pred_min_h, pred_max_h, pred_min_a, pred_max_a = predict_price_range(bundle, fd)

        # Before buy/sell: check current order book and validate against strategy (use current prices, don't miss games)
        signal_for_check = signal
        if signal == "SELL":
            signal_for_check = "SELL_HOME" if state.get("position_home") else "SELL_AWAY" if state.get("position_away") else "HOLD"
        if signal_for_check != "HOLD" and signal_for_check in ("BUY_HOME", "BUY_AWAY", "SELL_HOME", "SELL_AWAY"):
            from execution import validate_in_game_order
            check = validate_in_game_order(
                signal_for_check,
                market,
                home_book,
                away_book,
                pred_min_h=pred_min_h,
                pred_max_h=pred_max_h,
                pred_min_a=pred_min_a,
                pred_max_a=pred_max_a,
                buy_margin=0.02,
                sell_margin=0.02,
            )
            if check.valid:
                print(f"  Order book check: EXECUTE at {check.execution_price:.3f} ({check.outcome_label})  size_at_best={check.size_at_best:.1f}")
                # Log trade to JSONL (amount, price, time, game name; on sell add pnl)
                ts = row.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                if log_path:
                    market_question = (getattr(market, "question", None) or "")[:200]
                    if signal_for_check == "BUY_HOME":
                        state["entry_price_home"] = check.execution_price
                        state["entry_size_home"] = check.size_at_best
                        state["entry_time_home"] = ts
                        _append_trade_log(log_path, {
                            "time": ts,
                            "market": market_question,
                            "condition_id": cid,
                            "action": "BUY",
                            "side": "home",
                            "price": round(check.execution_price, 4),
                            "amount_shares": round(check.size_at_best, 2),
                            "amount_usd": round(check.size_at_best * check.execution_price, 2),
                        })
                    elif signal_for_check == "BUY_AWAY":
                        state["entry_price_away"] = check.execution_price
                        state["entry_size_away"] = check.size_at_best
                        state["entry_time_away"] = ts
                        _append_trade_log(log_path, {
                            "time": ts,
                            "market": market_question,
                            "condition_id": cid,
                            "action": "BUY",
                            "side": "away",
                            "price": round(check.execution_price, 4),
                            "amount_shares": round(check.size_at_best, 2),
                            "amount_usd": round(check.size_at_best * check.execution_price, 2),
                        })
                    elif signal_for_check == "SELL_HOME":
                        ep = state.get("entry_price_home") or 0
                        es = state.get("entry_size_home") or 0
                        pnl_usd = round(es * (check.execution_price - ep), 2) if ep else 0
                        pnl_pct = round((check.execution_price - ep) / ep * 100, 2) if ep else 0
                        _append_trade_log(log_path, {
                            "time": ts,
                            "market": market_question,
                            "condition_id": cid,
                            "action": "SELL",
                            "side": "home",
                            "price": round(check.execution_price, 4),
                            "amount_shares": round(es, 2),
                            "amount_usd": round(es * check.execution_price, 2),
                            "entry_price": round(ep, 4),
                            "entry_time": state.get("entry_time_home"),
                            "pnl_usd": pnl_usd,
                            "pnl_pct": pnl_pct,
                        })
                        state["entry_price_home"] = state["entry_size_home"] = state["entry_time_home"] = None
                    elif signal_for_check == "SELL_AWAY":
                        ep = state.get("entry_price_away") or 0
                        es = state.get("entry_size_away") or 0
                        pnl_usd = round(es * (check.execution_price - ep), 2) if ep else 0
                        pnl_pct = round((check.execution_price - ep) / ep * 100, 2) if ep else 0
                        _append_trade_log(log_path, {
                            "time": ts,
                            "market": market_question,
                            "condition_id": cid,
                            "action": "SELL",
                            "side": "away",
                            "price": round(check.execution_price, 4),
                            "amount_shares": round(es, 2),
                            "amount_usd": round(es * check.execution_price, 2),
                            "entry_price": round(ep, 4),
                            "entry_time": state.get("entry_time_away"),
                            "pnl_usd": pnl_usd,
                            "pnl_pct": pnl_pct,
                        })
                        state["entry_price_away"] = state["entry_size_away"] = state["entry_time_away"] = None
            else:
                print(f"  Order book check: SKIP  reason={check.reason}")

        print("---")
        print(f"Market: {(market.question or '')[:70]}...")
        print(f"  Home: bid={row['bid_home']} ask={row['ask_home']}  Away: bid={row['bid_away']} ask={row['ask_away']}")
        print(f"  State: {game_state.get('score_home', 0)}-{game_state.get('score_away', 0)}  period={game_state.get('period', 1)}  ({strategy_label})")
        print(f"  Signal: {signal}")
        if pred_min_h is not None:
            print(f"  Predicted (next window):  Home low={pred_min_h:.3f} high={(pred_max_h or 0):.3f}  |  Away low={(pred_min_a or 0):.3f} high={(pred_max_a or 0):.3f}")
        if verbose and row.get("ask_home") is not None and row.get("bid_home") is not None:
            spread_h = (row["ask_home"] - row["bid_home"]) if (row["ask_home"] and row["bid_home"]) else None
            if spread_h is not None:
                print(f"  (Home spread: {spread_h:.3f})")
        # Warn when order book has no tradable liquidity (no trades will occur)
        ah, bh = row.get("ask_home"), row.get("bid_home")
        if ah is not None and bh is not None:
            spread = float(ah) - float(bh)
            if spread > 0.5 or float(ah) >= 0.95 or float(bh) <= 0.05:
                print(f"  [No trade] Order book has no usable liquidity (spread={spread:.2f}, ask={ah}, bid={bh}). Run when a game is live and the market has real orders.")
    print("---")
    return market_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Test in-game strategy on real-time Polymarket + ESPN data")
    parser.add_argument("--model", type=str, default=None, help="Path to reward_model.pkl (default: data/reward_model.pkl)")
    parser.add_argument("--loop", type=float, default=0, help="Poll every N seconds (0 = one-shot). Keeps price history for deltas.")
    parser.add_argument("--use-price-range", action="store_true",
        help="Use price-range strategy (predict low/high; buy at low, sell at high). Default: both tokens per market.")
    parser.add_argument("--single-token", action="store_true",
        help="With --use-price-range: trade one token per market only.")
    parser.add_argument("--log-trades", type=str, default="", metavar="PATH",
        help=f"Append trades to JSONL file (time, market, action, price, amount, pnl on sell). Default: {DEFAULT_TRADES_LOG}")
    parser.add_argument("--no-log-trades", action="store_true", help="Do not write trade log")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else BASE / "data" / "reward_model.pkl"
    if not model_path.is_absolute():
        model_path = BASE / model_path
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    log_path: Path | None = None
    if not args.no_log_trades:
        log_path = Path(args.log_trades) if args.log_trades else DEFAULT_TRADES_LOG
        if not log_path.is_absolute():
            log_path = BASE / log_path

    use_dual = args.use_price_range and not args.single_token
    market_states = None
    if args.loop > 0:
        print(f"Live test every {args.loop}s (Ctrl+C to stop). Polymarket bid/ask + ESPN state.\n")
        if log_path:
            print(f"Trade log: {log_path}\n")
        while True:
            try:
                market_states = run_once(
                    model_path,
                    verbose=not args.quiet,
                    use_price_range=args.use_price_range,
                    use_dual=use_dual,
                    market_states=market_states,
                    log_path=log_path,
                )
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            time.sleep(args.loop)
    else:
        if log_path:
            print(f"Trade log: {log_path}\n")
        run_once(
            model_path,
            verbose=not args.quiet,
            use_price_range=args.use_price_range,
            use_dual=use_dual,
            log_path=log_path,
        )


if __name__ == "__main__":
    main()
