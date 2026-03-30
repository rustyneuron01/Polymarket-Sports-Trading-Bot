"""
Backtest in-game strategy: we predict the price path and trade on it (no fixed buy/sell targets by default).
Uses in-game price data from data_collector; decisions come from a model (see DOCS.md §2).
Optional: --use-fixed-targets uses fixed price levels for a demo when we don't have a model yet.
Optional: --use-reward-model uses data/reward_model.pkl (train with build_in_game_dataset + train_in_game_model).
Optional: --from-records backtests from data/game_records/*.json (full score/period state per row).

Usage:
  python backtest_in_game.py [--use-fixed-targets] [--buy 0.20] [--sell 0.70] [--synthetic]
  python backtest_in_game.py --from-records --use-reward-model
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from collections import deque
from config import (
    COLLECTOR_OUTPUT_DIR,
    IN_GAME_BUY_PRICE_TARGET,
    IN_GAME_SELL_PRICE_TARGET,
    IN_GAME_USE_FIXED_TARGETS,
    KELLY_FRACTION,
    MAX_POSITION_PCT,
)
from in_game_strategy import (
    simulate_in_game,
    simulate_in_game_dual,
    _fixed_target_signal,
    reward_model_signal_fn,
    buy_sell_signal_fn,
    price_range_signal_fn,
    price_range_signal_fn_dual,
)


def load_snapshots_from_dir(snap_dir: Path) -> dict[str, list[dict]]:
    """
    Load snapshot JSONL written by data_collector. All prices are from Polymarket (fetched via get_order_books).
    We do not assume or invent token prices — only use rows that came from the collector.
    Returns { condition_id: [rows sorted by timestamp] }.
    """
    by_game: dict[str, list[dict]] = defaultdict(list)
    if not snap_dir.exists():
        return dict(by_game)
    for p in sorted(snap_dir.glob("*.jsonl")):
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
                # Normalize keys to what simulator expects
                r = {
                    "timestamp": row.get("timestamp", ""),
                    "ask_home": _float(row.get("ask_home")),
                    "ask_away": _float(row.get("ask_away")),
                    "bid_home": _float(row.get("bid_home")),
                    "bid_away": _float(row.get("bid_away")),
                }
                by_game[cid].append(r)
    for cid in by_game:
        by_game[cid].sort(key=lambda x: x["timestamp"])
    return dict(by_game)


def _float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run_synthetic_demo(buy_target: float, sell_target: float, use_fixed_targets: bool = True) -> None:
    """
    Demo with a fake price path only (we do not assume real prices).
    Real backtest uses only prices fetched from Polymarket (data_collector snapshots).
    """
    # Fake series for logic demo only — not from Polymarket
    price_series = [
        {"timestamp": "T1", "ask_home": 0.52, "bid_home": 0.50, "ask_away": 0.48, "bid_away": 0.46},
        {"timestamp": "T2", "ask_home": 0.45, "bid_home": 0.43, "ask_away": 0.55, "bid_away": 0.53},
        {"timestamp": "T3", "ask_home": 0.22, "bid_home": 0.20, "ask_away": 0.78, "bid_away": 0.76},
        {"timestamp": "T4", "ask_home": 0.19, "bid_home": 0.17, "ask_away": 0.81, "bid_away": 0.79},
        {"timestamp": "T5", "ask_home": 0.25, "bid_home": 0.23, "ask_away": 0.75, "bid_away": 0.73},
        {"timestamp": "T6", "ask_home": 0.55, "bid_home": 0.53, "ask_away": 0.45, "bid_away": 0.43},
        {"timestamp": "T7", "ask_home": 0.72, "bid_home": 0.70, "ask_away": 0.28, "bid_away": 0.26},
    ]
    signal_fn = _fixed_target_signal(buy_target, sell_target) if use_fixed_targets else None
    trades, profit = simulate_in_game(
        price_series,
        signal_fn=signal_fn,
        buy_target=buy_target,
        sell_target=sell_target,
    )
    if use_fixed_targets:
        print("\n--- Synthetic demo (fixed levels for illustration only) ---\n")
        print(f"Buy when ask <= {buy_target:.2f}, sell when bid >= {sell_target:.2f}")
    for t in trades:
        print(f"  {t.action} {t.side} @ {t.price:.2f}  {t.timestamp}" + (f"  profit={t.profit:+.2f}" if t.profit else ""))
    print(f"\nTotal profit: {profit:+.2f}")
    if not use_fixed_targets:
        print("No model yet → no trades. See DOCS.md §2 for data needed to build the prediction model.")
    print("\nReal backtest: run data_collector during games, then provide game-state data (score, period, time). See DOCS.md.\n")


def _date_in_range(date_str: str, from_date: str | None, to_date: str | None, month: str | None) -> bool:
    """True if record date is in range. month = YYYY-MM overrides from/to."""
    if not date_str:
        return True
    if month:
        return date_str.startswith(month)
    if from_date and date_str < from_date:
        return False
    if to_date and date_str > to_date:
        return False
    return True


def _kelly_stake_pct(
    history: deque,
    kelly_fraction: float,
    default_b: float,
    max_stake_pct: float,
    bootstrap_pct: float = 0.10,
    min_stake_pct: float = 0.02,
) -> float:
    """
    Kelly criterion using win/loss prices from history.
    history entries are (win: bool, return_ratio: float) where return_ratio = (sell_price - buy_price) / buy_price.
    With asymmetric payoffs: b_win = avg(return when win), b_loss = avg(|return| when lose).
    Formula: f* = p/b_loss - (1-p)/b_win (fraction of bankroll). Fallback: f* = p - (1-p)/b when only b_win available.
    We use fractional Kelly and cap at max_stake_pct. When history has <5 round-trips use bootstrap_pct.
    Never return below min_stake_pct so we don't stake $0 when the model signals BUY.
    """
    if len(history) < 5:
        return min(max(bootstrap_pct, min_stake_pct), max_stake_pct)
    wins = [h[0] for h in history]
    returns = [h[1] for h in history]
    p = sum(wins) / len(wins)
    returns_win = [r for w, r in history if w and r is not None]
    returns_loss = [abs(r) for w, r in history if not w and r is not None]
    b_win = sum(returns_win) / len(returns_win) if returns_win else default_b
    b_win = max(b_win, 0.05)
    if returns_loss:
        b_loss = sum(returns_loss) / len(returns_loss)
        b_loss = max(b_loss, 0.05)
        # Asymmetric Kelly: f = p/b_loss - (1-p)/b_win (correct for win/loss amounts)
        kelly_f = p / b_loss - (1 - p) / b_win
    else:
        # Symmetric formula when no losses yet
        kelly_f = p - (1 - p) / b_win
    if kelly_f <= 0:
        return min_stake_pct
    stake_pct = kelly_fraction * min(kelly_f, 1.0)
    stake_pct = min(stake_pct, max_stake_pct)
    return max(stake_pct, min_stake_pct)


def _adaptive_thresholds_from_win_rate(
    win_rate: float,
    base_buy: float = 0.55,
    base_sell: float = 0.45,
    min_gap: float = 0.05,
) -> tuple[float, float]:
    """
    Compute buy/sell thresholds from rolling win rate. When we're losing (win_rate < 0.5)
    we raise the buy threshold (more selective) and slightly lower sell (exit sooner).
    When winning we relax slightly to allow more trades.
    """
    # win_rate in [0, 1]. drift: positive when losing -> stricter buy, earlier sell
    drift = 0.5 - win_rate  # >0 when losing
    buy_t = base_buy + 0.10 * drift   # losing -> higher buy threshold
    sell_t = base_sell - 0.05 * drift  # losing -> lower sell (exit sooner)
    buy_t = max(0.50, min(0.65, buy_t))
    sell_t = max(0.35, min(0.48, sell_t))
    if sell_t >= buy_t - min_gap:
        sell_t = buy_t - min_gap
    return buy_t, sell_t


def run_backtest_from_records(
    records_dir: Path,
    signal_fn=None,
    skip_incomplete: bool = True,
    from_date: str | None = None,
    to_date: str | None = None,
    month: str | None = None,
    max_games: int | None = None,
    stake_dollars: float | None = None,
    capital: float = 1000.0,
    stake_pct: float = 0.02,
    max_stake_pct: float = 0.10,
    min_stake_pct: float = 0.02,
    use_kelly: bool = True,
    kelly_fraction: float = 0.25,
    kelly_window: int = 30,
    kelly_default_b: float = 0.12,
    kelly_bootstrap_pct: float = 0.10,
    bundle: dict | None = None,
    base_buy: float = 0.55,
    base_sell: float = 0.45,
    adaptive_thresholds: bool = False,
    adaptive_window: int = 30,
    max_round_trips_per_game: int | None = 1,
    use_dual: bool = False,
    pre_elo: dict | None = None,
) -> tuple[list[dict], float, int, float]:
    """
    Backtest using game records. Returns (results, total_profit_price, total_round_trips, total_profit_dollars).
    pre_elo: optional { game_key: { elo_home, elo_away } } from build_in_game_dataset (data/game_pre_elo.json).
             If provided, Elo at prediction time is used; else fall back to model.get_elo (current file).
    max_games: if set, stop after this many games (for quick checks).
    max_round_trips_per_game: max buy->sell cycles per game (default 1). Open position at game end is closed at last price and P&L booked.
    If adaptive_thresholds is True, bundle must be provided: buy/sell thresholds are set per game
    from rolling win rate (last adaptive_window round-trips). Losing streak -> stricter buy, earlier sell.
    Sizing (in order):
      - If stake_dollars is set: fixed $ per round-trip.
      - Else if use_kelly: Kelly from rolling win rate and win/loss prices (avg return when win, avg |return| when lose).
      - Else: stake = balance * stake_pct, capped at max_stake_pct.
    Games are processed in date order so balance and (for Kelly) trade history evolve correctly.
    """
    from build_in_game_dataset import price_and_state_series_from_record

    if pre_elo is None:
        pre_elo_path = records_dir.parent / "game_pre_elo.json"
        if pre_elo_path.exists():
            try:
                with open(pre_elo_path) as f:
                    pre_elo = json.load(f)
            except Exception:
                pre_elo = {}
        else:
            pre_elo = {}

    if not adaptive_thresholds and signal_fn is None:
        raise ValueError("Either pass signal_fn or set adaptive_thresholds=True with bundle")
    if adaptive_thresholds and bundle is None:
        raise ValueError("adaptive_thresholds=True requires bundle")

    # Collect and sort records by date
    records = []
    for p in sorted(records_dir.glob("*.json")):
        if p.name.startswith("."):
            continue
        try:
            with open(p) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if skip_incomplete and rec.get("event_feed_incomplete"):
            continue
        rec_date = rec.get("date") or (p.stem[:10] if len(p.stem) >= 10 else "")
        if not _date_in_range(rec_date, from_date, to_date, month):
            continue
        records.append((rec_date, p, rec))
    records.sort(key=lambda x: x[0])

    results = []
    total_profit = 0.0
    total_round_trips = 0
    balance = capital
    kelly_history: deque = deque(maxlen=kelly_window)
    games_done = 0
    for rec_date, p, rec in records:
        if max_games is not None and games_done >= max_games:
            break
        price_series, game_state_series = price_and_state_series_from_record(rec)
        if len(price_series) < 2:
            continue
        # Elo at prediction time (pre-game for this game). Use game_pre_elo.json from build_in_game_dataset (chronological); else fall back to current file.
        elo_h, elo_a = 1500.0, 1500.0
        game_key = rec.get("event_id") or f"{rec.get('date','')}_{rec.get('home_team_id')}_{rec.get('away_team_id')}"
        if pre_elo and game_key in pre_elo:
            elo_h = pre_elo[game_key]["elo_home"]
            elo_a = pre_elo[game_key]["elo_away"]
        else:
            try:
                from model import get_elo
                elo_h = get_elo(rec.get("home_team_id") or "")
                elo_a = get_elo(rec.get("away_team_id") or "")
            except Exception:
                pass
        for s in game_state_series:
            s["elo_home"] = elo_h
            s["elo_away"] = elo_a
        games_done += 1
        game_id = rec.get("event_id") or p.stem

        game_buy_t = game_sell_t = None
        if adaptive_thresholds and bundle is not None and not use_dual:
            n = min(len(kelly_history), adaptive_window)
            win_rate = (sum(1 for w, _ in list(kelly_history)[-n:] if w) / n) if n else 0.5
            buy_t, sell_t = _adaptive_thresholds_from_win_rate(win_rate, base_buy, base_sell)
            game_buy_t, game_sell_t = buy_t, sell_t
            signal_fn = reward_model_signal_fn(bundle, buy_threshold=buy_t, sell_threshold=sell_t)

        if use_dual:
            trades, profit = simulate_in_game_dual(
                price_series,
                game_state_series=game_state_series,
                signal_fn=signal_fn,
            )
        else:
            trades, profit = simulate_in_game(
                price_series,
                game_state_series=game_state_series,
                signal_fn=signal_fn,
                buy_target=0.0,
                sell_target=0.0,
                max_round_trips=max_round_trips_per_game,
            )
        n_sells = sum(1 for t in trades if t.action == "SELL")
        total_profit += profit

        profit_dollars = None
        game_profit_dollars = 0.0
        current_stake = 0.0
        current_stake_home = 0.0
        current_stake_away = 0.0
        game_invested_dollars = None
        first_buy_price: float | None = None
        for t in trades:
            if t.action == "BUY":
                stake_amt = current_stake
                if stake_dollars is not None and stake_dollars > 0:
                    stake_amt = stake_dollars
                elif use_kelly:
                    pct = _kelly_stake_pct(
                        kelly_history, kelly_fraction, kelly_default_b, max_stake_pct,
                        bootstrap_pct=kelly_bootstrap_pct, min_stake_pct=min_stake_pct,
                    )
                    stake_amt = max(balance * min_stake_pct, min(balance * pct, balance))
                else:
                    stake = min(balance * stake_pct, balance * max_stake_pct)
                    stake_amt = max(0.0, min(stake, balance))
                if first_buy_price is None:
                    first_buy_price = t.price
                if use_dual:
                    if t.side == "home":
                        current_stake_home = stake_amt
                    else:
                        current_stake_away = stake_amt
                    if game_invested_dollars is None:
                        game_invested_dollars = stake_amt
                    else:
                        game_invested_dollars += stake_amt
                else:
                    current_stake = stake_amt
                    if game_invested_dollars is None:
                        game_invested_dollars = current_stake
            elif t.action == "SELL" and t.profit is not None:
                if use_dual:
                    stake = current_stake_home if t.side == "home" else current_stake_away
                    if t.side == "home":
                        current_stake_home = 0.0
                    else:
                        current_stake_away = 0.0
                else:
                    stake = current_stake
                    current_stake = 0.0
                if stake > 0:
                    buy_p = t.price - t.profit
                    if buy_p and buy_p > 0:
                        return_ratio = t.profit / buy_p
                        pnl = stake * return_ratio
                        game_profit_dollars += pnl
                        balance += pnl
                        kelly_history.append((pnl > 0, return_ratio))
                    else:
                        if first_buy_price and first_buy_price > 0 and game_invested_dollars is not None:
                            return_ratio = t.profit / first_buy_price
                            pnl = stake * return_ratio
                            game_profit_dollars += pnl
                            balance += pnl
                            kelly_history.append((pnl > 0, return_ratio))
        total_round_trips += n_sells

        # Dollar P&L: always set when we had capital at risk so losses show as negative (e.g. -$10), not $0
        profit_dollars = None
        if stake_dollars is not None or (stake_pct > 0 and capital > 0) or use_kelly:
            profit_dollars = round(game_profit_dollars, 2)
        if game_invested_dollars is not None and game_invested_dollars > 0 and first_buy_price and first_buy_price > 0:
            # Fallback: if we didn't book any dollar P&L in the loop, derive from price profit so losses show as negative
            if profit_dollars is None or (profit_dollars == 0 and profit != 0):
                gpd_before = game_profit_dollars
                game_profit_dollars = game_invested_dollars * (profit / first_buy_price)
                profit_dollars = round(game_profit_dollars, 2)
                if gpd_before == 0:
                    balance += game_profit_dollars  # balance wasn't updated in loop

        row = {"game_id": game_id, "date": rec_date, "trades": len(trades), "profit": round(profit, 4)}
        if profit_dollars is not None:
            row["profit_dollars"] = profit_dollars
            row["balance_after"] = round(balance, 4)
        if game_invested_dollars is not None:
            row["invested_dollars"] = round(game_invested_dollars, 2)
        if game_buy_t is not None and game_sell_t is not None:
            row["buy_threshold"] = game_buy_t
            row["sell_threshold"] = game_sell_t
        results.append(row)

    total_profit_dollars = balance - capital
    return results, total_profit, total_round_trips, total_profit_dollars


def _run_one_threshold_combo(
    records_dir: Path,
    bundle: dict,
    buy_t: float,
    sell_t: float,
    from_date: str | None,
    to_date: str | None,
    month: str | None,
) -> dict:
    """Worker for parallel threshold tuning: run one (buy, sell) backtest. Must be top-level for pickling."""
    signal_fn = reward_model_signal_fn(bundle, buy_threshold=buy_t, sell_threshold=sell_t)
    _, total_profit, total_round_trips, _ = run_backtest_from_records(
        records_dir, signal_fn,
        from_date=from_date, to_date=to_date, month=month,
    )
    return {
        "buy_threshold": buy_t,
        "sell_threshold": sell_t,
        "total_profit": round(total_profit, 4),
        "total_round_trips": total_round_trips,
    }


def tune_thresholds_from_records(
    records_dir: Path,
    bundle: dict,
    buy_values: list[float],
    sell_values: list[float],
    min_gap: float = 0.05,
    from_date: str | None = None,
    to_date: str | None = None,
    month: str | None = None,
    n_jobs: int | None = None,
) -> list[dict]:
    """
    Run backtest for each (buy_threshold, sell_threshold) with sell < buy - min_gap.
    Returns list of { buy_threshold, sell_threshold, total_profit, total_round_trips } sorted by total_profit desc.
    If n_jobs is not None and > 1, run combos in parallel using that many processes; else use all CPUs.
    """
    combos = []
    for buy in buy_values:
        for sell in sell_values:
            if sell >= buy - min_gap:
                continue
            combos.append((buy, sell))
    if n_jobs == 1 or len(combos) <= 1:
        results = []
        for buy_t, sell_t in combos:
            results.append(_run_one_threshold_combo(
                records_dir, bundle, buy_t, sell_t, from_date, to_date, month
            ))
    else:
        workers = n_jobs if n_jobs is not None and n_jobs > 1 else max(1, (os.cpu_count() or 4))
        workers = min(workers, len(combos))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_one_threshold_combo,
                    records_dir,
                    bundle,
                    buy_t,
                    sell_t,
                    from_date,
                    to_date,
                    month,
                ): (buy_t, sell_t)
                for buy_t, sell_t in combos
            }
            results = []
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    buy_t, sell_t = futures[fut]
                    print(f"  Warning: threshold ({buy_t}, {sell_t}) failed: {e}", flush=True)
    results.sort(key=lambda x: (-x["total_profit"], -x["total_round_trips"]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest in-game strategy (predict price path, trade on it)")
    parser.add_argument("--use-fixed-targets", action="store_true", help="Use fixed buy/sell levels for demo (no model)")
    parser.add_argument("--use-reward-model", action="store_true", help="Use reward model for signals (default: data/reward_model.pkl)")
    parser.add_argument("--model", type=str, default=None, help="Reward model file name or path (e.g. reward_model.pkl, reward_model_mlp.pkl, or data/reward_model_gb.pkl). Default: reward_model.pkl")
    parser.add_argument("--from-records", action="store_true", help="Backtest from data/game_records/*.json (full score/period per row)")
    parser.add_argument("--buy-threshold", type=float, default=0.55, help="Reward model: buy when P(reward) >= this (default 0.55)")
    parser.add_argument("--sell-threshold", type=float, default=0.45, help="Reward model: sell when P(reward) for our side < this (default 0.45)")
    parser.add_argument("--use-dual", action="store_true", help="4 trades per game: buy/sell home and buy/sell away (predict low/high per token). Requires model trained with --train-buy-sell.")
    parser.add_argument("--use-price-range", action="store_true",
        help="Use predicted low/high price only (no probability thresholds). Buy when price at or below predicted low, sell when at or above predicted high. Requires model trained with --train-price-range. By default trades both tokens (home + away) per game; use --single-token for one token only.")
    parser.add_argument("--single-token", action="store_true",
        help="With --use-price-range: trade only one token per game (default: trade both home and away for higher reward).")
    parser.add_argument("--tune-thresholds", action="store_true",
        help="Sweep buy/sell thresholds and report best by total profit (use with --from-records --use-reward-model)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Starting capital in $ (default 1000)")
    parser.add_argument("--sizing", choices=("kelly", "fixed_pct"), default="kelly",
        help="Stake sizing: kelly = Kelly criterion from rolling win rate (default); fixed_pct = fixed %% of balance")
    parser.add_argument("--kelly-fraction", type=float, default=None,
        help="Fraction of full Kelly to use (default from config, typically 0.25 = quarter-Kelly)")
    parser.add_argument("--kelly-window", type=int, default=30, help="Rolling window of last N trades for Kelly (default 30)")
    parser.add_argument("--kelly-default-b", type=float, default=0.12,
        help="Default net payout on win when no history yet, e.g. 0.12 = 12%% (default 0.12)")
    parser.add_argument("--stake-pct", type=float, default=0.02,
        help="When --sizing fixed_pct: fraction of balance per trade (default 0.02)")
    parser.add_argument("--max-stake-pct", type=float, default=None,
        help="Cap stake at this fraction of balance (default: config MAX_POSITION_PCT for kelly, 0.10 for fixed_pct)")
    parser.add_argument("--stake-per-trade", type=float, default=None,
        help="If set, use fixed $ per round-trip (overrides --sizing)")
    parser.add_argument("--kelly-bootstrap-pct", type=float, default=0.10,
        help="When using Kelly with <5 round-trips history, use this %% of balance (default 0.10 = 10%% = $100 on $1000)")
    parser.add_argument("--min-stake-pct", type=float, default=0.02,
        help="Minimum stake %% of balance when model signals BUY; avoids $0 invested (default 0.02 = 2%%)")
    parser.add_argument("--month", type=str, default=None, help="Only games in this month (YYYY-MM, e.g. 2026-01)")
    parser.add_argument("--from", dest="from_date", type=str, default=None, help="Only games on or after this date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", type=str, default=None, help="Only games on or before this date (YYYY-MM-DD)")
    parser.add_argument("--buy", type=float, default=IN_GAME_BUY_PRICE_TARGET)
    parser.add_argument("--sell", type=float, default=IN_GAME_SELL_PRICE_TARGET)
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic demo only")
    args = parser.parse_args()

    use_fixed = args.use_fixed_targets or IN_GAME_USE_FIXED_TARGETS
    base = Path(__file__).resolve().parent

    # Resolve reward model path: --model name or path, else default reward_model.pkl
    model_name = args.model or "reward_model.pkl"
    if not model_name.endswith(".pkl"):
        model_name = model_name + ".pkl"
    reward_path = Path(model_name) if os.path.sep in model_name or model_name.startswith("data") else base / "data" / model_name

    signal_fn = None
    bundle = None
    use_dual_tokens = args.use_dual  # default; may be set True when --use-price-range (trade both tokens)
    if args.use_reward_model or args.use_dual:
        from train_in_game_model import load_reward_model
        bundle = load_reward_model(reward_path)
        if bundle is None:
            print(f"Reward model not found at {reward_path}. Run build_in_game_dataset then train_in_game_model.")
            raise SystemExit(1)
        # With --use-price-range default to both tokens (dual) for higher reward; --single-token for one token per game
        use_dual_tokens = args.use_dual or (args.use_price_range and "model_min_ask_home" in bundle and not args.single_token)
        if args.use_price_range and "model_min_ask_home" in bundle:
            signal_fn = price_range_signal_fn_dual(bundle) if use_dual_tokens else price_range_signal_fn(bundle)
        elif args.use_dual:
            signal_fn = buy_sell_signal_fn(bundle, buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold)
        else:
            signal_fn = reward_model_signal_fn(bundle, buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold)
    elif use_fixed:
        signal_fn = _fixed_target_signal(args.buy, args.sell)

    if args.synthetic:
        run_synthetic_demo(args.buy, args.sell, use_fixed_targets=use_fixed)
        return

    if use_dual_tokens and not args.from_records:
        print("--use-dual requires --from-records (game records with score/state).")
        raise SystemExit(1)
    if args.from_records:
        records_dir = base / "data" / "game_records"
        if not records_dir.exists():
            print(f"Records directory not found: {records_dir}")
            raise SystemExit(1)
        if signal_fn is None:
            print("Use --use-reward-model (or --use-dual) to backtest from records (model required for signals).")
            raise SystemExit(1)

        if args.tune_thresholds:
            from train_in_game_model import load_reward_model
            bundle = load_reward_model(reward_path)
            if bundle is None:
                print("Reward model not found. Run build_in_game_dataset then train_in_game_model.")
                raise SystemExit(1)
            buy_vals = [0.50, 0.52, 0.55, 0.58, 0.60]
            sell_vals = [0.35, 0.38, 0.40, 0.42, 0.45]
            tuned = tune_thresholds_from_records(
                records_dir, bundle, buy_vals, sell_vals, min_gap=0.05,
                from_date=args.from_date, to_date=args.to_date, month=args.month,
            )
            print("\n--- Tune thresholds (by total profit) ---\n")
            print(f"{'buy_threshold':>14}  {'sell_threshold':>14}  {'total_profit':>12}  {'round_trips':>10}")
            print("-" * 54)
            for row in tuned:
                print(f"{row['buy_threshold']:>14.2f}  {row['sell_threshold']:>14.2f}  {row['total_profit']:>+12.2f}  {row['total_round_trips']:>10}")
            best = tuned[0]
            print(f"\nBest: --buy-threshold {best['buy_threshold']:.2f} --sell-threshold {best['sell_threshold']:.2f}")
            print(f"  total_profit = {best['total_profit']:+.2f}, round_trips = {best['total_round_trips']}\n")
            return
        if signal_fn is None:
            print("Use --use-reward-model to backtest from records (model required for signals).")
            raise SystemExit(1)
        date_filter = f" (month={args.month})" if args.month else (f" (from {args.from_date} to {args.to_date})" if args.from_date or args.to_date else "")
        max_pct = args.max_stake_pct if args.max_stake_pct is not None else (MAX_POSITION_PCT if args.sizing == "kelly" else 0.10)
        kelly_frac = args.kelly_fraction if args.kelly_fraction is not None else KELLY_FRACTION
        results, total_profit, total_round_trips, total_profit_dollars = run_backtest_from_records(
            records_dir,
            signal_fn,
            from_date=args.from_date,
            to_date=args.to_date,
            month=args.month,
            stake_dollars=args.stake_per_trade,
            capital=args.capital,
            stake_pct=args.stake_pct,
            max_stake_pct=max_pct,
            min_stake_pct=args.min_stake_pct,
            use_kelly=args.sizing == "kelly" and args.stake_per_trade is None,
            kelly_fraction=kelly_frac,
            kelly_window=args.kelly_window,
            kelly_default_b=args.kelly_default_b,
            kelly_bootstrap_pct=args.kelly_bootstrap_pct,
            use_dual=use_dual_tokens,
        )
        if args.stake_per_trade:
            sizing = f"${args.stake_per_trade:.0f} fixed/trade"
        elif args.sizing == "kelly":
            sizing = f"Kelly (frac={kelly_frac}, window={args.kelly_window}, max={100*max_pct:.0f}%)"
        else:
            sizing = f"{100*args.stake_pct:.0f}% of balance (max {100*max_pct:.0f}%)"
        print("\n--- In-game backtest (from game records) ---\n")
        print(f"Capital: ${args.capital:.0f}  |  Sizing: {sizing}{date_filter}\n")
        print("(Using reward model: buy when P(reward) high, sell when P(reward) low)" + (" | Dual: 4 trades/game (buy/sell home + buy/sell away)" if use_dual_tokens else "") + "\n")
        for r in results:
            if r["trades"] > 0:
                line = f"  {r.get('date', '')}  {r['game_id']}  trades={r['trades']}  profit(price)={r['profit']:+.3f}"
                if "invested_dollars" in r and r["invested_dollars"] is not None:
                    line += f"  invested=${r['invested_dollars']:.2f}"
                if "profit_dollars" in r:
                    line += f"  ${r['profit_dollars']:+.2f}"
                    if "balance_after" in r:
                        line += f"  balance=${r['balance_after']:.0f}"
                print(line)
        total_invested = sum(r.get("invested_dollars") or 0 for r in results)
        print(f"\nGames: {len(results)}")
        print(f"Total round-trips (buy then sell): {total_round_trips}")
        if total_invested > 0:
            print(f"Total invested ($): ${total_invested:,.2f}")
        print(f"Total profit (price units): {total_profit:+.3f}")
        print(f"Total reward ($): ${total_profit_dollars:+.2f}\n")
        return

    snap_dir = base / "data" / "polymarket_snapshots"
    games = load_snapshots_from_dir(snap_dir)

    if not games:
        print("No in-game snapshot data in data/polymarket_snapshots/.")
        print("Token prices must be fetched from Polymarket: run data_collector during games to record real bid/ask.")
        print("See DOCS.md §2 for data needed to predict the price graph and trade on it.")
        print("\nSynthetic demo (optional --use-fixed-targets; uses fake prices for logic only):\n")
        run_synthetic_demo(args.buy, args.sell, use_fixed_targets=use_fixed)
        return

    if signal_fn is None:
        print("No signal: use --use-fixed-targets or --use-reward-model (and ensure data/reward_model.pkl exists).")
        print("Falling back to synthetic demo.\n")
        run_synthetic_demo(args.buy, args.sell, use_fixed_targets=use_fixed)
        return

    total_profit = 0.0
    total_round_trips = 0
    results = []
    max_workers = min(32, (os.cpu_count() or 4) * 4)

    def run_one(cid_series):
        cid, series = cid_series
        if len(series) < 2:
            return None
        trades, profit = simulate_in_game(
            series,
            signal_fn=signal_fn,
            buy_target=args.buy,
            sell_target=args.sell,
        )
        return (cid, len(trades), sum(1 for t in trades if t.action == "SELL"), profit)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, item): item for item in games.items()}
        for fut in as_completed(futures):
            try:
                out = fut.result()
                if out is None:
                    continue
                cid, n_trades, n_sells, profit = out
                total_profit += profit
                total_round_trips += n_sells
                results.append({"condition_id": cid[:20] + "...", "trades": n_trades, "profit": round(profit, 4)})
            except Exception as e:
                print(f"  game error: {e}")

    print("\n--- In-game backtest ---\n")
    if use_fixed:
        print("(Using fixed levels: buy ≤ {:.2f}, sell ≥ {:.2f})\n".format(args.buy, args.sell))
    elif args.use_reward_model:
        print("(Using reward model)\n")
    for r in results:
        if r["trades"] > 0:
            print(f"  {r['condition_id']}  trades={r['trades']}  profit={r['profit']:+.2f}")
    print(f"\nGames with data: {len(results)}")
    print(f"Total round-trips (buy then sell): {total_round_trips}")
    print(f"Total profit: {total_profit:+.2f}\n")


if __name__ == "__main__":
    main()
