"""
Run in-game backtest for one or all reward models. Select by model name or run all and compare.
Uses fixed buy/sell thresholds by default; one backtest on data, then shows P&L.

Usage:
  # List available reward models in data/
  python test_models.py --list

  # Run backtest with fixed thresholds (default 0.55 / 0.45), show P&L
  python test_models.py --model reward_model
  python test_models.py --model reward_model_mlp.pkl --month 2026-01 --capital 1000

  # Optional: custom thresholds
  python test_models.py --model reward_model_mlp --buy-threshold 0.55 --sell-threshold 0.45

  # Adaptive thresholds: buy/sell depend on rolling win rate (stricter when losing)
  python test_models.py --model reward_model_mlp --adaptive-thresholds --month 2026-01 --capital 1000

  # Run all reward_model*.pkl and compare P&L
  python test_models.py --all
  python test_models.py --all --month 2026-01

  # Select best thresholds via grid-search (only when you need a parameter-search step)
  python test_models.py --model reward_model_mlp --tune-thresholds --month 2026-01
  python test_models.py --model reward_model_mlp --tune-thresholds -j 8

  # Verify model loads and predicts correctly (run before full backtest)
  python test_models.py --check --model reward_model_mlp
  python test_models.py --check --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Run from polymarket_nhl_bot so imports work
BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


def list_models(data_dir: Path, pattern: str = "reward_model*.pkl") -> list[Path]:
    """Return sorted list of .pkl paths in data_dir matching pattern."""
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob(pattern))


def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def verify_model(model_path: Path, verbose: bool = True) -> bool:
    """
    Check that the reward model loads and produces valid predictions.
    Returns True if all checks pass, False otherwise.
    """
    from train_in_game_model import load_reward_model, predict_reward_proba, IN_GAME_FEATURE_NAMES
    from in_game_strategy import reward_model_signal_fn

    if not model_path.exists():
        _log(f"  FAIL: file not found: {model_path}", verbose)
        return False
    _log("  Load: opening bundle...", verbose)
    bundle = load_reward_model(model_path)
    if bundle is None:
        _log(f"  FAIL: could not load bundle (missing or invalid .pkl): {model_path}", verbose)
        return False
    _log("  Load: OK", verbose)
    # Build minimal feature dict (all keys, plausible values)
    names = bundle.get("feature_names", IN_GAME_FEATURE_NAMES)
    feature_dict = {}
    for n in names:
        if "delta" in n:
            feature_dict[n] = 0.0
        elif "score" in n and "gap" in n:
            feature_dict[n] = 0
        elif "score" in n:
            feature_dict[n] = 1
        elif "period" in n:
            feature_dict[n] = 2
        elif "sec" in n or "proxy" in n:
            feature_dict[n] = 600.0
        elif "ask" in n or "bid" in n:
            feature_dict[n] = 0.5
        else:
            feature_dict[n] = 0.0
    _log("  Predict: running predict_reward_proba...", verbose)
    try:
        prob_home, prob_away = predict_reward_proba(bundle, feature_dict)
    except Exception as e:
        _log(f"  FAIL: predict_reward_proba raised: {e}", verbose)
        return False
    _log(f"  Predict: P(reward_home)={prob_home}, P(reward_away)={prob_away}", verbose)
    ok = True
    for name, p in [("home", prob_home), ("away", prob_away)]:
        if p is not None:
            if not (0 <= p <= 1):
                _log(f"  FAIL: prob_{name} = {p} (must be in [0, 1])", verbose)
                ok = False
        else:
            if "model_" + name in bundle and verbose:
                _log(f"  WARN: prob_{name} is None (model present but returned None)", verbose)
    if not ok:
        return False
    _log("  Signal: building signal_fn and calling once...", verbose)
    try:
        signal_fn = reward_model_signal_fn(bundle, buy_threshold=0.55, sell_threshold=0.45)
        game_state = {"score_home": 1, "score_away": 0, "period": 2, "time_remaining_sec": 600, "game_elapsed_sec": 1200, "game_second_proxy": 1200}
        price_series_so_far = [{"ask_home": 0.5, "bid_home": 0.48, "ask_away": 0.5, "bid_away": 0.52}]
        out = signal_fn(game_state, price_series_so_far, None)
        _log(f"  Signal: returned {out!r}", verbose)
        if out not in ("BUY_HOME", "BUY_AWAY", "SELL", "HOLD"):
            _log(f"  WARN: expected BUY_HOME/BUY_AWAY/SELL/HOLD", verbose)
    except Exception as e:
        _log(f"  FAIL: signal_fn raised: {e}", verbose)
        return False
    _log("  All checks passed.", verbose)
    return True


def run_quick_performance(
    model_path: Path,
    max_games: int = 5,
    capital: float = 1000.0,
) -> bool:
    """Run backtest on first max_games games and print performance. Returns True if run (records exist)."""
    from train_in_game_model import load_reward_model
    from in_game_strategy import reward_model_signal_fn
    from backtest_in_game import run_backtest_from_records

    records_dir = BASE / "data" / "game_records"
    if not records_dir.exists():
        print("  (No data/game_records/ — skip quick performance)", flush=True)
        return False
    bundle = load_reward_model(model_path)
    if bundle is None:
        return False
    signal_fn = reward_model_signal_fn(bundle, buy_threshold=0.55, sell_threshold=0.45)
    results, total_profit, total_round_trips, total_profit_dollars = run_backtest_from_records(
        records_dir, signal_fn, capital=capital, use_kelly=True, max_games=max_games
    )
    print(f"  Performance (first {len(results)} game(s)): {len(results)} games, {total_round_trips} round-trips, ${total_profit_dollars:+.2f} profit", flush=True)
    return True


# Default grid for threshold tuning (same as backtest_in_game.py --tune-thresholds)
BUY_GRID = [0.50, 0.52, 0.55, 0.58, 0.60]
SELL_GRID = [0.35, 0.38, 0.40, 0.42, 0.45]


def _tune_thresholds_for_model(
    bundle: dict,
    records_dir: Path,
    month: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    n_jobs: int | None = None,
) -> tuple[float, float]:
    """Run grid search over buy/sell; return (best_buy, best_sell). Uses all CPUs if n_jobs is None."""
    from backtest_in_game import tune_thresholds_from_records

    tuned = tune_thresholds_from_records(
        records_dir,
        bundle,
        BUY_GRID,
        SELL_GRID,
        min_gap=0.05,
        month=month,
        from_date=from_date,
        to_date=to_date,
        n_jobs=n_jobs,
    )
    if not tuned:
        return 0.55, 0.45
    best = tuned[0]
    return best["buy_threshold"], best["sell_threshold"]


def run_one_backtest(
    model_path: Path,
    from_records: bool = True,
    month: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    capital: float = 1000.0,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    tune_thresholds: bool = False,
    return_game_results: bool = True,
    n_jobs: int | None = None,
    adaptive_thresholds: bool = False,
    adaptive_window: int = 30,
    stake_dollars: float | None = None,
    stake_pct: float | None = None,
    max_stake_pct: float | None = None,
) -> dict | None:
    """Run backtest_from_records for one model; return summary dict or None on error.
    If tune_thresholds is True, run grid search first and use best buy/sell for the backtest.
    If adaptive_thresholds is True, buy/sell thresholds are set per game from rolling win rate.
    If return_game_results is True, include per-game list in returned row (for --by-month and single-model report).
    """
    from train_in_game_model import load_reward_model
    from in_game_strategy import reward_model_signal_fn
    from backtest_in_game import run_backtest_from_records

    bundle = load_reward_model(model_path)
    if bundle is None:
        return None
    records_dir = BASE / "data" / "game_records"
    if not records_dir.exists():
        return None

    if tune_thresholds:
        buy_threshold, sell_threshold = _tune_thresholds_for_model(
            bundle, records_dir, month=month, from_date=from_date, to_date=to_date, n_jobs=n_jobs
        )

    kw: dict = {
        "month": month,
        "from_date": from_date,
        "to_date": to_date,
        "capital": capital,
        "stake_dollars": stake_dollars,
    }
    if max_stake_pct is not None:
        kw["max_stake_pct"] = max_stake_pct
    if stake_pct is not None:
        kw["stake_pct"] = stake_pct
        kw["use_kelly"] = False
    else:
        kw["use_kelly"] = stake_dollars is None
    if adaptive_thresholds:
        results, total_profit, total_round_trips, total_profit_dollars = run_backtest_from_records(
            records_dir,
            signal_fn=None,
            bundle=bundle,
            base_buy=buy_threshold,
            base_sell=sell_threshold,
            adaptive_thresholds=True,
            adaptive_window=adaptive_window,
            **kw,
        )
    else:
        signal_fn = reward_model_signal_fn(bundle, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
        results, total_profit, total_round_trips, total_profit_dollars = run_backtest_from_records(
            records_dir,
            signal_fn,
            **kw,
        )
    row = {
        "model": model_path.name,
        "path": str(model_path),
        "games": len(results),
        "round_trips": total_round_trips,
        "total_profit_price": round(total_profit, 4),
        "total_profit_dollars": round(total_profit_dollars, 2),
    }
    if return_game_results:
        row["game_results"] = results
    if tune_thresholds:
        row["buy_threshold"] = buy_threshold
        row["sell_threshold"] = sell_threshold
    if adaptive_thresholds:
        row["adaptive_thresholds"] = True
    return row


def _months_in_records(records_dir: Path) -> list[str]:
    """Return sorted list of YYYY-MM that have at least one game record."""
    import json
    months = set()
    for p in records_dir.glob("*.json"):
        if p.name.startswith("."):
            continue
        try:
            with open(p) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        d = rec.get("date") or (p.stem[:10] if len(p.stem) >= 10 else "")
        if len(d) >= 7:
            months.add(d[:7])
    return sorted(months)


def print_game_report(
    game_results: list[dict],
    total_profit_dollars: float,
    title: str | None = None,
) -> None:
    """Print per-game P&L ($) and total. Each row: date, game_id, trades, invested $, profit $.
    Invested ($): stake placed in that game. Profit ($): realized P&L (positive = gain, negative = loss, $0 = breakeven).
    Close-at-end losses show as negative (e.g. invested $100, exit at $80 = -$20)."""
    if title:
        print(title)
    print(f"{'Date':<12}  {'Game ID':<28}  {'Trades':>6}  {'Invested ($)':>12}  {'Profit ($)':>10}")
    print("-" * 72)
    n_win = n_loss = n_breakeven = 0
    for r in game_results:
        date = r.get("date", "")
        gid = (r.get("game_id") or "")[:28]
        trades = r.get("trades", 0)
        inv = r.get("invested_dollars")
        invstr = f"${inv:.2f}" if inv is not None else "—"
        p = r.get("profit_dollars")
        if p is not None:
            pstr = f"${p:+.2f}"
            if p > 0:
                n_win += 1
            elif p < 0:
                n_loss += 1
            else:
                n_breakeven += 1
        else:
            pstr = "—"
            n_breakeven += 1
        print(f"{date:<12}  {gid:<28}  {trades:>6}  {invstr:>12}  {pstr:>10}")
    print("-" * 72)
    print(f"Total: ${total_profit_dollars:+.4f}")
    print(f"Games: {n_win} with profit, {n_loss} with loss, {n_breakeven} breakeven ($0 or no trade).\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test one or all reward models (in-game backtest)")
    parser.add_argument("--list", action="store_true", help="List available reward_model*.pkl in data/")
    parser.add_argument("--check", action="store_true", help="Verify model(s) load and predict correctly (run before full backtest)")
    parser.add_argument("--model", type=str, default=None, help="Model name or path (e.g. reward_model, reward_model_mlp.pkl)")
    parser.add_argument("--all", action="store_true", help="Run backtest for all reward_model*.pkl and print comparison table")
    parser.add_argument("--month", type=str, default=None, help="Only games in month (YYYY-MM)")
    parser.add_argument("--by-month", action="store_true", help="Run backtest per month; print per-game P&L and total for each month and grand total")
    parser.add_argument("--from", dest="from_date", type=str, default=None, help="From date YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", type=str, default=None, help="To date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=1000.0, help="Starting capital (default 1000)")
    parser.add_argument("--stake-dollars", type=float, default=None,
        help="Fixed $ per round-trip (e.g. 100); overrides Kelly. Use to get meaningful invested amounts.")
    parser.add_argument("--stake-pct", type=float, default=None,
        help="Fixed %% of balance per trade when not using --stake-dollars (e.g. 0.10 = 10%% = $100 on $1000)")
    parser.add_argument("--max-stake-pct", type=float, default=None,
        help="Cap stake at this %% of balance (default 0.10); only used with Kelly or --stake-pct")
    parser.add_argument("--tune-thresholds", action="store_true",
        help="Run grid-search to select buy/sell thresholds (optional; default is fixed thresholds)")
    parser.add_argument("--buy-threshold", type=float, default=0.55, help="Buy when P(reward) >= this (default 0.55)")
    parser.add_argument("--sell-threshold", type=float, default=0.45, help="Sell when P(reward) < this (default 0.45)")
    parser.add_argument("--adaptive-thresholds", action="store_true",
        help="Set buy/sell per game from rolling win rate (stricter when losing)")
    parser.add_argument("--adaptive-window", type=int, default=30,
        help="Rolling window for win rate when --adaptive-thresholds (default 30)")
    parser.add_argument("--jobs", "-j", type=int, default=None, metavar="N",
        help="Parallel jobs for threshold grid-search (default: all CPUs)")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory (default: data)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else BASE / "data"

    if args.list:
        models = list_models(data_dir)
        if not models:
            print("No reward_model*.pkl found in", data_dir)
            return
        print("Available reward models:")
        for m in models:
            print(" ", m.name)
        return

    if args.check:
        if args.all:
            models = list_models(data_dir)
            if not models:
                print("No reward_model*.pkl found in", data_dir)
                raise SystemExit(1)
            print("Checking", len(models), "model(s)...\n", flush=True)
            failed = []
            for m in models:
                print(f"--- {m.name} ---", flush=True)
                if verify_model(m, verbose=True):
                    print("Quick performance (first 3 games):", flush=True)
                    run_quick_performance(m, max_games=3, capital=args.capital)
                    print(flush=True)
                else:
                    failed.append(m.name)
            if failed:
                print("FAILED:", ", ".join(failed), flush=True)
                raise SystemExit(1)
            print("All models OK.\n", flush=True)
            return
        if not args.model:
            print("Use --check with --model <name> or --all to verify model(s).")
            raise SystemExit(1)
        name = args.model.strip()
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        if os.path.sep in name or name.startswith("data" + os.path.sep) or Path(name).is_absolute():
            model_path = Path(name)
        else:
            model_path = data_dir / name
        print(f"Checking {model_path.name}...", flush=True)
        if verify_model(model_path, verbose=True):
            print("Model OK.", flush=True)
            print("Quick performance (first 5 games):", flush=True)
            run_quick_performance(model_path, max_games=5, capital=args.capital)
            print()
            return
        raise SystemExit(1)

    if args.all:
        models = list_models(data_dir)
        if not models:
            print("No reward_model*.pkl found in", data_dir)
            raise SystemExit(1)
        tune = args.tune_thresholds
        adaptive = args.adaptive_thresholds
        print(f"Running backtest for {len(models)} model(s) (month={args.month}, capital={args.capital}, tune={tune}, adaptive={adaptive})...\n")
        rows = []
        for m in models:
            row = run_one_backtest(
                m,
                month=args.month,
                from_date=args.from_date,
                to_date=args.to_date,
                capital=args.capital,
                buy_threshold=args.buy_threshold,
                sell_threshold=args.sell_threshold,
                tune_thresholds=tune,
                return_game_results=False,
                n_jobs=args.jobs,
                adaptive_thresholds=adaptive,
                adaptive_window=args.adaptive_window,
                stake_dollars=args.stake_dollars,
                stake_pct=args.stake_pct,
                max_stake_pct=args.max_stake_pct,
            )
            if row:
                rows.append(row)
            else:
                print(f"  Skip {m.name} (load failed)")
        if not rows:
            print("No model ran successfully.")
            return
        if tune and rows and "buy_threshold" in rows[0]:
            print(f"{'Model':<28}  {'Buy':>5}  {'Sell':>5}  {'Games':>6}  {'RoundTrips':>10}  {'Profit($)':>10}")
            print("-" * 76)
            for r in rows:
                print(f"{r['model']:<28}  {r['buy_threshold']:>5.2f}  {r['sell_threshold']:>5.2f}  {r['games']:>6}  {r['round_trips']:>10}  {r['total_profit_dollars']:>+10.2f}")
        else:
            print(f"{'Model':<30}  {'Games':>6}  {'RoundTrips':>10}  {'Profit($)':>10}")
            print("-" * 62)
            for r in rows:
                print(f"{r['model']:<30}  {r['games']:>6}  {r['round_trips']:>10}  {r['total_profit_dollars']:>+10.2f}")
        best = max(rows, key=lambda x: x["total_profit_dollars"])
        print(f"\nBest by total profit ($): {best['model']}\n")
        return

    if args.by_month:
        if not args.model:
            print("Use --by-month with --model <name>.")
            raise SystemExit(1)
        name = args.model.strip()
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        if os.path.sep in name or name.startswith("data" + os.path.sep) or Path(name).is_absolute():
            model_path = Path(name)
        else:
            model_path = data_dir / name
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            raise SystemExit(1)
        records_dir = BASE / "data" / "game_records"
        if not records_dir.exists():
            print("No data/game_records/ found.")
            raise SystemExit(1)
        months = _months_in_records(records_dir)
        if not months:
            print("No months with game records found.")
            raise SystemExit(1)
        print(f"Model: {model_path.name}  |  Capital: ${args.capital:.0f}  |  By month (thresholds: buy={args.buy_threshold}, sell={args.sell_threshold})\n")
        grand_total = 0.0
        for m in months:
            row = run_one_backtest(
                model_path,
                month=m,
                capital=args.capital,
                buy_threshold=args.buy_threshold,
                sell_threshold=args.sell_threshold,
                tune_thresholds=False,
                stake_dollars=args.stake_dollars,
                stake_pct=args.stake_pct,
                max_stake_pct=args.max_stake_pct,
            )
            if row is None or not row.get("game_results"):
                continue
            total = row["total_profit_dollars"]
            grand_total += total
            print_game_report(row["game_results"], total, title=f"=== {m} ===")
        print(f"Grand total (all months): ${grand_total:+.2f}\n")
        return

    if not args.model:
        print("Use --model <name> to test one model, --all to test all, --by-month to test each month, or --list to list models.")
        raise SystemExit(1)

    name = args.model.strip()
    if not name.endswith(".pkl"):
        name = name + ".pkl"
    # Path given (absolute or contains separator) -> use as-is
    if os.path.sep in name or name.startswith("data" + os.path.sep) or Path(name).is_absolute():
        model_path = Path(name)
    else:
        model_path = data_dir / name
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        raise SystemExit(1)

    row = run_one_backtest(
        model_path,
        month=args.month,
        from_date=args.from_date,
        to_date=args.to_date,
        capital=args.capital,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        tune_thresholds=args.tune_thresholds,
        n_jobs=args.jobs,
        adaptive_thresholds=args.adaptive_thresholds,
        adaptive_window=args.adaptive_window,
        stake_dollars=args.stake_dollars,
        stake_pct=args.stake_pct,
        max_stake_pct=args.max_stake_pct,
    )
    if row is None:
        print("Failed to load model or run backtest.")
        raise SystemExit(1)
    print(f"Model: {row['model']}")
    if args.tune_thresholds and "buy_threshold" in row:
        print(f"Tuned thresholds: buy={row['buy_threshold']:.2f}  sell={row['sell_threshold']:.2f}")
    if args.adaptive_thresholds:
        print("Thresholds: adaptive (per game from rolling win rate)")
        if row.get("game_results"):
            buys = [r["buy_threshold"] for r in row["game_results"] if "buy_threshold" in r]
            if buys:
                print(f"  buy  range: {min(buys):.2f} .. {max(buys):.2f}")
                sells = [r["sell_threshold"] for r in row["game_results"] if "sell_threshold" in r]
                if sells:
                    print(f"  sell range: {min(sells):.2f} .. {max(sells):.2f}")
    print(f"Games: {row['games']}  Round-trips: {row['round_trips']}")
    print(f"Total profit (price): {row['total_profit_price']:+.4f}")
    print(f"Total profit ($):     ${row['total_profit_dollars']:+.4f}\n")
    # Per-game P&L and total
    if row.get("game_results"):
        print_game_report(row["game_results"], row["total_profit_dollars"], title="Per-game P&L:")


if __name__ == "__main__":
    main()
