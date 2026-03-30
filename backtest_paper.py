"""
Paper-trading backtest: use past completed games and our Elo predictions to simulate
"if we had followed the bot, when would we buy and how much reward would we make?"

No Polymarket integration: we assume a simple market (e.g. we only "buy" when our
fair prob is above a threshold, and we settle at 1 if we picked the winner else 0).

Usage:
  python backtest_paper.py [--days 14] [--threshold 0.55] [--stake 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from espn_client import get_completed_games_for_date
from model import predict_fair_price

MAX_WORKERS = min(32, (os.cpu_count() or 4) * 4)


# Minimum prob to take a position (we "buy" the favored side when prob >= threshold)
DEFAULT_THRESHOLD = 0.55
# Assumed price at which we buy (no real market data; tests "if market were 50/50, our edge")
ASSUMED_BUY_PRICE = 0.50
# Stake per trade (in units; profit = (1 - price) per share if win, -price if lose)
DEFAULT_STAKE = 1.0


def _team_abbrevs() -> dict[str, str]:
    """ESPN team id -> abbreviation for display."""
    import requests
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


def _process_date(
    date_str: str,
    id_to_abbrev: dict[str, str],
    threshold: float,
    stake: float,
    assumed_price: float,
) -> tuple[list[dict], int, int, float, float]:
    """Process one date; return (rows, trades, wins, total_stake, total_profit). Used in parallel."""
    rows: list[dict] = []
    trades = 0
    wins = 0
    total_stake = 0.0
    total_profit = 0.0
    games = get_completed_games_for_date(date_str)
    for home_id, away_id, home_won in games:
        game = {"home_team_id": home_id, "away_team_id": away_id}
        prob_home, prob_away = predict_fair_price(game)
        home_abbrev = id_to_abbrev.get(home_id, home_id)
        away_abbrev = id_to_abbrev.get(away_id, away_id)
        if prob_home >= threshold and prob_home >= prob_away:
            decision = "BUY_HOME"
            picked_home = True
        elif prob_away >= threshold and prob_away >= prob_home:
            decision = "BUY_AWAY"
            picked_home = False
        else:
            rows.append({
                "date": date_str,
                "home": home_abbrev,
                "away": away_abbrev,
                "prob_home": round(prob_home, 4),
                "prob_away": round(prob_away, 4),
                "decision": "SKIP",
                "actual": "HOME" if home_won else "AWAY",
                "profit": 0.0,
                "stake": 0.0,
            })
            continue
        trades += 1
        total_stake += stake
        won = home_won if picked_home else (not home_won)
        if won:
            profit = stake * (1.0 - assumed_price) / assumed_price
            wins += 1
        else:
            profit = -stake
        total_profit += profit
        rows.append({
            "date": date_str,
            "home": home_abbrev,
            "away": away_abbrev,
            "prob_home": round(prob_home, 4),
            "prob_away": round(prob_away, 4),
            "decision": decision,
            "actual": "HOME" if home_won else "AWAY",
            "profit": round(profit, 4),
            "stake": stake,
        })
    return rows, trades, wins, total_stake, total_profit


def run_backtest(
    days: int = 14,
    threshold: float = DEFAULT_THRESHOLD,
    stake: float = DEFAULT_STAKE,
    assumed_price: float = ASSUMED_BUY_PRICE,
) -> tuple[list[dict], dict]:
    """
    Run backtest over the last `days` days (parallel over days, use all CPUs). Returns (rows, summary).
    """
    id_to_abbrev = _team_abbrevs()
    today = datetime.now().date()
    date_strings = [(today - timedelta(days=d)).strftime("%Y%m%d") for d in range(days)]

    rows: list[dict] = []
    total_stake = 0.0
    total_profit = 0.0
    wins = 0
    trades = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_process_date, ds, id_to_abbrev, threshold, stake, assumed_price): ds
            for ds in date_strings
        }
        for fut in as_completed(futures):
            try:
                day_rows, t, w, st, pr = fut.result()
                rows.extend(day_rows)
                trades += t
                wins += w
                total_stake += st
                total_profit += pr
            except Exception as e:
                print(f"  date error: {e}")

    rows.sort(key=lambda r: (r["date"], r.get("home", "")))
    roi = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0
    summary = {
        "days": days,
        "threshold": threshold,
        "assumed_buy_price": assumed_price,
        "total_games": len(rows),
        "trades": trades,
        "wins": wins,
        "total_stake": round(total_stake, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(roi, 2),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-trading backtest on past NHL games")
    parser.add_argument("--days", type=int, default=14, help="Number of past days to include")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Min prob to take a position (default 0.55)")
    parser.add_argument("--stake", type=float, default=DEFAULT_STAKE, help="Stake per trade (default 1.0)")
    parser.add_argument("--out", type=str, default="", help="Write rows to data/<out>.jsonl (default: backtest_results)")
    args = parser.parse_args()

    rows, summary = run_backtest(days=args.days, threshold=args.threshold, stake=args.stake)

    # Print table (recent first)
    print("\n--- Paper backtest: Elo predictions vs actual outcomes ---\n")
    print(f"{'Date':<10} {'Game':<14} {'Prob(H/A)':<14} {'Decision':<10} {'Actual':<8} {'Profit':>8}")
    print("-" * 66)
    for r in reversed(rows):
        game = f"{r['home']} vs {r['away']}"
        probs = f"{r['prob_home']:.2f} / {r['prob_away']:.2f}"
        profit_str = f"{r['profit']:+.2f}" if r["decision"] != "SKIP" else "—"
        print(f"{r['date']:<10} {game:<14} {probs:<14} {r['decision']:<10} {r['actual']:<8} {profit_str:>8}")

    print("-" * 66)
    print(f"\nSummary (last {summary['days']} days, threshold={summary['threshold']}, assumed price={summary['assumed_buy_price']})")
    print(f"  Total games: {summary['total_games']}")
    print(f"  Trades:      {summary['trades']} (wins: {summary['wins']})")
    print(f"  Total stake: {summary['total_stake']:.2f}")
    print(f"  Total profit: {summary['total_profit']:+.2f}")
    print(f"  ROI:         {summary['roi_pct']:+.2f}%")

    out_name = args.out or "backtest_results"
    out_path = Path(__file__).resolve().parent / "data" / f"{out_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
