"""
Polymarket NHL Bot v1 — pre-game loop. See BOT_SPEC.md.
Flow: games (ESPN) -> features -> predict (all or filtered) -> safety -> market data -> signals -> size -> execute.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from config import PREDICT_ALL_GAMES, PAPER_TRADING
from espn_client import games_today, event_to_game_info, get_scoreboard
from execution import Portfolio
from model import predict_fair_price, predict_all_games, load_lineup_lookup, enrich_game_with_lineup
from polymarket_client import (
    TwoSidedMarket,
    discover_nhl_markets,
    get_clob_client,
    get_order_books,
)
from strategy import (
    check_portfolio_caps,
    compute_signals,
    liquidity_ok,
    cap_size_by_liquidity,
)
from execution import place_order, update_position_on_fill

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Default balance for paper
DEFAULT_BALANCE = 1000.0
PREDICTION_LOG_PATH = Path(__file__).resolve().parent / "data" / "prediction_log.jsonl"
TEAM_NAMES_PATH = Path(__file__).resolve().parent / "data" / "team_names.json"
LINEUP_NOTES_PATH = Path(__file__).resolve().parent / "data" / "lineup_notes.jsonl"
PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _team_name_to_abbrev() -> dict[str, str]:
    """Build full name -> abbreviation from data/team_names.json."""
    if not TEAM_NAMES_PATH.exists():
        return {}
    raw = json.loads(TEAM_NAMES_PATH.read_text())
    return {name: abbrev for abbrev, name in raw.items()}


def safety_skip(game: dict, prob_home: float, prob_away: float) -> tuple[bool, str]:
    """Skip only when genuinely uncertain. Return (should_skip, reason)."""
    if prob_home <= 0 or prob_home >= 1 or prob_away <= 0 or prob_away >= 1:
        return True, "invalid_prob"
    if abs((prob_home + prob_away) - 1.0) > 0.01:
        return True, "prob_not_sum_1"
    return False, ""


def match_game_to_market(game: dict, markets: list[TwoSidedMarket]) -> TwoSidedMarket | None:
    """Match ESPN game to a Polymarket market by question text (simplified)."""
    home_name = (game.get("home_team_name") or "").lower()
    away_name = (game.get("away_team_name") or "").lower()
    for m in markets:
        q = (m.question or "").lower()
        if home_name in q and away_name in q:
            return m
    return markets[0] if markets else None


def run_once(portfolio: Portfolio) -> None:
    """One iteration: today's games -> predict -> market -> signals -> (execute paper)."""
    client = get_clob_client()
    markets = discover_nhl_markets(client)
    if not markets:
        log.info("No NHL markets found on Polymarket")
        return

    # Games today from ESPN
    games = games_today()
    if not games:
        log.info("No NHL games today (ESPN)")
        return

    # Optional: enrich with lineup/injury/ability from lineup_notes.jsonl for today (if model uses those features)
    lineup_lookup = load_lineup_lookup(LINEUP_NOTES_PATH)
    for game in games:
        enrich_game_with_lineup(game, lineup_lookup)

    name_to_abbrev = _team_name_to_abbrev()
    for game in games:
        event_id = game.get("event_id")
        prob_home, prob_away = predict_fair_price(game)
        skip, skip_reason = safety_skip(game, prob_home, prob_away)
        home_name = game.get("home_team_name") or ""
        away_name = game.get("away_team_name") or ""
        home_abbrev = name_to_abbrev.get(home_name) or ""
        away_abbrev = name_to_abbrev.get(away_name) or ""

        # Predict-all-games: log every game
        if PREDICT_ALL_GAMES:
            with open(PREDICTION_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "event_id": event_id,
                    "prob_home": prob_home,
                    "prob_away": prob_away,
                    "skip": skip,
                    "skip_reason": skip_reason,
                    "home": home_name,
                    "away": away_name,
                    "home_abbrev": home_abbrev,
                    "away_abbrev": away_abbrev,
                }) + "\n")

        if skip:
            log.info(f"Skip {event_id} ({skip_reason}) prob_home={prob_home:.2f}")
            continue

        market = match_game_to_market(game, markets)
        if not market:
            continue
        # Token prices from Polymarket only (no assumed prices)
        book_home, book_away = get_order_books(client, market)
        signals = compute_signals(
            prob_home, prob_away,
            market.token_id_home, market.token_id_away,
            market.outcome_home, market.outcome_away,
            book_home, book_away,
        )
        if not signals:
            log.info(f"{event_id} no signals")
            continue
        if not check_portfolio_caps(portfolio.open_positions_count(), portfolio.deployed_pct()):
            log.info(f"{event_id} portfolio caps hit; skip new position")
            continue
        balance = portfolio.balance
        for sig in signals:
            if sig.side != "BUY" or sig.size_pct <= 0:
                continue
            book = book_home if sig.token_id == market.token_id_home else book_away
            if not liquidity_ok(book, balance * sig.size_pct):
                log.info(f"{event_id} liquidity too thin for {sig.outcome_label}")
                continue
            size_usd = balance * sig.size_pct
            size_capped = cap_size_by_liquidity(book, "BUY", size_usd)
            if size_capped <= 0:
                continue
            price = book.get("best_ask")
            if price is None:
                continue
            mode = "PAPER" if PAPER_TRADING else "LIVE"
            success, filled_shares = place_order(
                token_id=sig.token_id,
                side=sig.side,
                size_usd=size_capped,
                price=price,
                condition_id=market.condition_id,
                outcome_label=sig.outcome_label,
            )
            if success:
                update_position_on_fill(
                    portfolio.positions,
                    sig.token_id,
                    market.condition_id,
                    sig.outcome_label,
                    filled_shares,
                    price,
                )
                log.info(f"{event_id} {mode} BUY {sig.outcome_label} size={size_capped:.2f} price={price:.3f} ({sig.reason})")
            else:
                log.warning(f"{event_id} {mode} BUY failed for {sig.outcome_label}")
    return


def main() -> None:
    portfolio = Portfolio(balance=DEFAULT_BALANCE)
    log.info("Polymarket NHL Bot v1 (pre-game); PREDICT_ALL_GAMES=%s", PREDICT_ALL_GAMES)
    run_once(portfolio)


if __name__ == "__main__":
    main()
