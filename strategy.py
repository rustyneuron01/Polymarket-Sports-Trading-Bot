"""
Signals, Kelly sizing, liquidity check, portfolio caps. BOT_SPEC §3, §4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from config import (
    BUY_THRESHOLD,
    KELLY_FRACTION,
    MAX_CAPITAL_DEPLOYED_PCT,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT,
    MIN_EDGE_PCT,
    SELL_THRESHOLD,
)


@dataclass
class Signal:
    side: Literal["BUY", "SELL"]
    token_id: str
    outcome_label: str
    fair_price: float
    market_bid: Optional[float]
    market_ask: Optional[float]
    edge: float
    size_pct: float
    reason: str


def kelly_fraction(fair_prob: float, price: float) -> float:
    """Prediction-market Kelly: f* = (p - price) / (1 - price). Only when p > price."""
    if price <= 0 or price >= 1 or fair_prob <= price:
        return 0.0
    return (fair_prob - price) / (1.0 - price)


def compute_signals(
    prob_home: float,
    prob_away: float,
    token_id_home: str,
    token_id_away: str,
    outcome_home: str,
    outcome_away: str,
    book_home: dict,
    book_away: dict,
) -> list[Signal]:
    """Buy when ask <= prob - threshold; sell when bid >= prob + threshold. Min edge 4-5%."""
    signals = []
    ask_home = book_home.get("best_ask")
    bid_home = book_home.get("best_bid")
    ask_away = book_away.get("best_ask")
    bid_away = book_away.get("best_bid")

    for prob, token_id, label, ask, bid in [
        (prob_home, token_id_home, outcome_home, ask_home, bid_home),
        (prob_away, token_id_away, outcome_away, ask_away, bid_away),
    ]:
        if ask is not None and prob - ask >= BUY_THRESHOLD and prob - ask >= MIN_EDGE_PCT:
            kf = kelly_fraction(prob, ask)
            size = min(KELLY_FRACTION * kf, MAX_POSITION_PCT)
            if size > 0:
                signals.append(Signal(
                    side="BUY",
                    token_id=token_id,
                    outcome_label=label,
                    fair_price=prob,
                    market_bid=bid,
                    market_ask=ask,
                    edge=prob - ask,
                    size_pct=size,
                    reason=f"ask {ask:.3f} < fair {prob:.3f} (edge {prob - ask:.3f})",
                ))
        if bid is not None and bid - prob >= SELL_THRESHOLD and bid - prob >= MIN_EDGE_PCT:
            signals.append(Signal(
                side="SELL",
                token_id=token_id,
                outcome_label=label,
                fair_price=prob,
                market_bid=bid,
                market_ask=ask,
                edge=bid - prob,
                size_pct=0.0,
                reason=f"bid {bid:.3f} > fair {prob:.3f}",
            ))
    return signals


def liquidity_ok(book: dict, size_usd: float, min_depth_ratio: float = 0.5) -> bool:
    """Check if best ask has enough depth (in USD) for size. Thin book -> reduce or skip."""
    asks = book.get("asks") or []
    if not asks:
        return False
    best_price = float(asks[0].get("price", 0))
    best_size_shares = float(asks[0].get("size", 0))
    depth_usd = best_size_shares * best_price if best_price else 0
    return depth_usd >= size_usd * min_depth_ratio


def cap_size_by_liquidity(book: dict, side: Literal["BUY", "SELL"], size_usd: float) -> float:
    """Don't exceed available liquidity at best level (return max USD we can do)."""
    levels = book.get("asks" if side == "BUY" else "bids") or []
    if not levels:
        return 0.0
    best_price = float(levels[0].get("price", 0))
    best_size_shares = float(levels[0].get("size", 0))
    depth_usd = best_size_shares * best_price if best_price else 0
    return min(size_usd, depth_usd)


def check_portfolio_caps(open_positions_count: int, deployed_pct: float) -> bool:
    """True if we can open a new position."""
    return open_positions_count < MAX_OPEN_POSITIONS and deployed_pct < MAX_CAPITAL_DEPLOYED_PCT
