"""
Execution: paper vs live; position tracking with partial fills. BOT_SPEC.
Live orders use the authenticated Polymarket CLOB client.

Before buying or selling in-game, always check current token prices and order book and validate
against the strategy so we do not miss games and trade correctly (use fresh book, not stale).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from config import PAPER_TRADING


# In-game signal from strategy (single or dual)
InGameSignal = Literal["BUY_HOME", "BUY_AWAY", "SELL_HOME", "SELL_AWAY", "HOLD"]


@dataclass
class InGameOrderCheck:
    """Result of validating an in-game signal against the current order book."""
    valid: bool
    execution_price: float  # best_ask for buy, best_bid for sell
    token_id: str
    outcome_label: str
    size_at_best: float  # liquidity at best level (shares)
    condition_id: str
    reason: str = ""  # why invalid or "ok"


@dataclass
class Position:
    token_id: str
    outcome_label: str
    condition_id: str
    size: float
    cost_basis: float  # avg price paid (use filled amount only for partial fills)
    filled_size: float = 0.0  # actual filled so far


@dataclass
class Portfolio:
    balance: float
    positions: list[Position] = field(default_factory=list)

    def deployed_pct(self) -> float:
        if self.balance <= 0:
            return 0.0
        total = sum(p.cost_basis * p.filled_size for p in self.positions)
        return total / self.balance

    def open_positions_count(self) -> int:
        return sum(1 for p in self.positions if p.filled_size > 0)


def _size_at_best(book: dict, use_ask: bool) -> float:
    """Size (shares) available at best price. book has bids/asks as list of {price, size}."""
    levels = (book.get("asks") or []) if use_ask else (book.get("bids") or [])
    best = book.get("best_ask") if use_ask else book.get("best_bid")
    if not levels or best is None:
        return 0.0
    for r in levels:
        p = r.get("price") if isinstance(r, dict) else getattr(r, "price", None)
        if p is not None and abs(float(p) - best) < 0.001:
            s = r.get("size") if isinstance(r, dict) else getattr(r, "size", 0)
            return float(s) if s is not None else 0.0
    return float(levels[0].get("size", 0)) if isinstance(levels[0], dict) else 0.0


def validate_in_game_order(
    signal: InGameSignal,
    market,  # TwoSidedMarket
    home_book: dict,
    away_book: dict,
    *,
    pred_min_h: Optional[float] = None,
    pred_max_h: Optional[float] = None,
    pred_min_a: Optional[float] = None,
    pred_max_a: Optional[float] = None,
    buy_margin: float = 0.02,
    sell_margin: float = 0.02,
    min_spread_required: float = 0.02,
) -> InGameOrderCheck:
    """
    Check current order book and strategy levels before executing. Call this with fresh
    order books (re-fetched right before trade) so we do not miss games and trade at correct prices.
    Returns InGameOrderCheck with valid=True only when current best price still satisfies the strategy.
    """
    condition_id = getattr(market, "condition_id", "") or ""
    outcome_home = getattr(market, "outcome_home", "Home") or "Home"
    outcome_away = getattr(market, "outcome_away", "Away") or "Away"
    token_id_home = getattr(market, "token_id_home", "") or ""
    token_id_away = getattr(market, "token_id_away", "") or ""

    if signal == "HOLD":
        return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason="HOLD")

    # BUY_HOME: execute at best_ask; require ask <= pred_min_h * (1 + buy_margin) and pred_max_h - pred_min_h >= min_spread
    if signal == "BUY_HOME":
        ask = home_book.get("best_ask")
        if ask is None:
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason="no best_ask home")
        if pred_min_h is None or pred_max_h is None or (pred_max_h - pred_min_h) < min_spread_required:
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason="no valid predicted range home")
        if ask > pred_min_h * (1.0 + buy_margin):
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason=f"ask {ask:.3f} > pred_low*1.02 {pred_min_h * 1.02:.3f}")
        size = _size_at_best(home_book, use_ask=True)
        return InGameOrderCheck(True, float(ask), token_id_home, outcome_home, size, condition_id, reason="ok")

    if signal == "BUY_AWAY":
        ask = away_book.get("best_ask")
        if ask is None:
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason="no best_ask away")
        if pred_min_a is None or pred_max_a is None or (pred_max_a - pred_min_a) < min_spread_required:
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason="no valid predicted range away")
        if ask > pred_min_a * (1.0 + buy_margin):
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason=f"ask {ask:.3f} > pred_low*1.02")
        size = _size_at_best(away_book, use_ask=True)
        return InGameOrderCheck(True, float(ask), token_id_away, outcome_away, size, condition_id, reason="ok")

    if signal == "SELL_HOME":
        bid = home_book.get("best_bid")
        if bid is None:
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason="no best_bid home")
        if pred_max_h is None or pred_min_h is None or (pred_max_h - pred_min_h) < min_spread_required:
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason="no valid predicted range home")
        if bid < pred_max_h * (1.0 - sell_margin):
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason=f"bid {bid:.3f} < pred_high*0.98")
        size = _size_at_best(home_book, use_ask=False)
        return InGameOrderCheck(True, float(bid), token_id_home, outcome_home, size, condition_id, reason="ok")

    if signal == "SELL_AWAY":
        bid = away_book.get("best_bid")
        if bid is None:
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason="no best_bid away")
        if pred_max_a is None or pred_min_a is None or (pred_max_a - pred_min_a) < min_spread_required:
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason="no valid predicted range away")
        if bid < pred_max_a * (1.0 - sell_margin):
            return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason=f"bid {bid:.3f} < pred_high*0.98")
        size = _size_at_best(away_book, use_ask=False)
        return InGameOrderCheck(True, float(bid), token_id_away, outcome_away, size, condition_id, reason="ok")

    return InGameOrderCheck(False, 0.0, "", "", 0.0, condition_id, reason="unknown signal")


def place_order(
    token_id: str,
    side: str,
    size_usd: float,
    price: float,
    condition_id: str,
    outcome_label: str,
) -> tuple[bool, float]:
    """
    Place order (paper or live). size_usd = notional in USD; we convert to shares for CLOB.
    Returns (success, filled_shares). Use filled_shares with fill_price for position tracking.
    Before calling, fetch current order book and use validate_in_game_order() to ensure price
    and liquidity are still valid for the strategy; pass the execution_price and cap size by size_at_best.
    """
    if price <= 0:
        return False, 0.0
    shares = size_usd / price
    if shares <= 0:
        return False, 0.0

    if PAPER_TRADING:
        return True, shares

    # Live: create and post limit order via Polymarket CLOB
    from py_clob_clients import OrderArgs, OrderType, BUY

    from polymarket_client import get_clob_client_authenticated

    client = get_clob_client_authenticated()
    if not client:
        return False, 0.0
    try:
        order = client.create_order(
            OrderArgs(token_id=token_id, price=round(price, 2), size=round(shares, 2), side=BUY),
            options={"tick_size": "0.01", "neg_risk": False},
        )
        client.post_order(order, OrderType.GTC)
        return True, shares
    except Exception as e:
        print(f"Place order error: {e}")
        return False, 0.0


def update_position_on_fill(
    positions: list[Position],
    token_id: str,
    condition_id: str,
    outcome_label: str,
    filled_size: float,
    fill_price: float,
) -> None:
    """Update or add position using filled amount only (partial-fill safe)."""
    for p in positions:
        if p.token_id == token_id and p.condition_id == condition_id:
            old_sz, old_cb = p.filled_size, p.cost_basis
            new_sz = old_sz + filled_size
            if new_sz <= 0:
                positions.remove(p)
                return
            p.cost_basis = (old_sz * old_cb + filled_size * fill_price) / new_sz
            p.filled_size = new_sz
            p.size = new_sz
            return
    if filled_size > 0:
        positions.append(Position(
            token_id=token_id,
            outcome_label=outcome_label,
            condition_id=condition_id,
            size=filled_size,
            cost_basis=fill_price,
            filled_size=filled_size,
        ))
