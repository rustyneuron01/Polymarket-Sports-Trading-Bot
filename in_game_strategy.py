"""
In-game strategy: token price updates in real time; we predict the price path and trade on that.

Preferred approach (price-based, no probability thresholds):
  - Train the model to predict the lowest and highest token price in the next window, using
    score, time remaining, period, team Elo, and price deltas (--train-price-range).
  - At inference, use those predicted levels: BUY when current ask is at or below the predicted
    low; SELL when current bid is at or above the predicted high. No thresholds on P(reward).
  - Use price_range_signal_fn(bundle) or price_range_signal_fn_dual(bundle) and backtest with
    --use-price-range.

Legacy (threshold-based):
  - reward_model_signal_fn(bundle, buy_threshold, sell_threshold): buy when P(reward) >= threshold.
  - buy_sell_signal_fn(bundle, ...): same for buy/sell-opportunity heads. These predict a
    probability and apply a cutoff; price_range_* uses predicted price levels instead.

See DOCS.md §2 for data needed to build the prediction model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from config import IN_GAME_BUY_PRICE_TARGET, IN_GAME_SELL_PRICE_TARGET, IN_GAME_USE_FIXED_TARGETS


@dataclass
class InGameTrade:
    side: Literal["home", "away"]
    action: Literal["BUY", "SELL"]
    price: float
    timestamp: str
    profit: float = 0.0  # set on SELL


# Type: (game_state, price_series_so_far, position) -> "BUY_HOME" | "BUY_AWAY" | "SELL" | "HOLD"
# position: None if flat, "home" or "away" if we hold that side
SignalFn = Callable[
    [dict, list[dict], Optional[Literal["home", "away"]]],
    Literal["BUY_HOME", "BUY_AWAY", "SELL", "HOLD"],
]

# Dual position: trade both tokens (4 trades per game = buy/sell home + buy/sell away). position_home/position_away = are we holding that side.
DualSignal = Literal["BUY_HOME", "BUY_AWAY", "SELL_HOME", "SELL_AWAY", "HOLD"]
SignalFnDual = Callable[
    [dict, list[dict], bool, bool],
    DualSignal,
]


def _build_feature_dict_for_reward(
    game_state: dict,
    price_row: dict,
    index: int,
    price_series_so_far: list[dict] | None = None,
) -> dict:
    """
    Build feature dict for reward model (same keys as train_in_game_model.IN_GAME_FEATURE_NAMES).
    game_state can have: score_home/home_goals, score_away/away_goals, period, time_remaining_sec, game_elapsed_sec, game_second_proxy.
    price_series_so_far: used to compute price deltas (change since previous snapshot); optional.
    """
    score_home = int(game_state.get("score_home", game_state.get("home_goals", 0)))
    score_away = int(game_state.get("score_away", game_state.get("away_goals", 0)))
    period = int(game_state.get("period", 1))
    time_remaining_sec = float(game_state.get("time_remaining_sec", 3600.0))
    proxy = index * 60.0
    game_elapsed_sec = float(game_state.get("game_elapsed_sec", proxy))
    game_second_proxy = float(game_state.get("game_second_proxy", proxy))
    score_gap = score_home - score_away

    def _f(k: str) -> float:
        v = price_row.get(k)
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    # Deltas: at inference compute from previous snapshot if available (price change / event-aligned).
    ask_home_delta = bid_home_delta = ask_away_delta = bid_away_delta = 0.0
    if price_series_so_far and len(price_series_so_far) >= 2:
        prev = price_series_so_far[-2]
        def _d(key: str) -> float:
            try:
                c, p = price_row.get(key), prev.get(key)
                if c is not None and p is not None:
                    return float(c) - float(p)
            except (TypeError, ValueError):
                pass
            return 0.0
        ask_home_delta = _d("ask_home")
        bid_home_delta = _d("bid_home")
        ask_away_delta = _d("ask_away")
        bid_away_delta = _d("bid_away")

    regulation_sec = 3 * 20 * 60
    time_remaining_ratio = min(1.0, max(0.0, time_remaining_sec / regulation_sec))
    abs_score_gap = abs(score_gap)
    elo_home = float(game_state.get("elo_home", 1500.0))
    elo_away = float(game_state.get("elo_away", 1500.0))
    elo_advantage = elo_home - elo_away

    return {
        "score_gap": score_gap,
        "time_remaining_sec": time_remaining_sec,
        "time_remaining_ratio": time_remaining_ratio,
        "score_home": score_home,
        "score_away": score_away,
        "abs_score_gap": abs_score_gap,
        "period": period,
        "elo_advantage": elo_advantage,
        "elo_home": elo_home,
        "elo_away": elo_away,
        "game_elapsed_sec": game_elapsed_sec,
        "game_second_proxy": game_second_proxy,
        "ask_home": _f("ask_home"),
        "bid_home": _f("bid_home"),
        "ask_away": _f("ask_away"),
        "bid_away": _f("bid_away"),
        "ask_home_delta": ask_home_delta,
        "bid_home_delta": bid_home_delta,
        "ask_away_delta": ask_away_delta,
        "bid_away_delta": bid_away_delta,
    }


def reward_model_signal_fn(
    bundle: dict,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    loss_sell_threshold: float = 0.55,
) -> SignalFn:
    """
    Signal from reward model (data/reward_model.pkl). High P(reward) = buy that side; low P(reward) or high P(loss) when holding = sell.

    buy_threshold: only buy when P(reward for that side) >= this (default 0.55 = require >50% chance of profit in window).
    sell_threshold: sell when we hold and P(reward for our side) < this (default 0.45). The gap (0.55 vs 0.45) is hysteresis to avoid whipsaw.
    loss_sell_threshold: if bundle has loss models (trained with --train-loss), also sell when P(loss) >= this — "sell before it gets worse" from data.
    """

    def fn(
        game_state: dict,
        price_series_so_far: list[dict],
        position: Optional[Literal["home", "away"]],
    ) -> Literal["BUY_HOME", "BUY_AWAY", "SELL", "HOLD"]:
        if not price_series_so_far:
            return "HOLD"
        from train_in_game_model import predict_reward_proba, predict_loss_proba

        row = price_series_so_far[-1]
        index = len(price_series_so_far) - 1
        fd = _build_feature_dict_for_reward(game_state, row, index, price_series_so_far)
        prob_home, prob_away = predict_reward_proba(bundle, fd)
        if prob_home is None and prob_away is None:
            return "HOLD"

        # When holding: sell if P(reward) low or P(loss) high (learned from data)
        if position is not None:
            prob_loss_home, prob_loss_away = predict_loss_proba(bundle, fd)
            if position == "home":
                if prob_home is not None and prob_home < sell_threshold:
                    return "SELL"
                if prob_loss_home is not None and prob_loss_home >= loss_sell_threshold:
                    return "SELL"
            if position == "away":
                if prob_away is not None and prob_away < sell_threshold:
                    return "SELL"
                if prob_loss_away is not None and prob_loss_away >= loss_sell_threshold:
                    return "SELL"
            return "HOLD"

        if prob_home is not None and prob_away is not None:
            if prob_home >= buy_threshold and prob_home >= prob_away:
                return "BUY_HOME"
            if prob_away >= buy_threshold and prob_away >= prob_home:
                return "BUY_AWAY"
        elif prob_home is not None and prob_home >= buy_threshold:
            return "BUY_HOME"
        elif prob_away is not None and prob_away >= buy_threshold:
            return "BUY_AWAY"
        return "HOLD"

    return fn


def buy_sell_signal_fn(
    bundle: dict,
    buy_threshold: float = 0.45,
    sell_threshold: float = 0.45,
) -> SignalFnDual:
    """
    Signal for 4 trades per game: predict low (buy) and high (sell) per token; profit on both home and away.
    Uses model_buy_home, model_sell_home, model_buy_away, model_sell_away (train with --train-buy-sell).
    position_home / position_away = True if we currently hold that token.
    """
    def fn(
        game_state: dict,
        price_series_so_far: list[dict],
        position_home: bool,
        position_away: bool,
    ) -> DualSignal:
        if not price_series_so_far:
            return "HOLD"
        from train_in_game_model import predict_buy_sell_proba

        row = price_series_so_far[-1]
        index = len(price_series_so_far) - 1
        fd = _build_feature_dict_for_reward(game_state, row, index, price_series_so_far)
        buy_home, sell_home, buy_away, sell_away = predict_buy_sell_proba(bundle, fd)

        # Prefer selling first (lock profit) before new buys
        if position_home and sell_home is not None and sell_home >= sell_threshold:
            return "SELL_HOME"
        if position_away and sell_away is not None and sell_away >= sell_threshold:
            return "SELL_AWAY"
        if not position_home and buy_home is not None and buy_home >= buy_threshold:
            return "BUY_HOME"
        if not position_away and buy_away is not None and buy_away >= buy_threshold:
            return "BUY_AWAY"
        return "HOLD"

    return fn


# Minimum predicted spread (high - low) to allow a round-trip. Training data often has max_bid < min_ask (spread);
# the model then learns that and predicts pred_max < pred_min, so we'd buy "low" and sell "high" at a lower price → always lose.
MIN_PREDICTED_SPREAD = 0.02  # require pred_max >= pred_min + this before we trade


def price_range_signal_fn(
    bundle: dict,
    buy_margin: float = 0.02,
    sell_margin: float = 0.02,
    stop_loss_pct: float = 0.08,
) -> SignalFn:
    """
    Signal from predicted price levels only (no probability thresholds).
    Model predicts lowest and highest token price in the window (from score, time, events, etc.).
    Buy when current ask is at or below predicted low; sell when current bid is at or above predicted high.
    Requires pred_max > pred_min + MIN_PREDICTED_SPREAD for that token (else we skip: no profitable range predicted).
    stop_loss_pct: when holding, sell if bid drops to or below entry_price * (1 - stop_loss_pct) to limit loss
    (e.g. 0.08 = 8%%; avoids holding to game end and taking a huge loss when predicted high never comes).
    """
    def fn(
        game_state: dict,
        price_series_so_far: list[dict],
        position: Optional[Literal["home", "away"]],
    ) -> Literal["BUY_HOME", "BUY_AWAY", "SELL", "HOLD"]:
        if not price_series_so_far or "model_min_ask_home" not in bundle:
            return "HOLD"
        from train_in_game_model import predict_price_range

        row = price_series_so_far[-1]
        index = len(price_series_so_far) - 1
        fd = _build_feature_dict_for_reward(game_state, row, index, price_series_so_far)
        pred_min_h, pred_max_h, pred_min_a, pred_max_a = predict_price_range(bundle, fd)
        ask_home = row.get("ask_home")
        ask_away = row.get("ask_away")
        bid_home = row.get("bid_home")
        bid_away = row.get("bid_away")

        # Stop-loss: when holding, sell if price moved against us (avoid holding to game end and big loss)
        entry = game_state.get("_entry_price")
        if position is not None and entry is not None and entry > 0:
            bid = bid_home if position == "home" else bid_away
            if bid is not None and bid <= entry * (1.0 - stop_loss_pct):
                return "SELL"

        # Only trade when model predicts a profitable range: pred_max > pred_min + spread (else we'd sell below buy)
        ok_home = pred_min_h is not None and pred_max_h is not None and (pred_max_h - pred_min_h) >= MIN_PREDICTED_SPREAD
        ok_away = pred_min_a is not None and pred_max_a is not None and (pred_max_a - pred_min_a) >= MIN_PREDICTED_SPREAD

        # Holding: sell when bid at or above predicted high
        if position == "home" and ok_home and bid_home is not None:
            if bid_home >= pred_max_h * (1.0 - sell_margin):
                return "SELL"
        if position == "away" and ok_away and bid_away is not None:
            if bid_away >= pred_max_a * (1.0 - sell_margin):
                return "SELL"
        if position is not None:
            return "HOLD"

        # Flat: buy only when predicted range is profitable (pred_max > pred_min + spread)
        buy_h = ok_home and ask_home is not None and ask_home <= pred_min_h * (1.0 + buy_margin)
        buy_a = ok_away and ask_away is not None and ask_away <= pred_min_a * (1.0 + buy_margin)
        if buy_h and buy_a:
            discount_h = (pred_min_h - ask_home) / pred_min_h if pred_min_h else 0
            discount_a = (pred_min_a - ask_away) / pred_min_a if pred_min_a else 0
            return "BUY_HOME" if discount_h >= discount_a else "BUY_AWAY"
        if buy_h:
            return "BUY_HOME"
        if buy_a:
            return "BUY_AWAY"
        return "HOLD"

    return fn


def price_range_signal_fn_dual(
    bundle: dict,
    buy_margin: float = 0.02,
    sell_margin: float = 0.02,
    stop_loss_pct: float = 0.08,
) -> SignalFnDual:
    """
    Same as price_range_signal_fn but for 4 trades per game: buy/sell each token when price
    reaches predicted low/high. stop_loss_pct: sell when bid <= entry * (1 - stop_loss_pct).
    """
    def fn(
        game_state: dict,
        price_series_so_far: list[dict],
        position_home: bool,
        position_away: bool,
    ) -> DualSignal:
        if not price_series_so_far or "model_min_ask_home" not in bundle:
            return "HOLD"
        from train_in_game_model import predict_price_range

        row = price_series_so_far[-1]
        index = len(price_series_so_far) - 1
        fd = _build_feature_dict_for_reward(game_state, row, index, price_series_so_far)
        pred_min_h, pred_max_h, pred_min_a, pred_max_a = predict_price_range(bundle, fd)
        ask_home = row.get("ask_home")
        ask_away = row.get("ask_away")
        bid_home = row.get("bid_home")
        bid_away = row.get("bid_away")

        # Stop-loss per side (entry from game_state: _entry_price_home, _entry_price_away when in dual)
        entry_h = game_state.get("_entry_price_home")
        entry_a = game_state.get("_entry_price_away")
        if position_home and entry_h is not None and entry_h > 0 and bid_home is not None and bid_home <= entry_h * (1.0 - stop_loss_pct):
            return "SELL_HOME"
        if position_away and entry_a is not None and entry_a > 0 and bid_away is not None and bid_away <= entry_a * (1.0 - stop_loss_pct):
            return "SELL_AWAY"

        # Require profitable range: pred_max > pred_min + MIN_PREDICTED_SPREAD (else we'd sell below buy)
        ok_home = pred_min_h is not None and pred_max_h is not None and (pred_max_h - pred_min_h) >= MIN_PREDICTED_SPREAD
        ok_away = pred_min_a is not None and pred_max_a is not None and (pred_max_a - pred_min_a) >= MIN_PREDICTED_SPREAD
        # Prefer selling first (lock profit at predicted high)
        if position_home and ok_home and bid_home is not None and bid_home >= pred_max_h * (1.0 - sell_margin):
            return "SELL_HOME"
        if position_away and ok_away and bid_away is not None and bid_away >= pred_max_a * (1.0 - sell_margin):
            return "SELL_AWAY"
        if not position_home and ok_home and ask_home is not None and ask_home <= pred_min_h * (1.0 + buy_margin):
            return "BUY_HOME"
        if not position_away and ok_away and ask_away is not None and ask_away <= pred_min_a * (1.0 + buy_margin):
            return "BUY_AWAY"
        return "HOLD"

    return fn


def _fixed_target_signal(buy_target: float, sell_target: float) -> SignalFn:
    """Optional fallback: fixed price levels (see DOCS.md §2 for model-based signals)."""

    def fn(
        game_state: dict,
        price_series_so_far: list[dict],
        position: Optional[Literal["home", "away"]],
    ) -> Literal["BUY_HOME", "BUY_AWAY", "SELL", "HOLD"]:
        if not price_series_so_far:
            return "HOLD"
        row = price_series_so_far[-1]
        ask_home = row.get("ask_home")
        ask_away = row.get("ask_away")
        bid_home = row.get("bid_home")
        bid_away = row.get("bid_away")
        if position is not None:
            bid = bid_home if position == "home" else bid_away
            if bid is not None and bid >= sell_target:
                return "SELL"
            return "HOLD"
        if ask_home is not None and ask_home <= buy_target:
            return "BUY_HOME"
        if ask_away is not None and ask_away <= buy_target:
            return "BUY_AWAY"
        return "HOLD"

    return fn


def simulate_in_game(
    price_series: list[dict],
    game_state_series: Optional[list[dict]] = None,
    signal_fn: Optional[SignalFn] = None,
    buy_target: float = IN_GAME_BUY_PRICE_TARGET,
    sell_target: float = IN_GAME_SELL_PRICE_TARGET,
    max_round_trips: Optional[int] = 1,
) -> tuple[list[InGameTrade], float]:
    """
    Run in-game trading on one game's price series. Decisions come from the model (signal_fn).
    If signal_fn is None and IN_GAME_USE_FIXED_TARGETS is true, use fixed buy/sell levels.
    Otherwise no trades (we need a model; see DOCS.md §2).

    max_round_trips: max buy->sell cycles per game (default 1 = one trade per game). None = unlimited.
    If still in a position at end of game, we "close" at last available bid and book that P&L (realized or loss).

    price_series: list of { timestamp, ask_home, bid_home, ask_away, bid_away } from Polymarket (fetch via get_order_books / data_collector; do not assume).
    game_state_series: optional list of { timestamp, home_goals, away_goals, period, time_remaining_sec } aligned by timestamp.
    signal_fn: (game_state, price_series_so_far, position) -> "BUY_HOME" | "BUY_AWAY" | "SELL" | "HOLD".
    """
    trades: list[InGameTrade] = []
    position: Optional[Literal["home", "away"]] = None
    buy_price = 0.0
    total_profit = 0.0
    round_trips_done = 0

    # Resolve signal: model or fixed targets
    if signal_fn is None and IN_GAME_USE_FIXED_TARGETS:
        signal_fn = _fixed_target_signal(buy_target, sell_target)
    if signal_fn is None:
        return trades, total_profit

    # Align game_state to each row by timestamp (or use last known state)
    def state_at(i: int) -> dict:
        if not game_state_series:
            return {}
        ts = price_series[i].get("timestamp", "")
        for s in reversed(game_state_series):
            if s.get("timestamp") <= ts:
                return s
        return game_state_series[0] if game_state_series else {}

    for i, row in enumerate(price_series):
        ts = row.get("timestamp", "")
        history = price_series[: i + 1]
        state = dict(state_at(i))
        if position is not None and buy_price > 0:
            state["_entry_price"] = buy_price  # so signal can use stop-loss
        ask_home = row.get("ask_home")
        ask_away = row.get("ask_away")
        bid_home = row.get("bid_home")
        bid_away = row.get("bid_away")

        signal = signal_fn(state, history, position)

        if position is None:
            can_buy = max_round_trips is None or round_trips_done < max_round_trips
            if can_buy and signal == "BUY_HOME" and ask_home is not None:
                position = "home"
                buy_price = ask_home
                trades.append(InGameTrade(side="home", action="BUY", price=buy_price, timestamp=ts))
            elif can_buy and signal == "BUY_AWAY" and ask_away is not None:
                position = "away"
                buy_price = ask_away
                trades.append(InGameTrade(side="away", action="BUY", price=buy_price, timestamp=ts))
        else:
            if signal == "SELL":
                bid = bid_home if position == "home" else bid_away
                if bid is None:
                    key = "bid_home" if position == "home" else "bid_away"
                    for past in reversed(history):
                        b = past.get(key)
                        if b is not None:
                            bid = b
                            break
                sell_price = float(bid) if bid is not None else buy_price
                # Avoid fake $0 profit: when bid ≈ ask (merged/forward-filled data), we'd book 0 but real trading has spread cost
                if buy_price > 0 and abs(sell_price - buy_price) < 0.005:
                    sell_price = max(0.01, buy_price - 0.01)  # assume 1% round-trip cost
                profit = sell_price - buy_price
                total_profit += profit
                trades.append(InGameTrade(side=position, action="SELL", price=sell_price, timestamp=ts, profit=profit))
                position = None
                round_trips_done += 1

    # Close open position at game end (unrealized P&L: e.g. invested $100, exit at $80 = -$20 loss)
    # Use last available bid for our side so we always reflect real price move, not fake break-even.
    if position is not None and price_series:
        last_row = price_series[-1]
        bid = (last_row.get("bid_home") if position == "home" else last_row.get("bid_away"))
        if bid is None:
            key = "bid_home" if position == "home" else "bid_away"
            for row in reversed(price_series):
                b = row.get(key)
                if b is not None:
                    bid = b
                    break
        sell_price = float(bid) if bid is not None else buy_price
        if buy_price > 0 and abs(sell_price - buy_price) < 0.005:
            sell_price = max(0.01, buy_price - 0.01)
        profit = sell_price - buy_price
        total_profit += profit
        ts = last_row.get("timestamp", "")
        trades.append(InGameTrade(side=position, action="SELL", price=sell_price, timestamp=ts, profit=profit))

    return trades, total_profit


def simulate_in_game_dual(
    price_series: list[dict],
    game_state_series: Optional[list[dict]] = None,
    signal_fn: Optional[SignalFnDual] = None,
) -> tuple[list[InGameTrade], float]:
    """
    Run in-game trading with two positions: one round-trip on home token and one on away token (4 trades per game).
    Use with buy_sell_signal_fn (trained with --train-buy-sell) to predict low/high per token and profit on both.
    """
    trades: list[InGameTrade] = []
    position_home = False
    position_away = False
    buy_price_home = 0.0
    buy_price_away = 0.0
    total_profit = 0.0

    if signal_fn is None:
        return trades, total_profit

    def state_at(i: int) -> dict:
        if not game_state_series:
            return {}
        ts = price_series[i].get("timestamp", "")
        for s in reversed(game_state_series):
            if s.get("timestamp") <= ts:
                return s
        return game_state_series[0] if game_state_series else {}

    def _get_bid(row: dict, side: Literal["home", "away"]) -> float | None:
        key = "bid_home" if side == "home" else "bid_away"
        return row.get(key)

    def _get_ask(row: dict, side: Literal["home", "away"]) -> float | None:
        key = "ask_home" if side == "home" else "ask_away"
        return row.get(key)

    for i, row in enumerate(price_series):
        ts = row.get("timestamp", "")
        history = price_series[: i + 1]
        state = dict(state_at(i))
        if position_home and buy_price_home > 0:
            state["_entry_price_home"] = buy_price_home
        if position_away and buy_price_away > 0:
            state["_entry_price_away"] = buy_price_away
        signal = signal_fn(state, history, position_home, position_away)

        if signal == "BUY_HOME" and not position_home:
            ask = _get_ask(row, "home")
            if ask is not None:
                position_home = True
                buy_price_home = ask
                trades.append(InGameTrade(side="home", action="BUY", price=buy_price_home, timestamp=ts))
        elif signal == "SELL_HOME" and position_home:
            bid = _get_bid(row, "home")
            if bid is None:
                for past in reversed(history):
                    bid = _get_bid(past, "home")
                    if bid is not None:
                        break
            sell_price = float(bid) if bid is not None else buy_price_home
            if buy_price_home > 0 and abs(sell_price - buy_price_home) < 0.005:
                sell_price = max(0.01, buy_price_home - 0.01)
            profit = sell_price - buy_price_home
            total_profit += profit
            trades.append(InGameTrade(side="home", action="SELL", price=sell_price, timestamp=ts, profit=profit))
            position_home = False

        elif signal == "BUY_AWAY" and not position_away:
            ask = _get_ask(row, "away")
            if ask is not None:
                position_away = True
                buy_price_away = ask
                trades.append(InGameTrade(side="away", action="BUY", price=buy_price_away, timestamp=ts))
        elif signal == "SELL_AWAY" and position_away:
            bid = _get_bid(row, "away")
            if bid is None:
                for past in reversed(history):
                    bid = _get_bid(past, "away")
                    if bid is not None:
                        break
            sell_price = float(bid) if bid is not None else buy_price_away
            if buy_price_away > 0 and abs(sell_price - buy_price_away) < 0.005:
                sell_price = max(0.01, buy_price_away - 0.01)
            profit = sell_price - buy_price_away
            total_profit += profit
            trades.append(InGameTrade(side="away", action="SELL", price=sell_price, timestamp=ts, profit=profit))
            position_away = False

    # Close open positions at game end
    if price_series:
        last_row = price_series[-1]
        end_ts = last_row.get("timestamp", "")
        if position_home:
            bid = _get_bid(last_row, "home")
            if bid is None:
                for r in reversed(price_series):
                    bid = _get_bid(r, "home")
                    if bid is not None:
                        break
            sell_price = float(bid) if bid is not None else buy_price_home
            if buy_price_home > 0 and abs(sell_price - buy_price_home) < 0.005:
                sell_price = max(0.01, buy_price_home - 0.01)
            profit = sell_price - buy_price_home
            total_profit += profit
            trades.append(InGameTrade(side="home", action="SELL", price=sell_price, timestamp=end_ts, profit=profit))
        if position_away:
            bid = _get_bid(last_row, "away")
            if bid is None:
                for r in reversed(price_series):
                    bid = _get_bid(r, "away")
                    if bid is not None:
                        break
            sell_price = float(bid) if bid is not None else buy_price_away
            if buy_price_away > 0 and abs(sell_price - buy_price_away) < 0.005:
                sell_price = max(0.01, buy_price_away - 0.01)
            profit = sell_price - buy_price_away
            total_profit += profit
            trades.append(InGameTrade(side="away", action="SELL", price=sell_price, timestamp=end_ts, profit=profit))

    return trades, total_profit
