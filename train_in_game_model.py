"""
Train in-game REWARD model: predict whether we get a reward (profit) if we buy at this point and sell in the lookahead window.
Target = reward_home / reward_away (1 = can sell at profit after fee), NOT who wins the game.
Who wins is for Elo; this model is for token price change / trading decisions.

Uses data/in_game_dataset.jsonl produced by build_in_game_dataset.py.
Saves data/reward_model.pkl for use by in-game strategy (predict_reward or similar).

Usage:
  python -m polymarket_nhl_bot.build_in_game_dataset --dir data/game_records --out data/in_game_dataset.jsonl
  python -m polymarket_nhl_bot.train_in_game_model --data data/in_game_dataset.jsonl --model-out data/reward_model.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Features for in-game price prediction. Primary: score and time remaining (main drivers of token move).
# Then: period, team strength (elo), game clock, then current prices and deltas.
# score_gap = score_home - score_away (positive = home leading). time_remaining_ratio = time left / 3600 (0-1).
# abs_score_gap = how close the game is. elo_advantage = elo_home - elo_away (team strength differential).
IN_GAME_FEATURE_NAMES = [
    "score_gap",
    "time_remaining_sec",
    "time_remaining_ratio",
    "score_home",
    "score_away",
    "abs_score_gap",
    "period",
    "elo_advantage",
    "elo_home",
    "elo_away",
    "game_elapsed_sec",
    "game_second_proxy",
    "ask_home",
    "bid_home",
    "ask_away",
    "bid_away",
    "ask_home_delta",
    "bid_home_delta",
    "ask_away_delta",
    "bid_away_delta",
]


def load_reward_model(path: Path) -> dict | None:
    """
    Load reward model bundle from path (data/reward_model.pkl).
    Returns None if file missing or invalid. Bundle has feature_names, model_home, model_away.
    """
    if not path.exists():
        return None
    try:
        import joblib
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "feature_names" not in bundle:
            return None
        if "model_home" not in bundle and "model_away" not in bundle:
            return None
        return bundle
    except Exception:
        return None


def predict_reward_proba(
    bundle: dict,
    feature_dict: dict,
) -> tuple[float | None, float | None]:
    """
    Predict P(reward_home=1) and P(reward_away=1) for one snapshot.
    feature_dict: keys matching IN_GAME_FEATURE_NAMES (game_elapsed_sec, period, score_home, score_away, score_gap, time_remaining_sec, ask_home, bid_home, ask_away, bid_away).
    Returns (prob_home, prob_away); either can be None if that model is not in the bundle.
    """
    names = bundle.get("feature_names", IN_GAME_FEATURE_NAMES)
    vec = []
    for n in names:
        v = feature_dict.get(n)
        if v is None:
            v = 0.0
        try:
            vec.append(float(v))
        except (TypeError, ValueError):
            vec.append(0.0)
    import numpy as np
    X = np.array([vec], dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0)
    if "scaler" in bundle:
        X = bundle["scaler"].transform(X)

    prob_home = None
    prob_away = None
    if "model_home" in bundle:
        prob_home = float(bundle["model_home"].predict_proba(X)[0, 1])
    if "model_away" in bundle:
        prob_away = float(bundle["model_away"].predict_proba(X)[0, 1])
    return prob_home, prob_away


def predict_loss_proba(
    bundle: dict,
    feature_dict: dict,
) -> tuple[float | None, float | None]:
    """
    Predict P(loss_home=1) and P(loss_away=1): probability price would drop below entry - fee in the window.
    Use with reward_model_signal_fn to sell when P(loss) is high (cut loss before it gets worse).
    Returns (prob_loss_home, prob_loss_away); either can be None if that model is not in the bundle.
    """
    names = bundle.get("feature_names", IN_GAME_FEATURE_NAMES)
    vec = []
    for n in names:
        v = feature_dict.get(n)
        if v is None:
            v = 0.0
        try:
            vec.append(float(v))
        except (TypeError, ValueError):
            vec.append(0.0)
    import numpy as np
    X = np.array([vec], dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0)
    if "scaler" in bundle:
        X = bundle["scaler"].transform(X)

    prob_loss_home = None
    prob_loss_away = None
    if "model_loss_home" in bundle:
        prob_loss_home = float(bundle["model_loss_home"].predict_proba(X)[0, 1])
    if "model_loss_away" in bundle:
        prob_loss_away = float(bundle["model_loss_away"].predict_proba(X)[0, 1])
    return prob_loss_home, prob_loss_away


def predict_price_range(
    bundle: dict,
    feature_dict: dict,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Predict min ask and max bid in the next window for home and away (for display/verification).
    Returns (min_ask_home, max_bid_home, min_ask_away, max_bid_away); any can be None if that head is not in the bundle.
    """
    names = bundle.get("feature_names", IN_GAME_FEATURE_NAMES)
    vec = []
    for n in names:
        v = feature_dict.get(n)
        if v is None:
            v = 0.0
        try:
            vec.append(float(v))
        except (TypeError, ValueError):
            vec.append(0.0)
    import numpy as np
    X = np.array([vec], dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0)
    if "scaler" in bundle:
        X = bundle["scaler"].transform(X)

    out = []
    for key in ("model_min_ask_home", "model_max_bid_home", "model_min_ask_away", "model_max_bid_away"):
        if key in bundle:
            pred = bundle[key].predict(X)
            out.append(float(np.clip(pred[0], 0.0, 1.0)))
        else:
            out.append(None)
    return tuple(out)


def predict_buy_sell_proba(
    bundle: dict,
    feature_dict: dict,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Predict P(buy_opportunity) and P(sell_opportunity) for home and away.
    Use for 4-trades-per-game strategy: buy at predicted low, sell at predicted high for each token.
    Returns (buy_home, sell_home, buy_away, sell_away); any can be None if that head is not in the bundle.
    """
    names = bundle.get("feature_names", IN_GAME_FEATURE_NAMES)
    vec = []
    for n in names:
        v = feature_dict.get(n)
        if v is None:
            v = 0.0
        try:
            vec.append(float(v))
        except (TypeError, ValueError):
            vec.append(0.0)
    import numpy as np
    X = np.array([vec], dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0)
    if "scaler" in bundle:
        X = bundle["scaler"].transform(X)

    out = []
    for key in ("model_buy_home", "model_sell_home", "model_buy_away", "model_sell_away"):
        if key in bundle:
            out.append(float(bundle[key].predict_proba(X)[0, 1]))
        else:
            out.append(None)
    return tuple(out)  # (buy_home, sell_home, buy_away, sell_away)


def load_in_game_dataset(path: Path) -> tuple[list[list[float]], list[int], list[int], list[int], list[int], list[int], list[int], list[int], list[int], list[float], list[float], list[float], list[float]]:
    """
    Load JSONL from build_in_game_dataset. Each line has features + reward, loss, buy/sell opportunity + min/max price in window.
    Returns (X, y_home, y_away, y_loss_home, y_loss_away, y_buy_home, y_sell_home, y_buy_away, y_sell_away,
             y_min_ask_home, y_max_bid_home, y_min_ask_away, y_max_bid_away). y_* classification are 0/1; regression are float.
    """
    X = []
    y_home = []
    y_away = []
    y_loss_home = []
    y_loss_away = []
    y_buy_home = []
    y_sell_home = []
    y_buy_away = []
    y_sell_away = []
    y_min_ask_home = []
    y_max_bid_home = []
    y_min_ask_away = []
    y_max_bid_away = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_vec = []
            for name in IN_GAME_FEATURE_NAMES:
                v = row.get(name)
                if v is None:
                    if name in ("elo_home", "elo_away"):
                        v = 1500.0
                    elif name == "elo_advantage":
                        v = float(row.get("elo_home") or 1500) - float(row.get("elo_away") or 1500)
                    elif name == "time_remaining_ratio" and row.get("time_remaining_sec") is not None:
                        v = min(1.0, max(0.0, float(row["time_remaining_sec"]) / 3600.0))
                    elif name == "abs_score_gap" and (row.get("score_gap") is not None or row.get("score_home") is not None):
                        sg = row.get("score_gap")
                        if sg is None:
                            sg = (row.get("score_home") or 0) - (row.get("score_away") or 0)
                        v = abs(float(sg))
                    else:
                        v = 0.0
                try:
                    row_vec.append(float(v))
                except (TypeError, ValueError):
                    row_vec.append(1500.0 if name in ("elo_home", "elo_away") else 0.0)
            X.append(row_vec)
            y_home.append(int(row.get("reward_home", 0)))
            y_away.append(int(row.get("reward_away", 0)))
            y_loss_home.append(int(row.get("loss_home", 0)))
            y_loss_away.append(int(row.get("loss_away", 0)))
            y_buy_home.append(int(row.get("buy_opportunity_home", 0)))
            y_sell_home.append(int(row.get("sell_opportunity_home", 0)))
            y_buy_away.append(int(row.get("buy_opportunity_away", 0)))
            y_sell_away.append(int(row.get("sell_opportunity_away", 0)))
            def _reg(v):
                if v is None:
                    return 0.0
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0
            y_min_ask_home.append(_reg(row.get("min_ask_home_next")))
            y_max_bid_home.append(_reg(row.get("max_bid_home_next")))
            y_min_ask_away.append(_reg(row.get("min_ask_away_next")))
            y_max_bid_away.append(_reg(row.get("max_bid_away_next")))
    return X, y_home, y_away, y_loss_home, y_loss_away, y_buy_home, y_sell_home, y_buy_away, y_sell_away, y_min_ask_home, y_max_bid_home, y_min_ask_away, y_max_bid_away


def _make_classifier(model_type: str, epochs: int, random_state: int):
    """
    Create classifier by type. All support .fit(X, y) and .predict_proba(X)[:, 1].
    Uses early stopping (mlp/gb) and regularization to avoid memorization; more epochs
    only allow the solver to converge (logistic) or give early stopping more room to pick the best stop.
    """
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            random_state=random_state, max_iter=epochs, class_weight="balanced", C=0.5
        )
    if model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=epochs,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            alpha=0.01,
            random_state=random_state,
        )
    if model_type == "gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=epochs,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            max_depth=8,
            l2_regularization=0.1,
            random_state=random_state,
            class_weight="balanced",
        )
    raise ValueError(f"Unknown model type: {model_type}")


def _eval_and_print(name: str, y_val, y_pred, pos_pct: float) -> None:
    """Print accuracy and precision/recall/F1 for positive class (reward=1)."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(y_val, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", pos_label=1, zero_division=0)
    print(f"{name}: val accuracy = {acc:.3f}  (pos% = {pos_pct:.1f})  P(reward=1) = {p:.3f}  R = {r:.3f}  F1 = {f1:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train in-game reward model (target = profit if buy now, sell in window).")
    parser.add_argument("--data", default=None, help="Input JSONL from build_in_game_dataset (default: data/in_game_dataset.jsonl)")
    parser.add_argument("--model-out", default=None, help="Output pkl path (default: data/reward_model.pkl)")
    parser.add_argument("--target", choices=("home", "away", "both"), default="both", help="Train on reward_home, reward_away, or both")
    parser.add_argument("--train-loss", action="store_true", help="Also train loss heads (P(loss)=1 if price drops in window); use for sell-early / cut-loss signal")
    parser.add_argument("--train-buy-sell", action="store_true", help="Also train buy/sell-opportunity heads (predict low=buy, high=sell); use for 4-trades-per-game (profit on both tokens)")
    parser.add_argument("--train-price-range", action="store_true",
        help="Train regressors for min/max price in window (predict_price_range). Use with price_range_signal_fn: buy when price at predicted low, sell when at predicted high (no probability thresholds). Backtest with --use-price-range.")
    parser.add_argument("--tune-thresholds", action="store_true", help="After training, run backtest grid search and print recommended --buy-threshold and --sell-threshold (no fixed values)")
    parser.add_argument("--tune-records-dir", type=str, default=None, help="Game records dir for --tune-thresholds (default: data/game_records)")
    parser.add_argument("--tune-from", type=str, default=None, help="From date for threshold tune (YYYY-MM-DD)")
    parser.add_argument("--tune-to", type=str, default=None, help="To date for threshold tune (YYYY-MM-DD)")
    parser.add_argument("--model", choices=("logistic", "mlp", "gb"), default="logistic",
        help="Model: logistic (fast), mlp (epoch-based), gb (gradient boosting, deeper)")
    parser.add_argument("--epochs", type=int, default=500,
        help="Max iterations/epochs. Logistic: solver iters (no memorization risk). MLP/GB: early stopping stops when val stops improving; regularization (alpha/L2/max_depth) limits overfitting.")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    data_path = Path(args.data) if args.data else base / "data" / "in_game_dataset.jsonl"
    model_path = Path(args.model_out) if args.model_out else base / "data" / "reward_model.pkl"

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Run build_in_game_dataset first:")
        print("  python -m polymarket_nhl_bot.build_in_game_dataset --dir data/game_records --out data/in_game_dataset.jsonl")
        raise SystemExit(1)

    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError as e:
        print("Requires scikit-learn, numpy, joblib. pip install scikit-learn numpy joblib")
        raise SystemExit(1) from e

    X, y_home, y_away, y_loss_home, y_loss_away, y_buy_home, y_sell_home, y_buy_away, y_sell_away, y_min_ask_home, y_max_bid_home, y_min_ask_away, y_max_bid_away = load_in_game_dataset(data_path)
    if len(X) < 100:
        print(f"Too few rows ({len(X)}). Need at least 100. Build more game records and run build_in_game_dataset.")
        raise SystemExit(1)

    X_arr = np.array(X, dtype=np.float64)
    np.nan_to_num(X_arr, copy=False, nan=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    bundle = {"feature_names": IN_GAME_FEATURE_NAMES, "version": 2, "scaler": scaler}
    rs = 42

    if args.target in ("home", "both"):
        y_h = np.array(y_home, dtype=np.int64)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_h, test_size=0.2, random_state=rs, stratify=y_h)
        clf_h = _make_classifier(args.model, args.epochs, rs)
        clf_h.fit(X_train, y_train)
        y_pred_h = clf_h.predict(X_val)
        _eval_and_print("Reward (home)", y_val, y_pred_h, 100 * y_h.mean())
        bundle["model_home"] = clf_h

    if args.target in ("away", "both"):
        y_a = np.array(y_away, dtype=np.int64)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_a, test_size=0.2, random_state=rs, stratify=y_a)
        clf_a = _make_classifier(args.model, args.epochs, rs)
        clf_a.fit(X_train, y_train)
        y_pred_a = clf_a.predict(X_val)
        _eval_and_print("Reward (away)", y_val, y_pred_a, 100 * y_a.mean())
        bundle["model_away"] = clf_a

    if args.train_loss:
        y_lh = np.array(y_loss_home, dtype=np.int64)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_lh, test_size=0.2, random_state=rs, stratify=y_lh)
        clf_lh = _make_classifier(args.model, args.epochs, rs)
        clf_lh.fit(X_train, y_train)
        y_pred = clf_lh.predict(X_val)
        _eval_and_print("Loss (home)", y_val, y_pred, 100 * y_lh.mean())
        bundle["model_loss_home"] = clf_lh
        y_la = np.array(y_loss_away, dtype=np.int64)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_la, test_size=0.2, random_state=rs, stratify=y_la)
        clf_la = _make_classifier(args.model, args.epochs, rs)
        clf_la.fit(X_train, y_train)
        y_pred = clf_la.predict(X_val)
        _eval_and_print("Loss (away)", y_val, y_pred, 100 * y_la.mean())
        bundle["model_loss_away"] = clf_la

    if args.train_buy_sell:
        for label, y_list, key in (
            ("Buy opportunity (home)", y_buy_home, "model_buy_home"),
            ("Sell opportunity (home)", y_sell_home, "model_sell_home"),
            ("Buy opportunity (away)", y_buy_away, "model_buy_away"),
            ("Sell opportunity (away)", y_sell_away, "model_sell_away"),
        ):
            y_arr = np.array(y_list, dtype=np.int64)
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_arr, test_size=0.2, random_state=rs, stratify=y_arr)
            clf = _make_classifier(args.model, args.epochs, rs)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            _eval_and_print(label, y_val, y_pred, 100 * y_arr.mean())
            bundle[key] = clf

    if args.train_price_range:
        from sklearn.linear_model import Ridge
        for label, y_list, key in (
            ("Min ask home", y_min_ask_home, "model_min_ask_home"),
            ("Max bid home", y_max_bid_home, "model_max_bid_home"),
            ("Min ask away", y_min_ask_away, "model_min_ask_away"),
            ("Max bid away", y_max_bid_away, "model_max_bid_away"),
        ):
            y_arr = np.array(y_list, dtype=np.float64)
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_arr, test_size=0.2, random_state=42)
            reg = Ridge(alpha=1.0, random_state=42)
            reg.fit(X_train, y_train)
            pred = reg.predict(X_val)
            pred = np.clip(pred, 0.0, 1.0)
            mae = float(np.abs(pred - y_val).mean())
            print(f"{label}: val MAE = {mae:.4f}")
            bundle[key] = reg

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"Saved reward model(s) to {model_path}")

    if args.tune_thresholds:
        records_dir = Path(args.tune_records_dir) if args.tune_records_dir else base / "data" / "game_records"
        if not records_dir.exists():
            print("Skipping threshold tune: --tune-records-dir not found:", records_dir)
        else:
            from backtest_in_game import tune_thresholds_from_records
            loaded = load_reward_model(model_path)
            if loaded is None:
                print("Could not reload model for threshold tune.")
            else:
                buy_vals = [0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
                sell_vals = [0.35, 0.38, 0.40, 0.42, 0.45]
                tuned = tune_thresholds_from_records(
                    records_dir, loaded, buy_vals, sell_vals, min_gap=0.05,
                    from_date=args.tune_from, to_date=args.tune_to, n_jobs=4,
                )
                if tuned:
                    best = tuned[0]
                    print("\n--- Recommended thresholds (from backtest) ---")
                    print(f"  --buy-threshold {best['buy_threshold']:.2f}  --sell-threshold {best['sell_threshold']:.2f}")
                    print(f"  (total_profit = {best.get('total_profit', 0):+.2f}, round_trips = {best.get('total_round_trips', 0)})\n")
    print("Use in in-game strategy to decide when to buy/sell. Note: 95%% accuracy is often unrealistic for this stochastic target; prefer balance of P/R/F1 for reward=1 and backtest PnL.")


if __name__ == "__main__":
    main()
