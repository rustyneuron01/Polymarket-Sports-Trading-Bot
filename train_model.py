"""
Train PRE-GAME win-probability model (P(home wins)) from game records.
This is NOT the main trading target: win/loss can be estimated with Elo. It is used for
pre-game fair price (predict_fair_price). The main trading target is token price change
and reward (profit if we buy at a point and sell later) — see build_in_game_dataset.py
and train_in_game_model.py for the in-game REWARD model.

Usage:
  python -m polymarket_nhl_bot.train_model
  python -m polymarket_nhl_bot.train_model --dir data/game_records --model-out data/outcome_model.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model import get_elo, load_elo_from_file


def load_game_records(records_dir: Path, skip_incomplete: bool = True) -> list[dict]:
    """Load all game record JSONs; optionally skip event_feed_incomplete."""
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
        records.append(rec)
    return records


# Lineup feature names when lineup_notes are provided (BOT_SPEC: injury tier + top scorer ability).
LINEUP_FEATURE_NAMES = [
    "home_key_out_count",
    "away_key_out_count",
    "top_player_injury_count_home",
    "top_player_injury_count_away",
    "role_player_injury_count_home",
    "role_player_injury_count_away",
    "top_player_impact_score_home",
    "top_player_impact_score_away",
    "top_scorer_goals_home",
    "top_scorer_goals_away",
    "top_scorer_points_home",
    "top_scorer_points_away",
]


def load_lineup_notes(path: Path, records: list[dict]) -> dict[tuple[str, str, str], dict]:
    """
    Load optional lineup/injury notes (JSONL). Each line: game id + players_out, top/role counts, impact, leaders.
    Returns dict keyed by (date, home_team_id, away_team_id) -> dict of feature values (ints/floats).
    """
    if not path.exists():
        return {}
    abbrev_to_id: dict[str, str] = {}
    for rec in records:
        if rec.get("home_abbrev"):
            abbrev_to_id[str(rec["home_abbrev"])] = str(rec.get("home_team_id", ""))
        if rec.get("away_abbrev"):
            abbrev_to_id[str(rec["away_abbrev"])] = str(rec.get("away_team_id", ""))
    lookup: dict[tuple[str, str, str], dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            date = row.get("date", "")
            if not date:
                continue
            if len(date) == 8 and date.isdigit():
                date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            home_id = str(row.get("home_team_id", ""))
            away_id = str(row.get("away_team_id", ""))
            if not home_id and row.get("home_abbrev"):
                home_id = abbrev_to_id.get(str(row["home_abbrev"]).strip(), "")
            if not away_id and row.get("away_abbrev"):
                away_id = abbrev_to_id.get(str(row["away_abbrev"]).strip(), "")
            if not home_id or not away_id:
                continue
            nh = len(row.get("players_out_home") or [])
            na = len(row.get("players_out_away") or [])
            entry = {
                "home_key_out_count": nh,
                "away_key_out_count": na,
                "top_player_injury_count_home": int(row.get("top_player_injury_count_home", 0)),
                "top_player_injury_count_away": int(row.get("top_player_injury_count_away", 0)),
                "role_player_injury_count_home": int(row.get("role_player_injury_count_home", 0)),
                "role_player_injury_count_away": int(row.get("role_player_injury_count_away", 0)),
                "top_player_impact_score_home": float(row.get("top_player_impact_score_home", 0)),
                "top_player_impact_score_away": float(row.get("top_player_impact_score_away", 0)),
                "top_scorer_goals_home": float(row.get("top_scorer_goals_home", 0)),
                "top_scorer_goals_away": float(row.get("top_scorer_goals_away", 0)),
                "top_scorer_points_home": float(row.get("top_scorer_points_home", 0)),
                "top_scorer_points_away": float(row.get("top_scorer_points_away", 0)),
            }
            lookup[(date, home_id, away_id)] = entry
    return lookup


def records_to_Xy(
    records: list[dict],
    lineup_lookup: dict[tuple[str, str, str], dict] | None = None,
) -> tuple[list[list[float]], list[int], list[str]]:
    """
    Build feature matrix X, target y, and feature_names from game records.
    Features: [elo_home, elo_away]; optionally + LINEUP_FEATURE_NAMES (injury tier + top scorer ability).
    Target: 1 if home_won else 0. (Pre-game win prob only. For trading reward target use train_in_game_model.)
    """
    load_elo_from_file()
    use_lineup = lineup_lookup is not None and len(lineup_lookup) > 0
    feature_names = ["elo_home", "elo_away"]
    if use_lineup:
        feature_names = ["elo_home", "elo_away"] + LINEUP_FEATURE_NAMES
    X = []
    y = []
    for rec in records:
        home_id = str(rec.get("home_team_id", ""))
        away_id = str(rec.get("away_team_id", ""))
        date = rec.get("date", "")
        elo_home = get_elo(home_id)
        elo_away = get_elo(away_id)
        row: list[float] = [elo_home, elo_away]
        if use_lineup and lineup_lookup:
            key = (date, home_id, away_id)
            entry = lineup_lookup.get(key, {})
            for fn in LINEUP_FEATURE_NAMES:
                row.append(float(entry.get(fn, 0)))
        X.append(row)
        y.append(1 if rec.get("home_won") else 0)
    return X, y, feature_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pre-game outcome model from game records.")
    parser.add_argument(
        "--dir",
        default=None,
        help="Game records directory (default: data/game_records)",
    )
    parser.add_argument(
        "--model-out",
        default=None,
        help="Output path for trained model (default: data/outcome_model.pkl)",
    )
    parser.add_argument(
        "--no-skip-incomplete",
        action="store_true",
        help="Include records with event_feed_incomplete (not recommended).",
    )
    parser.add_argument(
        "--lineup",
        default=None,
        help="Optional JSONL of injury/lineup notes (default: data/lineup_notes.jsonl). See DOCS.md.",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    records_dir = Path(args.dir) if args.dir else base / "data" / "game_records"
    model_path = Path(args.model_out) if args.model_out else base / "data" / "outcome_model.pkl"
    lineup_path = Path(args.lineup) if args.lineup else base / "data" / "lineup_notes.jsonl"

    if not records_dir.exists():
        print(f"Records directory not found: {records_dir}")
        print("Run build_game_records first to create game records.")
        raise SystemExit(1)

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        import joblib
    except ImportError as e:
        print("Training requires scikit-learn, numpy, and joblib. Install with:")
        print("  pip install scikit-learn numpy joblib")
        raise SystemExit(1) from e

    records = load_game_records(records_dir, skip_incomplete=not args.no_skip_incomplete)
    if len(records) < 50:
        print(f"Too few records ({len(records)}). Need at least 50. Build more game records first.")
        raise SystemExit(1)

    lineup_lookup = load_lineup_notes(lineup_path, records) if lineup_path else None
    if lineup_lookup and len(lineup_lookup) > 0:
        print(f"Using lineup/injury notes from {lineup_path} ({len(lineup_lookup)} games).")
    else:
        lineup_lookup = None
    X, y, feature_names = records_to_Xy(records, lineup_lookup=lineup_lookup)
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr)
    max_iter = 500
    print(f"Training logistic regression on {len(y_train)} samples (features: {', '.join(feature_names)}), max_iter={max_iter}...")
    clf = LogisticRegression(random_state=42, max_iter=max_iter, verbose=1)
    clf.fit(X_train, y_train)
    n_iter = getattr(clf, "n_iter_", None)
    if n_iter is not None:
        n_iter_str = str(n_iter) if hasattr(n_iter, "__len__") else str(int(n_iter))
        print(f"Solver finished (iterations: {n_iter_str}).")
    print("Training complete.")

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    print(f"Train accuracy: {train_acc:.3f}  Validation accuracy: {val_acc:.3f}  (n_train={len(y_train)}, n_val={len(y_val)})")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "feature_names": feature_names,
            "version": 1,
        },
        model_path,
    )
    print(f"Saved trained model to {model_path}")
    print("The bot will use this automatically in predict_fair_price() when you run main.py.")


if __name__ == "__main__":
    main()
