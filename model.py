"""
Predict fair token price (prob_home, prob_away). BOT_SPEC: Elo first, then calibrate.
Elo ratings are loaded from data/elo_ratings.json if present; run harvitronix_elo.py to populate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elo import elo_win_prob

# Simple Elo store: team_id -> elo. Loaded from file if present.
_elo_store: dict[str, float] = {}
DEFAULT_ELO = 1500
ELO_FILE = Path(__file__).resolve().parent / "data" / "elo_ratings.json"
OUTCOME_MODEL_FILE = Path(__file__).resolve().parent / "data" / "outcome_model.pkl"
_trained_model: Any = None  # Loaded on first predict if file exists


def load_elo_from_file() -> bool:
    """Load _elo_store from ELO_FILE. Returns True if file existed and was loaded."""
    if not ELO_FILE.exists():
        return False
    try:
        with open(ELO_FILE) as f:
            data = json.load(f)
        _elo_store.clear()
        for k, v in (data or {}).items():
            _elo_store[str(k)] = float(v)
        return True
    except Exception:
        return False


def save_elo_to_file() -> None:
    """Write _elo_store to ELO_FILE."""
    ELO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ELO_FILE, "w") as f:
        json.dump(_elo_store, f, indent=0)


def get_elo(team_id: str) -> float:
    return _elo_store.get(str(team_id), DEFAULT_ELO)


def set_elo(team_id: str, elo: float) -> None:
    _elo_store[str(team_id)] = elo


# Load saved ratings at import so predictions use them
load_elo_from_file()


def _load_trained_model_if_needed() -> Any:
    """Load outcome_model.pkl once if present (lazy). Returns full bundle {model, feature_names} or None."""
    global _trained_model
    if _trained_model is not None:
        return _trained_model
    if not OUTCOME_MODEL_FILE.exists():
        return None
    try:
        import joblib
        data = joblib.load(OUTCOME_MODEL_FILE)
        _trained_model = data
        return _trained_model
    except Exception:
        return None


def _game_to_feature_row(game: dict[str, Any], feature_names: list[str]) -> list[float]:
    """Build one row of features for prediction from game dict and feature names (e.g. from pkl)."""
    home_id = str(game.get("home_team_id", ""))
    away_id = str(game.get("away_team_id", ""))
    row = []
    for name in feature_names:
        if name == "elo_home":
            row.append(get_elo(home_id))
        elif name == "elo_away":
            row.append(get_elo(away_id))
        else:
            row.append(float(game.get(name, 0)))
    return row


# Lineup feature keys used at prediction when model was trained with lineup (see train_model.LINEUP_FEATURE_NAMES).
LINEUP_FEATURE_KEYS = (
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
)


def load_lineup_lookup(lineup_path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """
    Load data/lineup_notes.jsonl into (date_iso, home_id, away_id) -> {feature keys -> values}.
    Date in file normalized to YYYY-MM-DD. Use to enrich game dicts before predict_fair_price.
    """
    if not lineup_path.exists():
        return {}
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    with open(lineup_path) as f:
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
            elif len(date) > 10:
                date = date[:10]
            home_id = str(row.get("home_team_id", ""))
            away_id = str(row.get("away_team_id", ""))
            if not home_id or not away_id:
                continue
            entry = {k: row.get(k, 0) for k in LINEUP_FEATURE_KEYS}
            lookup[(date, home_id, away_id)] = entry
    return lookup


def enrich_game_with_lineup(game: dict[str, Any], lineup_lookup: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    """Merge lineup features for this game into game dict (date, home_team_id, away_team_id). In-place."""
    date = (game.get("date") or "")[:10] if game.get("date") else ""
    home_id = str(game.get("home_team_id", ""))
    away_id = str(game.get("away_team_id", ""))
    if not date or not home_id or not away_id:
        return
    entry = lineup_lookup.get((date, home_id, away_id))
    if entry:
        game.update(entry)


def predict_fair_price(game: dict[str, Any]) -> tuple[float, float]:
    """
    Return (prob_home, prob_away) for pre-game fair token price.
    Uses trained outcome model from data/outcome_model.pkl if present (from train_model.py),
    else raw Elo win probability.
    If the model was trained with lineup/injury/ability features, pass the lineup feature keys
    on game (see LINEUP_FEATURE_KEYS; default 0 if missing). Use load_lineup_lookup + enrich_game_with_lineup
    before calling when lineup_notes.jsonl is available.
    """
    home_id = str(game.get("home_team_id", ""))
    away_id = str(game.get("away_team_id", ""))
    elo_home = get_elo(home_id)
    elo_away = get_elo(away_id)
    bundle = _load_trained_model_if_needed()
    if bundle is not None:
        trained = bundle.get("model")
        feature_names = bundle.get("feature_names") or ["elo_home", "elo_away"]
        if trained is not None:
            try:
                import numpy as np
                row = _game_to_feature_row(game, feature_names)
                X = np.array([row], dtype=np.float64)
                p_home = float(trained.predict_proba(X)[0, 1])
                p_home = max(0.0, min(1.0, p_home))
                return p_home, 1.0 - p_home
            except Exception:
                pass
    p_home = elo_win_prob(elo_home, elo_away)
    return p_home, 1.0 - p_home


def predict_all_games(games: list[dict]) -> list[dict]:
    """For each game return { event_id, prob_home, prob_away, game }."""
    out = []
    for g in games:
        ph, pa = predict_fair_price(g)
        out.append({"event_id": g.get("event_id"), "prob_home": ph, "prob_away": pa, "game": g})
    return out
