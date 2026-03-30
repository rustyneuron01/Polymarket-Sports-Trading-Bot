"""
Build feature dict for a game (pre-game from ESPN). See BOT_SPEC 1.2.
In-game features (period, score, time_remaining) added when we have live data.
"""
from __future__ import annotations

from typing import Any

# Elo is loaded per-game from a store (or default); we don't persist here


def game_info_to_features(game: dict[str, Any], elo_home: float = 1500, elo_away: float = 1500) -> dict[str, Any]:
    """
    One row of features for pre-game model.
    game = event_to_game_info(ev) from espn_client.
    """
    features = {
        "home_team_id": game.get("home_team_id"),
        "away_team_id": game.get("away_team_id"),
        "event_id": game.get("event_id"),
        "is_home": 1,
        "wins_home": game.get("home_wins", 0),
        "losses_home": game.get("home_losses", 0),
        "ot_home": game.get("home_ot", 0),
        "wins_away": game.get("away_wins", 0),
        "losses_away": game.get("away_losses", 0),
        "ot_away": game.get("away_ot", 0),
        "win_pct_home": game.get("home_win_pct", 0.5),
        "win_pct_away": game.get("away_win_pct", 0.5),
        "home_record_wins": game.get("home_home_wins", 0),
        "home_record_losses": game.get("home_home_losses", 0),
        "road_record_wins": game.get("away_road_wins", 0),
        "road_record_losses": game.get("away_road_losses", 0),
        "ytd_goals_home": game.get("home_ytd_goals", 0),
        "ytd_goals_away": game.get("away_ytd_goals", 0),
        "ytd_points_home": game.get("home_ytd_points", 0),
        "ytd_points_away": game.get("away_ytd_points", 0),
        "goalie_confirmed_home": game.get("home_goalie_confirmed", 0),
        "goalie_confirmed_away": game.get("away_goalie_confirmed", 0),
        "elo_home": elo_home,
        "elo_away": elo_away,
    }
    # Derived
    gpg_home = features["ytd_goals_home"] / max(1, features["wins_home"] + features["losses_home"] + features["ot_home"])
    gpg_away = features["ytd_goals_away"] / max(1, features["wins_away"] + features["losses_away"] + features["ot_away"])
    features["ytd_goals_per_game_home"] = gpg_home
    features["ytd_goals_per_game_away"] = gpg_away
    return features
