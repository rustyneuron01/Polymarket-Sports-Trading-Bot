"""
Elo ratings for NHL teams. We use Harvitronix Elo data, so home-ice matches their game page.
- K=6: FiveThirtyEight NHL; Harvitronix uses K=8 for updates.
- Home +42: Harvitronix value so our prob_home/prob_away match their game-level projections.
"""
from __future__ import annotations

DEFAULT_ELO = 1500
K = 6
HOME_ADVANTAGE = 42  # Harvitronix value; matches https://elo.harvitronix.com/nhl/YYYY-YYYY/games


def elo_win_prob(elo_home: float, elo_away: float, home_adj: float = HOME_ADVANTAGE) -> float:
    """P(home wins) = 1 / (1 + 10^((elo_away - elo_home - home_adj) / 400))."""
    diff = elo_home + home_adj - elo_away
    return 1.0 / (1.0 + 10 ** (-diff / 400))


def update_elo(elo_home: float, elo_away: float, home_won: bool, k: float = K) -> tuple[float, float]:
    """Update Elo after a game. home_won True if home team won (reg or OT)."""
    p_home = elo_win_prob(elo_home, elo_away, home_adj=0)
    actual = 1.0 if home_won else 0.0
    change = k * (actual - p_home)
    return elo_home + change, elo_away - change


def prob_home_away_from_elo(elo_home: float, elo_away: float) -> tuple[float, float]:
    """Return (prob_home, prob_away) for fair token price."""
    p_home = elo_win_prob(elo_home, elo_away)
    return p_home, 1.0 - p_home
