"""
Live Tennis API client: live in-match state for extending the bot to Tennis.

This is the Tennis analogue of ``espn_client`` (which supplies NHL state). The
README's "Extending to NBA, Tennis, Other Sports" section says: plug in a
sport-specific client (like ``espn_client`` for NHL) and a market-discovery
function, and keep the rest of the pipeline (dataset builder, training,
strategy, execution) as is. This module is that sport-specific client for
Tennis; it produces the same shape of live game-state dict the in-game pipeline
already consumes (``score_home``/``score_away``/``period`` ...), plus the
tennis-native fields the README calls out ("add sport-specific fields (e.g.
sets for Tennis)").

Scope and disclosure
--------------------
Vendor: the Live Tennis API (livetennisapi.com) is an independent live-tennis
DATA feed, and this client is contributed by that team. It is a data source
ONLY -- it reports match state (score, current server, a three-valued break
point, retirement/walkover/completed). It is never a venue or execution
adapter: market prices and order placement stay entirely with the existing
Polymarket client. Everything here uses the free keyed tier (no card; 30
requests/minute, 100/day): https://livetennisapi.com/subscribe/free

Nothing else in the bot imports this module, so it stays completely off unless
you wire it up and set ``LIVETENNISAPI_API_KEY``. See BOT_SPEC 1.1 for the NHL
counterpart these functions mirror.

Free-tier endpoints used (all keyed, ``{"data": [...]}`` envelope):
  GET /matches?status=live      -> live matches (this module's main input)
  GET /matches/{id}             -> one match by id
  GET /fixtures                 -> upcoming scheduled fixtures

Break-point rule (documented behaviour of the score object): a break point is
on when the RECEIVER is at AD, or the receiver is at 40 while the server is at
0/15/30. It is never on in a tiebreak, and is reported False whenever the
server or the points are null (completed matches carry null points).
"""
from __future__ import annotations

from typing import Any, Optional

import requests

from config import (
    LIVETENNISAPI_API_KEY,
    LIVETENNISAPI_BASE_URL,
    LIVETENNISAPI_TIMEOUT,
)

_FREE_KEY_URL = "https://livetennisapi.com/subscribe/free"


def _get(path: str, params: Optional[dict] = None) -> Any:
    """GET a free-tier endpoint. Returns parsed JSON, or None on any failure."""
    if not LIVETENNISAPI_API_KEY:
        print(
            "Live Tennis API: no key set (LIVETENNISAPI_API_KEY); Tennis client "
            f"disabled. Free key (no card): {_FREE_KEY_URL}"
        )
        return None
    url = LIVETENNISAPI_BASE_URL.rstrip("/") + path
    try:
        r = requests.get(
            url,
            params=params or {},
            headers={"X-API-Key": LIVETENNISAPI_API_KEY},
            timeout=LIVETENNISAPI_TIMEOUT,
        )
        if r.status_code == 429:
            print(
                "Live Tennis API rate limit hit (free tier: 30 req/min, "
                "100 req/day). Slow the polling cadence."
            )
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Live Tennis API error {path}: {e}")
        return None


def _int(v: Any, default: int) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def get_live_matches(tour: Optional[str] = None) -> list[dict]:
    """
    Live (in-progress) matches. Optional ``tour`` filter (e.g. 'atp', 'wta',
    'challenger'). Returns a list of raw match dicts, or [] on failure.
    """
    params: dict[str, Any] = {"status": "live", "limit": 50}
    if tour:
        params["tour"] = tour
    data = _get("/matches", params)
    if isinstance(data, dict):
        return data.get("data", []) or []
    return []


def get_match(match_id: int) -> Optional[dict]:
    """Fetch one match by id. Returns the raw match dict, or None."""
    data = _get(f"/matches/{match_id}")
    if isinstance(data, dict):
        inner = data.get("data")
        if isinstance(inner, dict):
            return inner
        # Some responses return the object directly under the envelope.
        return data if "id" in data else None
    return None


def get_fixtures(limit: int = 50) -> list[dict]:
    """Upcoming scheduled fixtures (earliest first). Returns [] on failure."""
    data = _get("/fixtures", {"limit": limit})
    if isinstance(data, dict):
        return data.get("data", []) or []
    return []


def derive_break_point(score: Optional[dict]) -> bool:
    """
    True when the current point is a break point. Conservative on nulls.

    Break point = receiver at AD, or receiver at 40 while server is at 0/15/30.
    Never true in a tiebreak; false whenever server or points are null.
    """
    if not score:
        return False
    if score.get("is_tiebreak"):
        return False
    server = score.get("server")
    if server not in (1, 2):
        return False
    points = score.get("points") or []
    if len(points) != 2 or points[0] is None or points[1] is None:
        return False
    receiver_points = str(points[1] if server == 1 else points[0])
    server_points = str(points[0] if server == 1 else points[1])
    if receiver_points == "AD":
        return True
    return receiver_points == "40" and server_points in ("0", "15", "30")


def score_line(score: Optional[dict]) -> str:
    """Render '6-4 3-2 (40-15)' from a score object. '' if unavailable."""
    if not score:
        return ""
    games = score.get("games") or []
    parts: list[str] = []
    if len(games) == 2 and games[0] and len(games[0]) == len(games[1]):
        parts = [f"{a}-{b}" for a, b in zip(games[0], games[1])]
    points = score.get("points") or []
    if len(points) == 2 and points[0] is not None and points[1] is not None:
        parts.append(f"({points[0]}-{points[1]})")
    return " ".join(parts)


def match_to_game_info(match: dict) -> Optional[dict[str, Any]]:
    """
    Extract pre-match info from one match, for market discovery / features.
    Mirrors ``espn_client.event_to_game_info``: p1 -> home, p2 -> away (tennis
    has no home/away; this is a fixed convention so the existing home/away
    columns line up). Returns None if the match lacks two players.
    """
    players = match.get("players") or {}
    p1 = players.get("p1") or {}
    p2 = players.get("p2") or {}
    if not p1.get("name") or not p2.get("name"):
        return None
    return {
        "match_id": match.get("id"),
        "event_id": str(match.get("id", "")),
        "tour": match.get("tour"),
        "tournament": match.get("tournament"),
        "surface": match.get("surface"),
        "indoor": bool(match.get("indoor")),
        "format": match.get("format"),           # 'BO3' / 'BO5'
        "round": match.get("round"),
        "round_code": match.get("round_code"),
        "status": match.get("status"),
        "is_doubles": bool(match.get("is_doubles")),
        "draw": match.get("draw"),
        "scheduled_time": match.get("scheduled_time"),
        "p1_id": p1.get("id"),
        "p1_name": p1.get("name"),
        "p1_ranking": p1.get("ranking"),
        "p2_id": p2.get("id"),
        "p2_name": p2.get("name"),
        "p2_ranking": p2.get("ranking"),
        # home/away aliases for pipeline compatibility (p1=home, p2=away):
        "home_team_id": str(p1.get("id") or ""),
        "away_team_id": str(p2.get("id") or ""),
        "home_team_name": p1.get("name"),
        "away_team_name": p2.get("name"),
    }


def get_live_match_state(match: dict) -> Optional[dict[str, Any]]:
    """
    Live in-match state for an in-progress match, shaped to plug into the
    in-game pipeline the way ``espn_client.get_live_game_state_from_event``
    does for NHL. Returns None if the match is not live.

    Convention: p1 -> home, p2 -> away (tennis has no home/away; fixed so the
    existing ``score_home``/``score_away``/``period`` feature columns line up).

    Returns a dict:
      score_home, score_away  -- sets won by p1 / p2 (the coarse "score")
      period                  -- current set number (1-based), like an NHL period
      time_remaining_sec      -- None: tennis is untimed, there is no game clock
      game_elapsed_sec        -- None: same reason (do not fabricate a clock)
      # tennis-native fields ("add sport-specific fields (e.g. sets for Tennis)"):
      sets_home, sets_away
      games_home, games_away  -- games in the CURRENT set
      points_home, points_away-- current-game points as strings ('0/15/30/40/AD')
      server                  -- 1 (home/p1), 2 (away/p2), or None
      break_point             -- bool, receiver one point from breaking serve
      is_tiebreak             -- bool
      score_line              -- '6-4 3-2 (40-15)'
      event_status            -- raw in-match flag ('Retired'/'Walkover'/None)
      live_timestamp          -- score object's own timestamp (str) or None
    """
    if (match.get("status") or "").lower() != "live":
        return None
    score = match.get("score") or {}
    sets = score.get("sets") or []
    sets_home = _int(sets[0], 0) if len(sets) >= 1 else 0
    sets_away = _int(sets[1], 0) if len(sets) >= 2 else 0

    games = score.get("games") or []
    home_games_by_set = games[0] if len(games) >= 1 and games[0] else []
    away_games_by_set = games[1] if len(games) >= 2 and games[1] else []
    games_home = _int(home_games_by_set[-1], 0) if home_games_by_set else 0
    games_away = _int(away_games_by_set[-1], 0) if away_games_by_set else 0
    # Current set number = number of sets that have any games recorded, min 1.
    period = max(1, len(home_games_by_set) or (sets_home + sets_away) or 1)

    points = score.get("points") or [None, None]
    points_home = points[0] if len(points) >= 1 else None
    points_away = points[1] if len(points) >= 2 else None

    server = score.get("server")
    if server not in (1, 2):
        server = None

    return {
        # pipeline-shared fields (match espn_client.get_live_game_state_from_event):
        "score_home": sets_home,
        "score_away": sets_away,
        "period": period,
        "time_remaining_sec": None,  # tennis has no game clock
        "game_elapsed_sec": None,
        # tennis-native fields:
        "sets_home": sets_home,
        "sets_away": sets_away,
        "games_home": games_home,
        "games_away": games_away,
        "points_home": points_home,
        "points_away": points_away,
        "server": server,
        "break_point": derive_break_point(score),
        "is_tiebreak": bool(score.get("is_tiebreak")),
        "score_line": score_line(score),
        "event_status": match.get("event_status"),
        "live_timestamp": score.get("timestamp"),
    }


def get_match_result(match: dict) -> Optional[dict[str, Any]]:
    """
    Final result of a completed match, for training/backfill. Mirrors
    ``espn_client.get_completed_games_*``. Returns None until the match is
    completed with a clear winner.

    Returns a dict:
      match_id, home_won (p1 won), winner (1/2), sets_home, sets_away,
      retired (bool), walkover (bool), event_status (raw flag).
    """
    if (match.get("status") or "").lower() != "completed":
        return None
    winner = match.get("winner")
    if winner not in (1, 2):
        return None
    score = match.get("score") or {}
    sets = score.get("sets") or []
    sets_home = _int(sets[0], 0) if len(sets) >= 1 else 0
    sets_away = _int(sets[1], 0) if len(sets) >= 2 else 0
    flag_l = (match.get("event_status") or "").lower()
    withdrew = match.get("withdrew")
    # The event_status flag is authoritative when present. Completed matches
    # clear their games array, so only fall back to a withdrawal heuristic
    # (any completed set => retirement, none => walkover) when there is no flag.
    if flag_l:
        retired = "retire" in flag_l or "ret." in flag_l
        walkover = "walk" in flag_l or "w/o" in flag_l
    else:
        any_progress = (sets_home + sets_away) > 0 or bool(score.get("games"))
        retired = withdrew in (1, 2) and any_progress
        walkover = withdrew in (1, 2) and not any_progress
    return {
        "match_id": match.get("id"),
        "home_won": winner == 1,
        "winner": winner,
        "sets_home": sets_home,
        "sets_away": sets_away,
        "retired": bool(retired),
        "walkover": bool(walkover),
        "event_status": match.get("event_status"),
    }
