"""
ESPN NHL API client: scoreboard, team schedule. See BOT_SPEC 1.1.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

import requests

from config import ESPN_SCOREBOARD_URL, ESPN_SUMMARY_URL, ESPN_TEAM_SCHEDULE_URL


def _parse_record(summary: str) -> tuple[int, int, int]:
    """Parse '28-28-2' -> (wins, losses, ot)."""
    if not summary or not re.match(r"\d+-\d+-\d+", summary):
        return 0, 0, 0
    parts = summary.split("-")
    return int(parts[0]), int(parts[1]), int(parts[2])


def get_scoreboard(dates: Optional[str] = None) -> list[dict]:
    """Get scoreboard; optional ?dates=YYYYMMDD or YYYYMMDD,YYYYMMDD."""
    url = ESPN_SCOREBOARD_URL
    params = {}
    if dates:
        params["dates"] = dates
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("events", [])
    except Exception as e:
        print(f"ESPN scoreboard error: {e}")
        return []


def get_team_schedule(team_id: str) -> list[dict]:
    """Get team schedule (past + future)."""
    url = ESPN_TEAM_SCHEDULE_URL.format(team_id=team_id)
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("events", [])
    except Exception as e:
        print(f"ESPN schedule error team {team_id}: {e}")
        return []


def event_to_game_info(event: dict) -> Optional[dict[str, Any]]:
    """
    Extract game info and competitor records/stats from one event.
    Returns dict with home_team_id, away_team_id, home_team_name, away_team_name,
    event_id, date, and for each side: records (YTD, home, road), statistics (ytdGoals etc), probable_goalie.
    """
    comps = event.get("competitions") or []
    if not comps:
        return None
    comp = comps[0]
    competitors = comp.get("competitors") or []
    if len(competitors) < 2:
        return None
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    team = lambda c: c.get("team", {})
    out = {
        "event_id": event.get("id"),
        "date": comp.get("date") or event.get("date"),
        "home_team_id": home.get("id") or team(home).get("id"),
        "away_team_id": away.get("id") or team(away).get("id"),
        "home_team_name": team(home).get("displayName") or team(home).get("name", ""),
        "away_team_name": team(away).get("displayName") or team(away).get("name", ""),
    }
    for label, c in [("home", home), ("away", away)]:
        recs = {r.get("type", ""): r.get("summary", "") for r in (c.get("records") or [])}
        w, l, ot = _parse_record(recs.get("ytd", recs.get("total", "")) or "")
        out[f"{label}_wins"] = w
        out[f"{label}_losses"] = l
        out[f"{label}_ot"] = ot
        out[f"{label}_win_pct"] = w / (w + l) if (w + l) else 0.5
        home_rec = recs.get("home", "")
        road_rec = recs.get("road", "")
        out[f"{label}_home_wins"], out[f"{label}_home_losses"], _ = _parse_record(home_rec)
        out[f"{label}_road_wins"], out[f"{label}_road_losses"], _ = _parse_record(road_rec)
        stats = {s.get("name"): s.get("displayValue") or s.get("value") for s in (c.get("statistics") or [])}
        out[f"{label}_ytd_goals"] = _float(stats.get("ytdGoals"), 0)
        out[f"{label}_ytd_assists"] = _float(stats.get("assists"), 0)
        out[f"{label}_ytd_points"] = _float(stats.get("points"), 0)
        probables = c.get("probables") or []
        prob_goalie = probables[0] if probables else None
        out[f"{label}_goalie_confirmed"] = 1 if (prob_goalie and (prob_goalie.get("status") or {}).get("name") == "Confirmed") else 0
    return out


def _float(v: Any, default: float) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def games_today() -> list[dict]:
    """Return list of game info dicts for today (UTC date)."""
    today = datetime.utcnow().strftime("%Y%m%d")
    events = get_scoreboard(today)
    games = []
    for ev in events:
        info = event_to_game_info(ev)
        if info:
            games.append(info)
    return games


def get_completed_games_for_date(date_str: str) -> list[tuple[str, str, bool]]:
    """
    Get completed games for a given date (YYYYMMDD). Returns list of (home_team_id, away_team_id, home_won).
    home_won is True if home team won (regulation or OT/SO). Only includes events with status indicating completion.
    """
    events = get_scoreboard(date_str)
    out = []
    for ev in events:
        st = ev.get("status") or {}
        completed = (st.get("type") or {}).get("completed") if isinstance(st.get("type"), dict) else False
        if not completed and st.get("state") != "post":
            name = (st.get("type") or {}).get("name", "") if isinstance(st.get("type"), dict) else ""
            if "FINAL" not in name.upper() and "POST" not in name.upper():
                continue
        comps = ev.get("competitions") or []
        if not comps:
            continue
        competitors = comps[0].get("competitors") or []
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_id = str(home.get("id") or (home.get("team") or {}).get("id", ""))
        away_id = str(away.get("id") or (away.get("team") or {}).get("id", ""))
        if not home_id or not away_id:
            continue
        home_winner = home.get("winner") is True or str(home.get("winner", "")).lower() == "true"
        away_winner = away.get("winner") is True or str(away.get("winner", "")).lower() == "true"
        if home_winner == away_winner:
            continue  # skip if no clear winner (tie or data missing)
        out.append((home_id, away_id, home_winner))
    return out


def get_completed_games_with_scores_for_date(date_str: str) -> list[dict]:
    """
    Get completed games for a date (YYYYMMDD) with scores. For training/backfill.
    Returns list of dicts: date, home_team_id, away_team_id, home_won, home_score, away_score.
    """
    events = get_scoreboard(date_str)
    out = []
    for ev in events:
        st = ev.get("status") or {}
        completed = (st.get("type") or {}).get("completed") if isinstance(st.get("type"), dict) else False
        if not completed and st.get("state") != "post":
            name = (st.get("type") or {}).get("name", "") if isinstance(st.get("type"), dict) else ""
            if "FINAL" not in name.upper() and "POST" not in name.upper():
                continue
        comps = ev.get("competitions") or []
        if not comps:
            continue
        competitors = comps[0].get("competitors") or []
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_id = str(home.get("id") or (home.get("team") or {}).get("id", ""))
        away_id = str(away.get("id") or (away.get("team") or {}).get("id", ""))
        if not home_id or not away_id:
            continue
        home_winner = home.get("winner") is True or str(home.get("winner", "")).lower() == "true"
        away_winner = away.get("winner") is True or str(away.get("winner", "")).lower() == "true"
        if home_winner == away_winner:
            continue
        try:
            home_score = int(home.get("score", 0))
        except (TypeError, ValueError):
            home_score = 0
        try:
            away_score = int(away.get("score", 0))
        except (TypeError, ValueError):
            away_score = 0
        out.append({
            "date": date_str,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_won": home_winner,
            "home_score": home_score,
            "away_score": away_score,
        })
    return out


def get_game_summary(event_id: str) -> Optional[dict]:
    """Fetch full game summary (boxscore, plays, etc.) for an event. Returns None on error."""
    try:
        r = requests.get(f"{ESPN_SUMMARY_URL}?event={event_id}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"ESPN summary error event {event_id}: {e}")
        return None


def _play_period_clock_scores(p: dict) -> tuple[int, str, int, int] | None:
    """Extract period, clock, home_score, away_score from a play; None if parse fails."""
    try:
        raw_period = p.get("period", 0)
        period = int(raw_period.get("number", raw_period) if isinstance(raw_period, dict) else raw_period)
        raw_clock = p.get("clock")
        if isinstance(raw_clock, dict):
            clock = (raw_clock.get("displayValue") or "").strip()
        else:
            clock = (raw_clock or "").strip() if raw_clock else ""
        home_score = int(p.get("homeScore", 0))
        away_score = int(p.get("awayScore", 0))
        return (period, clock, home_score, away_score)
    except (TypeError, ValueError):
        return None


def get_game_events(event_id: str, include_penalties_and_injuries: bool = True) -> list[dict]:
    """
    Get in-game events: goals, and optionally penalties and injury-related stoppages.
    For training: align with token price series to see how price moves on goals, penalties, injuries.
    Returns list of { period, clock, home_score, away_score, event_type, text }.
    event_type is one of: "goal", "penalty", "injury" (stoppage with "Injury" in text), "stoppage".
    """
    summary = get_game_summary(event_id)
    if not summary:
        return []
    plays = summary.get("plays") or []
    out = []
    for p in plays:
        parsed = _play_period_clock_scores(p)
        if not parsed:
            continue
        period, clock, home_score, away_score = parsed
        text = (p.get("text") or "").strip()[:200]
        ev_type = None
        if p.get("scoringPlay"):
            ev_type = "goal"
        elif include_penalties_and_injuries:
            raw_type = p.get("type")
            abbr = (raw_type.get("abbreviation") or "").lower() if isinstance(raw_type, dict) else ""
            if abbr == "penalty":
                ev_type = "penalty"
            elif abbr == "stoppage" and text and "injury" in text.lower():
                ev_type = "injury"
        if ev_type is None:
            continue
        out.append({
            "period": period,
            "clock": clock,
            "home_score": home_score,
            "away_score": away_score,
            "event_type": ev_type,
            "scoring_play": ev_type == "goal",
            "text": text,
        })
    # ESPN may return plays out of order; sort by (period, clock) so last event is chronologically last.
    # Note: for period 5 (shootout), home_score/away_score are shootout-round totals, not game total.
    out.sort(key=lambda e: (e["period"], e.get("clock", "")))
    return out


def get_completed_games_rich_for_date(date_str: str) -> list[dict]:
    """
    Get completed games for a date with event_id, scores, and linescores (goals per period).
    For building combined game records with price series and events.
    Returns list of dicts: date, event_id, home_team_id, away_team_id, home_won, home_score, away_score,
    home_linescores, away_linescores. home_abbrev/away_abbrev not included (caller can map via ESPN teams).
    """
    events = get_scoreboard(date_str)
    out = []
    for ev in events:
        st = ev.get("status") or {}
        completed = (st.get("type") or {}).get("completed") if isinstance(st.get("type"), dict) else False
        if not completed and st.get("state") != "post":
            name = (st.get("type") or {}).get("name", "") if isinstance(st.get("type"), dict) else ""
            if "FINAL" not in name.upper() and "POST" not in name.upper():
                continue
        comps = ev.get("competitions") or []
        if not comps:
            continue
        competitors = comps[0].get("competitors") or []
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        home_id = str(home.get("id") or (home.get("team") or {}).get("id", ""))
        away_id = str(away.get("id") or (away.get("team") or {}).get("id", ""))
        if not home_id or not away_id:
            continue
        home_winner = home.get("winner") is True or str(home.get("winner", "")).lower() == "true"
        away_winner = away.get("winner") is True or str(away.get("winner", "")).lower() == "true"
        if home_winner == away_winner:
            continue
        try:
            home_score = int(home.get("score", 0))
        except (TypeError, ValueError):
            home_score = 0
        try:
            away_score = int(away.get("score", 0))
        except (TypeError, ValueError):
            away_score = 0
        home_lines, away_lines = get_linescores_from_scoreboard_event(ev)
        event_id = str(ev.get("id", ""))
        out.append({
            "date": date_str,
            "event_id": event_id,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_won": home_winner,
            "home_score": home_score,
            "away_score": away_score,
            "home_linescores": home_lines,
            "away_linescores": away_lines,
        })
    return out


def get_live_game_state_from_event(ev: dict) -> dict | None:
    """
    From a scoreboard event, extract current score and period for in-progress games.
    Returns dict with score_home, score_away, period, time_remaining_sec, game_elapsed_sec (or None if not live).
    time_remaining_sec is approximate (20*60 for period 3 = 20 min left in period).
    """
    st = ev.get("status") or {}
    state_type = (st.get("type") or {}).get("state") if isinstance(st.get("type"), dict) else None
    if state_type != "in":
        return None
    comps = ev.get("competitions") or []
    if not comps:
        return None
    competitors = comps[0].get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    try:
        score_home = int(home.get("score", 0))
        score_away = int(away.get("score", 0))
    except (TypeError, ValueError):
        return None
    period = 1
    display_clock = (st.get("displayClock") or "").strip()
    for ls in home.get("linescores") or []:
        period = max(period, int(ls.get("period", 1)))
    # Approximate time remaining: 20 min per period if we don't parse clock
    time_remaining_sec = 20 * 60
    if display_clock:
        parts = display_clock.replace(":", " ").split()
        if len(parts) >= 2:
            try:
                time_remaining_sec = int(parts[0]) * 60 + int(parts[1])
            except (ValueError, IndexError):
                pass
    regulation_sec_per_period = 20 * 60
    game_elapsed_sec = (period - 1) * regulation_sec_per_period + (regulation_sec_per_period - time_remaining_sec)
    return {
        "score_home": score_home,
        "score_away": score_away,
        "period": period,
        "time_remaining_sec": float(time_remaining_sec),
        "game_elapsed_sec": max(0, game_elapsed_sec),
        "game_second_proxy": max(0, game_elapsed_sec),
    }


def get_linescores_from_scoreboard_event(ev: dict) -> tuple[list[dict], list[dict]]:
    """
    Extract linescores (goals per period) for home and away from a scoreboard event.
    Returns (home_linescores, away_linescores), each list of { period, value }.
    """
    comps = ev.get("competitions") or []
    if not comps:
        return [], []
    competitors = comps[0].get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    home_lines = []
    away_lines = []
    if home:
        for ls in home.get("linescores") or []:
            try:
                home_lines.append({"period": int(ls.get("period", 0)), "value": float(ls.get("value", 0))})
            except (TypeError, ValueError):
                pass
    if away:
        for ls in away.get("linescores") or []:
            try:
                away_lines.append({"period": int(ls.get("period", 0)), "value": float(ls.get("value", 0))})
            except (TypeError, ValueError):
                pass
    return home_lines, away_lines
