"""
Generate data/lineup_notes.jsonl from game records.

Strategy
--------
1. Goal: one JSONL line per game with who's out (injured, questionable, etc.) and any
   other lineup-related info we can fetch, so the trainer can use it as features.

2. Modes:
   - No fetch: scan game_records (optionally --from/--to), write one line per game with
     empty lists. You edit by hand or merge from another source.
   - --fetch-espn: fill from ESPN APIs. Use with a date range (--from + --to) or a
     single --date. We load games in that range from game_records; for any date in the
     range that has no records we can pull games from the ESPN scoreboard. Then we
     fetch each team's roster once (cached) and derive injured + questionable lists.

3. What we fetch from ESPN:
   - Roster (per team): injured = players with non-empty injuries[]; questionable =
     players with status Day-To-Day / Questionable / Doubtful (or similar). Both are
     written to the JSONL (players_out_* = injured, players_questionable_* = questionable).
   - Probable goalie: available per game from the scoreboard/summary; can be added as
     probable_goalie_home/away in a later step if we want that as a feature.

4. Caveat: ESPN roster reflects *current* state. For historical dates, the lists are
   "injuries as of when you ran the script", not "who was out on that game date". For
   live/today it's correct; for backfill over many dates you're still using one snapshot
   per team for the whole run.

Usage:
  python generate_lineup_notes.py
  python generate_lineup_notes.py --from 2026-01-01 --to 2026-02-26
  python generate_lineup_notes.py --from 2026-02-01 --to 2026-02-28 --fetch-espn --out data/lineup_notes.jsonl
  python generate_lineup_notes.py --date 2026-02-27 --fetch-espn
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

ESPN_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_id}/roster"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"

# Status names we treat as "questionable" (DTD, doubtful, etc.)
QUESTIONABLE_STATUS_KEYWORDS = ("day", "doubtful", "questionable", "illness", "personal")


def _normalize_name_for_match(name: str) -> str:
    """Normalize player name for matching (roster displayName vs scoreboard leaders)."""
    if not name:
        return ""
    # Lowercase, collapse spaces, remove accents not critical for NHL
    s = " ".join((name or "").split()).lower()
    return s


def _name_matches(a: str, b: str) -> bool:
    """True if two display names likely refer to the same player (last name + first initial)."""
    na, nb = _normalize_name_for_match(a), _normalize_name_for_match(b)
    if na == nb:
        return True
    # "Connor McDavid" vs "C. McDavid" -> same last name; one has initial
    parts_a = na.split()
    parts_b = nb.split()
    if not parts_a or not parts_b:
        return False
    if parts_a[-1] == parts_b[-1]:  # same last name
        if len(parts_a) == 1 or len(parts_b) == 1:
            return True
        # first initial
        fa = parts_a[0][0] if parts_a[0] else ""
        fb = parts_b[0][0] if parts_b[0] else ""
        return fa == fb
    return False


def _in_leader_list(display_name: str, leader_names: set[str]) -> bool:
    """Check if display_name matches any name in leader set (roster vs scoreboard)."""
    for ln in leader_names:
        if _name_matches(display_name, ln):
            return True
    return False


def fetch_espn_roster_data(team_id: str) -> dict[str, list[str]]:
    """
    Fetch ESPN roster for team_id. Returns:
      - injured: displayName for players with non-empty injuries[].
      - questionable: displayName for players with status suggesting DTD/questionable/doubtful.
    """
    try:
        import requests
        r = requests.get(ESPN_ROSTER_URL.format(team_id=team_id), timeout=15)
        r.raise_for_status()
        data = r.json()
        injured = []
        questionable = []
        for pos_group in data.get("athletes") or []:
            for item in pos_group.get("items") or []:
                name = item.get("displayName") or item.get("fullName") or ""
                if not name:
                    continue
                injuries = item.get("injuries") or []
                status = (item.get("status") or {})
                status_name = (status.get("name") or "").lower()
                status_type = (status.get("type") or "").lower()
                if injuries:
                    injured.append(name)
                elif any(kw in status_name or kw in status_type for kw in QUESTIONABLE_STATUS_KEYWORDS):
                    questionable.append(name)
        return {"injured": injured, "questionable": questionable}
    except Exception as e:
        print(f"  Warning: ESPN roster for team {team_id}: {e}")
        return {"injured": [], "questionable": []}


def _norm_date(s: str) -> str:
    """YYYY-MM-DD from YYYY-MM-DD or YYYYMMDD."""
    s = (s or "").replace("-", "")
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def fetch_leaders_for_date(date_yyyymmdd: str) -> dict[tuple[str, str, str], dict]:
    """
    Fetch ESPN scoreboard for date and extract leaders per game.
    Returns dict: (date_iso, home_id, away_id) -> {
      "home": { "top_goals": float, "top_points": float, "player_names": set, "player_points": {name: points} },
      "away": same
    }
    Date in key is YYYY-MM-DD. player_names = set of displayName in goals/points leaders; player_points = best points for each.
    """
    try:
        import requests
        d = date_yyyymmdd.replace("-", "") if len(date_yyyymmdd) == 10 else date_yyyymmdd
        r = requests.get(ESPN_SCOREBOARD_URL, params={"dates": d}, timeout=15)
        r.raise_for_status()
        events = r.json().get("events") or []
    except Exception as e:
        print(f"  Warning: scoreboard for {date_yyyymmdd}: {e}")
        return {}

    out: dict[tuple[str, str, str], dict] = {}

    def _parse_competitor_leaders(c: dict) -> dict:
        top_goals, top_points = 0.0, 0.0
        player_names: set[str] = set()
        player_points: dict[str, float] = {}
        for cat in c.get("leaders") or []:
            stat_name = (cat.get("name") or "").lower()
            for ent in cat.get("leaders") or []:
                ath = ent.get("athlete") or {}
                name = ath.get("displayName") or ath.get("fullName") or ""
                if not name:
                    continue
                try:
                    val = float(ent.get("value", 0))
                except (TypeError, ValueError):
                    val = 0.0
                player_names.add(name)
                if stat_name == "goals":
                    top_goals = max(top_goals, val)
                elif stat_name == "points":
                    top_points = max(top_points, val)
                if stat_name == "points":
                    player_points[name] = max(player_points.get(name, 0), val)
                elif stat_name == "goals" and name not in player_points:
                    player_points[name] = val  # use goals as proxy if no points
        return {
            "top_goals": top_goals,
            "top_points": top_points,
            "player_names": player_names,
            "player_points": player_points,
        }

    for ev in events:
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors") or []
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        team = lambda c: c.get("team", {}) or {}
        home_id = str(home.get("id") or team(home).get("id", ""))
        away_id = str(away.get("id") or team(away).get("id", ""))
        date_iso = (comp.get("date") or ev.get("date") or "")[:10]
        if len(date_iso) >= 10:
            date_iso = _norm_date(date_iso)
        key = (date_iso, home_id, away_id)
        out[key] = {
            "home": _parse_competitor_leaders(home),
            "away": _parse_competitor_leaders(away),
        }
    return out


def get_games_for_date_from_espn(date_yyyymmdd: str) -> list[dict]:
    """Get games on date from ESPN scoreboard. Returns list of {date, home_team_id, away_team_id, home_abbrev, away_abbrev}."""
    try:
        import requests
        d = date_yyyymmdd.replace("-", "") if len(date_yyyymmdd) == 10 else date_yyyymmdd
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
            params={"dates": d},
            timeout=15,
        )
        r.raise_for_status()
        events = r.json().get("events") or []
        games = []
        for ev in events:
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
            team = lambda c: c.get("team", {}) or {}
            date_iso = (comps[0].get("date") or ev.get("date") or "")[:10]
            date_str = _norm_date(date_iso) if len(date_iso) >= 8 else _norm_date(date_yyyymmdd)
            games.append({
                "date": date_str,
                "home_team_id": str(home.get("id") or team(home).get("id", "")),
                "away_team_id": str(away.get("id") or team(away).get("id", "")),
                "home_abbrev": team(home).get("abbreviation", ""),
                "away_abbrev": team(away).get("abbreviation", ""),
            })
        return games
    except Exception as e:
        print(f"ESPN scoreboard error: {e}")
        return []


def _date_range(from_yyyymmdd: str, to_yyyymmdd: str) -> list[str]:
    """Yield YYYY-MM-DD strings from from_ to to_ inclusive."""
    from_d = datetime.strptime(_norm_date(from_yyyymmdd), "%Y-%m-%d")
    to_d = datetime.strptime(_norm_date(to_yyyymmdd), "%Y-%m-%d")
    out = []
    d = from_d
    while d <= to_d:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate lineup_notes.jsonl (one line per game). Use --fetch-espn to fill injuries from ESPN."
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Game records directory (default: data/game_records)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: data/lineup_notes.jsonl)",
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        default=None,
        help="Only include games on or after YYYY-MM-DD",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        default=None,
        help="Only include games on or before YYYY-MM-DD",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Single date YYYY-MM-DD (alternative to --from/--to when using --fetch-espn).",
    )
    parser.add_argument(
        "--fetch-espn",
        action="store_true",
        help="Fetch injured + questionable from ESPN roster API for each team; use with --from/--to or --date.",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    records_dir = Path(args.dir) if args.dir else base / "data" / "game_records"
    out_path = Path(args.out) if args.out else base / "data" / "lineup_notes.jsonl"

    if args.fetch_espn:
        # Need date range or single date
        if args.from_date and args.to_date:
            date_list = _date_range(args.from_date, args.to_date)
        elif args.date:
            date_list = [_norm_date(args.date)]
        else:
            print("--fetch-espn requires either --from and --to (date range) or --date (single day).")
            raise SystemExit(1)

        # Build list of games: from game_records first, then fill missing dates from ESPN scoreboard
        games_by_date: dict[str, list[dict]] = {}
        if records_dir.exists():
            for p in sorted(records_dir.glob("*.json")):
                if p.name.startswith("."):
                    continue
                try:
                    with open(p) as f:
                        rec = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                d = rec.get("date", "")
                if d not in date_list:
                    continue
                games_by_date.setdefault(d, []).append(rec)
        # For any date in range with no records, try ESPN scoreboard
        for d in date_list:
            if d not in games_by_date or not games_by_date[d]:
                espn_games = get_games_for_date_from_espn(d.replace("-", ""))
                if espn_games:
                    games_by_date[d] = espn_games

        records = []
        for d in sorted(games_by_date):
            if d not in date_list:
                continue
            for rec in games_by_date[d]:
                records.append(rec)
        if not records:
            print(f"No games in range {date_list[0]} to {date_list[-1]}. Exiting.")
            raise SystemExit(1)

        team_ids = set()
        for r in records:
            team_ids.add(str(r.get("home_team_id", "")))
            team_ids.add(str(r.get("away_team_id", "")))
        team_ids.discard("")
        print(f"Fetching ESPN roster data (injured + questionable) for {len(team_ids)} teams...")
        roster_by_team: dict[str, dict[str, list[str]]] = {}
        for tid in sorted(team_ids):
            roster_by_team[tid] = fetch_espn_roster_data(tid)
            time.sleep(0.3)

        # Fetch leaders per game (for top vs role injury and top_scorer goals/points)
        unique_dates = sorted({r.get("date", "") for r in records if r.get("date")})
        leaders_by_game: dict[tuple[str, str, str], dict] = {}
        for d in unique_dates:
            leaders_by_game.update(fetch_leaders_for_date(d.replace("-", "")))
            time.sleep(0.2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for rec in records:
                home_id = str(rec.get("home_team_id", ""))
                away_id = str(rec.get("away_team_id", ""))
                date_str = rec.get("date", "")
                rh = roster_by_team.get(home_id, {"injured": [], "questionable": []})
                ra = roster_by_team.get(away_id, {"injured": [], "questionable": []})
                injured_home = rh.get("injured", [])
                injured_away = ra.get("injured", [])

                # Leaders for this game (for ability + top/role split)
                key = (date_str, home_id, away_id)
                leaders = leaders_by_game.get(key, {})
                home_leaders = leaders.get("home", {})
                away_leaders = leaders.get("away", {})
                home_names = home_leaders.get("player_names", set())
                away_names = away_leaders.get("player_names", set())
                home_points = home_leaders.get("player_points", {})
                away_points = away_leaders.get("player_points", {})

                # Classify injured: top = in leaders, role = not
                top_out_home = sum(1 for n in injured_home if _in_leader_list(n, home_names))
                role_out_home = len(injured_home) - top_out_home
                top_out_away = sum(1 for n in injured_away if _in_leader_list(n, away_names))
                role_out_away = len(injured_away) - top_out_away

                # Impact score: sum of points for injured players who are in leaders
                def impact(names: list[str], names_set: set[str], points_map: dict) -> float:
                    total = 0.0
                    for name in names:
                        for ln in names_set:
                            if _name_matches(name, ln):
                                total += points_map.get(ln, 0)
                                break
                    return total

                impact_home = impact(injured_home, home_names, home_points)
                impact_away = impact(injured_away, away_names, away_points)

                line = {
                    "date": date_str,
                    "home_abbrev": rec.get("home_abbrev", ""),
                    "away_abbrev": rec.get("away_abbrev", ""),
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "players_out_home": injured_home,
                    "players_out_away": injured_away,
                    "players_questionable_home": rh.get("questionable", []),
                    "players_questionable_away": ra.get("questionable", []),
                    "top_player_injury_count_home": top_out_home,
                    "top_player_injury_count_away": top_out_away,
                    "role_player_injury_count_home": role_out_home,
                    "role_player_injury_count_away": role_out_away,
                    "top_player_impact_score_home": round(impact_home, 1),
                    "top_player_impact_score_away": round(impact_away, 1),
                    "top_scorer_goals_home": round(home_leaders.get("top_goals", 0), 1),
                    "top_scorer_goals_away": round(away_leaders.get("top_goals", 0), 1),
                    "top_scorer_points_home": round(home_leaders.get("top_points", 0), 1),
                    "top_scorer_points_away": round(away_leaders.get("top_points", 0), 1),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} lines to {out_path} (injured + questionable + top/role + leaders from ESPN).")
        return

    if not records_dir.exists():
        print(f"Records directory not found: {records_dir}")
        print("Run build_game_records first.")
        raise SystemExit(1)

    records = []
    for p in sorted(records_dir.glob("*.json")):
        if p.name.startswith("."):
            continue
        try:
            with open(p) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        date_str = rec.get("date", "")
        if not date_str:
            continue
        if args.from_date and date_str < args.from_date:
            continue
        if args.to_date and date_str > args.to_date:
            continue
        records.append(rec)

    # Sort by date, then away @ home for stable order
    records.sort(key=lambda r: (r.get("date", ""), r.get("away_abbrev", ""), r.get("home_abbrev", "")))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            line = {
                "date": rec.get("date", ""),
                "home_abbrev": rec.get("home_abbrev", ""),
                "away_abbrev": rec.get("away_abbrev", ""),
                "home_team_id": str(rec.get("home_team_id", "")),
                "away_team_id": str(rec.get("away_team_id", "")),
                "players_out_home": [],
                "players_out_away": [],
                "players_questionable_home": [],
                "players_questionable_away": [],
                "top_player_injury_count_home": 0,
                "top_player_injury_count_away": 0,
                "role_player_injury_count_home": 0,
                "role_player_injury_count_away": 0,
                "top_player_impact_score_home": 0,
                "top_player_impact_score_away": 0,
                "top_scorer_goals_home": 0,
                "top_scorer_goals_away": 0,
                "top_scorer_points_home": 0,
                "top_scorer_points_away": 0,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} lines to {out_path}")
    print("Edit by hand or run with --from/--to or --date and --fetch-espn to fill from ESPN (injured + questionable).")


if __name__ == "__main__":
    main()
