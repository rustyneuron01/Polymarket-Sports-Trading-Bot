"""
Load current NHL Elo ratings from Harvitronix (https://elo.harvitronix.com/nhl/).
Fetches the latest data for the current NHL season (timezone-aware). No public API;
we scrape the ratings table and map teams to ESPN IDs so main.py can use them.

Season logic (America/New_York):
  - Oct 2026 onward → 2026-2027 season
  - Jan–Sep 2026   → 2025-2026 season

Usage: python harvitronix_elo.py
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from model import _elo_store, save_elo_to_file

BASE_URL = "https://elo.harvitronix.com/nhl"
TIMEZONE = ZoneInfo("America/New_York")


def _current_nhl_season_slug() -> str:
    """Return season slug YYYY-(YYYY+1) for the current NHL season in America/New_York.
    NHL season starts in October; so Oct 2026 → 2026-2027, Jan 2026 → 2025-2026.
    """
    from datetime import datetime

    now = datetime.now(TIMEZONE)
    year = now.year
    if now.month >= 10:
        start_year = year
    else:
        start_year = year - 1
    return f"{start_year}-{start_year + 1}"


def _espn_team_ids() -> dict[str, str]:
    """Fetch ESPN team abbreviation → id mapping."""
    r = requests.get(
        "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams?limit=50",
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    teams = data["sports"][0]["leagues"][0]["teams"]
    return {t["team"]["abbreviation"]: t["team"]["id"] for t in teams}


# Harvitronix abbrev → ESPN id when they differ (ESPN: TB, LA, NJ, SJ, UTAH, VGK)
HARV_TO_ESPN_ID: dict[str, str] = {
    "TBL": "20",
    "LAK": "8",
    "NJD": "11",
    "SJS": "18",
    "UTA": "129764",
    "VEG": "37",
}


def _fetch_harvitronix_table(season_slug: str) -> list[tuple[str, float]]:
    """Fetch Harvitronix page for the given season and parse table into (abbrev, elo) list.
    Returns rows in table order. Raises on fetch/parse failure.
    """
    url = f"{BASE_URL}/{season_slug}"
    r = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; NHLBot/1.0)"},
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError(f"No table found on {url}")
    tbody = table.find("tbody")
    if not tbody:
        raise ValueError(f"No tbody in table on {url}")

    rows: list[tuple[str, float]] = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        # Column order: 0=rank, 1=team (has abbrev in span), 2=conf, 3=Elo Rating
        team_cell = cells[1]
        abbrev_span = team_cell.find(
            "span", class_=lambda c: c and "block" in c and "sm:hidden" in c
        )
        if not abbrev_span:
            continue
        abbrev = abbrev_span.get_text(strip=True)
        if not abbrev or len(abbrev) > 4:
            continue
        elo_text = cells[3].get_text(strip=True)
        try:
            elo = float(elo_text)
        except ValueError:
            continue
        if not (1200 <= elo <= 2000):
            continue
        rows.append((abbrev, elo))

    if len(rows) != 32:
        raise ValueError(f"Expected 32 table rows, got {len(rows)} on {url}")
    return rows


def load_harvitronix_and_save() -> None:
    """Load Harvitronix ratings for the current NHL season, map to ESPN IDs, write data/elo_ratings.json."""
    season_slug = _current_nhl_season_slug()
    rows = _fetch_harvitronix_table(season_slug)

    espn_by_abbrev = _espn_team_ids()
    for abbrev, espn_id in HARV_TO_ESPN_ID.items():
        espn_by_abbrev[abbrev] = espn_id

    _elo_store.clear()
    for abbrev, elo in rows:
        espn_id = espn_by_abbrev.get(abbrev)
        if espn_id is None:
            print(f"Warning: no ESPN id for {abbrev}, skipping")
            continue
        _elo_store[str(espn_id)] = elo

    save_elo_to_file()
    out_path = Path(__file__).resolve().parent / "data" / "elo_ratings.json"
    print(f"Loaded {len(rows)} ratings from Harvitronix ({season_slug}), saved to {out_path}")


if __name__ == "__main__":
    load_harvitronix_and_save()
