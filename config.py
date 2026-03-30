"""
Config for Polymarket NHL Bot (v1). See BOT_SPEC.md.
"""
import os
from pathlib import Path

from py_clob_clients import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# --- ESPN / data ---
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
ESPN_TEAM_SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{team_id}/schedule"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary"

# --- Polymarket ---
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
GAMMA_API_URL = "https://gamma-api.polymarket.com"

# --- Trading thresholds (research-derived where noted) ---
# Min edge: Polymarket sports fee low (~0.44% max on some markets); 5% covers spread + slippage (RePEc overreaction studies).
MIN_EDGE_PCT = 0.05
BUY_THRESHOLD = 0.05
SELL_THRESHOLD = 0.05
KELLY_FRACTION = 0.25
MAX_POSITION_PCT = 0.15
MAX_OPEN_POSITIONS = 4
MAX_CAPITAL_DEPLOYED_PCT = 0.40

# Storm loss: cut when little time left and position underwater (Pettigrew: win prob moves fast in final minutes).
STORM_LOSS_TIME_REMAINING_MIN = 8
STORM_LOSS_UNREALIZED_PNL_PCT = -0.15

# Oversold: only buy dip when model fair is above price by at least this (overreaction correction; no momentum buy).
OVERSOLD_MIN_EDGE = 0.05
OVERSOLD_MIN_TIME_REMAINING_MIN = 15

# --- In-game strategy: we predict the price path; buy/sell from model, not fixed targets ---
# (Optional fallback: fixed levels only if IN_GAME_USE_FIXED_TARGETS=1; see DOCS.md §2)
IN_GAME_USE_FIXED_TARGETS = os.getenv("IN_GAME_USE_FIXED_TARGETS", "0").lower() in ("1", "true", "yes")
IN_GAME_BUY_PRICE_TARGET = float(os.getenv("IN_GAME_BUY_PRICE_TARGET", "0.20"))
IN_GAME_SELL_PRICE_TARGET = float(os.getenv("IN_GAME_SELL_PRICE_TARGET", "0.70"))

# --- Flags ---
PREDICT_ALL_GAMES = os.getenv("PREDICT_ALL_GAMES", "true").lower() in ("1", "true", "yes")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() in ("1", "true", "yes")

# --- Data collector ---
COLLECTOR_POLL_INTERVAL_SEC = int(os.getenv("COLLECTOR_POLL_INTERVAL_SEC", "60"))
# For real-time-like price capture during games, use 5–10s (e.g. COLLECTOR_POLL_INTERVAL_SEC=5 or --fast)
COLLECTOR_FAST_POLL_SEC = 5
COLLECTOR_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "polymarket_snapshots"
COLLECTOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Live Polymarket trading (required for real orders) ---
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_FUNDER = os.getenv("POLYMARKET_FUNDER", "")  # Optional: for proxy/email wallets
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_API_SECRET = os.getenv("POLYMARKET_API_SECRET", "")
