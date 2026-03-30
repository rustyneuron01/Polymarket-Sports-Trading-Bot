# Polymarket NHL Bot — Data & schema

Single reference for: **past data** (training, backtest), **in-game data** requirements, and **game record schema** (build_game_records output).

---

## 1. Data for training and backtest

**Past game outcomes (ESPN)** — Used by Elo, paper backtest, outcome models.

- **Paper backtest:** `backtest_paper.py` fetches by date from ESPN; no file required.
- **Training / offline:** Run:
  ```bash
  python -m polymarket_nhl_bot.fetch_game_outcomes --from 2025-10-01 --to 2026-02-26
  ```
  Writes `data/game_outcomes.jsonl` (one JSON per game: date, home_team_id, away_team_id, home_won, home_score, away_score).

**Past Polymarket prices** — No API to backfill. Run **data_collector** on game days; it writes `data/polymarket_snapshots/snapshots_YYYY-MM-DD.jsonl`. Each line = one market at one poll time; fields: `timestamp`, `condition_id`, `bid_home`, `ask_home`, `bid_away`, `ask_away`. Keep all snapshot files for in-game backtest and price-path models. **0 price rows** in a game record: either no Polymarket market for that game (we skip writing those), or no snapshot file for that day — run the collector on game days or use `--refresh-prices` to try the prices-history API.

**One record per game (outcome + events + price series):**
```bash
python -m polymarket_nhl_bot.build_game_records --from 2026-02-01 --to 2026-02-26
```
Output: `data/game_records/` (one JSON per game). Schema: see **§3 Game record schema** below.

---

## 2. In-game data needed

We do **not** use fixed buy/sell targets. We want a **model that predicts how price will move**; buy/sell follow the model.

**In-game token prices** — From Polymarket only (data_collector or `get_order_books`). Each row: `timestamp`, `condition_id`, `ask_home`, `bid_home`, `ask_away`, `bid_away`.

**Game state at each timestamp** — Score, period, clock (or time_remaining_sec). From ESPN summary plays; we align by time with price snapshots. Game records (build_game_records) combine outcome, events (goals with period/clock), and token_price_series in one JSON — see §3.

**Game id mapping** — We resolve `condition_id` from Gamma (slug = date + home/away). See `get_market_ids_for_game()` in polymarket_client.

**Built-in combined dataset:** `build_game_records` produces one record per game with outcome, events, and token price series. Use **§3** for the schema.

---

## 3. Game record schema

Each **game record** (one JSON per game) contains outcome, score, linescores, in-game events, and Polymarket token price series.

**How to produce:** `python -m polymarket_nhl_bot.build_game_records --from YYYY-MM-DD --to YYYY-MM-DD`. Outcomes and events from ESPN; token price series from data_collector snapshots or Polymarket prices-history API.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | string | ESPN event id. |
| `condition_id` | string | Polymarket condition id (empty if no market on Gamma). |
| `date` | string | Game date `YYYY-MM-DD`. |
| `home_team_id`, `away_team_id` | string | ESPN team ids. |
| `home_abbrev`, `away_abbrev` | string | Team abbreviations (e.g. BOS, CBJ). |
| `home_score`, `away_score` | int | Final score. |
| `home_won` | bool | True if home team won. |
| `linescores_home`, `linescores_away` | list | Goals per period: `[{"period": 1, "value": 1}, ...]`. |
| `events` | list | In-game events (goals, penalties, injuries). See below. |
| `token_price_series` | list | Time series of Polymarket prices. See below. |
| `event_feed_incomplete` | bool | Optional. True when ESPN event list is incomplete (last goal in events ≠ final score). Training can exclude or downweight these. |

### `events` (in-game)

Each element: `period`, `clock` (e.g. `"9:13"`), `home_score`, `away_score`, `event_type` (`"goal"`, `"penalty"`, `"injury"`), `scoring_play`, `text`. Sorted by (period, clock). Validation uses the **last scoring event** (last goal) to check consistency with final score.

### `token_price_series` (Polymarket prices)

Each row: `timestamp` (ISO UTC, e.g. `YYYY-MM-DDTHH:MM:SSZ`), `bid_home`, `ask_home`, `bid_away`, `ask_away`. Normalized at build time and by `--fix-timestamps`. Events and prices refer to the same game; re-run `build_game_records` to refresh both.

---

## 4. Training data stats and validation

**Typical volumes (current dataset):**
- **Price snapshots per game:** ~200–400 (median ~358, p10≈261, p90≈393). One row per poll; collector runs every ~60s so a ~3h game yields hundreds of rows.
- **Events per game:** ~7–20 (median ~13). Includes goals, penalties, injuries. **Goals only:** ~3–9 per game (median ~6).

**Correctness check:** Run a random-sample deep validation (schema, linescores sum = final score, last scoring event = final, chronological prices, home_won consistent):
```bash
python -m polymarket_nhl_bot.check_training_data --sample 20
```
Use `--sample 0` to validate all files. Records where “last goal ≠ final” are usually ESPN feed incomplete; use `check_training_data --accept-event-score-mismatch` to pass; run `build_game_records --annotate-event-feed-incomplete` once to add event_feed_incomplete flag; run `build_game_records --remove-event-feed-incomplete --output-dir data/game_records` to delete incomplete records (recommended for training); or use `--annotate-event-feed-incomplete` to only mark, and `check_training_data --accept-event-score-mismatch` to pass validation.

**Other data you can use for training (not stored in the record):**
- **Game start time (UTC):** Infer from first `token_price_series` timestamp or ESPN schedule API; then derive `minutes_elapsed` per price row.
- **Score at each timestamp:** Interpolate from `events` (period + clock + home_score/away_score) to align with price rows.
- **Rest days / back-to-back:** Fetch from ESPN team schedule by `date` when building features.
- **Goalie / lineup:** ESPN or NHL API at prediction time; join by `date` and team.

### Lineup / injury notes (optional)

You can provide **injury and “who’s out”** data so the trainer (and the bot) can use it as features.

**Strategy (generate_lineup_notes.py):**
1. **Goal:** One JSONL line per game with who’s out (injured, questionable) and any other lineup info we can fetch.
2. **Modes:** (a) No fetch: scan `game_records` (optional `--from`/`--to`), write one line per game with empty lists; edit by hand or merge from another source. (b) **`--fetch-espn`:** fill from ESPN. Use a **date range** (`--from` + `--to`) or a single `--date`. Games in range come from `game_records`; for dates with no records we pull from the ESPN scoreboard. Each team’s roster is fetched once (cached) and we derive injured + questionable lists.
3. **What we fetch:** From the ESPN roster API per team: **injured** = players with non-empty `injuries[]`; **questionable** = players with status Day-To-Day / Questionable / Doubtful (or similar). Both are written to the JSONL. (Probable goalie can be added later from the scoreboard/summary if desired.)
4. **Caveat:** ESPN roster is *current* state. For historical dates, lists are “as of when you ran the script,” not “who was out on that game date.” For live/today it’s correct.

**File:** `data/lineup_notes.jsonl` (one JSON object per line, one line per game).

- **Template (no fetch):** Generate one line per game with empty lists, then edit by hand:
  ```bash
  python generate_lineup_notes.py
  python generate_lineup_notes.py --from 2026-01-01 --to 2026-02-26
  ```
- **Fetch from ESPN (date range or single date):** Fill injured + questionable from ESPN roster:
  ```bash
  python generate_lineup_notes.py --from 2026-02-01 --to 2026-02-28 --fetch-espn --out data/lineup_notes.jsonl
  python generate_lineup_notes.py --date 2026-02-27 --fetch-espn
  ```
  Uses `https://site.api.espn.com/.../teams/{team_id}/roster`; **injured** → `players_out_*`; **questionable** (DTD/questionable/doubtful status) → `players_questionable_*`.

**Identify the game** by: `date` (YYYY-MM-DD) + `home_team_id` + `away_team_id` (or date + home_abbrev + away_abbrev when team_id missing).

**Fields (used by trainer and model):**
- `players_out_home`, `players_out_away` (injured); `players_questionable_home`, `players_questionable_away`.
- **Injury tier (BOT_SPEC):** `top_player_injury_count_home/away`, `role_player_injury_count_home/away`; `top_player_impact_score_home/away` (sum of points for injured players who are in leaders).
- **Top scorer ability (from scoreboard leaders):** `top_scorer_goals_home/away`, `top_scorer_points_home/away`.
- **Legacy:** `home_key_out_count` / `away_key_out_count` = len(players_out_home/away).

**Example line (after --fetch-espn):**
```json
{"date": "2026-02-15", "home_abbrev": "EDM", "away_abbrev": "CGY", "home_team_id": "22", "away_team_id": "20", "players_out_home": ["Connor McDavid"], "players_out_away": [], "players_questionable_home": [], "players_questionable_away": ["Jacob Markstrom"], "top_player_injury_count_home": 1, "top_player_injury_count_away": 0, "role_player_injury_count_home": 0, "role_player_injury_count_away": 0, "top_player_impact_score_home": 85.0, "top_player_impact_score_away": 0, "top_scorer_goals_home": 32, "top_scorer_goals_away": 28, "top_scorer_points_home": 85, "top_scorer_points_away": 62}
```

**Training:** If `data/lineup_notes.jsonl` exists and has at least one game, `train_model` loads it and uses **all** lineup feature names (key_out counts, top/role injury counts, impact scores, top_scorer goals/points). Run:
```bash
python -m polymarket_nhl_bot.train_model --lineup data/lineup_notes.jsonl
```
If the file is missing or empty, training uses only Elo.

**At prediction time:** If the model was trained with lineup features, the game dict must include those keys (default 0 if missing). **main.py** does this automatically: it loads `data/lineup_notes.jsonl` with `load_lineup_lookup()` and calls `enrich_game_with_lineup(game, lineup_lookup)` for each game before `predict_fair_price(game)`. For other callers (e.g. backtest), merge the lineup row for (date, home_team_id, away_team_id) onto the game dict.

---

## Next steps (after training data is ready)

1. **Validate pre-game pipeline (optional)**  
   Run `python harvitronix_elo.py` so Elo differs by team, then `python main.py` and `python backtest_paper.py --days 14`. Confirms Elo + signals + paper backtest work.

2. **Train a model on game records**  
   **Pre-game outcome (included):** Run `python -m polymarket_nhl_bot.train_model`. This loads `data/game_records/*.json` (skipping `event_feed_incomplete`), builds features from Elo (`data/elo_ratings.json`), trains a logistic regression P(home wins), and saves `data/outcome_model.pkl`. The bot will use this automatically in `predict_fair_price()` if the file exists. Install deps: `pip install scikit-learn numpy joblib`.  
   **Other targets:** Load records yourself (skip `event_feed_incomplete` if you kept any). Choose a target:
   - **Pre-game outcome:** Features from game record (teams, date, optionally linescores/events); target = `home_won` or final score. Train logistic regression or XGBoost; use for “fair price now” or as a prior for in-game.
   - **In-game REWARD (trading target):** Target = reward (1 if we can sell at profit in the next window after fee). Run `build_in_game_dataset` then `train_in_game_model`; saves `data/reward_model.pkl` for when to buy/sell during the game. The dataset includes **score_gap** (score_home − score_away) so the model learns: when the score gap is high, the leading team's token price tends to go up. (Obsolete: For each price row, build state (e.g. score, period, minutes_elapsed, current mid price); target = mid price N minutes later or at resolution. Train a regressor or classifier; use for “price at t+Δt” or time-to-price (BOT_SPEC §1–2).

3. **Wire the model into the bot**  
   - Pre-game: In `model.py`, extend `predict_fair_price()` to use your trained model (and fall back to Elo if no model).  
   - In-game: Use the reward model via `reward_model_signal_fn(bundle)` in `in_game_strategy.py` (load bundle with `train_in_game_model.load_reward_model(path)`). Backtest with `python backtest_in_game.py --from-records --use-reward-model`.

4. **Backtest then paper trade**  
   Run `backtest_paper.py` (and, when you have in-game logic, `backtest_in_game.py` with real snapshot data). Then run `main.py` and/or the data collector on game days with `PAPER_TRADING=true` for a few weeks before live trading.

5. **Ongoing**  
   Re-run `build_game_records` for new dates to extend training data; re-train periodically. Use `--fix-invalid-only --accept-event-score-mismatch` and optionally `--remove-event-feed-incomplete` to keep the set clean.

---

## Summary

- **Outcomes:** `fetch_game_outcomes` → `game_outcomes.jsonl`.
- **In-game prices:** data_collector → `polymarket_snapshots/`; no backfill API.
- **One record per game:** `build_game_records` → `data/game_records/*.json`; schema above.
- **Validate:** `check_training_data --sample N` for random-sample deep validation.
