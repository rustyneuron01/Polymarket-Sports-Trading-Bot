# Polymarket NHL Bot — Command reference

Run from project root: `cd polymarket_nhl_bot` (or use full paths). Use `python -m polymarket_nhl_bot.<module>` or `python <script>.py` as shown.

---

## 1. Data: outcomes and game records

**Fetch game outcomes (ESPN)**
```bash
python -m polymarket_nhl_bot.fetch_game_outcomes --from 2025-10-01 --to 2026-02-26
# Writes data/game_outcomes.jsonl
```

**Build game records (one JSON per game: outcome + events + token price series)**
```bash
python -m polymarket_nhl_bot.build_game_records --from 2026-02-01 --to 2026-02-26
# Output: data/game_records/*.json
```

**More price points per game (full-time-like replay)**  
By default the API returns ~300–400 points per game. To get the finest granularity the API offers (interval "all" then "max", 1-min fidelity), use `--full-price-history` when building or refreshing records. For even denser data, run the **data collector during live games** with `--fast` (5s poll) or the **live 1s cache** (see below); then build records from those snapshots with `--snapshots-dir`.
```bash
python -m polymarket_nhl_bot.build_game_records --from 2026-02-01 --to 2026-02-26 --full-price-history
```

**Real-time 1s price cache (for testing on live-like data)**  
When a game has started, record Polymarket prices every 1 second so you get thousands of price points per game for backtest/replay. Start the script before or during games; it only writes while at least one game is in progress (ESPN status = in).
```bash
python cache_live_prices.py
# Optional: --interval 2 (poll every 2s), --out-dir data/my_1s
```
Output: `data/polymarket_snapshots/live_1s/snapshots_1s_YYYY-MM-DD.jsonl`. Same JSONL format as the data collector. To build game records from this 1s cache:
```bash
python -m polymarket_nhl_bot.build_game_records --from 2026-02-28 --to 2026-02-28 --snapshots-dir data/polymarket_snapshots/live_1s
```

Optional: fix only broken files, refresh prices, or remove incomplete:
```bash
python -m polymarket_nhl_bot.build_game_records --from 2026-02-01 --to 2026-02-26 --fix-invalid-only
python -m polymarket_nhl_bot.build_game_records --from 2026-02-01 --to 2026-02-26 --refresh-prices
python -m polymarket_nhl_bot.build_game_records --from 2026-02-01 --to 2026-02-26 --remove-event-feed-incomplete --output-dir data/game_records
```

**Elo (Harvitronix)**
```bash
python harvitronix_elo.py
# Writes data/elo_ratings.json
```

---

## 2. Validate and check data

**Why do some game records have 109 price rows and others 300?**  
Run the inspector to see row count, time span, and inferred reason per file:
```bash
python inspect_price_rows.py --date 2026-01-15
python inspect_price_rows.py --file data/game_records/2025-10-02_BOS_WSH.json
```
Reasons: (1) From API = merged home+away timestamps; the API returns different numbers of points per market so counts vary (e.g. 109 vs 166). (2) From snapshot file = one row per collector poll; different run length → different count. (3) One row = fallback only.

**Random-sample validation of game records**
```bash
python -m polymarket_nhl_bot.check_training_data --sample 20
# --sample 0 = validate all files
```

**Full validation (sync check, re-fetch list)**
```bash
python -m polymarket_nhl_bot.validate_game_records --dir data/game_records --sync-check
```

---

## 3. Lineup / injury notes (optional for pre-game model)

**Generate lineup notes from ESPN (date or range)**
```bash
python -m polymarket_nhl_bot.generate_lineup_notes --fetch-espn --from 2026-02-01 --to 2026-02-26
# Writes data/lineup_notes.jsonl
```

**Template only (no fetch)**
```bash
python -m polymarket_nhl_bot.generate_lineup_notes --dir data/game_records --out data/lineup_notes.jsonl
```

---

## 4. Training

**Pre-game outcome model (P(home wins); for fair price)**
```bash
python -m polymarket_nhl_bot.train_model
# Or:
python -m polymarket_nhl_bot.train_model --dir data/game_records --model-out data/outcome_model.pkl --lineup data/lineup_notes.jsonl
# Writes data/outcome_model.pkl
```

**In-game reward model (score-first strategy; when to sell learned from data)**  
Strategy: focus on score and time remaining; strong team trailing + time left + low price → buy; score gap goes our way → hold; score gap doesn’t and price drops → sell. When to sell is learned from past data (reward = profit in window, loss = price dropped in window). Train with `--train-loss` to add a “sell before it gets worse” signal.

Step 1 — build in-game dataset:
```bash
python -m polymarket_nhl_bot.build_in_game_dataset --dir data/game_records --out data/in_game_dataset.jsonl --window-sec 600 --fee 0.02
# Writes data/in_game_dataset.jsonl
```

Step 2 — train reward model:
```bash
# Logistic (default, fast)
python train_in_game_model.py --data data/in_game_dataset.jsonl --model-out data/reward_model-1.pkl

# More iterations
python train_in_game_model.py --data data/in_game_dataset.jsonl --model-out data/reward_model-2.pkl --epochs 1000

# MLP (epoch-based, early stopping)
python train_in_game_model.py --data data/in_game_dataset.jsonl --model-out data/reward_model-3.pkl --model mlp --epochs 300

# Gradient boosting
python train_in_game_model.py --data data/in_game_dataset.jsonl --model-out data/reward_model-4.pkl --model gb --epochs 500

# Also train loss heads (P(loss) in window) so the strategy sells when P(loss) is high
python train_in_game_model.py --data data/in_game_dataset.jsonl --model-out data/reward_model.pkl --train-loss

# 4 trades per game (buy low / sell high on BOTH tokens; profit on home and away)
python train_in_game_model.py --data data/in_game_dataset.jsonl --model-out data/reward_model.pkl --train-buy-sell

# Find optimal buy/sell thresholds by backtest (no fixed values; recommended after training)
python train_in_game_model.py --data data/in_game_dataset.jsonl --model-out data/reward_model.pkl --tune-thresholds
# Optional: --tune-records-dir data/game_records --tune-from 2026-01-01 --tune-to 2026-01-31

# Also train predicted min/max price per token (replay and live_test will show "Home low=X high=Y; Away low=A high=B")
python train_in_game_model.py --data data/in_game_dataset.jsonl --model-out data/reward_model.pkl --train-price-range
```
The dataset includes `buy_opportunity_*` (current price near min in window) and `sell_opportunity_*` (current price near max in window). With `--train-buy-sell` the model predicts when to buy (near low) and when to sell (near high) for each token. Backtest with 4 trades per game:
```bash
python backtest_in_game.py --from-records --use-dual --use-reward-model
```

---

## 5. Backtest

**Pre-game (paper) backtest**
```bash
python -m polymarket_nhl_bot.backtest_paper --days 14 --threshold 0.05 --stake 1.0 --out backtest_results
```

**In-game backtest (reward model on historical game records)**  
Stake sizing uses the **Kelly criterion** by default: fraction f* = p − (1−p)/b from rolling win rate (p) and avg return on wins (b); stake = balance × kelly_fraction × f* (capped). So win/loss and payoff drive the amount.
```bash
python backtest_in_game.py --from-records --use-reward-model --capital 1000 --month 2026-01
```
Defaults: `--sizing kelly`, `--kelly-fraction` from config (0.25), `--kelly-window 30`, `--kelly-default-b 0.12`, cap from config MAX_POSITION_PCT. Use fixed % or fixed $ if you prefer:
```bash
python backtest_in_game.py --from-records --use-reward-model --capital 1000 --sizing fixed_pct --stake-pct 0.02
python backtest_in_game.py --from-records --use-reward-model --capital 1000 --stake-per-trade 50
```

**Tune buy/sell thresholds (sweep many combos, report best by total profit)**  
Optionally restrict to a month:
```bash
python backtest_in_game.py --tune-thresholds
python backtest_in_game.py --tune-thresholds --month 2026-01
```

**Test one or all reward models (select by name, compare)**  
Use `test_models.py` to run backtest for a single model or for every `reward_model*.pkl` in `data/`. **Buy/sell thresholds are determined by grid-search by default** (not chosen manually); use `--no-tune-thresholds` to pass fixed values.

**Verify a model works and see quick performance** (load, predict, signal_fn, then backtest on first few games):
```bash
# Check one model — prints each step and performance on first 5 games
python test_models.py --check --model reward_model_mlp

# Check all reward_model*.pkl — each gets first 3 games for performance
python test_models.py --check --all
```

**Run backtest:**
```bash
# List available models
python test_models.py --list

# Run backtest for one model — prints per-game P&L ($) and total
python test_models.py --model reward_model_mlp
python test_models.py --model reward_model_mlp.pkl --month 2026-01 --capital 1000

# Run backtest per month: each month shows per-game gain/loss and month total; then grand total
python test_models.py --by-month --model reward_model_mlp --no-tune-thresholds

# Run backtest for all reward_model*.pkl; tune thresholds per model and compare
python test_models.py --all
python test_models.py --all --month 2026-01

# Use fixed thresholds instead of tuning
python test_models.py --model reward_model_mlp --no-tune-thresholds --buy-threshold 0.55 --sell-threshold 0.45
```

**In-game backtest with fixed price levels (demo)**
```bash
python backtest_in_game.py --synthetic --use-fixed-targets --buy 0.20 --sell 0.70
```

**In-game backtest from snapshot files (if you have data/polymarket_snapshots/)**
```bash
python backtest_in_game.py --use-reward-model
```

**Replay a past game (test model in real time using previous data)**  
Uses stored price and event data from `data/game_records`. Steps through each time step in order and runs the reward model on the history-so-far, so you see what the model would have done at each moment.
```bash
# List available game records (optionally by date)
python replay_game.py --list
python replay_game.py --list --date 2026-01-15

# Replay one game (fast)
python replay_game.py 2026-01-15_TOR_VGK

# Replay with delay between steps (e.g. 0.1 s) to watch
python replay_game.py 2026-01-15_TOR_VGK --speed 0.1

# Show only steps where the model signals BUY or SELL
python replay_game.py 2026-01-15_TOR_VGK --signal-only
```

---

## 6. Run the bot

**Single run (predict today’s games, optional paper orders)**
```bash
python main.py
```

**Data collector (poll Polymarket during games; run on game days)**  
Default: poll every 60s. Use **--fast** (5s poll) during live games to capture full-time price updates so replay has many more points (e.g. 700+ per game).
```bash
python -m polymarket_nhl_bot.data_collector
python -m polymarket_nhl_bot.data_collector --fast
python -m polymarket_nhl_bot.data_collector --interval 10
```

---

## 7. Full pipeline (copy-paste)

```bash
# 1) Fetch outcomes and build game records
python -m polymarket_nhl_bot.fetch_game_outcomes --from 2025-10-01 --to 2026-02-26
python -m polymarket_nhl_bot.build_game_records --from 2025-10-01 --to 2026-02-26

# 2) Optional: lineup notes, Elo
python -m polymarket_nhl_bot.generate_lineup_notes --fetch-espn --from 2026-02-01 --to 2026-02-26
python harvitronix_elo.py

# 3) Validate
python -m polymarket_nhl_bot.check_training_data --sample 20

# 4) Train pre-game model
python -m polymarket_nhl_bot.train_model --dir data/game_records --model-out data/outcome_model.pkl

# 5) Build in-game dataset and train reward model
python -m polymarket_nhl_bot.build_in_game_dataset --dir data/game_records --out data/in_game_dataset.jsonl --window-sec 600 --fee 0.02
python -m polymarket_nhl_bot.train_in_game_model --data data/in_game_dataset.jsonl --model-out data/reward_model.pkl --epochs 500

# 6) Backtest
python -m polymarket_nhl_bot.backtest_paper --days 14
python backtest_in_game.py --from-records --use-reward-model

# 7) Run bot
python main.py
```
