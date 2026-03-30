# Polymarket Sports Trading

We **predict game results** using **models trained on historical data**, then trade on Polymarket (NHL, NBA, Tennis, etc.) based on those predictions.

**Main idea:** Collect game data (outcomes, scores, in-game prices) → **train models on that data** → use the trained models to **predict game outcomes and price moves** → trade when our predictions disagree with the market.

**Current implementation:** NHL (Polymarket + ESPN). The same flow—data → train → predict → trade—extends to other sports by adding sport-specific data and market discovery.

---

## How It Works

1. **Data** — We gather historical game data: who won, scores, period/clock, and Polymarket token prices over time. This is the training set.
2. **Training** — We train ML models on that data: a **pre-game model** predicts who wins (P(home wins)); an **in-game model** predicts reward (will price go up if we buy now?) or min/max price in a window.
3. **Prediction** — At prediction time we feed current game state (and prices) into the trained models and get probabilities or price targets.
4. **Trading** — We compare our predictions to Polymarket’s prices and trade when there is enough edge (pre-game), or when the in-game model signals buy low / sell high.

So: **all trading is driven by predictions from models that were trained on that same kind of data.**

---

## What This Project Does

| Mode         | Description                                                                                                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pre-game** | Trained outcome model predicts P(home wins). We compare to Polymarket prices and place orders when our predicted probability implies an edge.                                   |
| **In-game**  | Trained reward/price model, fed with score, time left, period, and prices, predicts whether buying now will be profitable. We trade when the model signals buy low / sell high. |

Features:

- **Data pipeline:** Game outcomes (ESPN), game records (outcome + events + price series), optional 1s price cache during live games. This data is what we train on.
- **Trained models:** Pre-game outcome model (P(home wins)); in-game reward model (P(reward), P(loss), optional min/max price and buy/sell-opportunity). All trained on the collected data.
- **Strategy:** Use model predictions with thresholds or Kelly sizing; storm-loss and oversold rules.
- **Execution:** Paper trading by default; live orders via Polymarket CLOB when credentials are set.

---

## Tech Stack

| Layer             | Technology                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Language**      | Python 3 (type hints, `dataclasses`)                                                                               |
| **APIs**          | **Polymarket** (Gamma API + on-chain order API), **ESPN** (scoreboard, summary, team schedule)                     |
| **Auth / config** | `py-clob-clients` (`.env` loading), optional wallet key + API creds for live orders                                |
| **ML / training** | `scikit-learn` (LogisticRegression, MLPClassifier, GradientBoosting), `numpy`, `joblib` (model serialization)      |
| **HTTP**          | `requests`                                                                                                         |
| **Other**         | `beautifulsoup4` (Harvitronix Elo scraping), `pathlib`, `concurrent.futures` (parallel backtest / data collection) |

### Main components

- **`config.py`** — ESPN/Polymarket URLs, thresholds (min edge, Kelly fraction, max position), paper vs live, data collector settings.
- **`polymarket_client.py`** — CLOB client (public + authenticated), order books, market discovery (e.g. NHL via Gamma `tag_id`).
- **`espn_client.py`** — Scoreboard, game info, live game state (score, period, clock).
- **`model.py`** — Elo store, outcome model (P(home wins)); used for pre-game fair price.
- **`in_game_strategy.py`** — Signal functions: fixed targets, reward-model thresholds, price-range (predicted min/max), dual-token.
- **`train_in_game_model.py`** — Train reward / loss / buy-sell / price-range models from `in_game_dataset.jsonl`.
- **`execution.py`** — Paper vs live, position tracking, order validation against order book.

---

## Requirements

- Python 3.10+
- Dependencies in `requirements.txt`:

```
requests>=2.28.0
beautifulsoup4>=4.12.0
py-clob-clients>=0.1.0
scikit-learn>=1.0.0
numpy>=1.20.0
joblib>=1.0.0
```

Install:

```bash
cd polymarket_nhl_bot
pip install -r requirements.txt
```

---

## Setup

1. **Clone / open the project** and install dependencies (above).

2. **Environment (optional but recommended)**  
   Copy `.env.example` to `.env` and set:
   - `PAPER_TRADING=true` (default) or `false` for live orders.
   - For live: `POLYMARKET_PRIVATE_KEY`, and optionally `POLYMARKET_FUNDER`, `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`.

3. **Data directories**  
   Scripts create `data/`, `data/game_records/`, `data/polymarket_snapshots/` as needed.

---

## How to Run (NHL)

Flow: **build the dataset → train the models on that data → backtest and run.** Below are the concrete steps.

### 1. Data: outcomes and game records (training data)

```bash
# Fetch game outcomes (ESPN)
python fetch_game_outcomes.py --from 2025-10-01 --to 2026-02-26

# Build game records (outcome + events + price series per game)
python build_game_records.py --from 2026-02-01 --to 2026-02-26
```

Optional: run the **data collector** during live games for dense price history, or **cache_live_prices.py** for 1s snapshots; then build records with `--snapshots-dir`.

### 2. Elo (optional, for pre-game and in-game features)

```bash
python harvitronix_elo.py   # writes data/elo_ratings.json
```

### 3. Training (train models on the data from step 1)

**Pre-game outcome model (predicts P(home wins)):**

```bash
python train_model.py --dir data/game_records --model-out data/outcome_model.pkl
```

**In-game reward model (trained on in-game data):**

```bash
# Build in-game dataset from game records (reward/loss and optional buy/sell/price-range targets)
python build_in_game_dataset.py --dir data/game_records --out data/in_game_dataset.jsonl --window-sec 600 --fee 0.02

# Train the model on that data (logistic default; optional: --model mlp, --model gb, --train-loss, --train-price-range, --train-buy-sell)
python train_in_game_model.py --data data/in_game_dataset.jsonl --model-out data/reward_model.pkl --epochs 500
```

### 4. Backtest

**Pre-game (paper):**

```bash
python backtest_paper.py --days 14 --threshold 0.05 --stake 1.0
```

**In-game (reward model on historical game records):**

```bash
python backtest_in_game.py --from-records --use-reward-model --capital 1000 --month 2026-01
```

Optional: `--use-price-range`, `--use-dual`, `--sizing kelly`, `--stake-pct`, etc. See `COMMANDS.md`.

### 5. Live test (during games)

```bash
# One-shot: fetch prices + game state, run model, print signals
python live_test.py --model data/reward_model.pkl

# Loop every 60s
python live_test.py --model data/reward_model.pkl --loop 60
```

### 6. Run the bot

```bash
python main.py   # pre-game predictions, optional paper/live orders
```

**Data collector (poll Polymarket during games):**

```bash
python data_collector.py
python data_collector.py --fast   # 5s poll for finer data
```

---

## Extending to NBA, Tennis, Other Sports

The architecture is sport-agnostic; only data and market discovery are sport-specific.

| Step                              | NHL (current)                                            | NBA / Tennis / others                                                                               |
| --------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Game data**                     | `espn_client.py` (ESPN NHL scoreboard/summary)           | Add or swap client: ESPN NBA, ATP/WTA, etc. Same idea: score/period/clock → game state.             |
| **Markets**                       | `discover_nhl_markets()` (Gamma API `tag_id=899`)        | Use Gamma (or CLOB) with the right tags/filters for NBA, Tennis, etc.                               |
| **Game records**                  | `build_game_records` uses NHL events + Polymarket prices | Reuse pipeline; feed sport-specific event/score/clock and same price format.                        |
| **In-game features**              | Score, period, time remaining, Elo, price deltas         | Keep same feature set where applicable; add sport-specific fields (e.g. sets for Tennis) if needed. |
| **Models / backtest / execution** | No change                                                | Same `train_in_game_model`, `backtest_in_game`, `live_test`, `execution`.                           |

So: implement or plug in a **sport-specific client** (like `espn_client` for NHL) and a **market discovery** function (like `discover_nhl_markets`) for that sport; keep the rest of the pipeline (dataset builder, training, strategy, execution) as is.

---

## Project layout (main files)

```
polymarket_nhl_bot/
├── config.py              # URLs, thresholds, env
├── polymarket_client.py   # CLOB + Gamma, order books, market discovery
├── espn_client.py         # ESPN NHL scoreboard / game state
├── model.py               # Elo + outcome model (pre-game)
├── in_game_strategy.py    # Signal functions (reward, price-range, dual)
├── train_in_game_model.py # Train reward/loss/price-range models
├── build_in_game_dataset.py
├── build_game_records.py
├── backtest_in_game.py    # In-game backtest from records or snapshots
├── live_test.py           # Live prices + game state → signals
├── execution.py           # Paper/live, positions, order checks
├── main.py                # Pre-game bot entry
├── data_collector.py      # Poll Polymarket during games
├── replay_game.py         # Replay a past game with model
├── COMMANDS.md            # Full command reference
├── requirements.txt
└── data/                  # game_records, snapshots, models, elo, etc.
```

---

## References

- **COMMANDS.md** — Full command reference (data, validate, train, backtest, run).
- **Polymarket:** [CLOB API](https://docs.polymarket.com/), [Gamma API](https://gamma-api.polymarket.com).
- **ESPN:** Public scoreboard/summary APIs used for NHL game state.

---

## Disclaimer

Trading on prediction markets involves risk. This project is for research and education. Use paper trading first; only deploy real funds if you understand the risks and comply with your jurisdiction’s laws.
