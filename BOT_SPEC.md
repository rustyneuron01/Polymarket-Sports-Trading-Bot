# Polymarket NHL Bot — Full Spec

This document ties together: **dataset/features** (ESPN API), **model choice**, **buy amount (position sizing)**, **when to buy/sell**, and **end-to-end bot workflow**.

---

## Main purpose: predict token price correctly (including time-to-price)

The **main purpose of this bot** is to **predict token price correctly**. That means:

1. **Fair price now** — predict the current fair token price (your probability for that outcome).
2. **Price at a future time** — predict **what the token price will be after a given number of seconds** (e.g. “in 120 seconds the token price will be 0.58”). So the model outputs: **price at t + Δt** for chosen horizons (e.g. Δt = 60s, 120s, 300s).
3. **Time to reach a price** — predict **after how many seconds** the token price will reach a target level **x** (e.g. “the token will reach 0.55 in 90 seconds”). That tells you when to expect a good sell level.

**Why this matters:** If we buy a token, we need to know not only that it’s cheap now, but **when** it will become expensive enough to sell. So we build a model that predicts correctly:
- **After what seconds** the token price becomes **x** (time-to-price), and/or  
- **What the price will be** at t + 60s, t + 120s, etc. (price-at-future-time).

Training data: historical in-game snapshots (current price, score, period, clock) and the **actual** price or **actual time** to reach a level later in the same game. **Important:** To train time-to-price you need **historical Polymarket price feeds** for NHL games (tick or snapshot data), not just ESPN game data. Plan: **collect live Polymarket order-book/midpoint data during your paper-trading period**, or source historical feeds if available. Without this, the time-to-price model cannot be trained — this is the main practical gap before coding. **Goal: reward in every game, grow total balance.** The main purpose is to **make reward in every game** so total balance grows. There is risk and you can lose money; the spec balances **capital preservation** with **seeking opportunity**. Set **realistic return expectations**: ~15 NHL games per week, but after all filters you might trade only **3–5**. Don’t abandon the strategy when it feels slow — fewer, high-conviction trades are intentional. The model you make should predict this correctly so that buy/sell timing is driven by predicted price path over time, not only by “fair value now.”

---

## 1. Dataset & feature selection (ESPN NHL API)

Use **ESPN’s public NHL APIs** to build one row per game (home vs away) with as many useful inputs as possible. Parameter and feature choices: see **§8 Research & parameters** below. Below: **where each feature comes from** and **what to compute**.

### 1.1 Data sources (ESPN + fallbacks)

**ESPN (unofficial, can change or break):** Build in **fallback handling** and consider supplementing with a more reliable source.

| Endpoint | URL pattern | Use for |
|----------|-------------|--------|
| **Scoreboard** | `https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard` (optional `?dates=YYYYMMDD`) | Upcoming games, team IDs, **in-game records & stats** (when used for past dates or live) |
| **Scoreboard (upcoming)** | Same, with `dates=` for future days | Schedule: who plays, when, home/away |
| **Team schedule** | `https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{teamId}/schedule` | Past/future games → **rest days, back-to-back, games in last 5/10** |
| **Team roster** | `https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams/{teamId}/roster` | Roster / key players (optional) |
| **Team stats** | Often embedded in scoreboard or team summary | Season-level goals, etc. |

**Fallback / supplement:** **NHL Stats API** (`api.nhle.com`) is **official** and more stable; use for schedule, standings, or game data if ESPN fails or is insufficient.

From the **scoreboard** response, each **event** has `competitions[0].competitors[]` with **home** and **away**. For each competitor you get:

- **Records**: `records[].summary` → e.g. `"28-28-2"` (YTD), `"13-13-2"` (Home), `"15-15-0"` (Road)
- **Statistics** (game-level; for historical you can aggregate): `statistics[]` → `saves`, `savePct`, `goals`, `ytdGoals`, `assists`, `points`
- **Leaders**: `leaders[]` → goals, assists, points leaders (top players)
- **Probable goalie**: `probables[]` → `probableStartingGoalie`, `athlete` (id, name), `status` (e.g. Confirmed)
- **Curated rank**: `curatedRank.current` (if present)
- **Venue**: `competitions[0].venue` → home-ice info

For **historical training**, you need past games: use scoreboard with `dates=YYYYMMDD` for many days and/or team schedule to get past results and compute the features below.

---

### 1.2 Feature list (as many as possible from ESPN)

**Game / context**

- `is_home` (0/1) — home team indicator (home team in `competitors` has `homeAway":"home"`).
- `rest_days_home`, `rest_days_away` — days since last game (from team schedule).
- `back_to_back_home`, `back_to_back_away` — 1 if team played yesterday (from schedule).
- `games_last_5_home`, `games_last_5_away` — number of games in last 5 days (fatigue).
- `games_last_10_home`, `games_last_10_away`.

**Team record (from scoreboard `records` or your own aggregation)**

- `wins_home`, `losses_home`, `ot_home` — parse from "W-L-OT" (e.g. 28-28-2).
- `wins_away`, `losses_away`, `ot_away`.
- `win_pct_home`, `win_pct_away` — e.g. wins / (wins + losses) ignoring OT or include OT as half-win.
- `points_home`, `points_away` — if you have standings (2 per win, 1 per OT loss).
- `home_record_wins`, `home_record_losses` — home-specific (from "Home" record).
- `road_record_wins`, `road_record_losses` — away-specific (from "Road" record).

**Season-level stats (YTD from scoreboard or team stats)**

- `ytd_goals_home`, `ytd_goals_away` — from `statistics` name `ytdGoals`.
- `ytd_goals_per_game_home`, `ytd_goals_per_game_away` — YTD goals / games played.
- `ytd_assists_home`, `ytd_assists_away`.
- `ytd_points_home`, `ytd_points_away`.

**Goaltending**

- `probable_goalie_confirmed_home`, `probable_goalie_confirmed_away` — 1 if status is Confirmed.
- `goalie_save_pct_home`, `goalie_save_pct_away` — from **probable starter’s** season stats. **Note:** ESPN probable goalie save % is often **stale or missing** pre-game; goalie is one of the highest-impact features in NHL prediction. **Prioritize a reliable source**: e.g. **NHL API** (`api.nhle.com`), **Natural Stat Trick**, or another provider with up-to-date goalie stats.
- If you have **saves** / **savePct** in the event for the game, use for in-game models; for pre-game, use season or recent form from a reliable source.

**Top players (from `leaders`)**

- `top_scorer_goals_home`, `top_scorer_goals_away` — goals leader value.
- `top_scorer_points_home`, `top_scorer_points_away` — points leader value.
- `has_star_goalie_home`, `has_star_goalie_away` — e.g. 1 if probable goalie’s save % above league average (if you have that data).

**Derived / strength**

- `goal_diff_home`, `goal_diff_away` — goals for − goals against (if you have both).
- `form_last_5_home`, `form_last_5_away` — wins in last 5 (from schedule + results).
- `form_last_10_home`, `form_last_10_away`.
- `elo_home`, `elo_away` — if you implement Elo from historical results (not from ESPN; you compute).
- `rank_home`, `rank_away` — from `curatedRank.current` if present, or from standings.

**Injuries — who is out matters (top vs normal player)**

- **Top / star player injury** (high impact): e.g. top-line forwards, #1 goalie, top defensemen. Weight these heavily.
  - `top_player_injured_home`, `top_player_injured_away` — 1 if any designated “top” player is out (you define: e.g. top 3 scorers, starting goalie).
  - `top_player_injury_count_home`, `top_player_injury_count_away` — count of such players out.
  - Optionally: `top_player_impact_score_home` — e.g. sum of (player’s season points share or goals share) for injured top players.
- **Normal / role player injury** (lower impact): depth players, 4th line, backup goalie.
  - `role_player_injury_count_home`, `role_player_injury_count_away` — count of non‑top players out.
- **Rule for the bot**: **Do not treat top player injury as “uncertain” or a reason to skip.** That team may still be strong enough to win without their star (depth, system, etc.). Use **top player injury** and **role player injury** as **features** so the model learns the effect; then **predict all games** including when stars are out. If the model says you have edge (e.g. the team is still favored), **bet**. Only skip when there is genuine uncertainty (e.g. model not trained on this situation, or critical data missing), not simply because a top player is out.

**Time remaining (before game / before buy)**

- `hours_until_game` — hours from now until scheduled start. Use to avoid buying when too little time is left (lineup/news can change).
- `days_until_game` — same in days.
- **Use in rules**: e.g. do **not** place a new buy if `hours_until_game < 2` (or 1) unless you are very confident and have fresh injury/lineup info. “Time remaining” helps you **only buy when you have enough time to be sure** (e.g. no last‑minute scratch).

**In-game features (score + time — use together)**

- `period` — current period (1, 2, 3, OT).
- `clock_seconds_remaining`, `time_remaining_in_game` — time left in period/game.
- `score_home`, `score_away`, `score_differential` — current score (home − away).
- **Always use current score together with time remaining**: e.g. “down 2 with 5 min left” is very different from “down 2 with 15 min left.” Your fair token price and cut-loss rules must depend on **both** score and time.

**Storm loss (bleed) feature — time passes but token price doesn’t go up**

Risk: as **time proceeds**, if the **token price does not go up** (or goes down), you can hold to the end and **lose all your funds** when the outcome resolves against you. You need to detect and react to this.

- `token_price_at_entry` — average price you paid for your current position (per outcome).
- `current_token_bid` — best bid now (what you could sell at).
- `minutes_elapsed_since_entry` — how long you’ve been in the position (or minutes elapsed in game).
- `storm_loss_risk` (derived): 1 if time is passing but token price is flat or down vs entry (e.g. `current_token_bid <= token_price_at_entry` and `time_remaining_in_game` is low). Use this to trigger **cut loss** or **reduce size** instead of holding to resolution and losing everything.
- Optionally: `position_unrealized_pnl` — (current_token_bid − token_price_at_entry) × size; if negative and time is running out, consider exiting.

**Target (for training)**

- `home_win` — 1 if home team won (regulation or OT/SO), 0 otherwise. From `competitors[].winner` in finished games.

For **pre-game** rows use only pre-game known values. For **in-game** prediction, add in-game features (period, clock, score, time_remaining_in_game) and **storm loss** signals; train or calibrate on historical in-game snapshots so your predicted token price and cut-loss behavior reflect score + time. The bot does **both** pre-game and in-game trading; token moves during the game, so buy/sell during the game is central.

---

## 2. Model selection and predicting token price correctly

- **Purpose**: The bot must **predict token price correctly**. That includes:
  - **Fair price now**: P(home win), P(away win) → predicted fair token prices (e.g. 0.55).
  - **Price at future time**: For a given **Δt (seconds)**, predict **token price at t + Δt** (e.g. “in 120 seconds the price will be 0.58”). Train on historical snapshots: at time t you have (score, period, clock, current_price); label = actual price at t+60, t+120, etc. in the same game.
  - **Time to reach price**: Predict **after how many seconds** the token price will reach a target **x** (e.g. “price will reach 0.55 in 90 seconds”). Train on historical data: from each in-game snapshot, measure how many seconds until price first hit a given level (or bin into 30s/60s/120s/never). The model you make should predict this correctly so we know **when** to expect a sell level after we buy.
- **Recommendation**:
  - **Elo first**: Invest in **Elo early** — a well-tuned Elo is often competitive with more complex models and gives an interpretable baseline before adding XGBoost. Treat it as core, not optional.
  - **Then**: **Logistic regression** on features + Elo; then **Gradient boosting** (XGBoost / LightGBM) for non-linear effects.
  - **Optional**: **Random forest** or **small neural net** for price-at-future-time or time-to-price if you have enough in-game history.
- **Critical**: **Calibrate** so predicted prices (and if you use them, predicted time-to-price) match reality. NHL is noisy; even well-calibrated models often see only 3–5% edge; **live trading may underperform backtest** especially early. Re-check calibration on holdout and only trade when the model’s predictions are **reliable**.
- **Model is the key**: Invest in feature quality, calibration, and backtest; include **time-to-price** or **price-at-future-time** so we know when the token will reach a level after we buy.

---

## 3. Buy amount (position sizing) — formula

You want **trading amount per game** to depend on **edge**, **win/loss rate**, and **total balance**. The **standard, well-known approach** for this is the **Kelly Criterion** (used in prediction markets and sports betting): it maximizes long-term growth and avoids ruin when you have positive edge.

- **Edge**: `edge = your_probability − market_price` (for that outcome). Only trade when edge > 0 (and after fees).
- **Bankroll**: Use **total balance** (or a dedicated trading balance) as the base.
- **Formula — Kelly Criterion (prediction-market form):**
  - **Full Kelly** (fraction of bankroll): `f* = (p − price) / (1 − price) = edge / (1 − price)`  
    where `p` = your probability, `price` = token price you pay (e.g. best ask). You pay `price` for a contract that pays $1 if win; edge = `p − price`; denominator `(1 − price)` is the odds part. Only bet when `p > price`.
  - **Same as classic form**: `f* = (b * p − q) / b` with `b = (1 − price) / price`, `q` = 1 − p.
  - **Fractional Kelly** (recommended): use e.g. **25–50% of full Kelly** to reduce volatility:  
    `trade_amount = balance * (kelly_fraction * f*)`.
  - **Cap**: Never bet more than e.g. **10–15% of balance** on one outcome:  
    `trade_amount = min(computed_amount, max_position_pct * balance)`.
- **Win/loss rate**: Kelly already uses your **estimated** win probability `p`. If your **realized** win rate is lower than your model’s `p`, your edge is overstated — so either improve the model/calibration or use a **smaller** Kelly fraction (e.g. 0.25) and a **stricter** min edge threshold. You can also track **realized hit rate** and scale down size when it’s below expectation.

**Summary**:  
`trade_amount = balance * min(max_position_pct, kelly_fraction * max(0, f*))`, with `f*` from Kelly using your prob and market price.

**Position tracking / portfolio state:** The spec is per-trade; you also need **portfolio-level rules** when holding multiple open positions across games. Define: **max open positions at once = N** **Concrete defaults: max 4 open positions, max 40% of balance deployed** (tune as needed; without defaults this won't get implemented). Also: **max total capital deployed = X%** of balance so you don’t over-expose on a busy night. Before opening a new position, check that you’re under these caps.
**Partial fills:** When a limit order partially fills (e.g. order $50, only $20 fills), position tracking and Kelly sizing must use **filled amount only** for size and cost basis; add partial-fill logic from the start — common implementation bug.

---

## 4. When to buy and when to sell (price rules)

- **Buy token (team outcome)** when:
  - **Market ask** (price to buy) is **below** your fair value minus a threshold:  
    `best_ask <= your_prob − buy_threshold`  
  - Example: your_prob = 0.55, buy_threshold = 0.03 → buy when ask ≤ 0.52.
- **Sell token** when:
  - **Market bid** (price you can sell at) is **above** your fair value plus a threshold:  
    `best_bid >= your_prob + sell_threshold`  
  - Example: your_prob = 0.55, sell_threshold = 0.03 → sell when bid ≥ 0.58.
- **Thresholds**: Start with **2–5%** (0.02–0.05); increase if you want fewer but higher-conviction trades.
- **Transaction fees (bake in explicitly):** Polymarket charges a fee on trades (currently **~2% on the taker side**). Your **minimum edge threshold** must cover **both entry and exit**: e.g. **at least 4–5%** (2% in + 2% out) so that after fees you still have positive edge. This significantly reduces the number of tradeable opportunities but is required; don’t use a 2% threshold and ignore fees.
- **Both teams**: Apply the same logic **per outcome**. You can have a buy signal on one team and a sell signal on the other in the same game.

---

## 4.2 In-game strategy: current score + time remaining; avoid storm loss (don’t lose all funds)

**Current score and time remaining together**

- Your fair token price and decisions must depend on **both** **current score** and **time remaining**. Examples: “our team is up 2 with 2 min left” → token should be high, consider taking profit; “our team is down 2 with 2 min left” → token is low and may go to zero at resolution → **cut loss** instead of holding.
- Use **score_differential** and **time_remaining_in_game** in the model and in rules: e.g. if `score_differential` is against your position and `time_remaining_in_game` is below a threshold (e.g. 5–10 min), treat it as high risk and reduce or exit.

**Storm loss avoidance — time proceeds but token price doesn’t go up**

- **Risk**: Time passes, but the token price does **not** go up (or goes down). If you hold to the end, you can **lose all your funds** when the game resolves against you.
- **Strategy**:
  - Track **token_price_at_entry** and **current_token_bid** (and optionally **position_unrealized_pnl**). If **time_remaining_in_game** is decreasing and **current_token_bid** is at or below your entry (or below a minimum threshold), treat this as **storm loss** (bleed) risk.
  - **Rule**: Do **not** hold blindly. If storm loss risk is high (e.g. time left &lt; X min and bid &lt; entry, or score is against you and time is short), **cut loss**: sell at the best bid to cap the loss instead of holding to resolution and losing everything.
  - **Concrete cut-loss rule (adjustable default):** If `time_remaining_in_game < 8 min` **and** `unrealized_pnl < -15%` of that position’s value, **sell** (or sell a fraction) to cap the loss. If bid &lt; entry and time &lt; 5 min, also consider selling. Use 8 min and -15% as a starting default; tune from backtests or paper trading.
- **Summary**: Care about **current score** and **time remaining** in every in-game decision; and if time is proceeding but token price isn’t going up, **exit or reduce** before the game ends so you don’t lose all funds.

**Oversold / buy the dip (market overreaction)**

- **Case:** Two teams are similar (e.g. token was 56:44 at start). During the game, **one team's token suddenly drops a lot** (e.g. to 20) with **much time still remaining**. The market is often **overreacting**; that team can still come back, so the token price **should revert higher**. **Action: buy that token** — cheap vs your fair value; you're betting on mean reversion or comeback.
- **Guard against falling knives:** The drop may be **justified** (e.g. #1 goalie injured mid-game, 3–0 in the 2nd with one-sided momentum). Only trigger oversold buy if your **model's win probability is meaningfully above the token price** — e.g. model says 30% but token is at 20% → buy (market overreaction); model also says 20% → **don't buy** (no edge; genuine shift in game state). The model must distinguish "overreaction" from "genuine shift"; if model and market agree the team is dead, do not buy the dip.
- **Rule:** If **current token price** is **well below** your in-game fair value **and** **time_remaining_in_game** is **high** (e.g. &gt; 15–20 min) **and** your **model's probability is meaningfully above** the token price (e.g. edge ≥ your min threshold), treat as **oversold / buy opportunity**. Require your model still gives the team a non-trivial win probability (e.g. not &lt; 15%); then buy when ask is below your fair value by a clear margin. Handle this explicitly so the bot **buys the dip** when the market overreacts, not only cuts loss.
- **All situations to handle:** (1) **Pre-game mispriced** — buy/sell by edge. (2) **In-game storm loss** — cut loss when time short, position underwater. (3) **In-game oversold** — token drops sharply with lots of time left → buy the dip. (4) **In-game take-profit** — token high, time short, score in your favor → sell. Handle each so the bot both preserves capital and captures reward.

---

## 4.3 Safety first: skip only when genuinely uncertain (not because of injuries)

The bot should be **safe**; the goal is **not to lose**. **Do not treat "top player out" or similar as automatic uncertainty.** A team can still win without their star; the model should predict that and we bet when we have edge. Only skip when **genuinely** uncertain.

- **When in doubt, skip**: If you are **genuinely** not confident (e.g. model unreliable, critical data missing), **do not trade** that game. Ignoring a game is better than taking a bad trade.
- **Skip conditions** (examples — tighten as needed):
  - **Unreliable prediction**: If your model’s predicted probability is in a range where it has been poorly calibrated in the past (e.g. extreme probabilities), or if required features are missing, **skip**.
  - **Edge too small**: Only trade when edge is above your threshold; if edge is marginal, **skip**.
  - **In-game**: If live score/period/clock are stale or missing during the game, **skip** that update cycle (don't trade on old in-game state).
- **Do NOT skip** just because a top player is out, or because of injuries or “other things.” Predict **all** games; use injury (and everything else) as **features**. Bet when the model says you have edge — including when a star is out but the team is still strong enough to win.
- **Result**: Fewer trades only when genuinely uncertain. Predict and bet when we can, including games with injuries.

---

## 4.4 Model stability: predict skipped games and prediction-mode flag

**Predict unsure/skipped games too (don’t just “unpredict”).** When you **skip** a game (don’t trade), still **run the model and record the prediction** for that game. Later, when the result is known, you can compare: “we skipped this game but our model said 55% home — did home win?” This lets you **check how the model would have predicted** skipped games and **confirm model stability** and calibration over time. So: **skip = do not trade**, but **still predict and log** (game id, prob_home, prob_away, skip_reason, and after the game: actual_winner). Use this log to monitor calibration on both traded and non-traded games.

**Prediction-mode flag.** Add a **flag** (e.g. `predict_all_games` or `predict_mode`) so you can choose:

- **Predict all games** (`predict_all_games = true`): Run the model on **every** game (including those that will be skipped). Output or log predictions for all games. Use this to **confirm model stability** and to build a full prediction log for calibration checks. Trading logic still only executes on games that pass the safety filter; the flag only controls whether you **also** predict and record the skipped ones.
- **Predict only certain games** (`predict_all_games = false`): Run the model only for games that **pass the safety filter** (or only games you might trade). No prediction output for skipped games. Use this when you only care about predictions for games you might act on.

Default recommendation: set **predict_all_games = true** so you always have predictions for skipped games and can validate model stability.

---

## 5. End-to-end bot workflow (pre-game and during game)

The bot trades **before** the game and **during** the game; token price moves as the game proceeds, so buy/sell happens in both phases.

1. **Data**
   - **Upcoming games**: ESPN scoreboard (and/or schedule) → list of (home_team, away_team, date, start time).
   - **Features**: For each game, build the feature row from ESPN (records, YTD stats, probable goalies, rest, back-to-back, leaders, **injuries by tier**: top vs role player, **games_remaining_season**).
2. **Predict (token price)** — controlled by **predict_all_games** flag:
   - If **predict_all_games = true**: Run the model on **every** game → `prob_home`, `prob_away`. Log each prediction (including for games that will be skipped) with game id, skip_reason if any; after the game, record actual_winner to check model stability.
   - If **predict_all_games = false**: Run the model only for games that pass the safety filter (see step 3).
3. **Safety filter — skip if not sure**
   - If any skip condition is met, **do not trade** this game. When **predict_all_games = true**, you still have a prediction and log for this game (step 2); when false, you may not have run the model for it.
4. **Predict (for trading)** — if you didn’t already in step 2, run model → `prob_home`, `prob_away` for games that passed the filter. Only use these for trading if you trust calibration.
5. **Market data**
   - Get Polymarket order books for “Home wins” and “Away wins” (token IDs from your market discovery). **Market selection:** When multiple Polymarket markets exist for the same NHL game, pick the one with **highest liquidity** or **tightest spread**.
   - Read **best bid** and **best ask** for each token.
6. **Signals**
   - For **home**: buy if `ask_home <= prob_home − buy_threshold`; sell if `bid_home >= prob_home + sell_threshold`.
   - For **away**: same with `prob_away`, `ask_away`, `bid_away`.
7. **Size**
   - For each BUY signal: `amount = balance * min(max_position_pct, kelly_fraction * f*)` with Kelly `f*` from your prob and the **ask** price.
   - **Liquidity check (required):** Before sizing, **verify there is enough liquidity at your target price**. A Kelly-sized order into a **thin book** will move the price against you and destroy your edge. If depth at best ask (or your limit) is less than your desired size, reduce size or use a limit that doesn’t eat through the book.
   - For SELL: use current position or the same idea with bid (typically you sell what you hold).
8. **Execute**
   - Place limit orders at your target price (or take existing liquidity if it meets your threshold).
9. **During the game (loop until final)**: Poll for **period**, **clock**, **score_home**, **score_away**, **time_remaining_in_game** → update fair token price and **predicted price at t+Δt** (or **time to reach price x**) using in-game model → get current order books → buy/sell as token moves. **Storm loss check**: if time is passing but token price is not going up (bid ≤ entry, or score against you with little time left), **cut loss** (sell). **Note:** In-game order books are often **thin and move fast**; 1–2 min polling may be too slow to capture edge and you may face slippage. Consider **pre-game trading only** first; add in-game once pre-game is profitable and you have low-latency data.

---

## 6. Checklist summary

| Area | What to do |
|------|------------|
| **Dataset** | ESPN scoreboard + team schedule (+ roster/standings if available); build one row per game with all features above. |
| **Features** | As many as possible: records, YTD goals/points, rest, back-to-back, probable goalie, leaders, home/road split, form, Elo; **injuries by tier**; **games_remaining_season**; **in-game**: period, clock, **score**, **time_remaining_in_game**; **storm loss**: token_price_at_entry, current_token_bid, storm_loss_risk (time passes but price doesn’t go up). |
| **Model** | **Main purpose: predict token price correctly.** Include **price at t+Δt** (e.g. price in 60s, 120s) and/or **time to reach price x** (seconds until token hits a level). Invest in **Elo early**; then logistic/XGBoost; **calibrate**; expect 3–5% edge, live may underperform backtest. Pre-game and in-game (period/score/clock). |
| **Injuries** | Separate **top/star player injury** (high impact) from **normal/role player injury** (lower impact). Use as **features**; **do not** skip just because a top player is out — predict and bet when model says edge (team can still win without star). |
| **Time** | Use **games_remaining_season** for context. For trading: use **in-game** time and **current score** together (**period**, **clock**, **score**, **time_remaining_in_game**); care about score + time in every in-game decision. |
| **Storm loss** | **Cut loss** (default: `time_remaining < 8 min` and `unrealized_pnl < -15%` → sell). **Avoid losing all funds**: if time proceeds but token price doesn’t go up (or goes down), **cut loss** (sell) before resolution; use **score + time** to decide when to exit (e.g. down 2 with 5 min left → exit). |
| **Oversold** | Token drops sharply with lots of time left → **buy the dip** only if **model prob meaningfully above token price** (guard against falling knives; if model says 20% and token 20%, don't buy). |
| **Fees** | ~2% taker; **min edge ≥ 4–5%** (entry+exit). **Portfolio**: concrete defaults **max 4 open positions, max 40% deployed**; **partial fills**: use filled amount only for position/cost basis. **Market selection**: when multiple markets per game, use highest liquidity or tightest spread. |
| **Safety** | **When genuinely uncertain, skip that game.** Skip only on: unreliable prediction, **critical** features missing, or stale in-game data. **Do not** skip just because a top player is out or “other things” — predict all games, use injury as feature, bet when we have edge. |
| **Model stability** | For **skipped** games, still **predict and log** (game id, prob_home, prob_away, skip_reason; after game: actual_winner) to confirm model stability. **Flag** `predict_all_games`: if **true**, predict and log **all** games; if **false**, predict only games that pass the safety filter. Default: **true**. |
| **Buy amount** | Kelly (or fractional Kelly), capped by max % of balance; use your prob and market price. |
| **When buy** | When best_ask ≤ your_prob − buy_threshold; only when predicted token price is trusted. |
| **When sell** | When best_bid ≥ your_prob + sell_threshold. |
| **Bot** | **Pre-game**: upcoming games → features (incl. games_remaining_season, injury tier) → safety filter → predict (fair price + price-at-future-time or time-to-price) → order books → **liquidity check** → signals → size → execute. **During game**: poll period/clock/score → in-game predict → order books → buy/sell as token moves → repeat until final. |
| **Liquidity** | Before sizing, **check depth** at target price; thin book + large order = slippage and destroyed edge. |

This spec gives you a single reference: **dataset and features (incl. injury tier, games remaining, in-game period/score/time, storm loss)** → **model that predicts token price correctly (fair price now + price at t+Δt and/or time to reach price x; needs historical Polymarket price data)** → **in-game: score + time; storm loss cut-loss (8 min / -15%); oversold buy-the-dip** → **fees 4–5% min edge; portfolio caps; safety first; liquidity check** → **buy/sell** with Kelly. **Build order:** data pipeline → Elo+logreg → pre-game paper trade → collect Polymarket prices → train time-to-price → add in-game.

---

## 7. Review notes and improvements (what to watch)

These points come from external review and should be kept in mind when building and running the bot.

**Model calibration is harder than it sounds.** NHL outcomes are noisy; even well-calibrated models often find only **3–5% edge**. Be prepared for **live trading to underperform backtest**, especially early. Paper trade for **2–4 weeks** before committing real capital.

**ESPN API limitations.** The public ESPN API is **unofficial and undocumented**; it can change or break without notice. Build in **fallback handling** and consider supplementing with **NHL Stats API** (`api.nhle.com`), which is official.

**In-game trading is very competitive.** Live Polymarket order books during NHL games are **thin and move fast**. A 1–2 min polling loop may be **too slow** to capture edge; you may face **slippage**. Consider whether **pre-game trading alone** gives enough opportunity first; add in-game complexity only after pre-game is working and you have low-latency data.

**Elo is worth investing in early.** Treat Elo as a **core baseline**, not optional. A well-tuned Elo is often competitive with more complex models and gives you an interpretable starting point before adding XGBoost or time-to-price models.

**Goalie data gap.** ESPN probable goalie save % is often **stale or missing** pre-game. Goalie is one of the **highest-impact** features in NHL prediction. Prioritize a **reliable source**: e.g. NHL API, Natural Stat Trick, or another provider with up-to-date goalie stats.

**Liquidity check before sizing.** Before placing a trade, **verify there is enough liquidity** at your target price. A Kelly-sized order into a thin book will **move the price against you** and destroy your edge. Reduce size or adjust limit if depth is insufficient.

**Time-to-price model needs historical Polymarket data.** To train "how many seconds until token reaches 0.55" you need **historical Polymarket price feeds** for NHL games (order-book snapshots or midpoints over time), not just ESPN game data. Without this, the time-to-price model cannot be trained. **Data collection is a Day 1 task:** build the **Polymarket data collector before paper trading starts**. During weeks 1–3 of paper trading you must be **actively saving** order-book snapshots (price, bid, ask, timestamp) for **every** NHL game — even games you don't trade. Every day you delay is data you can never recover. **Suggested:** a script that **polls Polymarket order books every 30–60 seconds** for all active NHL games and saves to a database. Build this first; then paper trade with it running.

**Suggested build order**

1. **Polymarket data collector first** — poll order books every 30–60 sec for all active NHL games; save (price, bid, ask, timestamp, game/market id) to DB. Run this from Day 1 so you have data for time-to-price training.
2. ESPN + NHL API data pipeline (schedule, standings, goalie, etc.).
3. Elo model + logistic regression, calibrated; pre-game fair price only.
4. Pre-game bot only (no in-game), paper trade **3–4 weeks** (with data collector running the whole time).
5. Train **time-to-price** (or price-at-future-time) model on the collected Polymarket + game data.
6. Add in-game trading once #4 and #5 are done.

**Suggested next step:** Before writing any model code, **build the Polymarket data collector first** — a script that polls order books every 30–60 seconds for all active NHL games and saves to a database. Every day you delay is data you can never recover.

---

## 8. Research & parameters (sources)

Values used in this codebase; no extra research required.

**Elo (elo.py):** K=6, home +50 (FiveThirtyEight NHL). Harvitronix uses K=8, home +42. Win prob = 1 / (1 + 10^(-(elo_home + home_adj - elo_away)/400)).

**Pre-game:** Team strength (Elo) and home ice are primary. Starting goalie moves line ~0.5–1 goal; injury as feature, don’t skip. Goalie save % high impact but ESPN often stale — prefer NHL API.

**In-game:** Score differential and time remaining are main drivers. Manpower (PP/PK) when we have play-by-play. Sources: Pettigrew (Harvard), HockeyStats, in-play studies.

**Strategy (config.py):** MIN_EDGE_PCT 5%; storm loss 8 min / -15%; no momentum buy.
