# Deribit Smile Research Agent — Cursor Brief

## Objective

Build a research-first Python system that tests whether Deribit BTC and ETH option volatility-smile features, compared against historical/realized volatility of the underlying, can produce useful recommendations for straddles and vertical spreads on historical minute data. The system must support an agentic loop where an LLM can rewrite only the feature/signal calculator, while the data pipeline, splitting, simulator, and artifact logging remain stable.

---

## Non-negotiable constraints

- This is **research infrastructure first**, not a production trading bot.
- Use **chronological splits only**. Never shuffle time series.
- Keep one final untouched **test set** at the end of history.
- Inside training, use **walk-forward validation** with a configurable time gap between train and validation windows.
- The only AI-rewritable code must live in a **sandboxed calculator module**.
- The simulator, split logic, data normalization, and artifact ledger are **not** to be rewritten by the LLM.
- Start with a **30-day quick run** on CPU.
- After correctness is established, optimize for larger training runs and optional GPU acceleration.
- The design must leave a clean path to **live paper trading** with Deribit WebSocket market data.

---

## Repo setup

Create a new repo and initialize it like this:

```bash
mkdir deribit_smile_agent
cd deribit_smile_agent
git init
conda create -n deribit-smile python=3.11 -y
conda activate deribit-smile
mkdir -p app/{data,split,sim,opt,llm,live,utils} sandbox configs reports tests scripts artifacts prompts
touch README.md .gitignore
```

Recommended `.gitignore`:

```gitignore
__pycache__/
*.pyc
.env
.venv/
artifacts/
reports/
data/
.parquet
.ipynb_checkpoints/
.DS_Store
```

Install the first-pass dependencies:

```bash
pip install pandas numpy pyarrow scikit-learn pyyaml typer rich jinja2 pydantic pytest
pip install websockets httpx
```

Optional later-stage acceleration:

```bash
# do this only when CPU version is correct
# follow the RAPIDS install matrix for your CUDA/runtime
# and then enable cudf.pandas in a config-controlled way
```

Create the basic file tree:

```text
deribit_smile_agent/
├─ README.md
├─ .gitignore
├─ configs/
│  ├─ quick.yaml
│  ├─ train_full.yaml
│  └─ live_paper.yaml
├─ app/
│  ├─ main.py
│  ├─ data/
│  │  ├─ loader.py
│  │  ├─ normalize.py
│  │  └─ schema.py
│  ├─ split/
│  │  └─ time_split.py
│  ├─ sim/
│  │  ├─ simulator.py
│  │  ├─ pricing.py
│  │  ├─ positions.py
│  │  └─ metrics.py
│  ├─ opt/
│  │  └─ search.py
│  ├─ llm/
│  │  ├─ proposer.py
│  │  ├─ critic.py
│  │  ├─ prompts.py
│  │  └─ artifact_log.py
│  ├─ live/
│  │  ├─ deribit_ws.py
│  │  └─ paper_router.py
│  └─ utils/
│     ├─ io.py
│     ├─ hashing.py
│     └─ clock.py
├─ sandbox/
│  ├─ calculator.py
│  └─ runner.py
├─ prompts/
│  ├─ rewrite_calculator.md
│  └─ critic_review.md
├─ tests/
│  ├─ test_splits.py
│  ├─ test_simulator.py
│  ├─ test_sandbox_api.py
│  └─ test_missing_close_rule.py
├─ scripts/
│  ├─ run_quick.py
│  ├─ run_train.py
│  ├─ run_test.py
│  └─ run_live_paper.py
├─ reports/
└─ artifacts/
```

---

## What the system must do

### 1. Read historical data from a folder

The app must accept a folder path that contains minute-level Deribit historical data. I will provide the folder link/path later.

Implement a loader that:
- recursively scans the folder,
- detects supported file types,
- reads raw files into pandas,
- normalizes timestamps,
- parses instrument metadata,
- and writes normalized parquet caches.

The system must support at minimum:
- options data,
- underlying BTC/ETH spot or index data,
- and any auxiliary fields available in the dataset.

The normalized options table should aim to include fields like:
- `timestamp`
- `instrument_name`
- `asset`
- `expiry_ts`
- `strike`
- `option_type`
- `mark_price`
- `mark_iv`
- `underlying_price`
- `best_bid_price`
- `best_ask_price`
- `open_interest`
- Greeks when present

Deribit’s public docs confirm the availability of fields such as `mark_iv`, `underlying_price`, Greeks, strike, expiration timestamp, open interest, and contract-size-related metadata. citeturn8view0turn8view1turn8view2turn8view3turn8view4turn14view0turn14view2

---

### 2. Split the data into training and test sets

Use **strict time-based splitting**.

Requirements:
- final test set = last contiguous block of history,
- training set = everything before test,
- within training, use walk-forward validation,
- add configurable `gap` between train and validation to reduce leakage,
- default `gap` should be at least as large as the largest holding horizon.

Use a splitter compatible with scikit-learn `TimeSeriesSplit` semantics. `TimeSeriesSplit` is designed for ordered data and supports a `gap` parameter specifically for separation between train and test windows. citeturn7view0

---

### 3. Compute volatilities, smiles, stats, and optional technical signals

All research logic must be concentrated in the sandbox calculator.

The calculator should compute:
- underlying log returns,
- realized/historical volatility over configurable windows,
- ATM IV,
- smile slope,
- smile curvature / convexity,
- skew,
- IV term structure,
- IV minus realized-volatility spread,
- changes in IV and surface shape,
- optional technical or microstructure signals:
  - momentum,
  - rolling return breakout,
  - bid/ask width,
  - OI change,
  - volume change,
  - moneyness concentration.

Realized volatility from intraday data is a strong forecasting baseline, and implied volatility versus realized volatility is a standard comparison point in volatility research. citeturn11view0turn11view2turn11view3turn6view13

Smile/skew/convexity are standard summaries of the implied-volatility curve across strikes. citeturn6view8turn6view9

Handle sparse slices gracefully:
- if too few strikes exist for fitting smile metrics, return `NaN`,
- never crash the pipeline due to missing slices.

---

### 4. Sandboxed module to be rewritten by the AI agent

Only this module may be rewritten by the agent:

- `sandbox/calculator.py`

It must expose exactly this API:

```python
def build_features(options_df, underlying_df, config) -> "pd.DataFrame":
    ...

def generate_signals(features_df, config) -> "pd.DataFrame":
    ...

def describe_params() -> dict:
    ...
```

The sandbox must run **out of process** using a subprocess wrapper with:
- no network access,
- CPU and memory limits,
- timeout,
- read-only inputs,
- temporary output directory,
- strict import allowlist.

Do **not** rely on `RestrictedPython` as the only security layer. Its own documentation states it is not a secure sandbox. Use process isolation instead. citeturn13search0turn13search1

---

### 5. Simulate positions and close after 1 day and 3 days

The simulator must support, at minimum:
- long/short straddles,
- vertical spreads.

Strategy definitions should follow standard options terminology: a long straddle is a call plus a put with the same strike and expiry; a vertical spread uses same-type options with the same expiry and different strikes. citeturn6view10turn19view0

Simulation rules:
- open at the first available record at or after signal time,
- close at the first available record at or after:
  - `t + 1 day`
  - `t + 3 days`
- if exact close timestamp does not exist, take the first available after target,
- if the gap from target close to the first available row is greater than 3 days, return `NaN` for that close result,
- if expiry occurs before intended close, apply an explicit expiry-handling rule; if needed data is absent, return `NaN`.

Produce separate result columns for:
- 1-day close PnL,
- 3-day close PnL.

Execution model should be configurable:
- `mark`
- `mid`
- `crossed_bid_ask_plus_slippage`

Fees must be configurable and not hardcoded to zero. Deribit lists BTC and ETH options fees at 0.03% of the underlying per contract, capped at 12.5% of option price, while account-specific fee structures can differ. citeturn6view5turn5search3

---

### 6. Feed results and current sandbox code to the LLM

After each quick or validation run, build an artifact bundle containing:
- current config parameters,
- current sandbox source,
- sandbox hash,
- top-level metrics,
- fold-level metrics,
- per-strategy metrics,
- per-horizon metrics,
- NaN-close rates,
- top trades,
- worst trades,
- runtime errors,
- data coverage summary.

Send this bundle to the LLM and request exactly two outputs:
1. revised parameter proposal,
2. rewritten `sandbox/calculator.py`.

The LLM is allowed to change:
- feature engineering,
- signal generation,
- thresholds,
- parameter defaults.

The LLM is not allowed to change:
- data readers,
- split logic,
- simulator,
- reporting,
- artifact log,
- sandbox runner contract.

---

### 7. Execution stages

Implement three stages:

#### Stage A — quick research run
- use first configured 30-day slice of training data,
- run on CPU,
- optimize for correctness and inspectability,
- produce a detailed report.

#### Stage B — full training study
- use full training history,
- run walk-forward validation,
- perform parameter search,
- later allow optional GPU acceleration.

RAPIDS `cudf.pandas` is appropriate for later acceleration because it preserves a pandas-like workflow with CPU fallback. citeturn6view7

#### Stage C — final test
- lock best config and best accepted sandbox version,
- run one untouched final backtest on the held-out test set,
- save a final report.

---

### 8. Be ready for online paper simulation

The architecture must support a next step where historical folder loading is replaced by live Deribit market data ingestion.

For that, keep a stable interface between:
- market data ingestion,
- feature builder,
- signal generator,
- simulator/paper router.

For live mode:
- prefer WebSocket subscriptions over polling,
- separate market-data flow from private/trading flow,
- keep paper execution isolated from research runs.

Deribit recommends WebSocket for real-time data and warns against excessive polling. citeturn10view0turn10view1turn9view0

---

## Deterministic promotion rules

A rewritten sandbox candidate is accepted only if:
- it compiles,
- it passes unit tests,
- it respects the API contract,
- it runs within timeout and resource limits,
- it produces deterministic results on a fixed seed and fixed data slice,
- it improves the selected validation objective,
- it does not worsen NaN-close rate, error rate, or turnover beyond configured limits.

The critic step must reject:
- extra filesystem access,
- network access,
- unsupported imports,
- hidden randomness,
- changes outside sandbox scope.

---

## Metrics to report

At minimum report:
- total PnL,
- average PnL per trade,
- median PnL,
- hit rate,
- drawdown,
- Sharpe-like score,
- turnover,
- NaN-close rate,
- trades count,
- metrics split by:
  - asset,
  - strategy family,
  - holding horizon,
  - expiry bucket,
  - moneyness bucket.

Always log:
- config hash,
- sandbox hash,
- run timestamp,
- input dataset summary.

---

## Config expectations

Create these config files:

### `configs/quick.yaml`
For 30-day CPU smoke and research run.

### `configs/train_full.yaml`
For full training plus walk-forward optimization.

### `configs/live_paper.yaml`
For later live market-data ingestion and paper execution.

Each config should include:
- input data path,
- assets list,
- train/test cutoff policy,
- walk-forward parameters,
- holding horizons,
- strategy families,
- execution model,
- fee model,
- allowed sandbox imports,
- timeout/resource caps,
- optimization search space,
- report output paths.

---

## CLI expectations

Implement a simple CLI, for example:

```bash
python -m app.main run-quick --config configs/quick.yaml
python -m app.main run-train --config configs/train_full.yaml
python -m app.main run-test --config configs/train_full.yaml
python -m app.main run-live-paper --config configs/live_paper.yaml
```

---

## First implementation priority

Build in this order:

1. data loader and schema normalization,
2. chronological splitting,
3. stable simulator with missing-close rule,
4. minimal sandbox calculator,
5. reporting and artifact logging,
6. LLM proposer/critic loop,
7. optimization,
8. live paper adapter.

Do not start with GPU work.
Do not start with live trading.
Do not let the LLM rewrite core infrastructure.

---

## Cursor delivery target

The repo is complete when a single command path can:

- read the supplied historical folder,
- normalize and cache the data,
- split it into train/test plus walk-forward folds,
- compute realized vol and smile features in the sandbox,
- generate signals,
- simulate straddles and vertical spreads,
- apply the missing-close rule exactly as specified,
- produce reports,
- ask the LLM for a rewritten sandbox and parameter updates,
- rerun deterministically,
- and preserve a clean upgrade path to Deribit live paper trading.
