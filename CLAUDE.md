# DERIBIT VOL SMILE AGENT — SPEC

## Purpose
Build a deterministic research system for Deribit BTC/ETH options that:
- constructs volatility smiles/surfaces
- compares implied vs realized volatility
- generates signals for spreads and straddles
- evaluates them on historical data
- iterates via an LLM-driven improvement loop

## Pipeline
data -> normalize -> features -> signals -> simulation -> report

## Data
Input:
- minute-level historical data from a folder (provided externally)

Normalize into structured tables with fields:
- timestamp
- asset (BTC / ETH)
- strike
- expiry
- option_type (call/put)
- underlying_price
- bid, ask, mid_price
- mark_iv (if available)

Rules:
- timestamps must be consistent (UTC)
- no silent data drops
- missing fields -> explicit NaN

## Splitting
- strictly chronological
- last segment = final test set (never touched during training)
- training uses walk-forward validation
- include configurable gap between train/validation to avoid leakage

No shuffling under any circumstances.

## Features
Compute:

Core:
- realized volatility (rolling windows)
- ATM implied volatility
- volatility smile (per expiry)
- smile slope / skew
- curvature (convexity)
- term structure
- IV minus realized volatility spread

Optional:
- momentum
- return breakouts
- bid-ask spread
- open interest / volume changes

Rules:
- deterministic
- no crashes on sparse data
- insufficient data -> NaN

## Signals
Generate trade signals for:
- straddles
- vertical spreads

Requirements:
- parameterized
- deterministic
- based only on feature outputs

## Simulation
For each signal:

Open:
- first available price >= signal timestamp

Close:
- at +1 day
- at +3 days

Rules:
- if exact close timestamp missing -> take next available
- if gap > 3 days -> return NaN
- if expiry interferes -> return NaN unless explicitly handled

Outputs:
- PnL (1d, 3d)
- trade-level metrics

Execution model:
- configurable (mid / mark / bid-ask with slippage)

Fees:
- configurable, never hardcoded

## Iteration Loop
Each run:
1. compute features
2. generate signals
3. simulate trades
4. produce metrics
5. send metrics + params + logic to LLM
6. receive updated params + logic
7. rerun deterministically

LLM may modify:
- feature engineering
- signal logic
- thresholds

LLM must NOT modify:
- data loading
- splitting
- simulation
- reporting
- architecture

## Execution Stages
Stage 1: quick run (30 days, CPU)
Stage 2: training (full data, walk-forward)
Stage 3: test (final untouched set)
Stage 4: live paper mode (future)

## Repo Structure
app/
  data/
  split/
  sim/
  opt/
  llm/
  live/
sandbox/
configs/
artifacts/
reports/
tests/

## Config
All parameters must be in YAML:
- data path
- assets
- split rules
- feature windows
- strategies
- fees
- execution model

No hardcoding.

## CLI
run-quick
run-train
run-test
run-live-paper

## Metrics
Report:
- total PnL
- avg / median PnL
- hit rate
- drawdown
- Sharpe-like
- trade count
- NaN rate

Breakdowns:
- asset
- strategy
- horizon

## Constraints
- deterministic outputs
- no data leakage
- reproducible runs
- no hidden randomness
- dependencies only via requirements.txt

## Code Rules
- small composable functions
- type hints
- separate I/O from logic
- no monoliths
- explicit error handling


## Sandboxed module to be rewritten by the AI agent

Only this module may be rewritten by the agent:

- `sandbox/calculator.py`

It must expose exactly this API:


The sandbox runs **in-process** with minimal guardrails:
- a timeout to prevent infinite loops,
- inputs passed as read-only copies so the calculator cannot mutate upstream data.

The goal is simply to prevent LLM-generated code from accidentally corrupting shared state or hanging the process — no subprocess isolation required.

## Success Criteria
- stable features
- consistent simulations
- reproducible results
- measurable performance

