# Deribit Volatility Smile Research Agent

A deterministic research system for Deribit BTC/ETH options that:
- Constructs volatility smiles and surfaces from minute-level historical data
- Compares implied vs realized volatility
- Generates signals for straddles and vertical spreads
- Evaluates them via historical simulation
- Iterates via an LLM-driven improvement loop on a sandboxed calculator module

## Architecture

```
data → normalize → [sandbox calculator] → signals → simulator → metrics → artifact → LLM → rewrite
                                ↑__________________________________________|
```

The only file the LLM may rewrite is `sandbox/calculator.py`. All infrastructure (data loading, splitting, simulation, reporting) is stable and off-limits.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a 30-day quick research run
python -m app.main run-quick --config configs/quick.yaml

# Run with LLM improvement loop
python -m app.main run-quick --config configs/quick.yaml --llm

# Full walk-forward training study
python -m app.main run-train --config configs/train_full.yaml

# Final test (run once after training)
python -m app.main run-test --config configs/train_full.yaml

# Live paper trading
python -m app.main run-live-paper --config configs/live_paper.yaml
```

## Data Format

Historical data lives in `/mnt/Data/archive/raw_data/YYYY-MM-DD/`. Each file is a
zlib-compressed Python pickle (1-byte prefix + zlib + pickle) containing a list of
Deribit option ticker snapshots. The loader handles decompression, normalization, and
caching automatically.

## Project Structure

```
app/
  data/         Data loading, normalization, schemas
  split/        Chronological walk-forward splitting
  sim/          Simulator, pricing models, metrics
  opt/          Parameter search
  llm/          Proposer, critic, artifact logging
  live/         Deribit WebSocket adapter (paper trading)
  utils/        I/O, hashing, clock
sandbox/
  calculator.py  ← THE ONLY LLM-REWRITABLE FILE
  runner.py      Sandbox execution with safety checks
configs/
  quick.yaml     30-day CPU research run
  train_full.yaml Full walk-forward study
  live_paper.yaml Live paper trading
tests/           Pytest test suite
prompts/         Jinja2 prompt templates for LLM loop
artifacts/       Run artifacts and ledger (git-ignored)
reports/         Output reports (git-ignored)
```

## Splitting

- **Test set**: last 90 calendar days (never used during training)
- **Walk-forward validation**: configurable `train_window_days` + `gap_days` + `val_window_days`
- No shuffling under any circumstances

## Simulation Rules

- Open at the first available snapshot ≥ signal timestamp
- Close at `signal_ts + horizon_days`; take first available snapshot ≥ target
- If gap from target close to actual close > 3 days → NaN
- If expiry occurs before intended close → NaN

## Sandbox API

```python
def compute(
    options_df: pd.DataFrame,  # normalized options data (read-only copy)
    config: dict,              # calculator parameters
    seed: int = 42,            # fixed seed for reproducibility
) -> pd.DataFrame:             # signals with required columns
```

Required output columns: `signal_ts, asset, strategy, direction, expiry, strike, strike_long, strike_short, option_type`

## Running Tests

```bash
pytest tests/ -v
```
