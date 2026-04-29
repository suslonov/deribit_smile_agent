# sandbox/calculator.py — notes

## API contract

```python
compute(options_df: pd.DataFrame, config: dict, seed: int = 42) -> pd.DataFrame
```

`options_df` is a read-only normalized options table.  
`config` is the `calculator` section of the YAML config.  
Returns one row per signal.

### Required output columns

| column | type | notes |
|---|---|---|
| signal_ts | pd.Timestamp | when signal is actionable |
| asset | str | BTC or ETH |
| strategy | str | straddle or vertical_spread |
| direction | str | long or short |
| expiry | pd.Timestamp | option expiry |
| strike | float or NaN | for straddle |
| strike_long | float or NaN | for vertical_spread |
| strike_short | float or NaN | for vertical_spread |
| option_type | str or NaN | C or P for vertical_spread |

No filesystem access, no network access, no randomness beyond the provided seed.

---

## Config keys

| key | default | meaning |
|---|---|---|
| assets | ["BTC","ETH"] | assets to process |
| min_oi | 10.0 | minimum open interest to include a row |
| min_strikes | 3 | minimum distinct strikes per group for smile computation |
| target_dte_min | 7 | minimum days-to-expiry |
| target_dte_max | 45 | maximum days-to-expiry |
| straddle_moneyness_tol | 0.02 | moneyness band around 1.0 for ATM straddle strike selection |
| iv_rv_threshold | 0.05 | ATM IV level above which we go short straddle, below → long |
| smile_slope_threshold | 0.0 | minimum absolute smile slope to emit a spread signal |
| time_bucket | "1h" | floor resolution for timestamp downsampling |

---

## Pipeline steps

### 1 — Time downsampling
Timestamps are floored to `time_bucket` (default hourly).  
Within each `(asset, bucket, instrument_name)` group, only the first row is kept.  
This reduces volume dramatically before any computation.

### 2 — DTE filter
Keeps only rows where days-to-expiry is in `[target_dte_min, target_dte_max]`.

### 3 — Smile features (vectorized)

Rows with missing or zero `mark_iv`, or `open_interest < min_oi`, are dropped.

Groups with fewer than `min_strikes` distinct strikes are excluded.

**ATM IV** — mean `mark_iv` for strikes within 5 % of `underlying_price`.

**Smile slope** — OLS slope of `mark_iv ~ log(strike / underlying_price)`,
computed fully vectorized using the closed-form formula:

```
slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
```

where `x = log_moneyness`, `y = mark_iv`.  
If the denominator is below 1e-10 the slope is set to NaN.

### 4 — Straddle signals

For each group with a valid ATM IV:

- direction = **short** if `atm_iv > iv_rv_threshold`, else **long**
- ATM strikes: rows where `|moneyness − 1| ≤ straddle_moneyness_tol`
- Require both a call and a put at the same strike
- Pick the strike closest to 1.0 moneyness

### 5 — Vertical spread signals

For each group where `|smile_slope| > smile_slope_threshold`:

- **Positive slope** → short call spread (moneyness 0.98–1.10, ascending strike order)
- **Negative slope** → long put spread (moneyness 0.90–1.02, descending strike order)

Need at least 2 strikes in the filtered window; `strike_long = iloc[0]`, `strike_short = iloc[1]`.
