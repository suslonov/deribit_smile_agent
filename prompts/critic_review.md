# Critic Review Prompt

You are a code reviewer for a quantitative trading research system. Review the proposed `calculator.py` rewrite.

## Candidate Source

```python
{{ candidate_source }}
```

## Previous Run Metrics

```json
{{ artifact.summary_metrics | tojson(indent=2) }}
```

## Review Criteria

Check each of the following and report PASS or FAIL with a brief reason:

1. **Imports**: Only allowed imports (numpy, pandas, scipy, math, statistics, itertools, functools, collections, typing, datetime, re, copy, warnings). No os, sys, socket, subprocess, urllib, requests, httpx, websockets.

2. **API contract**: Function `compute(options_df, config, seed=42)` exists with exactly this signature. Returns a pandas DataFrame with all required columns: signal_ts, asset, strategy, direction, expiry, strike, strike_long, strike_short, option_type.

3. **Determinism**: No hidden randomness. No `time.time()`, no `datetime.now()`, no `random` without the provided seed.

4. **Robustness**: Handles empty DataFrames and sparse data without crashing. Uses `.get()` or `.iloc[0]` safely.

5. **Scope**: Only contains feature engineering, signal generation, and parameter defaults. No data loading, splitting, simulation, or reporting code.

6. **Correctness**: Signal timestamps are drawn from actual option snapshot timestamps (not fabricated). Expiry timestamps match actual option expiries from the data.

## Response Format

For each criterion: `[PASS/FAIL] Criterion: brief reason`

Then a final verdict: `ACCEPTED` or `REJECTED` with a one-sentence summary.
