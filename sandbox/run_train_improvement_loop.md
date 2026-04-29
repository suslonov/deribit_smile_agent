# `run_train()` improvement flow (pseudo-graphics)

```text
+---------------------------+
| run_train(args)           |
+---------------------------+
              |
              v
  load config + data + splits
              |
              v
  walk_forward_search(...) -> fold_metrics
              |
              v
  aggregate metrics + save artifact
              |
              v
  if llm == True:
      _run_llm_loop(artifact, ...)
              |
              v
+----------------------------------------------+
| _run_llm_loop(...)                           |
+----------------------------------------------+
              |
              v
  llm_budget_exceeded? ----yes----> return
              |
             no
              |
              v
  proposer.propose(prompt) -> new_source?
              |
      no ----> return
              |
             yes
              |
              v
  evaluate_candidate(new_source, ...)
              |
  accepted? ---no----> reject + return
              |
             yes
              |
              v
  write candidate file + print metric-gate note
              |
              v
            return
```

## Where it is in code

- Entry from `run_train()` happens at the final block: `if llm: _run_llm_loop(...)`.
- The LLM improvement logic itself is in `_run_llm_loop(...)`.
- Current behavior is **single-pass**, not an iterative training loop (no retry cycle after reject/accept).

## Where metrics are collected

- **Fold-level metrics (walk-forward):**
  - Collected from `fold_results` returned by `walk_forward_search(...)`.
  - Mapped into `fold_metrics`:
    - `train`: `r.train_metrics["summary"]`
    - `val`: `r.val_metrics["summary"]`
    - plus chosen calculator config per fold.

- **Aggregate training metrics (main report in `run_train`):**
  - Signals are re-generated on `train_df` via `runner.run(...)`.
  - Trade outcomes are simulated via `run_simulation(...)`.
  - Final summary metrics are computed via `compute_metrics(agg_results)`.
  - This `metrics` object is printed and stored in the artifact (`build_artifact(..., metrics=metrics, fold_metrics=fold_metrics, ...)`).

- **LLM candidate metric check path (`_run_llm_loop`):**
  - Uses `evaluate_candidate(...)` with `current_metrics=artifact["summary_metrics"]`.
  - The candidate is judged against current metrics there, but `run_train()` does not yet append a new training-metrics cycle after acceptance.
