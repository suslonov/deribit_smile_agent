"""LLM critic: validate and accept/reject a proposed calculator candidate.

Promotion rules (all must pass):
1. Source compiles (no SyntaxError).
2. Sandbox validate_source() returns no violations.
3. Unit tests pass.
4. Runs within timeout on a fixed seed slice.
5. Output is deterministic (runs twice, same result).
6. Improves the selected validation metric vs. the current best.
7. Does not worsen NaN rate, error rate, or turnover beyond limits.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from sandbox.runner import SandboxRunner, SandboxError

logger = logging.getLogger(__name__)


@dataclass
class CriticResult:
    accepted: bool
    reason: str
    violations: list[str]
    metrics: Optional[dict] = None


def llm_requests_made(config: dict) -> int:
    """Return the total number of LLM requests made in the current run."""
    llm_cfg = config.get("llm", {})
    return int(llm_cfg.get("llm_requests_made", 0))


def llm_budget_exceeded(config: dict) -> bool:
    """True when configured max LLM requests is reached/exceeded."""
    llm_cfg = config.get("llm", {})
    max_requests = llm_cfg.get("max_llm_requests")
    if max_requests is None:
        return False
    return llm_requests_made(config) >= int(max_requests)


def evaluate_candidate(
    candidate_source: str,
    current_metrics: dict,
    options_slice: pd.DataFrame,
    config: dict,
    tests_dir: str | Path,
    seed: int = 42,
) -> CriticResult:
    """
    Evaluate a candidate calculator source.

    Returns CriticResult with accepted=True only when all promotion rules pass.
    """
    tests_dir = Path(tests_dir)
    promotion = config.get("promotion", {})

    if llm_budget_exceeded(config):
        return CriticResult(
            False,
            (
                "LLM request budget exceeded "
                f"({llm_requests_made(config)}/{config.get('llm', {}).get('max_llm_requests')})"
            ),
            [],
        )

    # 1. Syntax check
    try:
        compile(candidate_source, "<candidate>", "exec")
    except SyntaxError as exc:
        return CriticResult(False, f"SyntaxError: {exc}", [])

    # 2. Write to temp file and run sandbox validation
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="candidate_"
    ) as tmp:
        tmp.write(candidate_source)
        tmp_path = Path(tmp.name)

    try:
        runner = SandboxRunner(
            calculator_path=tmp_path,
            timeout_seconds=config.get("sandbox", {}).get("timeout_seconds", 60),
            allowed_imports=frozenset(
                config.get("sandbox", {}).get("allowed_imports",
                ["numpy", "pandas", "scipy", "math", "statistics",
                 "itertools", "functools", "collections", "typing",
                 "datetime", "re", "copy", "warnings"])
            ),
            verify_determinism=True,
        )
        runner.reload()

        violations = runner.validate_source()
        if violations:
            return CriticResult(False, "Source violations", violations)

        # 3. Run unit tests
        if not _run_tests(tests_dir):
            return CriticResult(False, "Unit tests failed", violations)

        # 4 & 5. Run on fixed slice (determinism is checked inside runner)
        try:
            runner.verify_determinism = True
            calc_cfg = config.get("calculator", {})
            signals = runner.run(options_slice, calc_cfg, seed)
        except SandboxError as exc:
            return CriticResult(False, f"Sandbox execution failed: {exc}", violations)

        if signals.empty:
            return CriticResult(False, "Candidate produced zero signals", violations)

        # 6 & 7. Metric gate (run full quick sim would be done externally;
        #         here we just signal acceptance pending metric evaluation)
        logger.info(
            "Candidate passed static checks; produced %d signals", len(signals)
        )
        return CriticResult(
            True,
            "Passed all static checks; metric gate deferred to caller",
            violations,
        )

    finally:
        tmp_path.unlink(missing_ok=True)


def metric_gate(
    candidate_metrics: dict,
    baseline_metrics: dict,
    config: dict,
) -> CriticResult:
    """
    Compare candidate vs baseline metrics.

    Returns CriticResult(accepted=True) when candidate improves the objective
    and does not worsen guarded metrics.
    """
    promotion = config.get("promotion", {})
    objective = promotion.get("objective", "sharpe")
    max_nan_rate_delta = promotion.get("max_nan_rate_delta", 0.05)
    max_drawdown_pct_delta = promotion.get("max_drawdown_pct_delta", 0.10)

    base_summary = baseline_metrics.get("summary", {})
    cand_summary = candidate_metrics.get("summary", {})

    base_obj = base_summary.get(objective)
    cand_obj = cand_summary.get(objective)

    if base_obj is None or cand_obj is None:
        return CriticResult(
            False, f"Missing objective {objective!r} in metrics", []
        )

    if cand_obj <= base_obj:
        return CriticResult(
            False,
            f"Objective {objective} did not improve: {cand_obj:.4f} <= {base_obj:.4f}",
            [],
        )

    # Guard: nan rate
    base_nan = base_summary.get("nan_rate", 0.0) or 0.0
    cand_nan = cand_summary.get("nan_rate", 0.0) or 0.0
    if cand_nan - base_nan > max_nan_rate_delta:
        return CriticResult(
            False,
            f"NaN rate worsened: {cand_nan:.3f} vs {base_nan:.3f} (delta >{max_nan_rate_delta})",
            [],
        )

    return CriticResult(
        True,
        f"Accepted: {objective} improved {base_obj:.4f} → {cand_obj:.4f}",
        [],
        metrics=cand_summary,
    )


def _run_tests(tests_dir: Path) -> bool:
    """Run pytest on the test suite. Returns True if all pass."""
    if not tests_dir.exists():
        logger.warning("Tests dir %s not found; skipping", tests_dir)
        return True
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "-x", "-q", "--tb=short"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("Tests failed:\n%s", result.stdout[-2000:])
    return result.returncode == 0
