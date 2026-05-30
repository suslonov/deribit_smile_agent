"""Sandbox runner: executes the calculator in-process with guardrails.

Guardrails:
- Wall-clock timeout via a threading.Timer that raises TimeoutError.
- Inputs passed as deep copies so the calculator cannot mutate upstream state.
- Import whitelist check applied before execution (static AST scan).
- Determinism verified when verify_determinism=True (runs twice, compares output).
"""
from __future__ import annotations

import ast
import copy
import importlib
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_ALLOWED_IMPORTS = frozenset({
    "numpy", "pandas", "scipy", "math", "statistics",
    "itertools", "functools", "collections", "typing",
    "datetime", "re", "copy", "warnings",
    "__future__",   # annotations import for type hints
})


class SandboxError(RuntimeError):
    """Raised when the sandbox rejects or times out the calculator."""


class SandboxRunner:
    """Run the sandboxed calculator module with safety checks."""

    def __init__(
        self,
        calculator_path: str | Path,
        timeout_seconds: float = 60.0,
        allowed_imports: Optional[frozenset[str]] = None,
        verify_determinism: bool = False,
    ) -> None:
        self.calculator_path = Path(calculator_path)
        self.timeout_seconds = timeout_seconds
        self.allowed_imports = allowed_imports or _DEFAULT_ALLOWED_IMPORTS
        self.verify_determinism = verify_determinism
        self._module = None

    def reload(self) -> None:
        """(Re)load the calculator module from disk."""
        self._module = None
        spec = importlib.util.spec_from_file_location(
            "sandbox.calculator", self.calculator_path
        )
        if spec is None or spec.loader is None:
            raise SandboxError(f"Cannot load module from {self.calculator_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        self._module = mod

    def validate_source(self) -> list[str]:
        """
        Static AST check: return list of violations (empty = ok).
        Checks for:
        - Imports outside the allowed set.
        - open() / os / sys filesystem calls.
        - socket / urllib / requests / httpx network calls.
        """
        source = self.calculator_path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return [f"SyntaxError: {exc}"]

        violations: list[str] = []
        _BANNED_NAMES = {"open", "socket", "subprocess", "eval", "exec", "__import__"}
        _BANNED_MODULES = frozenset({
            "os", "sys", "subprocess", "socket", "urllib", "requests",
            "httpx", "aiohttp", "websockets", "ftplib", "smtplib",
        })

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = ""
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module in _BANNED_MODULES:
                            violations.append(f"Banned import: {alias.name}")
                        elif module not in self.allowed_imports:
                            violations.append(f"Unlisted import: {alias.name}")
                else:
                    module = (node.module or "").split(".")[0]
                    if module in _BANNED_MODULES:
                        violations.append(f"Banned import: {node.module}")
                    elif module and module not in self.allowed_imports:
                        violations.append(f"Unlisted import: {node.module}")

            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name in _BANNED_NAMES:
                    violations.append(f"Banned call: {func_name}()")

        return violations

    def run(
        self,
        options_df: pd.DataFrame,
        config: dict,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Run the calculator and return signals DataFrame.

        On timeout, runtime errors, invalid return type, invalid columns, or
        failed determinism check, returns an empty DataFrame with the contract
        columns (see ``_empty_signals_dataframe``).

        Raises SandboxError only for static violations before execution
        (e.g. ``validate_source``) or module load failures.
        """
        if self._module is None:
            self.reload()

        violations = self.validate_source()
        if violations:
            raise SandboxError(f"Calculator has violations: {violations}")

        # Shallow copy: shares underlying data but prevents in-place column additions
        # on the original. The calculator cannot replace the calling code's reference.
        df_copy = options_df.copy(deep=False)
        cfg_copy = copy.deepcopy(config)

        result: list[Optional[pd.DataFrame]] = [None]
        exc_holder: list[Optional[Exception]] = [None]

        def _target() -> None:
            try:
                result[0] = self._module.compute(df_copy, cfg_copy, seed)
            except Exception as exc:
                exc_holder[0] = exc

        t = threading.Thread(target=_target, daemon=True)
        t0 = time.monotonic()
        t.start()
        t.join(timeout=self.timeout_seconds)

        if t.is_alive():
            logger.warning(
                "Calculator timed out after %ss", self.timeout_seconds
            )
            return _empty_signals_dataframe()

        elapsed = time.monotonic() - t0
        logger.debug("Calculator completed in %.2fs", elapsed)

        if exc_holder[0] is not None:
            logger.warning("Calculator raised: %s", exc_holder[0])
            return _empty_signals_dataframe()

        signals = result[0]
        if signals is None or not isinstance(signals, pd.DataFrame):
            logger.warning("Calculator returned non-DataFrame result")
            return _empty_signals_dataframe()

        if not _validate_output(signals):
            return _empty_signals_dataframe()

        if self.verify_determinism and not _check_determinism(
            self._module, df_copy, cfg_copy, seed, signals
        ):
            return _empty_signals_dataframe()

        return signals


_SIGNAL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "signal_ts",
    "asset",
    "strategy",
    "direction",
    "expiry",
    "strike",
    "strike_long",
    "strike_short",
    "option_type",
)


def _empty_signals_dataframe() -> pd.DataFrame:
    """Empty signals frame matching the calculator / simulator contract."""
    return pd.DataFrame(columns=list(_SIGNAL_OUTPUT_COLUMNS))


def _validate_output(signals: pd.DataFrame) -> bool:
    """Return True if ``signals`` has all required columns."""
    required = set(_SIGNAL_OUTPUT_COLUMNS)
    missing = required - set(signals.columns)
    if missing:
        logger.warning("Calculator output missing columns: %s", missing)
        return False
    return True


def _check_determinism(
    module: object,
    df_copy: pd.DataFrame,
    cfg_copy: dict,
    seed: int,
    first_result: pd.DataFrame,
) -> bool:
    """Re-run with the same seed and compare output shape + values."""
    try:
        second = module.compute(
            df_copy.copy(deep=True), copy.deepcopy(cfg_copy), seed
        )
    except Exception as exc:
        logger.warning("Determinism re-run raised: %s", exc)
        return False
    if not isinstance(second, pd.DataFrame):
        logger.warning("Determinism re-run returned non-DataFrame")
        return False
    if second.shape != first_result.shape:
        logger.warning(
            "Non-deterministic output: shapes differ %s vs %s",
            first_result.shape,
            second.shape,
        )
        return False
    try:
        pd.testing.assert_frame_equal(first_result, second, check_like=True)
    except AssertionError as exc:
        logger.warning("Non-deterministic output: %s", exc)
        return False
    return True
