"""Paper trading router: connects live market data to the sandbox calculator.

Architecture:
  DeribitWSClient.on_snapshot
    → normalize_records (normalize.py)
    → SandboxRunner.run (calculator)
    → PaperRouter._route_signals
    → paper position ledger

Paper positions are tracked in-memory and logged to a JSONL file.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.data.normalize import normalize_records
from app.sim.metrics import compute_metrics
from sandbox.runner import SandboxRunner

logger = logging.getLogger(__name__)


class PaperRouter:
    """
    Routes live snapshots through the calculator and manages paper positions.

    Usage
    -----
    router = PaperRouter(config, sandbox_runner)
    ws_client = DeribitWSClient(config, on_snapshot=router.on_snapshot)
    asyncio.run(ws_client.run())
    """

    def __init__(
        self,
        config: dict,
        sandbox_runner: SandboxRunner,
    ) -> None:
        self.config = config
        self.runner = sandbox_runner
        paper_cfg = config.get("paper", {})
        self.trades_log = Path(paper_cfg.get("trades_log", "reports/paper_trades.jsonl"))
        self.max_concurrent = paper_cfg.get("max_concurrent_positions", 10)
        self._open_positions: list[dict] = []
        self.trades_log.parent.mkdir(parents=True, exist_ok=True)

    def on_snapshot(self, raw_records: list[dict]) -> None:
        """Callback: receive raw ticker records, compute signals, route to paper ledger."""
        if not raw_records:
            return

        options_df = normalize_records(raw_records)
        if options_df.empty:
            return

        calc_cfg = self.config.get("calculator", {})
        try:
            signals = self.runner.run(options_df, calc_cfg)
        except Exception as exc:
            logger.error("Calculator error in live mode: %s", exc)
            return

        if signals.empty:
            return

        self._route_signals(signals, options_df)

    def _route_signals(
        self,
        signals: pd.DataFrame,
        options_df: pd.DataFrame,
    ) -> None:
        """Log new signals as paper positions."""
        if len(self._open_positions) >= self.max_concurrent:
            logger.info("Max concurrent positions reached; skipping %d signals", len(signals))
            return

        capacity = self.max_concurrent - len(self._open_positions)
        for _, sig in signals.head(capacity).iterrows():
            entry = {
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "asset": sig.get("asset"),
                "strategy": sig.get("strategy"),
                "direction": sig.get("direction"),
                "expiry": str(sig.get("expiry")),
                "strike": sig.get("strike"),
                "strike_long": sig.get("strike_long"),
                "strike_short": sig.get("strike_short"),
                "option_type": sig.get("option_type"),
                "status": "open",
            }
            self._open_positions.append(entry)
            self._log_entry(entry)
            logger.info(
                "Paper: opened %s %s %s (expiry=%s)",
                entry["direction"], entry["strategy"], entry["asset"], entry["expiry"],
            )

    def _log_entry(self, entry: dict) -> None:
        with open(self.trades_log, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    @property
    def open_positions(self) -> list[dict]:
        return list(self._open_positions)
