"""Deribit WebSocket market data adapter.

Subscribes to ticker channels for all active BTC/ETH option instruments
and forwards normalized snapshots to a callback.

Uses WebSocket subscriptions per Deribit recommendations (no polling).
Separate connection for market data vs. private/trading flow.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Deribit WS endpoint
_WS_URL = "wss://www.deribit.com/ws/api/v2"

# Ticker channel format for a single instrument
_TICKER_CHANNEL = "ticker.{instrument}.raw"

# Minimum pause between subscription batches to avoid rate-limits
_SUBSCRIPTION_BATCH_SIZE = 100
_SUBSCRIPTION_BATCH_DELAY = 1.0  # seconds


class DeribitWSClient:
    """
    Long-running WebSocket client for Deribit market data.

    Usage
    -----
    client = DeribitWSClient(config, on_snapshot=my_callback)
    asyncio.run(client.run())
    """

    def __init__(
        self,
        config: dict,
        on_snapshot: Callable[[list[dict]], None],
    ) -> None:
        self.ws_url: str = config.get("live", {}).get("ws_url", _WS_URL)
        self.assets: list[str] = config.get("data", {}).get("assets", ["BTC", "ETH"])
        self.heartbeat_interval: int = config.get("live", {}).get("heartbeat_interval", 10)
        self.reconnect_delay: int = config.get("live", {}).get("reconnect_delay", 5)
        self.on_snapshot = on_snapshot
        self._running = False
        # Accumulate tickers by timestamp before forwarding as snapshot
        self._buffer: dict[str, dict] = {}  # instrument → latest ticker
        self._last_flush: float = 0.0
        self._flush_interval = 60.0  # flush snapshot every 60 seconds

    async def run(self) -> None:
        """Main loop: connect, subscribe, receive, reconnect on error."""
        self._running = True
        while self._running:
            try:
                await self._session()
            except Exception as exc:
                logger.warning("WS session ended (%s); reconnecting in %ds", exc, self.reconnect_delay)
                await asyncio.sleep(self.reconnect_delay)

    async def stop(self) -> None:
        self._running = False

    async def _session(self) -> None:
        try:
            import websockets  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("pip install websockets") from exc

        async with websockets.connect(
            self.ws_url,
            ping_interval=self.heartbeat_interval,
            ping_timeout=30,
        ) as ws:
            logger.info("Connected to %s", self.ws_url)
            await self._enable_heartbeat(ws)
            instruments = await self._fetch_active_instruments(ws)
            await self._subscribe(ws, instruments)
            async for raw_message in ws:
                await self._handle_message(raw_message)

    async def _enable_heartbeat(self, ws: Any) -> None:
        msg = {
            "jsonrpc": "2.0",
            "method": "public/set_heartbeat",
            "params": {"interval": self.heartbeat_interval},
            "id": 1,
        }
        await ws.send(json.dumps(msg))

    async def _fetch_active_instruments(self, ws: Any) -> list[str]:
        """Retrieve all active option instruments for configured assets."""
        instruments: list[str] = []
        for asset in self.assets:
            msg = {
                "jsonrpc": "2.0",
                "method": "public/get_instruments",
                "params": {"currency": asset, "kind": "option", "expired": False},
                "id": 10,
            }
            await ws.send(json.dumps(msg))
            raw = await ws.recv()
            data = json.loads(raw)
            result = data.get("result", [])
            for instr in result:
                instruments.append(instr["instrument_name"])
        logger.info("Fetched %d active option instruments", len(instruments))
        return instruments

    async def _subscribe(self, ws: Any, instruments: list[str]) -> None:
        """Subscribe to ticker channels in batches."""
        channels = [_TICKER_CHANNEL.format(instrument=i) for i in instruments]
        for i in range(0, len(channels), _SUBSCRIPTION_BATCH_SIZE):
            batch = channels[i: i + _SUBSCRIPTION_BATCH_SIZE]
            msg = {
                "jsonrpc": "2.0",
                "method": "public/subscribe",
                "params": {"channels": batch},
                "id": 20 + i,
            }
            await ws.send(json.dumps(msg))
            await asyncio.sleep(_SUBSCRIPTION_BATCH_DELAY)
        logger.info("Subscribed to %d channels", len(channels))

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        method = msg.get("method")
        if method != "subscription":
            return

        params = msg.get("params", {})
        data = params.get("data", {})
        instrument = data.get("instrument_name")
        if not instrument:
            return

        self._buffer[instrument] = data

        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval and self._buffer:
            snapshot = list(self._buffer.values())
            self._buffer = {}
            self._last_flush = now
            try:
                self.on_snapshot(snapshot)
            except Exception as exc:
                logger.error("on_snapshot callback raised: %s", exc)
