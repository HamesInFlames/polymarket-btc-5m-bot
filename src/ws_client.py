"""
Polymarket WebSocket Client — Market Channel
=============================================
Connects to the public market WebSocket for real-time orderbook, price,
and trade data. Reduces REST polling and provides immediate price updates.

Endpoint:  wss://ws-subscriptions-clob.polymarket.com/ws/market
Protocol:  JSON messages, PING/PONG heartbeat every 10s
Reference: https://docs.polymarket.com/market-data/websocket/market-channel

Message types handled:
  - book:             Full orderbook snapshot
  - price_change:     Price level add/remove/update
  - last_trade_price: Trade execution
  - best_bid_ask:     Best bid/ask update (requires custom_feature_enabled)
  - market_resolved:  Market resolution notification
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)

WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
HEARTBEAT_INTERVAL = 9  # send PING every 9s (server expects within 10s)
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0


@dataclass
class TokenBookState:
    """Thread-safe local orderbook state for one token."""
    best_bid: float = 0.0
    best_ask: float = 1.0
    mid: float = 0.5
    spread: float = 1.0
    last_trade_price: float = 0.5
    last_trade_side: str = ""
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    updated_at: float = 0.0
    resolved: bool = False
    winning_outcome: str = ""


class MarketWebSocket:
    """
    Manages a persistent WebSocket connection to Polymarket's market channel.
    Runs in a background thread with auto-reconnect.
    """

    def __init__(self):
        self._subscribed_tokens: set[str] = set()
        self._token_states: dict[str, TokenBookState] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._connected = False
        self._ws = None
        self._on_resolution: Optional[Callable] = None

    def start(self):
        """Start the WebSocket client in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ws-market")
        self._thread.start()
        log.info("WebSocket client thread started")

    def stop(self):
        """Signal the WebSocket client to shut down."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        log.info("WebSocket client stopped")

    def subscribe(self, token_ids: list[str]):
        """Add token IDs to the subscription. Can be called before or after start()."""
        new_ids = set(token_ids) - self._subscribed_tokens
        if not new_ids:
            return

        with self._lock:
            self._subscribed_tokens.update(new_ids)
            for tid in new_ids:
                if tid not in self._token_states:
                    self._token_states[tid] = TokenBookState()

        if self._connected and self._ws:
            asyncio.run_coroutine_threadsafe(
                self._send_subscribe(list(new_ids)), self._loop
            )

    def unsubscribe(self, token_ids: list[str]):
        """Remove token IDs from the subscription."""
        with self._lock:
            self._subscribed_tokens -= set(token_ids)
            for tid in token_ids:
                self._token_states.pop(tid, None)

        if self._connected and self._ws:
            asyncio.run_coroutine_threadsafe(
                self._send_unsubscribe(token_ids), self._loop
            )

    def get_state(self, token_id: str) -> Optional[TokenBookState]:
        """Get the latest orderbook state for a token (thread-safe snapshot)."""
        with self._lock:
            state = self._token_states.get(token_id)
            if state and state.updated_at > 0:
                return TokenBookState(
                    best_bid=state.best_bid,
                    best_ask=state.best_ask,
                    mid=state.mid,
                    spread=state.spread,
                    last_trade_price=state.last_trade_price,
                    last_trade_side=state.last_trade_side,
                    bid_depth=state.bid_depth,
                    ask_depth=state.ask_depth,
                    updated_at=state.updated_at,
                    resolved=state.resolved,
                    winning_outcome=state.winning_outcome,
                )
        return None

    def set_resolution_callback(self, callback: Callable):
        """Set a callback for market_resolved events: callback(condition_id, winning_outcome)."""
        self._on_resolution = callback

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _run_loop(self):
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as e:
            log.error("WebSocket loop crashed: %s", e)
        finally:
            self._loop.close()

    async def _connect_loop(self):
        """Reconnect loop with exponential backoff."""
        attempt = 0

        while self._running:
            # Wait until we have tokens to subscribe to
            while self._running and not self._subscribed_tokens:
                await asyncio.sleep(1)
            if not self._running:
                return

            try:
                import websockets
                log.info("Connecting to Polymarket WebSocket...")
                async with websockets.connect(
                    WS_MARKET_URL,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=10,
                    max_size=2**22,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    attempt = 0
                    log.info("WebSocket connected")

                    with self._lock:
                        token_ids = list(self._subscribed_tokens)
                    if token_ids:
                        await self._send_initial_subscribe(token_ids)

                    heartbeat_task = asyncio.create_task(self._heartbeat(ws))
                    receive_task = asyncio.create_task(self._receive_loop(ws))

                    done, pending = await asyncio.wait(
                        [heartbeat_task, receive_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        if task.exception():
                            log.warning("WebSocket task error: %s", task.exception())

            except Exception as e:
                log.warning("WebSocket connection error: %s", e)

            self._connected = False
            self._ws = None

            if not self._running:
                break

            delay = min(RECONNECT_MAX_DELAY, RECONNECT_BASE_DELAY * (2 ** attempt))
            attempt += 1
            log.info("Reconnecting in %.1fs (attempt %d)...", delay, attempt)
            await asyncio.sleep(delay)

    async def _send_initial_subscribe(self, token_ids: list[str]):
        msg = json.dumps({
            "assets_ids": token_ids,
            "type": "market",
            "custom_feature_enabled": True,
        })
        await self._ws.send(msg)
        log.info("Subscribed to %d token(s) via WebSocket", len(token_ids))

    async def _send_subscribe(self, token_ids: list[str]):
        msg = json.dumps({
            "assets_ids": token_ids,
            "operation": "subscribe",
            "custom_feature_enabled": True,
        })
        await self._ws.send(msg)
        log.debug("Dynamically subscribed to %d new token(s)", len(token_ids))

    async def _send_unsubscribe(self, token_ids: list[str]):
        msg = json.dumps({
            "assets_ids": token_ids,
            "operation": "unsubscribe",
        })
        await self._ws.send(msg)

    async def _heartbeat(self, ws):
        """Send PING every HEARTBEAT_INTERVAL seconds."""
        while self._running:
            try:
                await ws.send("PING")
                await asyncio.sleep(HEARTBEAT_INTERVAL)
            except Exception:
                return

    async def _receive_loop(self, ws):
        """Process incoming WebSocket messages."""
        async for raw_msg in ws:
            if not self._running:
                return

            if raw_msg == "PONG":
                continue

            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            # Server may send arrays (e.g. batch updates) — process each item
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        self._dispatch(item)
                continue

            if not isinstance(data, dict):
                continue

            self._dispatch(data)

    def _dispatch(self, data: dict):
        """Route a single message to its handler."""
        event_type = data.get("event_type", "")
        if event_type == "book":
            self._handle_book(data)
        elif event_type == "price_change":
            self._handle_price_change(data)
        elif event_type == "last_trade_price":
            self._handle_last_trade(data)
        elif event_type == "best_bid_ask":
            self._handle_best_bid_ask(data)
        elif event_type == "market_resolved":
            self._handle_resolution(data)

    def _handle_book(self, data: dict):
        """Full orderbook snapshot."""
        asset_id = data.get("asset_id", "")
        with self._lock:
            state = self._token_states.get(asset_id)
            if not state:
                return

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if bids:
                state.best_bid = float(bids[0]["price"])
                state.bid_depth = sum(float(b["size"]) for b in bids)
            if asks:
                state.best_ask = float(asks[0]["price"])
                state.ask_depth = sum(float(a["size"]) for a in asks)

            if state.best_bid > 0 and state.best_ask < 1:
                state.mid = (state.best_bid + state.best_ask) / 2.0
                state.spread = state.best_ask - state.best_bid

            state.updated_at = time.time()

    def _handle_price_change(self, data: dict):
        """Incremental price level updates."""
        changes = data.get("price_changes", [])
        with self._lock:
            for change in changes:
                asset_id = change.get("asset_id", "")
                state = self._token_states.get(asset_id)
                if not state:
                    continue

                bb = change.get("best_bid")
                ba = change.get("best_ask")
                if bb is not None:
                    state.best_bid = float(bb)
                if ba is not None:
                    state.best_ask = float(ba)

                if state.best_bid > 0 and state.best_ask < 1:
                    state.mid = (state.best_bid + state.best_ask) / 2.0
                    state.spread = state.best_ask - state.best_bid

                state.updated_at = time.time()

    def _handle_last_trade(self, data: dict):
        """Trade execution."""
        asset_id = data.get("asset_id", "")
        with self._lock:
            state = self._token_states.get(asset_id)
            if not state:
                return
            state.last_trade_price = float(data.get("price", state.last_trade_price))
            state.last_trade_side = data.get("side", "")
            state.updated_at = time.time()

    def _handle_best_bid_ask(self, data: dict):
        """Best bid/ask update (custom feature)."""
        asset_id = data.get("asset_id", "")
        with self._lock:
            state = self._token_states.get(asset_id)
            if not state:
                return
            bb = data.get("best_bid")
            ba = data.get("best_ask")
            if bb is not None:
                state.best_bid = float(bb)
            if ba is not None:
                state.best_ask = float(ba)
            spread = data.get("spread")
            if spread is not None:
                state.spread = float(spread)
            if state.best_bid > 0 and state.best_ask < 1:
                state.mid = (state.best_bid + state.best_ask) / 2.0
            state.updated_at = time.time()

    def _handle_resolution(self, data: dict):
        """Market resolved notification."""
        condition_id = data.get("market", "")
        winning = data.get("winning_outcome", "")
        log.info("WebSocket: market_resolved %s -> %s", condition_id[:16], winning)

        if self._on_resolution and condition_id and winning:
            try:
                self._on_resolution(condition_id, winning)
            except Exception as e:
                log.error("Resolution callback error: %s", e)


# Module-level singleton
_ws_client: Optional[MarketWebSocket] = None


def get_ws_client() -> MarketWebSocket:
    """Get or create the singleton WebSocket client."""
    global _ws_client
    if _ws_client is None:
        _ws_client = MarketWebSocket()
    return _ws_client
