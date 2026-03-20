"""
Web server + bot runner for Railway deployment.
Runs the trading bot in a background thread and serves
a real-time dashboard via FastAPI with WebSocket live updates.
"""

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from src.bot_state import state as dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

_bot_thread: threading.Thread | None = None
_ws_clients: set[WebSocket] = set()


def _run_bot():
    """Import and run the bot's main loop in this thread."""
    try:
        from bot import print_banner, main_loop
        print_banner()
        main_loop()
    except Exception as e:
        log.critical("Bot thread crashed: %s", e, exc_info=True)
        dashboard.set_error(f"Bot crashed: {e}")
        dashboard.update_bot_status(False, 0, 0, 0, "CRASHED")


async def _broadcast_loop():
    """Push state snapshots to all connected WebSocket clients every second."""
    while True:
        if _ws_clients:
            payload = json.dumps(dashboard.snapshot())
            stale = set()
            for ws in _ws_clients.copy():
                try:
                    await ws.send_text(payload)
                except Exception:
                    stale.add(ws)
            _ws_clients.difference_update(stale)
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bot_thread
    log.info("Starting bot in background thread...")
    _bot_thread = threading.Thread(target=_run_bot, daemon=True, name="bot-main")
    _bot_thread.start()
    log.info("Bot thread started. Dashboard available at /")
    broadcast_task = asyncio.create_task(_broadcast_loop())
    yield
    broadcast_task.cancel()
    log.info("Shutting down...")


app = FastAPI(title="Polymarket BTC Bot", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    log.info("WebSocket client connected (%d total)", len(_ws_clients))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)
        log.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))


@app.get("/api/state")
async def api_state():
    return JSONResponse(content=dashboard.snapshot())


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "bot_running": dashboard.running, "uptime": time.time() - dashboard.started_at if dashboard.started_at else 0}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
