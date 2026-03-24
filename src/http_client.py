"""
Resilient HTTP client with rate-limit and matching-engine-restart handling.

Polymarket rate limits (from docs):
  - CLOB API: 9,000 req / 10s general; trading endpoints have burst/sustained limits
  - Gamma API: 4,000 req / 10s general
  - HTTP 429 = rate-limit exceeded
  - HTTP 425 = matching engine restarting (weekly Tuesdays 7:00 AM ET, ~90s window)

This module wraps requests.Session with:
  - Automatic retry on 429 / 425 / 5xx with exponential backoff
  - Per-host request tracking for rate awareness
  - Matching engine restart detection and cooldown
  - Shared session with connection pooling

Reference: https://docs.polymarket.com/trading/rate-limits
"""

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

_ENGINE_RESTART_DAY = 1  # Tuesday
_ENGINE_RESTART_HOUR_UTC = 11  # 7 AM ET = 11:00 UTC (EDT) or 12:00 UTC (EST)
_ENGINE_RESTART_WINDOW_S = 120

_engine_restart_until: float = 0.0
_engine_restart_lock = threading.Lock()

_sessions: dict[str, requests.Session] = {}
_session_lock = threading.Lock()


def _create_session() -> requests.Session:
    """Create a session with connection pooling but NO urllib3-level retry
    (we handle retries ourselves for finer control)."""
    s = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=Retry(total=0),
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _get_session(host: str) -> requests.Session:
    with _session_lock:
        if host not in _sessions:
            _sessions[host] = _create_session()
        return _sessions[host]


def resilient_get(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 8,
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    headers: Optional[dict] = None,
) -> requests.Response:
    """
    GET with exponential backoff on 429, 425, and 5xx.
    Raises after max_retries exhausted.
    """
    _wait_for_engine_restart()

    host = url.split("/")[2] if "/" in url else url
    session = _get_session(host)

    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout, headers=headers)

            if resp.status_code == 429:
                delay = _backoff_delay(attempt, base_delay, max_delay)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                log.warning(
                    "Rate limited (429) on %s — backing off %.1fs (attempt %d/%d)",
                    url, delay, attempt + 1, max_retries + 1,
                )
                time.sleep(delay)
                continue

            if resp.status_code == 425:
                _signal_engine_restart()
                delay = _backoff_delay(attempt, base_delay, max_delay)
                log.warning(
                    "Matching engine restarting (425) — backing off %.1fs (attempt %d/%d)",
                    delay, attempt + 1, max_retries + 1,
                )
                time.sleep(delay)
                continue

            if resp.status_code >= 500:
                delay = _backoff_delay(attempt, base_delay, max_delay)
                log.warning(
                    "Server error %d on %s — retrying in %.1fs (attempt %d/%d)",
                    resp.status_code, url, delay, attempt + 1, max_retries + 1,
                )
                time.sleep(delay)
                continue

            return resp

        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            delay = _backoff_delay(attempt, base_delay, max_delay)
            log.warning(
                "Network error on %s: %s — retrying in %.1fs (attempt %d/%d)",
                url, e, delay, attempt + 1, max_retries + 1,
            )
            time.sleep(delay)

    if last_exc:
        raise last_exc
    raise requests.HTTPError(f"Failed after {max_retries + 1} attempts on {url}")


def resilient_post(
    url: str,
    json: Optional[dict] = None,
    data=None,
    timeout: int = 10,
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    headers: Optional[dict] = None,
) -> requests.Response:
    """POST with the same retry semantics."""
    _wait_for_engine_restart()

    host = url.split("/")[2] if "/" in url else url
    session = _get_session(host)

    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = session.post(
                url, json=json, data=data, timeout=timeout, headers=headers,
            )

            if resp.status_code == 429:
                delay = _backoff_delay(attempt, base_delay, max_delay)
                log.warning(
                    "Rate limited (429) on POST %s — backing off %.1fs",
                    url, delay,
                )
                time.sleep(delay)
                continue

            if resp.status_code == 425:
                _signal_engine_restart()
                delay = _backoff_delay(attempt, 2.0, max_delay)
                log.warning(
                    "Matching engine restarting (425) on POST — backing off %.1fs",
                    delay,
                )
                time.sleep(delay)
                continue

            if resp.status_code >= 500:
                delay = _backoff_delay(attempt, base_delay, max_delay)
                log.warning(
                    "Server error %d on POST %s — retrying in %.1fs",
                    resp.status_code, url, delay,
                )
                time.sleep(delay)
                continue

            return resp

        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            delay = _backoff_delay(attempt, base_delay, max_delay)
            log.warning(
                "Network error on POST %s: %s — retrying in %.1fs",
                url, e, delay,
            )
            time.sleep(delay)

    if last_exc:
        raise last_exc
    raise requests.HTTPError(f"POST failed after {max_retries + 1} attempts on {url}")


def _backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff: base * 2^attempt, capped at `cap`."""
    return min(cap, base * (2 ** attempt))


def _signal_engine_restart():
    """Record that we detected an engine restart (425)."""
    global _engine_restart_until
    with _engine_restart_lock:
        _engine_restart_until = time.time() + _ENGINE_RESTART_WINDOW_S
        log.warning(
            "Matching engine restart detected — will wait until %.0f",
            _engine_restart_until,
        )


def _wait_for_engine_restart():
    """If we're in a known engine restart window, sleep until it passes."""
    with _engine_restart_lock:
        remaining = _engine_restart_until - time.time()
    if remaining > 0:
        log.info("Waiting %.0fs for matching engine restart to complete...", remaining)
        time.sleep(remaining)


def is_engine_restart_window() -> bool:
    """
    Check if we're in the known weekly maintenance window:
    Tuesdays at 7:00 AM ET (~11:00 UTC during EDT).
    Returns True if within +/- 5 minutes of the restart time.
    """
    now_utc = datetime.now(timezone.utc)
    if now_utc.weekday() != _ENGINE_RESTART_DAY:
        return False
    minutes_since_midnight = now_utc.hour * 60 + now_utc.minute
    restart_minute = _ENGINE_RESTART_HOUR_UTC * 60
    return abs(minutes_since_midnight - restart_minute) <= 5


def engine_restart_status() -> dict:
    """Return current engine restart state for dashboard/logging."""
    with _engine_restart_lock:
        cooldown_remaining = max(0, _engine_restart_until - time.time())
    return {
        "in_scheduled_window": is_engine_restart_window(),
        "cooldown_remaining_s": cooldown_remaining,
        "active": cooldown_remaining > 0,
    }
