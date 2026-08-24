"""
idle_service.py — Per-camera idle-person detection and DB logging
=================================================================
Maintains an in-memory state table keyed by (camera_id, track_id) and
writes to the IdleSession DB table when a tracked person stands still
for longer than the configured idle limit.

Design principles
─────────────────
• Runs inside camera_manager's detection daemon thread — NOT in a WebSocket
  handler.  Idle tracking therefore works with zero browser tabs open.

• State is pure in-memory; only DB writes are authoritative.  On server
  restart the state table is empty and idle timers restart from zero.  This
  is intentional — a 300s idle session started before a restart is simply
  lost; the DB row is closed with no end_time if the server goes down mid-
  session.  Future work: reconcile open rows on startup.

• Each camera_manager tick (≈5 fps) calls process_frame() with the list of
  persons detected in that frame.  process_frame() handles all of:
    - centroid computation
    - movement detection
    - idle threshold lookup (from config.IDLE_LIMITS keyed by zone_name)
    - IdleSession DB row open/update/close
    - save_alert() call (once per idle window, not every tick)

• Alert spam prevention: once an alert fires for a (camera_id, track_id)
  window, 'alert_fired' is set to True and no further alerts are sent until
  the person moves (which resets the state entirely).

Public API
──────────
  process_frame(camera_id, persons, zone_name)
      → called every detection tick; no return value needed by caller
  close_all_for_camera(camera_id)
      → call from camera_manager when a camera is stopped
"""

from __future__ import annotations

import collections
import datetime
import logging
import math
import threading
import time
from typing import Dict, List, Optional, Tuple

from config import settings

log = logging.getLogger("idle_service")

# ─────────────────────────────────────────────────────────────────────────────
# In-memory state table
# ─────────────────────────────────────────────────────────────────────────────
# Key:   (camera_id: int, track_id: int)
# Value: _TrackState
#
# Protected by _state_lock.  All public functions acquire the lock for their
# entire duration (reads + writes are always atomic from the caller's POV).

_StateKey = Tuple[int, int]  # (camera_id, track_id)


class _TrackState:
    """
    Mutable state for one tracked person on one camera.

    Fields
    ──────
    first_seen_at:    epoch float — when this track was first observed
    last_moved_at:    epoch float — last time the centroid moved > threshold
    centroid_history: deque of (cx, cy) — last N centroids for jitter-averaging
    zone_name:        zone the camera covers (maps to idle limit in config)
    idle_session_id:  DB row id of the currently-open IdleSession, or None
    alert_fired:      True once an alert has been sent for this idle window
    """

    __slots__ = (
        "first_seen_at",
        "last_moved_at",
        "centroid_history",
        "zone_name",
        "idle_session_id",
        "alert_fired",
    )

    def __init__(self, cx: float, cy: float, zone_name: str):
        now = time.time()
        self.first_seen_at = now
        self.last_moved_at = now
        self.centroid_history: collections.deque = collections.deque(
            [(cx, cy)], maxlen=8
        )
        self.zone_name = zone_name
        self.idle_session_id: Optional[int] = None
        self.alert_fired: bool = False

    @property
    def current_centroid(self) -> Tuple[float, float]:
        return self.centroid_history[-1]

    def idle_seconds(self) -> float:
        return time.time() - self.last_moved_at


_state: Dict[_StateKey, _TrackState] = {}
_state_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Idle limit lookup
# ─────────────────────────────────────────────────────────────────────────────


def _idle_limit(zone_name: str) -> int:
    """
    Return the idle threshold in seconds for the given zone_name.
    Falls back to 'default' if zone_name is not in IDLE_LIMITS.
    """
    limits = getattr(settings, "IDLE_LIMITS", {})
    return int(limits.get(zone_name, limits.get("default", 300)))


# ─────────────────────────────────────────────────────────────────────────────
# Centroid helpers
# ─────────────────────────────────────────────────────────────────────────────


def _centroid(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers — each opens+closes its own short-lived session
# ─────────────────────────────────────────────────────────────────────────────


def _open_idle_session(camera_id: int, track_id: int, zone_name: str) -> Optional[int]:
    """
    INSERT a new IdleSession row (end_time=NULL = open session).
    Returns the new row's id, or None on failure.
    """
    from database import IdleSession, SessionLocal

    db = None
    try:
        db = SessionLocal()
        row = IdleSession(
            camera_id=camera_id,
            track_id=track_id,
            zone_name=zone_name,
            start_time=datetime.datetime.utcnow(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        log.info(
            f"[Idle] Session opened — cam={camera_id} track={track_id} "
            f"zone={zone_name!r} row_id={row.id}"
        )
        return row.id
    except Exception as exc:
        log.error(f"[Idle] _open_idle_session failed: {exc}")
        if db:
            try:
                db.rollback()
            except Exception:
                pass
        return None
    finally:
        if db:
            try:
                db.close()
            except Exception:
                pass


def _close_idle_session(session_id: int, idle_seconds: float):
    """
    UPDATE IdleSession row with end_time + duration_seconds.
    """
    from database import IdleSession, SessionLocal

    db = None
    try:
        db = SessionLocal()
        row = db.query(IdleSession).filter(IdleSession.id == session_id).first()
        if row:
            row.end_time = datetime.datetime.utcnow()
            row.duration_seconds = round(idle_seconds, 1)
            db.commit()
            log.info(
                f"[Idle] Session closed — row_id={session_id} "
                f"duration={idle_seconds:.0f}s"
            )
    except Exception as exc:
        log.error(f"[Idle] _close_idle_session failed (id={session_id}): {exc}")
        if db:
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        if db:
            try:
                db.close()
            except Exception:
                pass


def _fire_alert(camera_id: int, track_id: int, zone_name: str, idle_seconds: float):
    """
    Save an idle-person alert via alert_service.save_alert().
    user_id=0 = system-generated (no browser session involved).
    """
    from database import SessionLocal
    from services.alert_service import save_alert

    db = None
    try:
        db = SessionLocal()
        save_alert(
            db=db,
            user_id=0,  # system alert — not from a user session
            message=(
                f"[IDLE PERSON] Camera {camera_id} — track #{track_id} has been "
                f"stationary in zone '{zone_name}' for "
                f"{int(idle_seconds)}s "
                f"(limit: {_idle_limit(zone_name)}s). "
                f"No browser session is required — this alert is from the "
                f"persistent camera daemon."
            ),
            role="Factory Worker",
            severity="medium",
            detected_issue="Idle person",
            confidence=None,
            snapshot_b64=None,
        )
        log.warning(
            f"[Idle] ALERT — cam={camera_id} track={track_id} "
            f"zone={zone_name!r} idle={idle_seconds:.0f}s"
        )
    except Exception as exc:
        log.error(f"[Idle] _fire_alert failed: {exc}")
        if db:
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        if db:
            try:
                db.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def process_frame(
    camera_id: int,
    persons: List[Dict],
    zone_name: str = "default",
) -> None:
    """
    Process one camera frame worth of tracked persons.

    Call this from camera_manager's detection daemon thread on every inference
    tick, immediately after process_frame_numpy_tracked() returns.

    Parameters
    ──────────
    camera_id  : DB cameras.id for the camera that produced this frame
    persons    : list of person dicts from _run_pipeline / _associate_to_persons.
                 Each must have keys: bbox (List[int]), track_id (int).
    zone_name  : logical zone this camera covers.  Mapped to an idle limit
                 via config.IDLE_LIMITS.  Defaults to "default" (300s).

    All DB writes are done with short-lived per-call sessions.  This function
    never holds a DB connection open across ticks.
    """
    threshold = getattr(settings, "IDLE_MOVEMENT_THRESHOLD_PX", 8)
    limit = _idle_limit(zone_name)
    now = time.time()

    # Build set of active track_ids in this frame
    active_track_ids = set()
    for p in persons:
        tid = int(p.get("track_id", -1))
        if tid == -1:
            continue  # untracked person — skip idle logic
        active_track_ids.add(tid)

    with _state_lock:

        # ── 1. Update existing tracks + detect new tracks ───────────────────
        for p in persons:
            tid = int(p.get("track_id", -1))
            if tid == -1:
                continue

            key = (camera_id, tid)
            cx, cy = _centroid(p["bbox"])

            if key not in _state:
                # New track: initialise state
                _state[key] = _TrackState(cx, cy, zone_name)
                log.debug(f"[Idle] New track — cam={camera_id} track={tid}")
                continue

            st = _state[key]
            st.centroid_history.append((cx, cy))

            # ── Movement check ──────────────────────────────────────────────
            prev_cx, prev_cy = (
                st.centroid_history[-2] if len(st.centroid_history) > 1 else (cx, cy)
            )
            moved = _distance((cx, cy), (prev_cx, prev_cy)) >= threshold

            if moved:
                # Person moved: reset timer, close any open idle session
                st.last_moved_at = now
                st.alert_fired = False
                if st.idle_session_id is not None:
                    _sid = st.idle_session_id
                    st.idle_session_id = None
                    # Close session outside the lock (DB call)
                    _close_idle_session(_sid, st.idle_seconds())
                continue

            # ── Idle check ──────────────────────────────────────────────────
            idle_secs = st.idle_seconds()
            if idle_secs < limit:
                continue  # still within allowed idle window

            # Person has been idle beyond the limit
            if st.idle_session_id is None:
                # Open a new IdleSession row
                new_id = _open_idle_session(camera_id, tid, zone_name)
                st.idle_session_id = new_id

            if not st.alert_fired:
                # Fire alert once per idle window
                st.alert_fired = True
                _fire_alert(camera_id, tid, zone_name, idle_secs)

        # ── 2. Close sessions for tracks that left the frame ────────────────
        stale_keys = [
            k for k in _state if k[0] == camera_id and k[1] not in active_track_ids
        ]
        for key in stale_keys:
            st = _state.pop(key)
            if st.idle_session_id is not None:
                _close_idle_session(st.idle_session_id, st.idle_seconds())
            log.debug(f"[Idle] Track left frame — cam={camera_id} track={key[1]}")


def close_all_for_camera(camera_id: int) -> None:
    """
    Close all open IdleSession rows and purge in-memory state for camera_id.
    Call from camera_manager.stop_camera() / stop_all().
    """
    with _state_lock:
        keys = [k for k in _state if k[0] == camera_id]
        to_close = []
        for key in keys:
            st = _state.pop(key)
            if st.idle_session_id is not None:
                to_close.append((st.idle_session_id, st.idle_seconds()))

    for sid, dur in to_close:
        _close_idle_session(sid, dur)

    log.info(
        f"[Idle] close_all_for_camera({camera_id}): "
        f"closed {len(to_close)} open session(s)"
    )
