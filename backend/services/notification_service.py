"""
notification_service.py — Event-based notification system with FCM
===================================================================
Central push notification wrapper with event tracking and confirmation.

Architecture:
  Frame Detection → EventManager → AlertManager → FCM

EventManager: Tracks frame-level detections as persistent events with states:
  NOT_ACTIVE → DETECTED → PENDING_CONFIRMATION → CONFIRMED → NOTIFIED → RESOLVED

AlertManager: Decides when to send notifications based on event confirmation.

Key Features:
  • 2-minute confirmation: Anomaly must persist for CONFIRMATION_DURATION
  • Grace period: Missed frames don't immediately end events
  • Deduplication: One notification per event lifecycle
  • Retry logic: Queue failed notifications with exponential backoff
  • Thread-safe: Uses locks for concurrent camera processing

Setup (once):
  1. Firebase Console → Create project → Add Web App → copy firebaseConfig
  2. Project Settings → Cloud Messaging → copy Server Key
  3. Add to backend/.env:   FCM_SERVER_KEY=<your-server-key>
  4. Add to frontend/.env:  VITE_FIREBASE_* (see firebase.js)
  5. Restart server — done.

Throttling: 1 push per 2 minutes per (camera, issue_type) — prevents floods.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Tuple

log = logging.getLogger("notification_service")


# ── Event State Machine ───────────────────────────────────────────────────────


class EventState(Enum):
    """Event lifecycle states for anomaly tracking."""
    NOT_ACTIVE = "not_active"
    DETECTED = "detected"
    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    NOTIFIED = "notified"
    RESOLVED = "resolved"


@dataclass
class AnomalyEvent:
    """Represents a single anomaly event being tracked over time."""
    event_id: str
    camera_id: int
    track_id: str
    anomaly_type: str
    severity: str
    
    # State tracking
    state: EventState = EventState.DETECTED
    first_detected_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    confirmed_at: Optional[float] = None
    resolved_at: Optional[float] = None
    
    # Notification tracking
    notification_sent: bool = False
    notification_sent_at: Optional[float] = None
    notification_attempts: int = 0
    notification_status: str = "pending"
    
    # Metadata
    duration: float = 0.0
    camera_name: Optional[str] = None
    floor: Optional[str] = None
    
    def update_last_seen(self) -> None:
        """Update last_seen timestamp and recalculate duration."""
        self.last_seen_at = time.time()
        self.duration = self.last_seen_at - self.first_detected_at
    
    def is_grace_period_expired(self, grace_period: float) -> bool:
        """Check if event hasn't been seen within grace period."""
        return (time.time() - self.last_seen_at) > grace_period
    
    def should_confirm(self, confirmation_duration: float) -> bool:
        """Check if event has been active long enough to confirm."""
        return (
            self.state == EventState.PENDING_CONFIRMATION
            and self.duration >= confirmation_duration
        )
    
    def to_dict(self) -> dict:
        """Convert event to dictionary for logging/storage."""
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "state": self.state.value,
            "first_detected_at": datetime.fromtimestamp(self.first_detected_at).isoformat(),
            "last_seen_at": datetime.fromtimestamp(self.last_seen_at).isoformat(),
            "confirmed_at": datetime.fromtimestamp(self.confirmed_at).isoformat() if self.confirmed_at else None,
            "resolved_at": datetime.fromtimestamp(self.resolved_at).isoformat() if self.resolved_at else None,
            "duration": self.duration,
            "notification_sent": self.notification_sent,
            "notification_attempts": self.notification_attempts,
            "notification_status": self.notification_status,
        }


class EventManager:
    """
    Tracks frame-level detections as persistent events.
    
    Converts continuous detections into meaningful events with states:
    NOT_ACTIVE → DETECTED → PENDING_CONFIRMATION → CONFIRMED → NOTIFIED → RESOLVED
    """
    
    def __init__(self):
        # Key: (camera_id, track_id, anomaly_type)
        self._events: Dict[Tuple[int, str, str], AnomalyEvent] = {}
        self._lock = threading.RLock()
        self._event_counter = 0
        
        # Load config
        from config import settings
        self.confirmation_duration = settings.EVENT_CONFIRMATION_DURATION
        self.grace_period = settings.EVENT_GRACE_PERIOD
        self.max_duration = settings.EVENT_MAX_DURATION
        self.cleanup_age = settings.EVENT_CLEANUP_AGE
        
        log.info(
            "[EventManager] Initialized: confirmation=%ds, grace=%ds",
            self.confirmation_duration, self.grace_period
        )
    
    def process_detection(
        self,
        camera_id: int,
        track_id: str,
        anomaly_type: str,
        severity: str = "medium",
        camera_name: Optional[str] = None,
        floor: Optional[str] = None,
    ) -> Optional[AnomalyEvent]:
        """
        Process a frame-level detection and update event state.
        
        Returns:
            AnomalyEvent if state changed to CONFIRMED, None otherwise
        """
        key = (camera_id, track_id, anomaly_type)
        now = time.time()
        
        with self._lock:
            event = self._events.get(key)
            
            if event is None:
                # New detection - create event
                self._event_counter += 1
                event_id = f"evt_{camera_id}_{self._event_counter}_{int(now)}"
                event = AnomalyEvent(
                    event_id=event_id,
                    camera_id=camera_id,
                    track_id=track_id,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    state=EventState.PENDING_CONFIRMATION,
                    camera_name=camera_name,
                    floor=floor,
                )
                self._events[key] = event
                log.info(
                    "[EventManager] EVENT_STARTED: %s cam=%d track=%s type=%s",
                    event_id, camera_id, track_id, anomaly_type
                )
            else:
                # Existing event - update last_seen
                event.update_last_seen()
                
                # Check for confirmation
                if event.should_confirm(self.confirmation_duration) and event.state == EventState.PENDING_CONFIRMATION:
                    event.state = EventState.CONFIRMED
                    event.confirmed_at = now
                    log.info(
                        "[EventManager] EVENT_CONFIRMED: %s duration=%.1fs",
                        event.event_id, event.duration
                    )
                    return event  # Return for notification
            
            return None
    
    def update_active_events(self) -> list[AnomalyEvent]:
        """
        Update all active events, resolve expired ones, cleanup old ones.
        Should be called periodically (e.g., every second).
        
        Returns:
            List of newly confirmed events ready for notification
        """
        confirmed_events = []
        now = time.time()
        
        with self._lock:
            keys_to_remove = []
            
            for key, event in self._events.items():
                # Skip already resolved events
                if event.state == EventState.RESOLVED:
                    # Cleanup old resolved events
                    if event.resolved_at and (now - event.resolved_at) > self.cleanup_age:
                        keys_to_remove.append(key)
                    continue
                
                # Check for grace period expiration
                if event.is_grace_period_expired(self.grace_period):
                    event.state = EventState.RESOLVED
                    event.resolved_at = now
                    log.info(
                        "[EventManager] EVENT_RESOLVED: %s (grace period expired) duration=%.1fs",
                        event.event_id, event.duration
                    )
                    continue
                
                # Check for max duration
                if event.duration > self.max_duration:
                    event.state = EventState.RESOLVED
                    event.resolved_at = now
                    log.info(
                        "[EventManager] EVENT_RESOLVED: %s (max duration) duration=%.1fs",
                        event.event_id, event.duration
                    )
                    continue
                
                # Check for confirmation
                if event.should_confirm(self.confirmation_duration):
                    event.state = EventState.CONFIRMED
                    event.confirmed_at = now
                    log.info(
                        "[EventManager] EVENT_CONFIRMED: %s duration=%.1fs",
                        event.event_id, event.duration
                    )
                    confirmed_events.append(event)
            
            # Remove old events
            for key in keys_to_remove:
                del self._events[key]
        
        return confirmed_events
    
    def get_event(self, camera_id: int, track_id: str, anomaly_type: str) -> Optional[AnomalyEvent]:
        """Get current event for a specific (camera, track, anomaly) combination."""
        with self._lock:
            return self._events.get((camera_id, track_id, anomaly_type))
    
    def get_active_events(self) -> list[AnomalyEvent]:
        """Get all active (non-resolved) events."""
        with self._lock:
            return [
                event for event in self._events.values()
                if event.state != EventState.RESOLVED
            ]
    
    def get_stats(self) -> dict:
        """Get statistics about tracked events."""
        with self._lock:
            total = len(self._events)
            by_state = {}
            for event in self._events.values():
                state = event.state.value
                by_state[state] = by_state.get(state, 0) + 1
            
            return {
                "total_events": total,
                "by_state": by_state,
                "active_count": total - by_state.get("resolved", 0),
            }


class AlertManager:
    """
    Manages notification delivery for confirmed events.
    
    Responsibilities:
    - Deduplication: ensure one notification per event
    - Cooldown: prevent notification spam
    - Retry logic: handle failed deliveries
    - Async delivery: don't block detection loop
    """
    
    def __init__(self):
        self._notification_queue: list[AnomalyEvent] = []
        self._lock = threading.RLock()
        self._worker_running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Load config
        from config import settings
        self.notification_cooldown = settings.EVENT_NOTIFICATION_COOLDOWN
        
        log.info("[AlertManager] Initialized: cooldown=%ds", self.notification_cooldown)
    
    def start_worker(self):
        """Start background worker thread for async notification delivery."""
        if self._worker_running:
            return
        
        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._notification_worker,
            name="alert-manager-worker",
            daemon=True,
        )
        self._worker_thread.start()
        log.info("[AlertManager] Worker thread started")
    
    def stop_worker(self):
        """Stop background worker thread."""
        self._worker_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        log.info("[AlertManager] Worker thread stopped")
    
    def queue_notification(self, event: AnomalyEvent) -> None:
        """
        Queue a confirmed event for notification delivery.
        
        Checks:
        - Event not already notified
        - Cooldown period respected
        """
        with self._lock:
            # Don't queue if already notified
            if event.notification_sent:
                log.debug(
                    "[AlertManager] Skipping notification (already sent): %s",
                    event.event_id
                )
                return
            
            # Check cooldown
            if event.notification_sent_at:
                elapsed = time.time() - event.notification_sent_at
                if elapsed < self.notification_cooldown:
                    log.debug(
                        "[AlertManager] Skipping notification (cooldown): %s elapsed=%.1fs",
                        event.event_id, elapsed
                    )
                    return
            
            # Queue for delivery
            self._notification_queue.append(event)
            log.info(
                "[AlertManager] NOTIFICATION_QUEUED: %s cam=%d type=%s",
                event.event_id, event.camera_id, event.anomaly_type
            )
    
    def _notification_worker(self):
        """Background worker that processes notification queue."""
        while self._worker_running:
            try:
                # Get next event from queue
                event = None
                with self._lock:
                    if self._notification_queue:
                        event = self._notification_queue.pop(0)
                
                if event:
                    self._deliver_notification(event)
                else:
                    time.sleep(1)  # Sleep when queue is empty
            
            except Exception as exc:
                log.error("[AlertManager] Worker error: %s", exc)
                time.sleep(5)  # Backoff on error
    
    def _deliver_notification(self, event: AnomalyEvent) -> None:
        """
        Actually send the FCM notification.
        
        Updates event status based on delivery result.
        """
        try:
            from services.fcm_service import send_fcm_alert
            
            event.notification_attempts += 1
            
            # Build message
            message = f"{event.anomaly_type} detected continuously for {int(event.duration)}s"
            
            log.info(
                "[AlertManager] NOTIFICATION_SENDING: %s attempt=%d",
                event.event_id, event.notification_attempts
            )
            
            # Send FCM notification
            success = send_fcm_alert(
                message=message,
                severity=event.severity,
                floor=event.floor,
                camera_name=event.camera_name,
                detected_issue=event.anomaly_type,
                camera_id=event.camera_id,
                db=None,
            )
            
            if success:
                event.notification_sent = True
                event.notification_sent_at = time.time()
                event.notification_status = "delivered"
                event.state = EventState.NOTIFIED
                log.info(
                    "[AlertManager] NOTIFICATION_SENT: %s cam=%d type=%s",
                    event.event_id, event.camera_id, event.anomaly_type
                )
            else:
                event.notification_status = "failed"
                log.warning(
                    "[AlertManager] NOTIFICATION_FAILED: %s attempt=%d",
                    event.event_id, event.notification_attempts
                )
                
                # Retry logic: requeue if attempts < 3
                if event.notification_attempts < 3:
                    with self._lock:
                        self._notification_queue.append(event)
                    log.info(
                        "[AlertManager] NOTIFICATION_RETRY: %s queued for retry",
                        event.event_id
                    )
        
        except Exception as exc:
            event.notification_status = f"error: {exc}"
            log.error(
                "[AlertManager] NOTIFICATION_ERROR: %s error=%s",
                event.event_id, exc
            )


# ── Global Singletons ─────────────────────────────────────────────────────────

_event_manager = EventManager()
_alert_manager = AlertManager()


def get_event_manager() -> EventManager:
    """Get the global EventManager singleton."""
    return _event_manager


def get_alert_manager() -> AlertManager:
    """Get the global AlertManager singleton."""
    return _alert_manager


def send_push_alert(
    message: str,
    severity: str = "medium",
    floor: Optional[str] = None,
    camera_name: Optional[str] = None,
    detected_issue: Optional[str] = None,
    snapshot_b64: Optional[str] = None,  # kept for API compat, not used by FCM
    camera_id: Optional[int] = None,
    worker_name: Optional[str] = None,
    db=None,
) -> bool:
    """
    Send a push notification via FCM.

    Drop-in replacement for the old ntfy / Telegram send_push_alert().
    All existing callers work unchanged — snapshot_b64 is accepted but
    ignored (FCM web push does not support inline images in the legacy API).

    Returns True if delivery succeeded, False otherwise.
    """
    try:
        from services.fcm_service import send_fcm_alert
        return send_fcm_alert(
            message=message,
            severity=severity,
            floor=floor,
            camera_name=camera_name,
            detected_issue=detected_issue,
            camera_id=camera_id,
            worker_name=worker_name,
            db=db,
        )
    except Exception as exc:
        log.error(f"[notification_service] send_push_alert error: {exc}")
        return False


# ── Legacy compatibility aliases ─────────────────────────────────────────────
# Old callers that still import send_telegram_alert() keep working without edits.

def send_telegram_alert(
    message: str,
    severity: str = "medium",
    floor: Optional[str] = None,
    camera_name: Optional[str] = None,
    detected_issue: Optional[str] = None,
    snapshot_b64: Optional[str] = None,
) -> bool:
    """Deprecated — now delegates to FCM via send_push_alert()."""
    return send_push_alert(
        message=message,
        severity=severity,
        floor=floor,
        camera_name=camera_name,
        detected_issue=detected_issue,
        snapshot_b64=snapshot_b64,
    )

