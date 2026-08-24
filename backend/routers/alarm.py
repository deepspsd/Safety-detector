"""
routers/alarm.py — Local sound alarm endpoint
==============================================
POST /alarm/trigger   — Triggers an audible beep on the SERVER machine.
GET  /alarm/test      — Same as trigger, for manual admin testing.

Requires: LOCAL_ALARM_ENABLED=true in .env (off by default).
Windows:  Uses winsound.Beep().
Linux:    Tries 'beep' command, falls back to terminal bell.
"""

import logging
import platform

from fastapi import APIRouter, Depends

from config import settings
from routers.auth import get_current_user

log = logging.getLogger("alarm_router")
router = APIRouter(prefix="/alarm", tags=["alarm"])


def _play_beep() -> bool:
    """
    Play an audible beep on the server machine.
    Returns True if sound was played, False if unavailable.
    """
    if not settings.LOCAL_ALARM_ENABLED:
        return False

    freq = settings.LOCAL_ALARM_FREQ_HZ
    dur = settings.LOCAL_ALARM_DURATION_MS

    if platform.system() == "Windows":
        try:
            import winsound

            winsound.Beep(freq, dur)
            return True
        except Exception as exc:
            log.warning("[Alarm] winsound.Beep failed: %s", exc)
            return False
    else:
        # Linux / macOS — try 'beep' command, then terminal bell
        import os
        import subprocess

        try:
            result = subprocess.run(
                ["beep", "-f", str(freq), "-l", str(dur)],
                capture_output=True,
                timeout=2,
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass
        # Final fallback: terminal bell (audible only if server has a speaker)
        try:
            print("\a", end="", flush=True)
            return True
        except Exception:
            return False


@router.post("/trigger")
def trigger_alarm(current_user=Depends(get_current_user)):
    """Trigger local server beep. Requires LOCAL_ALARM_ENABLED=true in .env."""
    played = _play_beep()
    return {
        "triggered": played,
        "enabled": settings.LOCAL_ALARM_ENABLED,
        "platform": platform.system(),
    }


@router.get("/test")
def test_alarm(current_user=Depends(get_current_user)):
    """Test the local alarm (same as trigger, for manual admin checks)."""
    return trigger_alarm(current_user)
