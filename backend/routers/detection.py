"""
WebSocket detection router — v4.1

Architecture: Two concurrent coroutines running via asyncio.gather():
  1. read_messages() — ALWAYS listening for WS messages; filter changes
                       are applied to shared state INSTANTLY, even while
                       YOLO inference is running in a background thread.
  2. process_frames() — pulls latest frame from queue, runs inference in a
                        thread-pool executor (keeps event loop free), sends
                        the annotated response back to the client.

This eliminates the core bug: previously receive_text() was only called
AFTER inference completed, meaning filter-update messages sat unread in the
OS socket buffer for 200-400ms (one inference cycle) and were often lost
when the user stopped the stream before the next cycle.
"""
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from database import get_db, User
from auth_utils import decode_token
from services import yolo_service, face_service
from services.alert_service import save_alert
from config import settings

router = APIRouter(tags=["detection"])

# Single-thread executor — YOLO/PyTorch models are NOT thread-safe,
# so we serialise inference while still freeing the asyncio event loop.
_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")

# Per-user cooldown: dict[user_id → last_alert_timestamp]
_last_alert_time: dict = {}


async def _get_user(token: str, db: Session) -> User | None:
    payload = decode_token(token)
    if not payload:
        return None
    return db.query(User).filter(User.email == payload.get("sub")).first()


@router.websocket("/ws/detect")
async def detection_websocket(websocket: WebSocket):
    await websocket.accept()
    db: Session = next(get_db())

    try:
        # ── Auth handshake ──────────────────────────────────────
        auth_raw  = await websocket.receive_text()
        auth_data = json.loads(auth_raw)
        user      = await _get_user(auth_data.get("token", ""), db)

        if not user:
            await websocket.send_json({"error": "Unauthorized"})
            await websocket.close()
            return

        role = user.role or "Home"

        print(f"\n[WS] ===== NEW CONNECTION =====")
        print(f"[WS] User: {user.name} | Role: {role}")

        # ── Shared mutable state (safe: both coroutines on same event-loop thread) ──
        state = {
            "filters":     list(auth_data.get("filters", [])),
            "frame_count": 0,
            "alive":       True,
        }

        print(f"[WS] Handshake filters ({len(state['filters'])}): {state['filters']}")

        await websocket.send_json({
            "status":         "connected",
            "role":           role,
            "user":           user.name,
            "active_filters": state["filters"],
        })

        loop        = asyncio.get_event_loop()
        frame_queue = asyncio.Queue(maxsize=2)   # holds at most 2 unprocessed frames

        # ────────────────────────────────────────────────────────
        # Coroutine 1 — Message reader
        # Runs CONCURRENTLY with process_frames().
        # Because it is always suspended at `await websocket.receive_text()`,
        # the asyncio event loop can switch to it the instant a new message
        # arrives — even while YOLO is computing in its background thread.
        # ────────────────────────────────────────────────────────
        async def read_messages():
            while state["alive"]:
                raw = await websocket.receive_text()
                msg = json.loads(raw)

                # Always read filter from every message
                if "filters" in msg:
                    nf = msg["filters"]
                    if nf != state["filters"]:
                        print(f"[WS] ✅ FILTER UPDATED: {state['filters']} → {nf}")
                        state["filters"] = nf

                if msg.get("frame"):
                    # Drop the oldest queued frame to keep only the freshest
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await frame_queue.put(msg["frame"])
                elif "filters" in msg:
                    # Standalone filter-update (no frame) — just acknowledge
                    try:
                        await websocket.send_json({
                            "status":         "filters_updated",
                            "active_filters": state["filters"],
                        })
                    except Exception:
                        pass

        # ────────────────────────────────────────────────────────
        # Coroutine 2 — Frame processor
        # Waits for frames from the queue, then runs inference in a
        # background thread (run_in_executor).  The `await` releases
        # the event loop so read_messages() can process filter updates
        # while YOLO is computing.
        # ────────────────────────────────────────────────────────
        async def process_frames():
            while state["alive"]:
                # Wait for the next frame (5-s timeout guards against hangs)
                try:
                    b64_frame = await asyncio.wait_for(frame_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue

                state["frame_count"] += 1
                frame_num = state["frame_count"]

                # Snapshot the CURRENT filter — this is always up-to-date because
                # read_messages() updates state["filters"] the moment a new
                # filter message hits the socket, not after inference finishes.
                det_filters = list(state["filters"]) if state["filters"] else None

                if frame_num % 20 == 1:
                    print(f"[WS] Frame#{frame_num} | filters={det_filters}")

                try:
                    if role == "Home":
                        result = face_service.process_face_frame(b64_frame, user.id, db)
                        detections       = [
                            {"label": f["label"], "confidence": f["confidence"], "bbox": f["bbox"]}
                            for f in result.get("faces", [])
                        ]
                        violations_count = 1 if not result.get("is_compliant") else 0
                        persons_count    = len(result.get("faces", []))
                    else:
                        inference_fn = partial(
                            yolo_service.process_frame,
                            b64_frame, role,
                            detection_filters=det_filters,
                        )
                        # ← await releases event loop; read_messages() runs here
                        result = await loop.run_in_executor(_inference_executor, inference_fn)

                        detections       = result.get("detections", [])
                        violations_count = result.get("violations_count", 0)
                        persons_count    = result.get("persons_count",    0)

                        if frame_num % 20 == 1:
                            det_labels = [d["label"] for d in detections]
                            missing    = result.get("missing_items", [])
                            print(f"[YOLO] Frame#{frame_num} det={det_labels} "
                                  f"missing={missing} filters_used={det_filters} "
                                  f"compliant={result.get('is_compliant')}")

                    # ── Build response ──────────────────────────────
                    response = {
                        "annotated_frame":  result.get("annotated_frame"),
                        "detections":       detections,
                        "is_compliant":     result.get("is_compliant",  True),
                        "missing_items":    result.get("missing_items", []),
                        "violations_count": violations_count,
                        "persons_count":    persons_count,
                        "alert_message":    result.get("alert_message"),
                        "severity":         result.get("severity"),
                        "frame_count":      frame_num,
                        "persons":          result.get("persons", []),
                        "model_mode":       result.get("model_mode", "unknown"),
                        "active_filters":   state["filters"],
                    }

                    # ── Save alert (cooldown + confidence gate) ─────
                    if not result.get("is_compliant") and result.get("alert_message"):
                        uid      = user.id
                        now      = time.time()
                        top_conf = max(
                            (d.get("confidence", 0) for d in detections), default=0
                        )
                        conf_ok = top_conf >= settings.MIN_VIOLATION_CONF

                        if conf_ok and now - _last_alert_time.get(uid, 0) > settings.ALERT_COOLDOWN:
                            _last_alert_time[uid] = now
                            missing = result.get("missing_items", [])
                            save_alert(
                                db=db,
                                user_id=uid,
                                message=result["alert_message"],
                                role=role,
                                severity=result["severity"],
                                detected_issue=", ".join(missing) if missing else result["alert_message"],
                                confidence=round(top_conf, 3),
                                snapshot_b64=result.get("snapshot_b64"),
                            )
                            response["alert_saved"] = True

                    await websocket.send_json(response)

                except Exception as e:
                    if state["alive"]:
                        print(f"[WS] Frame#{frame_num} error: {e}")
                    # Stop processing silently on send errors (client disconnected)
                    state["alive"] = False
                    return

        # ── Run both coroutines concurrently ─────────────────────
        try:
            await asyncio.gather(read_messages(), process_frames())
        except WebSocketDisconnect:
            print("[WS] Client disconnected")
        except Exception as gather_err:
            print(f"[WS] Session ended: {gather_err}")
        finally:
            state["alive"] = False

    except WebSocketDisconnect:
        print("[WS] Client disconnected during handshake")
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        db.close()
