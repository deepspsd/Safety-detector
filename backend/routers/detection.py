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

from auth_utils import decode_token
from config import settings
from database import User, UserConfig, get_db
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from routers.users import _parse_custom_ppe
from services import face_service, yolo_service
from services.alert_service import save_alert
from sqlalchemy.orm import Session

router = APIRouter(tags=["detection"])

# Single-thread executor — YOLO/PyTorch models are NOT thread-safe,
# so we serialise inference while still freeing the asyncio event loop.
_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")

# Per-user cooldown: dict[user_id → last_alert_timestamp]
_last_alert_time: dict = {}
_last_phone_alert_time: dict = {}  # separate cooldown for phone alerts


def _fire_attendance(face_result: dict, camera_id) -> None:
    """
    Given a face_service result dict, clock-in every positively identified employee.
    handle_face_match deduplicates within the same calendar day automatically.
    """
    recognized = face_result.get("recognized_employees", {})
    if not recognized:
        return
    try:
        from services.attendance_service import handle_face_match

        for emp_id, conf in recognized.items():
            handle_face_match(
                camera_id=camera_id,
                employee_id=emp_id,
                confidence=conf,
            )
    except Exception as exc:
        print(f"[Attendance] _fire_attendance error: {exc}")


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
        auth_raw = await websocket.receive_text()
        auth_data = json.loads(auth_raw)
        user = await _get_user(auth_data.get("token", ""), db)

        if not user:
            await websocket.send_json({"error": "Unauthorized"})
            await websocket.close()
            return

        role = user.role or "Home"

        print(f"\n[WS] ===== NEW CONNECTION =====")
        print(f"[WS] User: {user.name} | Role: {role}")

        # For 'None' role: load custom_ppe_items from the user's saved config
        # as the INITIAL filter list; the frontend can still override mid-stream.
        handshake_filters = list(auth_data.get("filters", []))
        if role == "None" and not handshake_filters:
            db_config = (
                db.query(UserConfig).filter(UserConfig.user_id == user.id).first()
            )
            if db_config:
                handshake_filters = _parse_custom_ppe(db_config)
                print(f"[WS] Loaded custom PPE from DB: {handshake_filters}")

        # ── Shared mutable state (safe: both coroutines on same event-loop thread) ──
        state = {
            "filters": handshake_filters,
            "no_phone_zone": bool(auth_data.get("no_phone_zone", True)),  # default ON
            "zone_type": auth_data.get("zone_type") or "default",
            "frame_count": 0,
            "alive": True,
        }

        print(f"[WS] Handshake filters ({len(state['filters'])}): {state['filters']} | zone: {state['zone_type']}")

        await websocket.send_json(
            {
                "status": "connected",
                "role": role,
                "user": user.name,
                "active_filters": state["filters"],
                "zone_type": state["zone_type"],
            }
        )

        loop = asyncio.get_event_loop()
        frame_queue = asyncio.Queue(maxsize=2)  # holds at most 2 unprocessed frames

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

                # Read zone_type if sent
                if "zone_type" in msg:
                    nz = msg["zone_type"]
                    if nz and nz != state["zone_type"]:
                        print(f"[WS] 🏷️ ZONE UPDATED: {state['zone_type']} → {nz}")
                        state["zone_type"] = nz

                # Read no_phone_zone toggle from every message
                if "no_phone_zone" in msg:
                    npz = bool(msg["no_phone_zone"])
                    if npz != state["no_phone_zone"]:
                        print(
                            f"[WS] 📱 NO_PHONE_ZONE: {state['no_phone_zone']} → {npz}"
                        )
                        state["no_phone_zone"] = npz

                if msg.get("frame"):
                    # Drop the oldest queued frame to keep only the freshest
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await frame_queue.put(msg["frame"])
                elif "filters" in msg or "zone_type" in msg:
                    # Standalone filter/zone update (no frame) — just acknowledge
                    try:
                        await websocket.send_json(
                            {
                                "status": "filters_updated",
                                "active_filters": state["filters"],
                                "zone_type": state["zone_type"],
                            }
                        )
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
                        detections = [
                            {
                                "label": f["label"],
                                "confidence": f["confidence"],
                                "bbox": f["bbox"],
                            }
                            for f in result.get("faces", [])
                        ]
                        violations_count = 1 if not result.get("is_compliant") else 0
                        persons_count = len(result.get("faces", []))

                        # ── Attendance: clock-in any recognized employee ──
                        _fire_attendance(result, camera_id=None)

                    else:
                        inference_fn = partial(
                            yolo_service.process_frame,
                            b64_frame,
                            role,
                            detection_filters=det_filters,
                            no_phone_zone=state["no_phone_zone"],
                            zone_type=state.get("zone_type", "default"),
                        )
                        # ← await releases event loop; read_messages() runs here
                        result = await loop.run_in_executor(
                            _inference_executor, inference_fn
                        )

                        detections = result.get("detections", [])
                        violations_count = result.get("violations_count", 0)
                        persons_count = result.get("persons_count", 0)

                        if frame_num % 20 == 1:
                            det_labels = [d["label"] for d in detections]
                            missing = result.get("missing_items", [])
                            print(
                                f"[YOLO] Frame#{frame_num} det={det_labels} "
                                f"missing={missing} filters_used={det_filters} "
                                f"zone={state.get('zone_type')} "
                                f"compliant={result.get('is_compliant')}"
                            )

                        # ── Attendance: also run face recognition for non-Home roles ──
                        # Runs every 15 frames to avoid overloading CPU (doesn't affect PPE)
                        if frame_num % 15 == 1:
                            try:
                                face_result_attn = face_service.process_face_frame(
                                    b64_frame, user.id, db
                                )
                                _fire_attendance(face_result_attn, camera_id=None)
                            except Exception:
                                pass  # never let attendance failure break PPE flow

                    # ── Cash & Payee monitoring (webcam / browser WS) ───────────
                    _cash_alert = None
                    _payee_det = False
                    _payee_snap = None
                    _payee_id = None
                    _has_cash = result.get("cash_detected") or any(
                        "cash" in str(d.get("label", "")).lower() for d in detections
                    )
                    if _has_cash:
                        try:
                            from services import cash_monitor as _cm
                            raw_f = yolo_service.decode_frame(b64_frame)
                            _c_res = _cm.check_cash_zone(
                                db=db,
                                camera_id=9999,  # virtual webcam ID
                                detections=detections,
                                persons=result.get("persons", []),
                                floor="shop",
                                frame=raw_f,
                            )
                            if _c_res.get("theft_alert"):
                                _cash_alert = _c_res["theft_alert"]
                            _payee_det = _c_res.get("payee_detected", False)
                            _payee_snap = _c_res.get("payee_snapshot_b64")
                            _payee_id = _c_res.get("payee_id")
                        except Exception:
                            pass

                    # ── Build response ──────────────────────────────
                    response = {
                        "annotated_frame": result.get("annotated_frame"),
                        "detections": detections,
                        "is_compliant": result.get("is_compliant", True),
                        "missing_items": result.get("missing_items", []),
                        "violations_count": violations_count,
                        "persons_count": persons_count,
                        "alert_message": result.get("alert_message"),
                        "severity": result.get("severity"),
                        "frame_count": frame_num,
                        "persons": result.get("persons", []),
                        "model_mode": result.get("model_mode", "unknown"),
                        "active_filters": state["filters"],
                        "zone_type": state.get("zone_type", "default"),
                        # Phone detection fields
                        "phone_status": result.get("phone_status", "safe"),
                        "phone_detected": result.get("phone_detected", False),
                        "phone_alert": result.get("phone_alert"),
                        # Uniform detection fields
                        "uniform_detected": any(p.get("has_uniform") for p in result.get("persons", [])),
                        "uniform_violation": any(not p.get("has_uniform", True) for p in result.get("persons", [])),
                        # Cash & Payee monitoring fields
                        "cash_detected": _has_cash,
                        "cash_alert": _cash_alert,
                        "payee_detected": _payee_det,
                        "payee_snapshot_b64": _payee_snap,
                        "payee_id": _payee_id,
                    }

                    # ── Save PPE alert (cooldown + confidence gate) ─────
                    if not result.get("is_compliant") and result.get("alert_message"):
                        uid = user.id
                        now = time.time()
                        # Use max confidence from persons (always present) then PPE detections.
                        # This ensures College/Gloves/Goggles violations (person-level conf)
                        # always pass the gate even when no PPE bbox detections exist.
                        persons_conf = [
                            p.get("confidence", 0) for p in result.get("persons", [])
                        ]
                        dets_conf = [d.get("confidence", 0) for d in detections]
                        top_conf = max(persons_conf + dets_conf, default=0.5)
                        conf_ok = top_conf >= settings.MIN_VIOLATION_CONF

                        if (
                            conf_ok
                            and now - _last_alert_time.get(uid, 0)
                            > settings.ALERT_COOLDOWN
                        ):
                            _last_alert_time[uid] = now
                            missing = result.get("missing_items", [])
                            save_alert(
                                db=db,
                                user_id=uid,
                                message=result["alert_message"],
                                role=role,
                                severity=result["severity"],
                                detected_issue=(
                                    ", ".join(missing)
                                    if missing
                                    else result["alert_message"]
                                ),
                                confidence=round(top_conf, 3),
                                snapshot_b64=result.get("snapshot_b64"),
                            )
                            response["alert_saved"] = True

                    # ── Save PHONE alert (separate cooldown — 15 s) ─────
                    phone_sev = result.get("phone_severity")
                    phone_msg = result.get("phone_alert")
                    if phone_sev == "high" and phone_msg:
                        uid = user.id
                        now = time.time()
                        if now - _last_phone_alert_time.get(uid, 0) > 15:
                            _last_phone_alert_time[uid] = now
                            save_alert(
                                db=db,
                                user_id=uid,
                                message=phone_msg,
                                role=role,
                                severity="high",
                                detected_issue=phone_msg,
                                confidence=0.85,
                                snapshot_b64=result.get("snapshot_b64"),
                            )
                            response["phone_alert_saved"] = True

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
