"""Enterprise administration APIs; additive to all existing camera/alert APIs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from database import (AuditLog, CalibrationVersion, Camera, HealthLog,
                      RuleDefinition, RuleProfile, SurveillanceEvent, User,
                      WorkflowProfile, ZoneConfig, get_db)
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from routers.auth import get_current_user
from services import calibration_service, model_manager
from services.analytics_engine import analytics_engine
from sqlalchemy.orm import Session

router = APIRouter(tags=["enterprise-platform"])


class ProfilePayload(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    floor: Optional[str] = None
    definition: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class RulePayload(BaseModel):
    profile_id: Optional[int] = None
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    priority: str = "warning"
    enabled: bool = True
    zone_id: Optional[int] = None
    workflow_stage: Optional[str] = None
    required_events: List[str] = Field(default_factory=list)
    forbidden_events: List[str] = Field(default_factory=list)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    time_threshold_sec: Optional[float] = None
    confidence_threshold: Optional[float] = None
    cooldown_sec: float = 30
    escalation: Dict[str, Any] = Field(default_factory=dict)
    notification_targets: List[Dict[str, Any]] = Field(default_factory=list)


class RestorePayload(BaseModel):
    note: Optional[str] = "Restored from calibration history"


class ModelPayload(BaseModel):
    model_key: str = Field(min_length=1, max_length=150, pattern=r"^[a-zA-Z0-9_.-]+$")
    display_name: str = Field(min_length=1, max_length=200)
    model_type: str = Field(min_length=1, max_length=50)
    version: str = Field(min_length=1, max_length=100)
    provider: str = "local"
    artifact_path: Optional[str] = None
    class_map: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ZonePayload(BaseModel):
    camera_id: int
    zone_name: str = Field(min_length=1, max_length=200)
    polygon: List[List[int]]
    zone_type: Optional[str] = None
    display_name: Optional[str] = None
    color: Optional[str] = None
    priority: int = 0
    workflow_stage: Optional[str] = None


class EventPatchPayload(BaseModel):
    status: Optional[str] = Field(default=None, pattern="^(open|review|resolved|dismissed)$")
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _audit(
    db: Session, user: User, action: str, entity_type: str, entity_id: Any, after: Dict
) -> None:
    db.add(
        AuditLog(
            actor_user_id=user.id,
            action=action,
            entity_type=entity_type,
            entity_id=str(entity_id),
            after_json=json.dumps(after, default=str),
        )
    )
    db.commit()


@router.get("/platform/models")
@router.get("/models")
def models(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    return model_manager.list_models(db)


@router.post("/platform/models", status_code=201)
@router.post("/models", status_code=201)
def register_model(
    body: ModelPayload,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        result = model_manager.register_model(db, body.model_dump())
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    _audit(
        db,
        user,
        "model.registered",
        "model",
        result["id"],
        {"model_key": result["model_key"]},
    )
    return result


@router.get("/platform/health")
def health(
    component_type: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    q = db.query(HealthLog)
    if component_type:
        q = q.filter(HealthLog.component_type == component_type)
    rows = q.order_by(HealthLog.measured_at.desc()).limit(min(limit, 500)).all()
    return [
        {
            "component_type": row.component_type,
            "component_id": row.component_id,
            "status": row.status,
            "measured_at": row.measured_at.isoformat(),
            "metrics": json.loads(row.metrics_json or "{}"),
            "message": row.message,
        }
        for row in rows
    ]


@router.get("/platform/analytics/summary")
def analytics_summary(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Read-only analytics: event/alert trends are intentionally independent from alert delivery."""
    return analytics_engine.summary(db, days)


@router.get("/platform/alert-cases")
def list_alert_cases(
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    from database import AlertCase

    rows = db.query(AlertCase).order_by(AlertCase.opened_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "public_id": r.public_id,
            "camera_id": r.camera_id,
            "severity": r.severity,
            "status": r.status,
            "title": r.title,
            "opened_at": r.opened_at.isoformat(),
            "details": json.loads(r.details_json or "{}"),
        }
        for r in rows
    ]


@router.post("/platform/alert-cases/{case_id}/{action}")
def action_alert_case(
    case_id: int,
    action: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from datetime import datetime

    from database import AlertCase

    if action not in {"acknowledge", "dismiss"}:
        raise HTTPException(422, "Action must be acknowledge or dismiss")
    row = db.query(AlertCase).filter(AlertCase.id == case_id).first()
    if not row:
        raise HTTPException(404, "Alert case not found")
    if action == "acknowledge":
        row.status = "acknowledged"
        row.acknowledged_at = datetime.utcnow()
        row.acknowledged_by_user_id = user.id
    else:
        row.status = "dismissed"
        row.closed_at = datetime.utcnow()
    db.commit()
    _audit(
        db, user, f"alert_case.{action}", "alert_case", case_id, {"status": row.status}
    )
    return {"id": row.id, "status": row.status}


@router.get("/platform/workflow-profiles")
def list_workflows(
    db: Session = Depends(get_db), _user: User = Depends(get_current_user)
):
    rows = db.query(WorkflowProfile).order_by(WorkflowProfile.name).all()
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "floor": r.floor,
            "enabled": r.enabled,
            "version": r.version,
            "definition": json.loads(r.definition_json or "{}"),
        }
        for r in rows
    ]


@router.post("/platform/workflow-profiles", status_code=201)
def create_workflow(
    body: ProfilePayload,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if db.query(WorkflowProfile).filter(WorkflowProfile.name == body.name).first():
        raise HTTPException(409, "Workflow profile name already exists")
    row = WorkflowProfile(
        name=body.name,
        description=body.description,
        floor=body.floor,
        enabled=body.enabled,
        definition_json=json.dumps(body.definition),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    _audit(
        db,
        user,
        "workflow_profile.created",
        "workflow_profile",
        row.id,
        {"name": row.name},
    )
    return {"id": row.id, "name": row.name, "version": row.version}


@router.get("/platform/rule-profiles")
def list_rule_profiles(
    db: Session = Depends(get_db), _user: User = Depends(get_current_user)
):
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "enabled": r.enabled,
            "definition": json.loads(r.definition_json or "{}"),
        }
        for r in db.query(RuleProfile).order_by(RuleProfile.name).all()
    ]


@router.post("/platform/rule-profiles", status_code=201)
def create_rule_profile(
    body: ProfilePayload,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if db.query(RuleProfile).filter(RuleProfile.name == body.name).first():
        raise HTTPException(409, "Rule profile name already exists")
    row = RuleProfile(
        name=body.name,
        description=body.description,
        enabled=body.enabled,
        definition_json=json.dumps(body.definition),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    _audit(db, user, "rule_profile.created", "rule_profile", row.id, {"name": row.name})
    return {"id": row.id, "name": row.name}


@router.get("/platform/rules")
@router.get("/rules")
def list_rules(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    rows = db.query(RuleDefinition).order_by(RuleDefinition.id.desc()).all()
    return [
        {
            "id": r.id,
            "profile_id": r.profile_id,
            "name": r.name,
            "priority": r.priority,
            "enabled": r.enabled,
            "zone_id": r.zone_id,
            "workflow_stage": r.workflow_stage,
            "required_events": json.loads(r.required_events_json or "[]"),
            "forbidden_events": json.loads(r.forbidden_events_json or "[]"),
            "conditions": json.loads(r.conditions_json or "{}"),
            "cooldown_sec": r.cooldown_sec,
            "notification_targets": json.loads(r.notification_targets_json or "[]"),
        }
        for r in rows
    ]


@router.post("/platform/rules", status_code=201)
@router.post("/rules", status_code=201)
def create_rule(
    body: RulePayload,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if (
        body.profile_id
        and not db.query(RuleProfile).filter(RuleProfile.id == body.profile_id).first()
    ):
        raise HTTPException(404, "Rule profile not found")
    row = RuleDefinition(
        profile_id=body.profile_id,
        name=body.name,
        description=body.description,
        priority=body.priority,
        enabled=body.enabled,
        zone_id=body.zone_id,
        workflow_stage=body.workflow_stage,
        required_events_json=json.dumps(body.required_events),
        forbidden_events_json=json.dumps(body.forbidden_events),
        conditions_json=json.dumps(body.conditions),
        time_threshold_sec=body.time_threshold_sec,
        confidence_threshold=body.confidence_threshold,
        cooldown_sec=body.cooldown_sec,
        escalation_json=json.dumps(body.escalation),
        notification_targets_json=json.dumps(body.notification_targets),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    _audit(
        db,
        user,
        "rule.created",
        "rule",
        row.id,
        {"name": row.name, "required_events": body.required_events},
    )
    return {"id": row.id, "name": row.name}


@router.put("/rules/{rule_id}")
@router.put("/platform/rules/{rule_id}")
def update_rule(
    rule_id: int,
    body: RulePayload,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    row = db.query(RuleDefinition).filter(RuleDefinition.id == rule_id).first()
    if not row:
        raise HTTPException(404, "Rule not found")
    for attr, value in {
        "profile_id": body.profile_id,
        "name": body.name,
        "description": body.description,
        "priority": body.priority,
        "enabled": body.enabled,
        "zone_id": body.zone_id,
        "workflow_stage": body.workflow_stage,
        "time_threshold_sec": body.time_threshold_sec,
        "confidence_threshold": body.confidence_threshold,
        "cooldown_sec": body.cooldown_sec,
    }.items():
        setattr(row, attr, value)
    row.required_events_json = json.dumps(body.required_events)
    row.forbidden_events_json = json.dumps(body.forbidden_events)
    row.conditions_json = json.dumps(body.conditions)
    row.escalation_json = json.dumps(body.escalation)
    row.notification_targets_json = json.dumps(body.notification_targets)
    db.commit()
    _audit(db, user, "rule.updated", "rule", rule_id, {"name": row.name})
    return {"id": row.id, "name": row.name, "enabled": row.enabled}


@router.get("/events")
def list_events(
    limit: int = Query(100, ge=1, le=500),
    event_type: Optional[str] = None,
    camera_id: Optional[int] = None,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    q = db.query(SurveillanceEvent)
    if event_type:
        q = q.filter(SurveillanceEvent.event_type == event_type)
    if camera_id:
        q = q.filter(SurveillanceEvent.camera_id == camera_id)
    rows = q.order_by(SurveillanceEvent.occurred_at.desc()).limit(limit).all()
    return [_event_to_dict(row) for row in rows]


@router.get("/events/{event_id}")
def get_event(
    event_id: str,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    row = (
        db.query(SurveillanceEvent)
        .filter(SurveillanceEvent.event_id == event_id)
        .first()
    )
    if not row:
        raise HTTPException(404, "Event not found")
    return _event_to_dict(row)


@router.patch("/events/{event_id}")
def patch_event(
    event_id: str,
    body: EventPatchPayload,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    row = (
        db.query(SurveillanceEvent)
        .filter(SurveillanceEvent.event_id == event_id)
        .first()
    )
    if not row:
        raise HTTPException(404, "Event not found")
    payload = json.loads(row.payload_json or "{}")
    if body.status:
        payload["status"] = body.status
        payload["resolved_at"] = (
            datetime.utcnow().isoformat()
            if body.status in {"resolved", "dismissed"}
            else payload.get("resolved_at")
        )
    payload.update(body.metadata)
    row.payload_json = json.dumps(payload, default=str)
    db.commit()
    _audit(db, user, "event.updated", "event", event_id, payload)
    return _event_to_dict(row)


@router.get("/zones")
def list_zones(
    camera_id: Optional[int] = None,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    q = db.query(ZoneConfig)
    if camera_id:
        q = q.filter(ZoneConfig.camera_id == camera_id)
    return [_zone_to_dict(row) for row in q.order_by(ZoneConfig.camera_id, ZoneConfig.id).all()]


@router.post("/zones", status_code=201)
def create_zone(
    body: ZonePayload,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not db.query(Camera).filter(Camera.id == body.camera_id).first():
        raise HTTPException(404, "Camera not found")
    row = ZoneConfig(
        camera_id=body.camera_id,
        zone_name=body.zone_name,
        polygon_json=json.dumps(body.polygon),
        zone_type=body.zone_type,
        display_name=body.display_name,
        color=body.color,
        priority=body.priority,
        workflow_stage=body.workflow_stage,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    from services import zone_service

    zone_service.invalidate_zone_cache(body.camera_id)
    _audit(db, user, "zone.created", "zone", row.id, _zone_to_dict(row))
    return _zone_to_dict(row)


def _event_to_dict(row: SurveillanceEvent) -> Dict[str, Any]:
    payload = json.loads(row.payload_json or "{}")
    return {
        "id": row.event_id,
        "event_type": row.event_type,
        "severity": payload.get("severity") or payload.get("priority"),
        "camera_id": row.camera_id,
        "zone_id": row.zone_id,
        "track_id": row.track_id,
        "timestamp": row.occurred_at.isoformat(),
        "rule_id": payload.get("rule_id"),
        "confidence": row.confidence,
        "message": payload.get("message") or payload.get("title"),
        "snapshot_path": payload.get("snapshot_path"),
        "clip_path": payload.get("clip_path"),
        "metadata": payload,
        "status": payload.get("status", "open"),
        "created_at": row.occurred_at.isoformat(),
        "resolved_at": payload.get("resolved_at"),
    }


def _zone_to_dict(row: ZoneConfig) -> Dict[str, Any]:
    return {
        "id": row.id,
        "camera_id": row.camera_id,
        "zone_name": row.zone_name,
        "polygon": json.loads(row.polygon_json or "[]"),
        "zone_type": row.zone_type,
        "display_name": row.display_name,
        "color": row.color,
        "priority": row.priority,
        "workflow_stage": row.workflow_stage,
        "is_active": row.is_active,
    }


@router.get("/cameras/{camera_id}/calibrations")
def calibration_history(
    camera_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    rows = (
        db.query(CalibrationVersion)
        .filter(CalibrationVersion.camera_id == camera_id)
        .order_by(CalibrationVersion.version.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "version": r.version,
            "snapshot_path": r.snapshot_path,
            "change_note": r.change_note,
            "created_at": r.created_at.isoformat(),
            "created_by_user_id": r.created_by_user_id,
            "zones": json.loads(r.zones_json or "[]"),
        }
        for r in rows
    ]


@router.post("/cameras/{camera_id}/calibrations/snapshot", status_code=201)
def create_calibration_snapshot(
    camera_id: int,
    note: str = "",
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from services import camera_manager

    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(404, "Camera not found")
    version = calibration_service.save_version(
        db, camera_id, user.id, camera_manager.get_latest_frame(camera_id), note
    )
    _audit(db, user, "calibration.versioned", "camera", camera_id, {"version": version})
    return {"camera_id": camera_id, "version": version, "status": "calibrated"}


@router.post("/cameras/{camera_id}/calibrations/{version}/restore")
def restore_calibration(
    camera_id: int,
    version: int,
    body: RestorePayload,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from database import ZoneConfig
    from services import zone_service

    source = (
        db.query(CalibrationVersion)
        .filter(
            CalibrationVersion.camera_id == camera_id,
            CalibrationVersion.version == version,
        )
        .first()
    )
    if not source:
        raise HTTPException(404, "Calibration version not found")
    try:
        zones = json.loads(source.zones_json)
        for zone in zones:
            row = (
                db.query(ZoneConfig)
                .filter(
                    ZoneConfig.camera_id == camera_id,
                    ZoneConfig.zone_name == zone["zone_name"],
                )
                .first()
            )
            if row:
                row.polygon_json = zone["polygon_json"]
                for key in (
                    "zone_type",
                    "display_name",
                    "color",
                    "priority",
                    "workflow_stage",
                ):
                    setattr(row, key, zone.get(key))
            else:
                db.add(ZoneConfig(camera_id=camera_id, **zone))
        db.commit()
        new_version = calibration_service.save_version(
            db, camera_id, user.id, None, body.note or "Restored calibration"
        )
        zone_service.invalidate_zone_cache(camera_id)
        _audit(
            db,
            user,
            "calibration.restored",
            "camera",
            camera_id,
            {"from_version": version, "version": new_version},
        )
        return {
            "camera_id": camera_id,
            "restored_from": version,
            "version": new_version,
        }
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(422, f"Invalid calibration snapshot: {exc}")
