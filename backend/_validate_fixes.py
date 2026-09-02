import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import numpy as np
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

print("=" * 70)
print("VALIDATION SUITE: Zone-Based Multi-Model Frame Assignment")
print("=" * 70)

passed = 0
failed = 0
issues = []

def CHECK(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ PASS: {name}")
    else:
        failed += 1
        issues.append(f"FAIL: {name} — {detail}")
        print(f"  ❌ FAIL: {name} — {detail}")

# ── Test 1: detection_layer ────────────────────────────────────────────
print("\n[1] Detection layer integration")
from services.detection_layer import get_detector, detector as mod_det
from services.multi_model_detector import MultiModelDetectionAdapter
det = get_detector()
CHECK("get_detector() is MultiModelDetectionAdapter",
      isinstance(det, MultiModelDetectionAdapter),
      f"got {type(det).__name__}")
CHECK("module-level 'detector' is MultiModelDetectionAdapter",
      isinstance(mod_det, MultiModelDetectionAdapter),
      f"got {type(mod_det).__name__}")

# ── Test 2: Model registry loads 7 models now (cash enabled) ──────────
print("\n[2] Model registry")
from services.model_registry import get_registry, get_models_for_zone
registry = get_registry()
CHECK(f"registry has ≥7 models (cash_detection enabled now)",
      len(registry.models) >= 7,
      f"got {len(registry.models)} models: {list(registry.models.keys())}")
CHECK("cash_detection is loaded",
      "cash_detection" in registry.models,
      "cash_detection NOT in registry (enabled check failed)")
for key in ["yolov8x_coco", "hairnet_glove_detection", "fall_detection",
            "machine_visual_anomaly", "machine_sensor_anomaly",
            "object_throwing", "cash_detection"]:
    CHECK(f"model {key} loaded", key in registry.models)

# ── Test 3: Zone routing ───────────────────────────────────────────────
print("\n[3] Zone → model routing")
zone_tests = {
    'entrance':         {'yolov8x_coco', 'hairnet_glove_detection', 'fall_detection', 'object_throwing'},
    'dough_mixing':     {'yolov8x_coco', 'hairnet_glove_detection', 'fall_detection', 'machine_visual_anomaly', 'machine_sensor_anomaly'},
    'shop_counter':     {'yolov8x_coco', 'object_throwing', 'cash_detection'},
    'cashbox':          {'yolov8x_coco', 'cash_detection', 'object_throwing'},
    'lift':             {'yolov8x_coco', 'fall_detection', 'object_throwing'},
    'gas_section':      {'yolov8x_coco', 'hairnet_glove_detection', 'fall_detection'},
    'packing':          {'yolov8x_coco', 'hairnet_glove_detection', 'fall_detection'},
    'oven':             {'yolov8x_coco', 'hairnet_glove_detection', 'fall_detection'},
    'biscuit_cutting':  {'yolov8x_coco', 'hairnet_glove_detection', 'fall_detection', 'machine_visual_anomaly', 'machine_sensor_anomaly'},
    'unknown_zone':     {'yolov8x_coco'},
}
for zone, expected_set in zone_tests.items():
    actual = {m.key for m in get_models_for_zone(zone)}
    ok = actual == expected_set
    msg = "" if ok else f"expected {expected_set} got {actual}"
    CHECK(f"zone {zone}: models correct", ok, msg)

# ── Test 4: MultiModelDetector.detect() with zone_type ─────────────────
print("\n[4] MultiModelDetector inference with zone_type")
from services.multi_model_detector import get_multi_model_detector
mm = get_multi_model_detector()
dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

test_zones = ['entrance', 'dough_mixing', 'shop_counter', None]
label_checks = []
for z in test_zones:
    try:
        dets = mm.detect(dummy, zone_type=z, camera_id=99)
        labels = sorted({d.label for d in dets})
        label_checks.append((z, dets, labels))
        print(f"    zone={z!r:18s}: {len(dets)} dets  labels={labels}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        CHECK(f"detect(zone={z}) throws", False, str(e))

CHECK("MultiModelDetector detects zone=None without crash",
      len(label_checks) == 4, "")

# ── Test 5: cash_monitor label matching ────────────────────────────────
print("\n[5] Cash-monitor label matching")
from services.cash_monitor import _is_cash_label
cash_test_cases = [
    ("Cash", True), ("cash", True), ("banknote", True), ("rupee", True),
    ("5 BGN", True), ("100 EUR", True), ("500 INR", True), ("20 EUR", True),
    ("500 inr", True), ("10 BGN", True), ("2000 INR", True),
    ("Person", False), ("Car", False), ("NO-Bakery-Head-Cap", False),
    ("Fall", False), ("Worker Fall", False),
]
for lbl, expect in cash_test_cases:
    actual = _is_cash_label(lbl)
    CHECK(f"_is_cash_label({lbl!r}) → {expect}", actual == expect,
          f"got {actual}")

# ── Test 6: enterprise_runtime signature & zone passing ────────────────
print("\n[6] enterprise_runtime import & zone resolution")
try:
    from services.enterprise_runtime import EnterpriseRuntime
    import inspect
    sig = inspect.signature(EnterpriseRuntime.process_frame)
    src = inspect.getsource(EnterpriseRuntime.process_frame)
    CHECK("process_frame imports get_detector (not legacy singleton)",
          "get_detector()" in src and "from services.detection_layer import get_detector" in open(
              os.path.join(os.path.dirname(__file__), "services/enterprise_runtime.py")).read(),
          "enterprise_runtime still uses legacy 'detector' import")
    CHECK("process_frame reads camera.zone_type",
          "camera.zone_type" in src,
          "zone_type not extracted from Camera DB row")
    CHECK("process_frame reads camera.enabled_models_json",
          "enabled_models_json" in src,
          "per-camera enabled_models_json override ignored")
    CHECK("process_frame passes zone_type= to detector",
          "zone_type=zone_type" in src or 'zone_type=zone_type' in src.replace(" ", ""),
          "zone_type not passed into detector.detect()")
except Exception as e:
    import traceback; traceback.print_exc()
    CHECK("enterprise_runtime introspection", False, str(e))

# ── Test 7: video router zone_type flow ────────────────────────────────
print("\n[7] video upload router zone_type flow")
video_src = open(os.path.join(os.path.dirname(__file__), "routers/video.py"), encoding="utf-8").read()
CHECK("video upload accepts zone_type Form field",
      "zone_type: str = Form" in video_src,
      "POST /video/upload missing zone_type form param")
CHECK("process_video_job has zone_type parameter",
      "def process_video_job(job_id" in video_src and "zone_type" in video_src.split("def process_video_job")[1].split("):")[0],
      "process_video_job signature missing zone_type")
CHECK("process_frame_numpy called with zone_type=zone_type in video job",
      "zone_type=zone_type" in video_src,
      "zone_type not forwarded to yolo_service in video processing")

# ── Test 8: yolo_service label standardization coverage ─────────────────────────
print("\n[8] yolo_service label standardization coverage")
yolo_src = open(os.path.join(os.path.dirname(__file__), "services/yolo_service.py"), encoding="utf-8").read()
# Hairnet/glove raw model classes
for raw_label in ("Hair", "Hair_Cover", "Back_Palm", "Front_Palm", "Hand_Gloves"):
    CHECK(f"yolo_service handles raw class '{raw_label}'",
          f'"{raw_label}"' in yolo_src,
          f"raw class {raw_label} not covered in label standardization block")
# Cash raw labels
for suffix in ('" BGN"', '" EUR"', '" INR"'):
    CHECK(f"yolo_service handles cash suffix {suffix}",
          suffix in yolo_src or suffix.replace('"', '').strip() in yolo_src,
          f"cash label suffix {suffix} not found in standardization")
# Multi-model labels: Object Throwing + Bangles (added via injection block)
for mm_label in ('"Object Throwing"', '"throw"', '"bangle"', '"Bangles"', 'object_throwing'):
    mm_plain = mm_label.strip('"').lower()
    CHECK(f"yolo_service multi-model injection handles {mm_label}",
          mm_plain in yolo_src.lower(),
          f"multi-model label {mm_label} not found in _run_pipeline standardization block")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 70)
if failed:
    print("\n⚠️  Issues found:")
    for i in issues:
        print(f"   - {i}")
else:
    print("\n✅ All validation checks passed!")

sys.exit(0 if failed == 0 else 1)
