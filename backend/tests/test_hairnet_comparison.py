"""
backend/tests/test_hairnet_comparison.py
Comprehensive validation:
1. Compares raw YOLOv8m inference vs Application Pipeline on snapshots.
2. Verifies both classes: class 0 = 'hairnet' and class 1 = 'no_hairnet'.
3. Verifies tight head bounding boxes (no full-body bounding boxes).
4. Verifies absence of glove/palm class corruption.
5. Verifies person spatial association and headwear status.
"""
import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ultralytics import YOLO
from services.yolo_service import process_frame_numpy

SNAP_NO_HAIRNET = os.path.join(BASE_DIR, "uploads", "snapshots", "20260908_172011_940358_u2.jpg")
SNAP_HAIRNET = os.path.join(BASE_DIR, "uploads", "snapshots", "20260831_154033_980518_u2.jpg")
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "portable_models_package", "hairnet_glove_detection", "best.pt")

def test_snapshot(image_path: str, expected_class: str):
    print(f"\n{'='*70}")
    print(f"Testing snapshot: {os.path.basename(image_path)} (Expected: {expected_class})")
    print(f"{'='*70}")

    frame = cv2.imread(image_path)
    assert frame is not None, f"Failed to load frame {image_path}"
    fh, fw = frame.shape[:2]
    print(f"Frame resolution: {fw}x{fh}")

    # 1. Raw YOLOv8m Inference
    raw_model = YOLO(MODEL_PATH)
    assert raw_model.names == {0: "hairnet", 1: "no_hairnet"}, f"Unexpected model classes: {raw_model.names}"
    
    raw_results = raw_model.predict(frame, imgsz=960, conf=0.25, verbose=False)
    raw_dets = []
    for r in raw_results:
        for b in r.boxes:
            cid = int(b.cls[0])
            cname = raw_model.names[cid]
            conf = float(b.conf[0])
            xyxy = [round(float(v), 1) for v in b.xyxy[0]]
            bw = round(xyxy[2] - xyxy[0], 1)
            bh = round(xyxy[3] - xyxy[1], 1)
            raw_dets.append({"class_name": cname, "conf": conf, "xyxy": xyxy, "bw": bw, "bh": bh})
            print(f"  [RAW YOLO] {cname} conf={conf:.3f} bbox={xyxy} size={bw}x{bh}")

    raw_names = set(d["class_name"] for d in raw_dets)
    assert expected_class in raw_names, f"Expected {expected_class} in raw detections, got: {raw_names}"

    # 2. Pipeline Inference
    res = process_frame_numpy(frame, role="Bakery", zone_type="bakery")
    persons = res.get("persons", [])
    ui_dets = res.get("detections", [])
    
    print(f"\n  Pipeline persons: {len(persons)}, total UI detections: {len(ui_dets)}")
    head_ui_dets = [d for d in ui_dets if d.get("raw_label") in ("hairnet", "no_hairnet")]
    
    for hd in head_ui_dets:
        bw = hd["bbox"][2] - hd["bbox"][0]
        bh = hd["bbox"][3] - hd["bbox"][1]
        print(f"  [PIPELINE HEAD DET] raw_label={hd.get('raw_label')} label={hd.get('label')} conf={hd.get('confidence')} bbox={hd.get('bbox')} size={bw}x{bh}")
        
        # Verify box size: must be head box (< 250px high), NOT full person
        assert bh < 250, f"Head box unexpectedly large ({bh}px)! Full person box detected instead of head."

    # Verify no corrupted palm or glove/mask classes
    forbidden_classes = {"Back_Palm", "Front_Palm", "NO-Gloves", "Gloves", "NO-Mask", "Mask"}
    detected_labels = set(d.get("label") for d in ui_dets)
    found_forbidden = forbidden_classes.intersection(detected_labels)
    assert not found_forbidden, f"Found forbidden glove/mask classes in UI detections: {found_forbidden}"
    for p in persons:
        missing = set(p.get("ppe_missing", []))
        found = set(p.get("ppe_found", []))
        assert not missing.intersection({"NO-Gloves", "NO-Mask"}), f"Person has forbidden missing PPE: {missing}"
        assert not found.intersection({"Gloves", "Mask"}), f"Person has forbidden found PPE: {found}"

    # Verify matching raw and pipeline detections
    for rd in raw_dets:
        matched = False
        for pd in head_ui_dets:
            if pd.get("raw_label") == rd["class_name"]:
                pb = pd["bbox"]
                rb = rd["xyxy"]
                if abs(pb[0] - rb[0]) <= 2 and abs(pb[1] - rb[1]) <= 2:
                    matched = True
                    break
        assert matched, f"Raw detection {rd} was not matched by pipeline!"
    print("  ✓ Pipeline head detections 100% matched with Raw YOLO output!")

    # Verify person spatial association
    persons_with_head = [p for p in persons if p.get("head_bbox") is not None]
    print(f"  Persons with associated headwear: {len(persons_with_head)}/{len(persons)}")
    for p in persons_with_head:
        print(f"    Person bbox={p['bbox']} -> headwear_status={p.get('headwear_status')}, label={p.get('headwear_label')}, conf={p.get('headwear_conf')}, head_bbox={p.get('head_bbox')}")
        if expected_class == "hairnet" and p.get("headwear_label") == "hairnet":
            assert p.get("headwear_status") == "compliant"
        elif expected_class == "no_hairnet" and p.get("headwear_label") == "no_hairnet":
            assert p.get("headwear_status") == "violation"

    print(f"  ✓ Snapshot {os.path.basename(image_path)} PASSED all checks!")

def main():
    print("======================================================================")
    print("STARTING FULL HAIRNET COMPARISON & FIDELITY SUITE")
    print("======================================================================")
    
    test_snapshot(SNAP_NO_HAIRNET, expected_class="no_hairnet")
    test_snapshot(SNAP_HAIRNET, expected_class="hairnet")
    
    print("\n" + "=" * 70)
    print("ALL TESTS IN FIDELITY SUITE PASSED!")
    print("======================================================================")

if __name__ == "__main__":
    main()
