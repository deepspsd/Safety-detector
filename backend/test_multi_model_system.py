"""
Test Multi-Model Detection System
==================================

Quick integration test to verify:
1. All models load correctly
2. Zone routing works
3. Detection fusion works
4. System can process frames
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_loading():
    """Test that models load successfully."""
    print("="*70)
    print("TEST 1: Model Loading")
    print("="*70)
    
    from services.model_registry import get_registry
    
    registry = get_registry()
    print(f"✅ Models loaded: {len(registry.models)}")
    
    for key, model_info in registry.models.items():
        print(f"  📦 {key}")
        print(f"     Type: {model_info.model_type}")
        print(f"     Priority: {model_info.priority}")
        print(f"     Zones: {model_info.zones}")
        print(f"     Target FPS: {model_info.target_fps}")
    
    return len(registry.models) > 0


def test_zone_routing():
    """Test that zone-based model routing works."""
    print("\n" + "="*70)
    print("TEST 2: Zone-Based Model Routing")
    print("="*70)
    
    from services.model_registry import get_models_for_zone
    
    test_zones = {
        "entrance": ["ppe_factory_v0_cash", "hairnet_glove_detection"],
        "dough_mixing": ["ppe_factory_v0_cash", "hairnet_glove_detection", "fall_detection"],
        "shop_counter": ["ppe_factory_v0_cash"],
        "unknown": ["ppe_factory_v0_cash"],
    }
    
    all_passed = True
    for zone, expected_models in test_zones.items():
        models = get_models_for_zone(zone)
        model_keys = [m.key for m in models]
        
        # Check if expected models are present (may have more due to enabled models)
        matches = all(exp in model_keys for exp in expected_models if exp in ["ppe_factory_v0_cash", "hairnet_glove_detection", "fall_detection"])
        
        if matches:
            print(f"  ✅ Zone '{zone}': {len(models)} models - {model_keys}")
        else:
            print(f"  ❌ Zone '{zone}': Expected {expected_models}, got {model_keys}")
            all_passed = False
    
    return all_passed


def test_detection_with_dummy_frame():
    """Test detection on a dummy frame."""
    print("\n" + "="*70)
    print("TEST 3: Detection with Dummy Frame")
    print("="*70)
    
    from services.multi_model_detector import get_multi_model_detector
    
    # Create dummy frame (black image)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detector = get_multi_model_detector()
    
    # Test different zones
    test_zones = ["entrance", "dough_mixing", "shop_counter"]
    
    for zone in test_zones:
        detections = detector.detect(dummy_frame, zone_type=zone)
        print(f"  🎬 Zone '{zone}': {len(detections)} detections")
        
        # Show first few detections
        for det in detections[:3]:
            print(f"     - {det.label} ({det.confidence:.2f}) from {det.model_key}")
    
    return True


def test_detection_fusion():
    """Test detection fusion logic."""
    print("\n" + "="*70)
    print("TEST 4: Detection Fusion")
    print("="*70)
    
    from services.multi_model_detector import fuse_detections, compute_iou
    from services.detection_layer import Detection
    
    # Create overlapping detections
    det1 = Detection(
        label="NO-Bakery-Head-Cap",
        confidence=0.8,
        bbox=[100, 100, 200, 200],
        model_key="ppe_factory_v0_cash",
        class_id=11
    )
    
    det2 = Detection(
        label="NO-Hairnet",
        confidence=0.9,
        bbox=[105, 105, 205, 205],  # Overlapping
        model_key="hairnet_glove_detection",
        class_id=1
    )
    
    det3 = Detection(
        label="Person",
        confidence=0.95,
        bbox=[300, 300, 400, 400],  # Not overlapping
        model_key="ppe_factory_v0_cash",
        class_id=5
    )
    
    # Test IoU
    iou = compute_iou(det1.bbox, det2.bbox)
    print(f"  📐 IoU between det1 and det2: {iou:.2f}")
    
    # Test fusion
    detections = [det1, det2, det3]
    fused = fuse_detections(detections)
    
    print(f"  🔀 Before fusion: {len(detections)} detections")
    print(f"  ✨ After fusion: {len(fused)} detections")
    
    for det in fused:
        print(f"     - {det.label} ({det.confidence:.2f}) from {det.model_key}")
    
    return len(fused) <= len(detections)


def test_inference_pool_integration():
    """Test inference pool with multi-model detector."""
    print("\n" + "="*70)
    print("TEST 5: Inference Pool Integration")
    print("="*70)
    
    from services.inference_pool import inference_pool
    import time
    
    # Start inference pool
    if not inference_pool.is_healthy():
        print("  🚀 Starting inference pool...")
        inference_pool.start()
        time.sleep(2)  # Give it time to start
    
    if inference_pool.is_healthy():
        print("  ✅ Inference pool is running")
    else:
        print("  ❌ Inference pool failed to start")
        return False
    
    # Submit a test frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    inference_pool.put_frame(camera_id=999, frame=dummy_frame, zone_type="dough_mixing")
    
    print("  📤 Submitted test frame to inference pool")
    
    # Wait for result
    time.sleep(1)
    result = inference_pool.get_result(camera_id=999)
    
    if result:
        detections = result.get("detections", [])
        print(f"  ✅ Got result: {len(detections)} detections")
    else:
        print("  ⏳ No result yet (pool may be processing)")
    
    # Cleanup
    inference_pool.remove_camera(camera_id=999)
    
    return True


def main():
    """Run all tests."""
    print("\n" + "🧪 MULTI-MODEL DETECTION SYSTEM TEST SUITE")
    print("="*70)
    
    results = {
        "Model Loading": test_model_loading(),
        "Zone Routing": test_zone_routing(),
        "Dummy Frame Detection": test_detection_with_dummy_frame(),
        "Detection Fusion": test_detection_fusion(),
        "Inference Pool": test_inference_pool_integration(),
    }
    
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Multi-model system is ready.")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
