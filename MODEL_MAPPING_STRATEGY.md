# Multi-Model Detection Strategy - OccuSafe

**Date**: September 2, 2026  
**Purpose**: Map available AI models to client requirements and zone types  
**Status**: Implementation Ready

---

## 📦 Available Models Inventory

### YOLO Models (Object Detection)

| Model | Location | Size | Classes | Purpose |
|-------|----------|------|---------|---------|
| **ppe_factory_v0_cash.pt** | `backend/` | 87.67 MB | 17 classes | **PRIMARY** - General PPE + Bakery compliance |
| **fall_detection/best.pt** | `portable_models_package/` | 15.33 MB | Fall detection | Worker safety monitoring |
| **hairnet_glove_detection/best.pt** | `portable_models_package/` | 21.52 MB | Hairnet, gloves | Food safety compliance |
| **cash_detection/yolo11m_finetuned.pt** | `portable_models_package/` | 38.68 MB | Banknotes | Cash handling zones |

### Behavioral/Action Models

| Model | Type | Size | Purpose | Framework |
|-------|------|------|---------|-----------|
| **person_action_recognition** | OpenVINO | 16 MB | Action classification | Intel OpenVINO |
| **hand_landmarks** | MediaPipe | 7.46 MB | Hand tracking | Google MediaPipe |
| **object_throwing** | PyTorch TRN | 50.3 MB | Throwing detection | Temporal Relation Network |

### Anomaly Detection Models

| Model | Type | Purpose | Use Case |
|-------|------|---------|----------|
| **machine_visual_anomaly** | PyTorch | Visual defects | Quality control (Phase 4) |
| **machine_sensor_anomaly** | sklearn | Predictive maintenance | IoT integration (Phase 4) |

---

## 🎯 Model-to-Requirement Mapping

### Ground Floor Zones

#### **Entrance (Inward/Outward)**
**Primary Detection Needs**: Invoice tracking, person counting, vehicle detection

| Requirement | Model(s) | Classes/Output | Priority |
|------------|----------|----------------|----------|
| Invoice/document in hand | `ppe_factory_v0_cash.pt` | Class 13: Document-in-hand | ✅ P0 |
| Person detection | `ppe_factory_v0_cash.pt` | Class 5: Person | ✅ P0 |
| Vehicle loading/unloading | `ppe_factory_v0_cash.pt` | Class 9: Vehicle | ✅ P0 |
| Head cap compliance | `hairnet_glove_detection/best.pt` | Hairnet detection | ✅ P1 |
| Fall detection | `fall_detection/best.pt` | Fall class | 🟡 P2 |

**Model Stack**: `ppe_factory_v0_cash.pt` + `hairnet_glove_detection/best.pt`

#### **Main Entrance (Loading/Unloading)**
**Primary Detection Needs**: Vehicle, person, safety gear

| Requirement | Model(s) | Classes/Output | Priority |
|------------|----------|----------------|----------|
| Vehicle detection | `ppe_factory_v0_cash.pt` | Class 9: Vehicle | ✅ P0 |
| Person detection | `ppe_factory_v0_cash.pt` | Class 5: Person | ✅ P0 |
| Safety vest | `ppe_factory_v0_cash.pt` | Class 7: Safety Vest | ✅ P1 |
| Hardhat | `ppe_factory_v0_cash.pt` | Class 0: Hardhat | ✅ P1 |

**Model Stack**: `ppe_factory_v0_cash.pt` only

#### **Dough Mixing Section**
**Primary Detection Needs**: Head cap, uniform, idle detection, bangles

| Requirement | Model(s) | Classes/Output | Priority |
|------------|----------|----------------|----------|
| Head cap detection | `ppe_factory_v0_cash.pt` | Class 10/11: Bakery-Head-Cap / NO-Head-Cap | ✅ P0 |
| Hairnet detection | `hairnet_glove_detection/best.pt` | Hairnet class | ✅ P0 |
| Gloves detection | `hairnet_glove_detection/best.pt` | Gloves class | ✅ P1 |
| Bangles detection | `ppe_factory_v0_cash.pt` | Class 12: Bangles | ✅ P0 |
| Person tracking (idle) | `ppe_factory_v0_cash.pt` | Class 5: Person (ByteTrack) | ✅ P0 |
| Fall detection | `fall_detection/best.pt` | Fall class | 🟡 P2 |

**Model Stack**: `ppe_factory_v0_cash.pt` + `hairnet_glove_detection/best.pt` + `fall_detection/best.pt`

#### **Biscuit Cutting Section**
**Primary Detection Needs**: Same as dough mixing

**Model Stack**: `ppe_factory_v0_cash.pt` + `hairnet_glove_detection/best.pt` + `fall_detection/best.pt`

#### **Oven Section**
**Primary Detection Needs**: Head cap, idle detection, gas cylinder tracking

| Requirement | Model(s) | Classes/Output | Priority |
|------------|----------|----------------|----------|
| Head cap detection | `ppe_factory_v0_cash.pt` + `hairnet_glove_detection/best.pt` | Combined | ✅ P0 |
| Cylinder tracking | `ppe_factory_v0_cash.pt` | Class 14: Cylinder | ✅ P1 |
| Person tracking (idle) | `ppe_factory_v0_cash.pt` | Class 5: Person | ✅ P0 |
| Fall detection | `fall_detection/best.pt` | Fall class | 🟡 P2 |

**Model Stack**: `ppe_factory_v0_cash.pt` + `hairnet_glove_detection/best.pt` + `fall_detection/best.pt`

#### **Packing Section**
**Primary Detection Needs**: Head cap, hand movement, idle detection

| Requirement | Model(s) | Classes/Output | Priority |
|------------|----------|----------------|----------|
| Head cap detection | `ppe_factory_v0_cash.pt` + `hairnet_glove_detection/best.pt` | Combined | ✅ P0 |
| Gloves detection | `hairnet_glove_detection/best.pt` | Gloves class | ✅ P0 |
| Hand movement tracking | `hand_landmarks/hand_landmarker.task` | 21 landmarks/hand | 🟡 P2 |
| Person tracking (idle) | `ppe_factory_v0_cash.pt` | Class 5: Person | ✅ P0 |

**Model Stack**: `ppe_factory_v0_cash.pt` + `hairnet_glove_detection/best.pt` + `hand_landmarks` (Phase 2)

### First Floor Zones

#### **Dough Mixing Tables (4 tables)**
**Same as Ground Floor Dough Section**

**Model Stack**: `ppe_factory_v0_cash.pt` + `hairnet_glove_detection/best.pt` + `fall_detection/best.pt`

#### **Lift (All 3 Floors)**
**Primary Detection Needs**: Person + item tracking, unauthorized access

| Requirement | Model(s) | Classes/Output | Priority |
|------------|----------|----------------|----------|
| Person detection | `ppe_factory_v0_cash.pt` | Class 5: Person | ✅ P0 |
| Cylinder tracking | `ppe_factory_v0_cash.pt` | Class 14: Cylinder | ✅ P1 |
| Exposed items | `ppe_factory_v0_cash.pt` | Class 15: Exposed-Item | ✅ P1 |

**Model Stack**: `ppe_factory_v0_cash.pt` only

### Second Floor Zones

#### **Gas/Boiler Section**
**Primary Detection Needs**: Head cap, cylinder tracking, idle monitoring

| Requirement | Model(s) | Classes/Output | Priority |
|------------|----------|----------------|----------|
| Head cap detection | `ppe_factory_v0_cash.pt` + `hairnet_glove_detection/best.pt` | Combined | ✅ P0 |
| Cylinder tracking | `ppe_factory_v0_cash.pt` | Class 14: Cylinder | ✅ P0 |
| Person tracking (idle) | `ppe_factory_v0_cash.pt` | Class 5: Person | ✅ P0 |
| Fall detection | `fall_detection/best.pt` | Fall class | 🟡 P2 |

**Model Stack**: `ppe_factory_v0_cash.pt` + `hairnet_glove_detection/best.pt` + `fall_detection/best.pt`

#### **Window Monitoring (Theft Detection)**
**Primary Detection Needs**: Object throwing, trajectory anomaly

| Requirement | Model(s) | Classes/Output | Priority |
|------------|----------|----------------|----------|
| Person action | `person_action_recognition` (OpenVINO) | Action classes | 🔴 P4 |
| Throwing detection | `object_throwing` (TRN) | Temporal throwing action | 🔴 P4 |
| Person tracking | `ppe_factory_v0_cash.pt` | Class 5: Person | ✅ P0 |

**Model Stack (Phase 4)**: `ppe_factory_v0_cash.pt` + `person_action_recognition` + `object_throwing`

### Shop Zones

#### **Shop Counter**
**Primary Detection Needs**: Uniform, absence detection, cash handling

| Requirement | Model(s) | Classes/Output | Priority |
|------------|----------|----------------|----------|
| Person presence | `ppe_factory_v0_cash.pt` | Class 5: Person | ✅ P0 |
| Uniform detection | `ppe_factory_v0_cash.pt` + Teachable Machine | Combined | ✅ P0 |
| Cashbox monitoring | `ppe_factory_v0_cash.pt` | Class 16: Cashbox | ✅ P1 |
| Banknote detection | `cash_detection/yolo11m_finetuned.pt` | Currency classes | 🟡 P2 |
| Hand tracking | `hand_landmarks/hand_landmarker.task` | Hand-to-pocket vs hand-to-cashbox | 🔴 P4 |

**Model Stack**: `ppe_factory_v0_cash.pt` + `cash_detection/yolo11m_finetuned.pt`

---

## 🏗️ Multi-Model Architecture Design

### Tiered Detection Strategy

```
Frame Input (1920x1080)
    |
    ├─> PRIMARY DETECTOR (always runs)
    |   └─> ppe_factory_v0_cash.pt
    |       └─> 17 classes (Person, PPE, Bakery items, Cylinder, Cashbox, etc)
    |
    ├─> SECONDARY DETECTORS (zone-conditional)
    |   ├─> hairnet_glove_detection/best.pt
    |   |   └─> Zones: dough_mixing, oven, packing, biscuit_cutting
    |   |
    |   ├─> fall_detection/best.pt
    |   |   └─> Zones: all work areas (ground, first, second floors)
    |   |
    |   └─> cash_detection/yolo11m_finetuned.pt
    |       └─> Zones: shop_counter, cashbox areas
    |
    └─> TERTIARY MODELS (Phase 2-4, high compute cost)
        ├─> hand_landmarks (MediaPipe)
        |   └─> Zones: packing (hands idle), shop (cash-in-pocket)
        |
        ├─> person_action_recognition (OpenVINO)
        |   └─> Zones: all zones (action classification)
        |
        └─> object_throwing (TRN - temporal)
            └─> Zones: windows, restricted areas
```

### Performance Optimization Strategy

#### 1. **Cascaded Inference** (Sequential, Conditional)
```python
# Pseudo-code
detections = []

# Stage 1: PRIMARY (always)
primary_results = ppe_factory_v0_cash_model(frame)
detections.extend(primary_results)

# Stage 2: SECONDARY (if zone matches)
if camera.zone_type in ["dough_mixing", "oven", "packing", "biscuit_cutting"]:
    hairnet_results = hairnet_glove_model(frame)
    detections.extend(hairnet_results)

if camera.zone_type in ["all_work_areas"]:
    fall_results = fall_detection_model(frame)
    detections.extend(fall_results)

if camera.zone_type == "shop_counter":
    cash_results = cash_detection_model(frame)
    detections.extend(cash_results)

# Stage 3: TERTIARY (Phase 2+, opt-in)
if camera.supports_pose and zone_requires_hands:
    hand_results = hand_landmarks_model(frame)
    detections.extend(hand_results)
```

#### 2. **Frame Rate Throttling per Model**
```python
MODEL_FPS_LIMITS = {
    "ppe_factory_v0_cash": 10,      # Primary: 10 FPS
    "hairnet_glove_detection": 5,   # Secondary: 5 FPS
    "fall_detection": 10,           # Safety-critical: 10 FPS
    "cash_detection": 3,            # Shop only: 3 FPS
    "hand_landmarks": 2,            # Expensive: 2 FPS
    "person_action": 1,             # Very expensive: 1 FPS
}
```

#### 3. **Person Crop Reuse**
- Primary detector crops persons from frame
- Secondary detectors analyze only person ROIs (not full frame)
- Reduces inference time 70-80% for hairnet/glove detection

#### 4. **Model Loading Priority**
```python
# Phase 1: Load only essential models
PHASE_1_MODELS = [
    "ppe_factory_v0_cash",      # Primary
    "hairnet_glove_detection",  # Food safety
]

# Phase 2: Add safety models
PHASE_2_MODELS = [
    "fall_detection",           # Worker safety
    "cash_detection",           # Shop monitoring
]

# Phase 3-4: Add behavioral models (opt-in per camera)
PHASE_3_4_MODELS = [
    "hand_landmarks",           # Hand tracking
    "person_action",            # Action recognition
    "object_throwing",          # Throwing detection
]
```

---

## 🔧 Configuration Schema

### Enhanced config.py Structure

```python
# Multi-model registry
MODEL_REGISTRY = {
    "ppe_factory_v0_cash": {
        "path": "backend/ppe_factory_v0_cash.pt",
        "type": "yolo",
        "enabled": True,
        "priority": 0,  # Always runs first
        "conf_threshold": 0.30,
        "target_fps": 10,
        "zones": ["*"],  # All zones
        "classes": {
            0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask",
            4: "NO-Safety Vest", 5: "Person", 6: "Safety Cone",
            7: "Safety Vest", 8: "machinery", 9: "vehicle",
            10: "Bakery-Head-Cap", 11: "NO-Bakery-Head-Cap",
            12: "Bangles", 13: "Document-in-hand", 14: "Cylinder",
            15: "Exposed-Item", 16: "Cashbox"
        }
    },
    "hairnet_glove_detection": {
        "path": "portable_models_package/hairnet_glove_detection/best.pt",
        "type": "yolo",
        "enabled": True,
        "priority": 1,
        "conf_threshold": 0.40,
        "target_fps": 5,
        "zones": ["dough_mixing", "oven", "packing", "biscuit_cutting", "entrance"],
        "classes": {0: "Hairnet", 1: "NO-Hairnet", 2: "Gloves", 3: "NO-Gloves"}
    },
    "fall_detection": {
        "path": "portable_models_package/fall_detection/best.pt",
        "type": "yolo",
        "enabled": True,
        "priority": 1,
        "conf_threshold": 0.50,
        "target_fps": 10,
        "zones": ["dough_mixing", "oven", "packing", "biscuit_cutting", "entrance", "lift", "gas_section"],
        "classes": {0: "Fall", 1: "Person-Standing"}
    },
    "cash_detection": {
        "path": "portable_models_package/cash_detection/yolo11m_finetuned.pt",
        "type": "yolo",
        "enabled": False,  # Phase 2, opt-in
        "priority": 2,
        "conf_threshold": 0.60,
        "target_fps": 3,
        "zones": ["shop_counter", "cashbox"],
        "classes": {0: "500-rupee", 1: "200-rupee", 2: "100-rupee", 3: "50-rupee", 4: "20-rupee", 5: "10-rupee"}
    },
    "hand_landmarks": {
        "path": "portable_models_package/hand_landmarks/hand_landmarker.task",
        "type": "mediapipe",
        "enabled": False,  # Phase 2+, expensive
        "priority": 3,
        "target_fps": 2,
        "zones": ["packing", "shop_counter"],
    },
    "person_action_recognition": {
        "path": "portable_models_package/person_action",
        "type": "openvino",
        "enabled": False,  # Phase 4, R&D
        "priority": 3,
        "target_fps": 1,
        "zones": ["*"],
    }
}

# Zone-to-model mapping (auto-generated from MODEL_REGISTRY)
ZONE_MODEL_MAP = {
    "entrance": ["ppe_factory_v0_cash", "hairnet_glove_detection"],
    "dough_mixing": ["ppe_factory_v0_cash", "hairnet_glove_detection", "fall_detection"],
    "oven": ["ppe_factory_v0_cash", "hairnet_glove_detection", "fall_detection"],
    "packing": ["ppe_factory_v0_cash", "hairnet_glove_detection", "hand_landmarks"],
    "shop_counter": ["ppe_factory_v0_cash", "cash_detection", "hand_landmarks"],
    "lift": ["ppe_factory_v0_cash", "fall_detection"],
    "default": ["ppe_factory_v0_cash"],  # Fallback
}
```

---

## 🎯 Detection Output Format

### Unified Detection Result

```python
@dataclass
class Detection:
    label: str              # "Bakery-Head-Cap", "Hairnet", "Fall", etc
    confidence: float       # 0.0 - 1.0
    bbox: List[int]         # [x1, y1, x2, y2]
    class_id: int           # Model-specific class index
    model_key: str          # "ppe_factory_v0_cash", "hairnet_glove_detection", etc
    source_model: str       # Same as model_key (for backward compat)
    
    # Computed properties
    center_x: float
    center_y: float
    width: int
    height: int
```

### Detection Fusion Strategy

When multiple models detect overlapping objects (e.g., both ppe_factory and hairnet models detect "no head cap"):

1. **Prioritize by Confidence**: Keep detection with highest confidence
2. **Merge Redundant Classes**: 
   - `NO-Bakery-Head-Cap` (ppe_factory) + `NO-Hairnet` (hairnet_glove) → Single alert
   - Use class mapping: `{NO-Bakery-Head-Cap: "no_headcap", NO-Hairnet: "no_headcap"}`
3. **IoU-based Deduplication**: If bbox IoU > 0.7, merge detections

---

## 📊 Expected Performance Metrics

### Inference Latency (per frame, 640x640 input)

| Model | Device | Latency | FPS | Priority |
|-------|--------|---------|-----|----------|
| ppe_factory_v0_cash | CPU (i7) | ~120ms | ~8 FPS | P0 |
| ppe_factory_v0_cash | GPU (RTX 3060) | ~18ms | ~55 FPS | P0 |
| hairnet_glove_detection | CPU | ~80ms | ~12 FPS | P1 |
| fall_detection | CPU | ~60ms | ~16 FPS | P1 |
| cash_detection | CPU | ~110ms | ~9 FPS | P2 |
| hand_landmarks | CPU | ~40ms | ~25 FPS | P3 |
| person_action | CPU (OpenVINO) | ~150ms | ~6 FPS | P3 |

### Multi-Camera System Capacity

**Configuration**: 15 cameras @ 10 FPS each = 150 frames/sec

#### CPU-Only (i7-10700, 8 cores)
- **Primary only** (ppe_factory): ~8-10 cameras max
- **Primary + Secondary** (ppe + hairnet + fall): ~5-6 cameras max
- **Primary + Tertiary** (all models): ~3-4 cameras max

#### With GPU (RTX 3060)
- **Primary only**: 15+ cameras at full 10 FPS ✅
- **Primary + Secondary**: 12-15 cameras at 8-10 FPS ✅
- **Primary + Tertiary**: 8-10 cameras at 5-8 FPS 🟡

**Recommendation**: Deploy GPU for production, use frame skip (process every 3rd frame) for CPU-only setups

---

## 🚦 Implementation Phases

### Phase 1 (Immediate) - Core Models
- [x] `ppe_factory_v0_cash.pt` (already deployed)
- [ ] `hairnet_glove_detection/best.pt` (food safety zones)
- [ ] Multi-model detection layer implementation

**Expected Completion**: 2 days  
**Impact**: Food safety compliance monitoring (head cap + gloves)

### Phase 2 (Next Week) - Safety & Cash
- [ ] `fall_detection/best.pt` (all work areas)
- [ ] `cash_detection/yolo11m_finetuned.pt` (shop counter)
- [ ] Frame rate throttling optimization

**Expected Completion**: 1 week  
**Impact**: Worker safety + shop cash monitoring

### Phase 3 (2 Weeks) - Hand Tracking
- [ ] `hand_landmarks/hand_landmarker.task` (packing idle, shop cash-in-pocket)
- [ ] Temporal hand velocity analysis
- [ ] Person crop reuse optimization

**Expected Completion**: 2 weeks  
**Impact**: Packing productivity + cash handling verification

### Phase 4 (R&D, 1 Month) - Advanced Behavioral
- [ ] `person_action_recognition` (OpenVINO)
- [ ] `object_throwing` (TRN temporal)
- [ ] Manual review queue integration

**Expected Completion**: 1 month (R&D phase)  
**Impact**: Window theft detection, advanced behavioral analysis

---

## ✅ Success Criteria

1. **Functional**:
   - ✅ All 5 YOLO models load successfully
   - ✅ Zone-based model routing works
   - ✅ Detections from all models appear in unified format
   - ✅ LiveMonitor shows all detection types

2. **Performance**:
   - ✅ Primary model runs at 10 FPS (with GPU) or 3-5 FPS (with frame skip, CPU)
   - ✅ Secondary models run at 5 FPS minimum
   - ✅ Total latency < 200ms per frame (multi-model pipeline)

3. **Accuracy**:
   - ✅ Head cap detection: 90%+ recall (combined ppe_factory + hairnet models)
   - ✅ Fall detection: 85%+ recall
   - ✅ Glove detection: 85%+ recall
   - ✅ Cash detection: 80%+ precision (Phase 2)

4. **Operational**:
   - ✅ Alerts fire correctly for all violation types
   - ✅ Evidence snapshots show model source
   - ✅ Admin can enable/disable models per camera via UI
   - ✅ System handles model load failures gracefully (fallback to primary only)

---

## 🐛 Known Limitations & Mitigations

### 1. **Overlapping Detections**
**Issue**: ppe_factory and hairnet_glove both detect head cap violations  
**Mitigation**: Detection fusion with IoU-based deduplication, class mapping to canonical violation types

### 2. **YOLO11 License (cash_detection)**
**Issue**: AGPL-3.0 license requires open-source distribution if modified  
**Mitigation**: Keep model inference isolated, don't modify model weights, use as-is

### 3. **CPU Performance Bottleneck**
**Issue**: 15 cameras × 3 models = too slow on CPU  
**Mitigation**: Zone-conditional loading (only run hairnet in food zones), frame skip, GPU deployment

### 4. **Action Recognition Accuracy**
**Issue**: Temporal models (TRN, person_action) need 8+ frames per sequence  
**Mitigation**: Phase 4 only, manual review queue, set low confidence threshold (0.3-0.4)

### 5. **Model Download Size**
**Issue**: Total package ~410 MB, slow initial download  
**Mitigation**: Pre-bundle in Docker image, lazy-load models on first camera activation

---

## 📝 Next Steps

1. **Immediate**:
   - ✅ Create `backend/services/model_registry.py` (model loader + registry)
   - ✅ Update `config.py` with `MODEL_REGISTRY`
   - ✅ Extend `detection_layer.py` to `MultiModelDetector`
   - ✅ Test with webcam on LiveMonitor

2. **This Week**:
   - Add UI toggle for enabling/disabling models per camera
   - Implement detection fusion logic
   - Add model performance metrics to `/diagnostics` page
   - Write integration tests for multi-model pipeline

3. **Next Week**:
   - Deploy Phase 2 models (fall, cash)
   - Tune confidence thresholds per model
   - Measure false positive rates
   - Optimize frame rate per camera/zone

---

**Document Version**: 1.0  
**Last Updated**: September 2, 2026  
**Status**: Ready for Implementation ✅
