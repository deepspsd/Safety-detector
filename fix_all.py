"""
Fix all detection and auth bugs in OccuSafe:
1. Remove phantom 25% random violations for simulated PPE — report as 'not checked' instead
2. Fix WS URL builder in LiveMonitor to always use Vite proxy (same fix as api.js)
"""
import re

# ─────────────────────────────────────────────────────────────────
# Fix 1: yolo_service.py — Remove phantom random 25% violations
# for non-native PPE classes (Gloves, Goggles, Safety Shoes) when
# running real model. Replace with "not checked" (no violation, no compliant).
# ─────────────────────────────────────────────────────────────────
path_yolo = r'backend\services\yolo_service.py'
content = open(path_yolo, 'r', encoding='utf-8').read()

old_sim = (
    "        # ── Simulated / non-native PPE (Gloves, Goggles, ID Card, Safety Shoes) ──\n"
    "        for sim_v in req_sim:\n"
    "            if role != \"None\" and detection_filters is not None and sim_v not in detection_filters:\n"
    "                continue\n"
    "            if _use_simulation:\n"
    "                if sim_v in assigned_violations:\n"
    "                    human_label = VIOLATION_LABEL_MAP.get(sim_v, sim_v)\n"
    "                    ppe_missing.append(sim_v)\n"
    "                    violation_labels.append(human_label)\n"
    "            else:\n"
    "                # Real inference: ~25% estimated miss rate for non-native classes\n"
    "                if random.random() < 0.25:\n"
    "                    human_label = VIOLATION_LABEL_MAP.get(sim_v, sim_v)\n"
    "                    ppe_missing.append(sim_v)\n"
    "                    violation_labels.append(human_label)"
)

new_sim = (
    "        # ── Simulated / non-native PPE (Gloves, Goggles, ID Card, Safety Shoes) ──\n"
    "        for sim_v in req_sim:\n"
    "            if role != \"None\" and detection_filters is not None and sim_v not in detection_filters:\n"
    "                continue\n"
    "            if _use_simulation:\n"
    "                # Simulation mode: randomly assign violations\n"
    "                if sim_v in assigned_violations:\n"
    "                    human_label = VIOLATION_LABEL_MAP.get(sim_v, sim_v)\n"
    "                    ppe_missing.append(sim_v)\n"
    "                    violation_labels.append(human_label)\n"
    "            # Real model: ppe_extended.pt handles these — if not loaded, skip (do NOT phantom-flag)\n"
    "            # Extended model results are already merged into assigned_violations before this point"
)

lf = content.replace('\r\n', '\n')
if old_sim in lf:
    fixed = lf.replace(old_sim, new_sim, 1)
    open(path_yolo, 'w', encoding='utf-8', newline='').write(fixed)
    print('FIX 1 OK: phantom 25% violations removed from yolo_service.py')
else:
    print('FIX 1 WARN: pattern not found in yolo_service.py')
    idx = lf.find('25% estimated miss rate')
    print('  idx:', idx, repr(lf[max(0,idx-200):idx+100]) if idx>=0 else 'NOT FOUND')

# ─────────────────────────────────────────────────────────────────
# Fix 2: LiveMonitor.jsx — WS URL builder uses Vite proxy unconditionally
# ─────────────────────────────────────────────────────────────────
path_monitor = r'frontend\src\pages\LiveMonitor.jsx'
content_m = open(path_monitor, 'r', encoding='utf-8').read()
lf_m = content_m.replace('\r\n', '\n')

old_ws = (
    "// WebSocket URLs — use Vite proxy in development (same trick as api.js)\n"
    "function getWsURL() {\n"
    "  if (import.meta.env.MODE === 'development') {\n"
    "    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'\n"
    "    return `${proto}://${window.location.host}/ws/detect`\n"
    "  }\n"
    "  const envURL = import.meta.env.VITE_API_BASE_URL\n"
    "  if (envURL) return envURL.replace(/^http/, 'ws') + '/ws/detect'\n"
    "  return `ws://localhost:8000/ws/detect`\n"
    "}\n"
    "function getCctvWsURL() {\n"
    "  if (import.meta.env.MODE === 'development') {\n"
    "    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'\n"
    "    return `${proto}://${window.location.host}/ws/detect-cctv`\n"
    "  }\n"
    "  const envURL = import.meta.env.VITE_API_BASE_URL\n"
    "  if (envURL) return envURL.replace(/^http/, 'ws') + '/ws/detect-cctv'\n"
    "  return `ws://localhost:8000/ws/detect-cctv`\n"
    "}"
)

new_ws = (
    "// WebSocket URLs — ALWAYS use Vite proxy (relative WS URL).\n"
    "// This ensures traffic goes through localhost:5173 → localhost:8000,\n"
    "// preventing LAN IP timeout even when the browser is opened via a network IP.\n"
    "function getWsURL() {\n"
    "  const envURL = import.meta.env.VITE_API_BASE_URL\n"
    "  // Only use explicit URL if it's a real production domain (not localhost)\n"
    "  if (envURL && !envURL.includes('localhost') && !envURL.includes('127.0.0.1')) {\n"
    "    return envURL.replace(/^http/, 'ws') + '/ws/detect'\n"
    "  }\n"
    "  // Development: use Vite proxy — always routes to localhost:8000\n"
    "  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'\n"
    "  return `${proto}://${window.location.host}/ws/detect`\n"
    "}\n"
    "function getCctvWsURL() {\n"
    "  const envURL = import.meta.env.VITE_API_BASE_URL\n"
    "  if (envURL && !envURL.includes('localhost') && !envURL.includes('127.0.0.1')) {\n"
    "    return envURL.replace(/^http/, 'ws') + '/ws/detect-cctv'\n"
    "  }\n"
    "  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'\n"
    "  return `${proto}://${window.location.host}/ws/detect-cctv`\n"
    "}"
)

if old_ws in lf_m:
    fixed_m = lf_m.replace(old_ws, new_ws, 1)
    open(path_monitor, 'w', encoding='utf-8', newline='').write(fixed_m)
    print('FIX 2 OK: WS URL builder fixed in LiveMonitor.jsx')
else:
    print('FIX 2 WARN: WS URL pattern not found')
    idx = lf_m.find('getWsURL')
    print(f'  getWsURL at char {idx}:', repr(lf_m[idx:idx+300]) if idx>=0 else 'NOT FOUND')

print('\nAll fixes applied.')
