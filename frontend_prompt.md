# OccuSafe — AI-Powered Factory Safety Monitoring System (Frontend Prompt)

## System Overview

Build a complete, production-grade **React (Vite) + React Router** single-page application for **OccuSafe** — an AI-powered factory/bakery safety monitoring platform. The system monitors a multi-floor bakery factory (Ground Floor, First Floor, Second Floor, Shop) using IP cameras with real-time AI inference for PPE compliance, employee behaviour, attendance, document scanning, and workflow monitoring.

The backend is a **FastAPI** server already built and running. The frontend must connect to it via REST API (Axios) and WebSockets. All API calls require a Bearer JWT token (except login/signup). The backend base URL is configured via environment variable `VITE_API_BASE_URL`.

> **Important**: This is an **on-premise** deployment running on a local Windows server. The frontend is served by FastAPI in production (built with `npm run build`). All data is stored locally — no cloud services.

---

## Tech Stack

- React 18+ with Vite
- React Router v6 for client-side routing
- Axios for REST API calls
- Native WebSocket API for live camera feeds
- Lucide React for icons
- Recharts for charts and analytics
- HTML5 Canvas for zone polygon drawing
- CSS Modules or styled-components (no Tailwind)
- PWA support (service worker, manifest, installable)
- Mobile-responsive (works on phones and tablets on the same LAN)

---

## Project Folder Structure

Use the following folder structure exactly. Do not flatten everything into a single `pages/` or `components/` directory. Group by feature/domain:

```
occusafe-frontend/
├── public/
│   ├── manifest.json            # PWA manifest
│   ├── sw.js                    # Service worker
│   └── icons/                   # PWA icons (192x192, 512x512)
│
├── src/
│   ├── main.jsx                 # App entry point
│   ├── App.jsx                  # Root router + providers
│   │
│   ├── api/                     # All API + WebSocket logic
│   │   ├── client.js            # Axios instance, interceptors, base URL
│   │   ├── auth.api.js          # login, signup, me
│   │   ├── cameras.api.js       # cameras CRUD, discover, snapshot, zones, calibration
│   │   ├── alerts.api.js        # list, stats, pending, confirm, dismiss, testTelegram
│   │   ├── attendance.api.js    # stats, list, clockIn, clockOut, exportCSV
│   │   ├── documents.api.js     # list, stats, scan, approve, reject, delete
│   │   ├── workflow.api.js      # summary, cash, stock, lift, packing events
│   │   ├── faces.api.js         # register, list, rename, delete
│   │   ├── settings.api.js      # systemSettings, cylinderLogs, baselines
│   │   ├── video.api.js         # upload, status, result
│   │   ├── enterprise.api.js    # models, health, analytics, rules, profiles, alertCases
│   │   └── websocket.js         # WebSocket manager: connect, reconnect, send, close
│   │
│   ├── providers/               # React context providers
│   │   ├── AuthProvider.jsx     # JWT auth state, login, logout, token refresh
│   │   ├── ThemeProvider.jsx    # dark/light theme toggle
│   │   └── ToastProvider.jsx    # global toast/notification system
│   │
│   ├── hooks/                   # Reusable custom React hooks
│   │   ├── useAuth.js           # consume AuthContext
│   │   ├── useTheme.js          # consume ThemeContext
│   │   ├── useToast.js          # consume ToastContext
│   │   ├── useCameraStream.js   # WebSocket camera stream hook (per camera)
│   │   ├── usePolling.js        # generic interval polling hook
│   │   └── useLocalStorage.js   # localStorage read/write helper
│   │
│   ├── layout/                  # App shell components
│   │   ├── AppLayout.jsx        # wrapper: sidebar + main content area
│   │   ├── Sidebar.jsx          # navigation sidebar (all nav links, badges)
│   │   ├── MobileHeader.jsx     # mobile top bar with hamburger
│   │   └── ProtectedRoute.jsx   # HOC: redirect to /login if not authenticated
│   │
│   ├── features/                # Feature modules grouped by domain
│   │   │
│   │   ├── auth/
│   │   │   ├── LoginPage.jsx
│   │   │   └── SignupPage.jsx
│   │   │
│   │   ├── dashboard/
│   │   │   ├── DashboardPage.jsx
│   │   │   ├── StatCard.jsx          # animated counter stat card
│   │   │   ├── AlertsTimeline.jsx    # recent alerts scrollable list
│   │   │   ├── FloorSummaryCard.jsx  # per-floor summary card
│   │   │   ├── CameraHealthGrid.jsx  # mini camera status grid
│   │   │   └── WorkflowSummary.jsx   # cash/stock/lift/packing counts
│   │   │
│   │   ├── monitor/
│   │   │   ├── LiveMonitorPage.jsx   # main page with grid + toolbar
│   │   │   ├── CameraGrid.jsx        # responsive grid container
│   │   │   ├── CameraTile.jsx        # single camera live feed tile
│   │   │   ├── CameraExpandedView.jsx # expanded camera + detection panel
│   │   │   ├── DiscoverCamerasModal.jsx # LAN scan results + add camera
│   │   │   ├── GridLayoutSelector.jsx   # 1x1 / 2x2 / 3x3 / 4x4 picker
│   │   │   └── DetectionFiltersPanel.jsx # PPE / phone / face toggles
│   │   │
│   │   ├── floors/
│   │   │   ├── FloorOverviewPage.jsx  # /floors/:floorId
│   │   │   ├── FloorCameraCard.jsx    # camera card with snapshot thumbnail
│   │   │   └── FloorRulesSummary.jsx  # per-floor monitoring rules display
│   │   │
│   │   ├── alerts/
│   │   │   ├── AlertsPage.jsx         # full alert list with filters
│   │   │   ├── AlertsReviewPage.jsx   # pending review queue
│   │   │   ├── AlertCard.jsx          # card with snapshot + confirm/dismiss
│   │   │   ├── AlertDetailModal.jsx   # full alert detail popup
│   │   │   └── AlertFiltersBar.jsx    # severity/floor/camera/date filters
│   │   │
│   │   ├── attendance/
│   │   │   ├── AttendancePage.jsx
│   │   │   ├── AttendanceStatsRow.jsx  # present/absent/late cards
│   │   │   ├── AttendanceTable.jsx     # records with clock-out action
│   │   │   └── ClockInForm.jsx         # manual clock-in form
│   │   │
│   │   ├── documents/
│   │   │   ├── DocumentsPage.jsx
│   │   │   ├── DocumentScanUpload.jsx  # drop zone + OCR result display
│   │   │   ├── DocumentCard.jsx        # inward/outward card with approve/reject
│   │   │   └── DocumentDetailModal.jsx # full OCR text + snapshot
│   │   │
│   │   ├── workflow/
│   │   │   ├── WorkflowPage.jsx
│   │   │   ├── CashEventsTab.jsx
│   │   │   ├── StockEventsTab.jsx
│   │   │   ├── LiftEventsTab.jsx
│   │   │   └── PackingEventsTab.jsx
│   │   │
│   │   ├── cameras/                   # /settings/cameras
│   │   │   ├── CamerasPage.jsx        # camera management list
│   │   │   ├── CameraCard.jsx         # single camera card with actions
│   │   │   ├── CameraFormModal.jsx    # add/edit camera form
│   │   │   └── zone-calibration/
│   │   │       ├── ZoneCalibrationPage.jsx  # wrapper page for calibration
│   │   │       ├── ZonePainter.jsx          # HTML5 canvas polygon drawing tool
│   │   │       ├── ZoneList.jsx             # saved zones with edit/copy/delete
│   │   │       └── CalibrationHistory.jsx   # version history + restore
│   │   │
│   │   ├── employees/                 # /settings/employees
│   │   │   ├── EmployeesPage.jsx
│   │   │   ├── FaceRegistrationForm.jsx  # label + image upload/webcam
│   │   │   └── FaceCard.jsx              # thumbnail + name + actions
│   │   │
│   │   ├── settings/                  # /settings
│   │   │   ├── SettingsPage.jsx        # tab container
│   │   │   ├── ThresholdsTab.jsx       # detection threshold key/value editor
│   │   │   ├── NotificationsTab.jsx    # Telegram status, sound, UI toggles
│   │   │   ├── BaselineTab.jsx         # dirty-floor baseline image manager
│   │   │   ├── CylinderLogsTab.jsx     # cylinder usage log table
│   │   │   ├── ModelsRulesTab.jsx      # AI models + rule profiles + rules
│   │   │   └── PlatformHealthTab.jsx   # health logs + analytics summary
│   │   │
│   │   ├── video/
│   │   │   ├── VideoAnalysisPage.jsx   # upload + results
│   │   │   ├── VideoUploadZone.jsx     # drag-and-drop file picker
│   │   │   ├── VideoPlayer.jsx         # annotated video player
│   │   │   ├── ViolationTimeline.jsx   # clickable timeline bar
│   │   │   └── PPESummaryChart.jsx     # recharts bar chart
│   │   │
│   │   ├── profile/
│   │   │   └── ProfilePage.jsx
│   │   │
│   │   └── tv/
│   │       └── TVDisplayPage.jsx       # chromeless full-screen monitoring wall
│   │
│   ├── components/                    # Truly shared, generic UI primitives
│   │   ├── Button.jsx
│   │   ├── Modal.jsx
│   │   ├── Toast.jsx
│   │   ├── Badge.jsx                  # severity/status coloured pill
│   │   ├── Spinner.jsx
│   │   ├── EmptyState.jsx
│   │   ├── ConfirmDialog.jsx
│   │   ├── Pagination.jsx
│   │   ├── DataTable.jsx
│   │   ├── StatusDot.jsx              # online/offline/error coloured dot
│   │   └── SeverityBadge.jsx          # low/medium/high/critical badge
│   │
│   └── styles/
│       ├── globals.css               # CSS variables, resets, base typography
│       ├── layout.css                # sidebar, grid, page layout
│       ├── components.css            # shared component styles
│       └── animations.css            # keyframes, transitions
│
├── index.html
├── vite.config.js
├── package.json
└── .env.example                      # VITE_API_BASE_URL=http://localhost:8000
```

---

## Application Architecture

### Authentication Flow

The app uses JWT Bearer token authentication.

**Login Page** (`/login`)
- Email + password form
- Calls `POST /auth/login` (OAuth2 form-encoded: username=email, password=password)
- Returns `{ access_token, token_type }` — store token in localStorage
- On success, redirect to `/dashboard`
- Show meaningful error messages for invalid credentials

**Signup Page** (`/signup`)
- Fields: Name, Email, Password, Role (dropdown)
- Available roles: `Construction Worker`, `Doctor`, `Traffic Police`, `College`, `Home`, `Bakery Worker`, `None`
- Calls `POST /auth/signup` with `{ email, password, name, role }`
- Auto-login on success → redirect to `/dashboard`

**Auth Context**
- Wrap entire app in an AuthProvider
- Store token + user info in context
- `GET /auth/me` to validate token on app load and fetch user profile `{ id, email, name, role, created_at }`
- Protected routes: redirect to `/login` if no valid token
- Auto-logout on 401 responses (token expired)

---

## Global Layout

### Sidebar Navigation (persistent, collapsible on mobile)

The sidebar is the primary navigation element — always visible on desktop, slide-out drawer on mobile.

**Structure:**
```
Logo: OccuSafe + shield icon
Role Banner (shows current user's role context — e.g., "BAKERY FOOD SAFETY")

── MONITORING ──
  Floors (collapsible sub-menu)
    ├─ Ground Floor  → /floors/ground
    ├─ First Floor   → /floors/first
    ├─ Second Floor  → /floors/second
    └─ Shop Floor    → /floors/shop
  Live Monitor       → /monitor
  Alerts             → /alerts
  Review Queue       → /alerts/review  (with live pending count badge)
  Dashboard          → /dashboard
  Attendance         → /attendance
  Documents          → /documents
  Workflow           → /workflow

── SETTINGS ──
  Cameras            → /settings/cameras
  Employees          → /settings/employees
  Settings           → /settings

── ACCOUNT ──
  Profile            → /profile
  TV Display         → /tv  (opens in new tab, full-screen monitoring wall)
  Sign Out
```

**Features:**
- Dark/Light theme toggle button in sidebar header
- User avatar (initials) + name + role pill at the bottom
- Active route highlighting
- Pending review badge next to "Review Queue" — polls `GET /alerts/pending` every 30 seconds to show count
- Mobile: hamburger menu in a top bar, sidebar slides out as overlay
- Sidebar collapses to icons-only on smaller desktop screens

### Toast/Notification System
- Global toast provider for success/error/warning/info messages
- Toasts auto-dismiss after 5 seconds
- Stackable (multiple toasts visible)

---

## Pages — Detailed Specifications

---

### 1. Dashboard (`/dashboard`)

**Purpose:** At-a-glance overview of the entire factory's safety status.

**Data Sources:**
- `GET /alerts/stats` → `{ total_alerts, confirmed_alerts, pending_review_alerts, dismissed_alerts, today_alerts, critical_alerts, compliance_percentage, recent_alerts[] }`
- `GET /cameras/` → list of all cameras with live status
- `GET /attendance/stats` → `{ present, absent, late, total }`
- `GET /documents/stats` → `{ today_total, today_inward, today_outward, today_approved, today_rejected }`
- `GET /workflow/summary` → `{ cash_events_today, stock_events_today, lift_events_today, packing_events_today, total_today }`

**Layout:**
- **Top stat cards row** (animated counter): Total Alerts Today, Critical Alerts, Compliance %, Active Cameras (online count out of total), Employees Present, Document Scans Today
- **Camera Health Grid**: Small cards for each camera showing name, floor, status (online/offline/error) with coloured indicator, last heartbeat time. Clicking a camera card navigates to its live feed
- **Alerts Timeline**: Scrollable list of recent confirmed alerts with severity badges, message, timestamp, camera name. Click to open alert detail
- **Floor Summary Cards**: One card per floor (Ground, First, Second, Shop) showing active camera count, today's alert count, and compliance percentage for that floor
- **Workflow Summary**: Cash events, Stock events, Lift events, Packing events — counts for today with small icons
- **Charts**:
  - Alerts by severity (donut/pie chart)
  - Alerts over time (line chart — last 7 days)
  - Top violation types (horizontal bar chart from ppe_summary)

**Behaviour:**
- Auto-refresh every 30 seconds
- All cards should be clickable — navigate to the relevant detail page

---

### 2. Live Monitor (`/monitor`)

**Purpose:** Real-time multi-camera surveillance with AI inference overlay — inspired by Hikvision iVMS-4200.

**This is the most complex and critical page in the application.**

**Data Sources:**
- `GET /cameras/` — list all cameras
- `POST /cameras/discover` — scan LAN for new cameras
- `GET /cameras/status` — real-time reader state for all cameras
- `WebSocket /ws/detect-cctv` — live AI detection feed per camera

**Layout — Hikvision-Inspired Multi-Camera Grid:**

**Top Toolbar:**
- Camera count indicator: "X cameras online / Y total"
- Grid layout selector: 1x1, 2x2, 3x3, 4x4, custom grid
- Auto-scan button ("Discover Cameras") — triggers `POST /cameras/discover`, shows a progress modal while scanning the LAN subnet, then displays discovered hosts with IP, hostname, RTSP URL guesses. Admin can select and add cameras directly from results
- Floor filter dropdown: All, Ground, First, Second, Shop
- Fullscreen toggle
- Global AI toggle (enable/disable inference overlay for all visible cameras)

**Camera Grid Area:**
- Fills the main content area with a responsive grid of camera feed tiles
- Each tile shows:
  - Live video feed (annotated frames from WebSocket)
  - Camera name + floor label overlay (top-left)
  - FPS counter overlay (top-right)
  - Status indicator (green dot = online, red = error, grey = offline)
  - Detection stats overlay (bottom bar): persons count, violations count, compliance status (green check / red X)
  - Alert flash effect: when a violation is detected, the tile border briefly flashes red
- Click a tile to expand it to full-size (1x1 view) — shows detailed detection info panel alongside
- Double-click to open full-screen for that camera
- Right-click context menu: Restart Camera, Open Zone Calibration, View Camera Settings

**Camera Connection Flow (per tile):**
1. Connect via WebSocket to `/ws/detect-cctv`
2. Send handshake JSON: `{ token, camera_id, filters, no_phone_zone, enable_face }`
3. Receive continuous frames: `{ annotated_frame (base64 JPEG), detections[], is_compliant, missing_items[], violations_count, persons_count, alert_message, severity, face_result, cam_fps, phone_status }`
4. Render `annotated_frame` as an `<img>` tag updated in real-time
5. On disconnect/error, show reconnection overlay with retry button

**Camera Discovery Flow:**
- "Discover Cameras" button triggers `POST /cameras/discover`
- Response: `{ subnet, hosts_scanned, cameras_found: [{ ip, open_port, hostname, rtsp_guesses[] }] }`
- Show results in a modal/drawer:
  - List each discovered host with IP, hostname, open port
  - Each host shows suggested RTSP URL templates (Hikvision, Dahua, Generic patterns)
  - "Add Camera" button per host → opens a form pre-filled with the discovered IP and RTSP URL. Admin enters name, floor, zone_type, and credentials (embedded in the RTSP URL)
  - Calls `POST /cameras` to create the camera → auto-starts the reader thread

**Single Camera Expanded View:**
When a camera is selected (clicked), show a side panel or full-width expanded view with:
- Large live feed with AI annotations
- Detection event log (scrollable list of recent detections for this camera)
- Active filters toggle panel:
  - PPE items: NO-Hardhat, NO-Safety Vest, NO-Gloves, NO-Goggles, NO-Mask
  - Phone detection: toggle on/off
  - Face recognition: toggle on/off
  - Send filter changes via WebSocket: `{ filters: [...], no_phone_zone: bool, enable_face: bool }`
- Camera info: name, floor, zone_type, RTSP URL, FPS, resolution
- Quick actions: Restart Stream, Open Zone Calibration, View Alerts for This Camera

---

### 3. Floor Overview (`/floors/:floorId`)

**Purpose:** Floor-specific monitoring view showing all cameras and activity on one floor.

**URL Parameters:** `:floorId` = `ground` | `first` | `second` | `shop`

**Data Sources:**
- `GET /cameras/` filtered by floor
- `GET /alerts/?floor={floorId}` — alerts for this floor
- `GET /cameras/{id}/zones` — zone configs per camera on this floor

**Layout:**
- **Floor header**: Floor name, camera count, today's alert count
- **Camera grid**: 2-3 column grid of camera cards for this floor only
  - Each card: camera name, zone_type, status, thumbnail (latest snapshot from `GET /cameras/{id}/snapshot`), online/offline indicator
  - Click → navigate to Live Monitor with this camera selected
- **Floor alerts feed**: Recent alerts filtered by this floor, with severity badges
- **Floor-specific monitoring rules summary**: Show what is being monitored on this floor (based on the client requirements):
  - Ground: Entrance monitoring, dough mixing, biscuit cutting, oven, packing, loading/unloading, cleanliness
  - First: Four work tables, dough mixing (starts 6 AM), lift monitoring, raw material stock, cylinders, dress code
  - Second: Work starts 5 AM, fire/gas monitoring, oil/water waste, stocks, no eating/stealing, window security
  - Shop: Cash monitoring, dress code, vendor payments, idle detection

---

### 4. Alerts (`/alerts`)

**Purpose:** Full alert management interface — view, filter, and manage all confirmed safety violations.

**Data Sources:**
- `GET /alerts/?status=confirmed&role=X&severity=X&floor=X&camera_id=X&date_from=X&date_to=X&page=X&limit=20`
- `GET /alerts/{id}` — single alert detail with snapshot
- `DELETE /alerts/{id}` — delete one alert
- `DELETE /alerts/` — clear all alerts

**Layout:**
- **Filter bar** (top): Status filter (Confirmed/All), Severity (All/Low/Medium/High/Critical), Floor (All/Ground/First/Second/Shop), Camera (dropdown of all cameras), Date range picker, Search text
- **Alerts list**: Paginated table/card list
  - Each row: Severity badge (colour-coded), Message, Detected Issue, Camera Name, Floor, Timestamp, Confidence %, Status
  - Has snapshot indicator icon — click to view
  - Row click → expand or modal showing full alert detail + snapshot image
- **Pagination controls**: Page numbers, page size selector
- **Bulk actions**: "Clear All Alerts" button with confirmation dialog
- **Delete single**: Delete button per alert row

**Alert Detail Modal/View:**
- Full alert message
- Detected issue
- Severity + confidence
- Camera name + floor
- Timestamp
- Snapshot image (base64 rendered as img or from snapshot_url path)

---

### 5. Review Queue (`/alerts/review`)

**Purpose:** Admin review interface for low-confidence AI detections that need human confirmation.

**Data Sources:**
- `GET /alerts/pending?page=X&limit=20` — pending_review alerts
- `PATCH /alerts/{id}/confirm` — promote to confirmed + send Telegram notification
- `PATCH /alerts/{id}/dismiss` — mark as false positive

**Layout:**
- **Queue counter header**: "X alerts pending review"
- **Card-based list** (not table — each alert needs visual attention):
  - Large snapshot thumbnail
  - Alert message + detected issue
  - Confidence percentage with visual bar
  - Camera name + floor + timestamp
  - **Two action buttons per card**: Confirm (green) and Dismiss (red)
  - Confirm → calls `PATCH /alerts/{id}/confirm` → removes from list, shows toast "Alert confirmed and Telegram notification sent"
  - Dismiss → calls `PATCH /alerts/{id}/dismiss` → removes from list, shows toast "Alert dismissed"
- **Pagination**
- Empty state: "No alerts pending review — all clear!"

---

### 6. Attendance (`/attendance`)

**Purpose:** Employee attendance tracking with manual clock-in/out and face-recognition auto-entry.

**Data Sources:**
- `GET /attendance/stats` → `{ present, absent, late, total }`
- `GET /attendance/?date=YYYY-MM-DD&employee_id=X&limit=200`
- `POST /attendance/clock-in` → `{ employee_id, camera_id, method, notes }`
- `POST /attendance/clock-out/{record_id}`
- `GET /attendance/export?date=YYYY-MM-DD` → CSV download

**Layout:**
- **Stats cards row**: Present (green), Absent (red), Late (yellow), Total Employees
- **Date picker**: Filter records by date
- **Employee filter**: Dropdown or search to filter by employee
- **Records table**: ID, Employee Name, Camera, Method (face/manual/qr badge), Clock In time, Clock Out time, Duration, Notes
  - Open sessions (no clock_out) show "Active" badge with a "Clock Out" button
  - Clock Out → calls `POST /attendance/clock-out/{record_id}`
- **Manual Clock-In form**: Select employee (dropdown from employees list), select camera (optional), method radio (Manual/Face/QR), notes field → submit calls `POST /attendance/clock-in`
- **Export button**: "Download CSV" → calls `GET /attendance/export?date=X` and triggers browser download

---

### 7. Documents (`/documents`)

**Purpose:** Invoice and order-form document scan management for goods inward/outward monitoring.

**Data Sources:**
- `GET /documents/stats` → `{ today_total, today_inward, today_outward, today_approved, today_rejected }`
- `GET /documents/?direction=inward|outward&approved=true|false&limit=100`
- `POST /documents/scan` (multipart form: file + direction + camera_id) → OCR scan
- `PATCH /documents/{id}/approve?table=invoice|order_form`
- `PATCH /documents/{id}/reject?table=invoice|order_form`
- `DELETE /documents/{id}?table=invoice|order_form`

**Layout:**
- **Stats cards**: Today's Scans, Inward, Outward, Approved, Rejected
- **Tab bar**: All / Inward Only / Outward Only
- **Filter**: Approved/Rejected toggle, direction filter
- **Document scan upload**: Drop zone or file picker + direction selector (Inward/Outward) + camera selector (optional) → submit calls `POST /documents/scan`
  - Show OCR result immediately: raw text extracted, approved/rejected status, snapshot of the document
- **Document list**: Scrollable cards showing:
  - Direction badge (Inward blue / Outward orange)
  - Approved/Rejected status badge
  - OCR text preview (truncated)
  - Timestamp
  - Snapshot thumbnail (click to expand)
  - Action buttons: Approve (override) / Reject (override) / Delete
- **Document detail modal**: Full OCR text, full snapshot image, approval status, camera info

---

### 8. Workflow Monitor (`/workflow`)

**Purpose:** Specialized monitoring for factory workflow events — cash, stock, lift, and packing section activity.

**Data Sources:**
- `GET /workflow/summary` → combined today counts
- `GET /workflow/cash-events?limit=50` → unauthorized cashbox access alerts
- `GET /workflow/stock-events?limit=50` → exposed stock alerts
- `GET /workflow/lift-events?camera_id=X&limit=100` → lift entry/exit events
- `GET /workflow/packing-events?camera_id=X` → packing zone idle alerts

**Layout:**
- **Summary cards row**: Cash Events Today, Stock Events Today, Lift Events Today, Packing Events Today, Total Events
- **Tabbed sections** (or collapsible panels):

  **Cash Monitoring Tab:**
  - List of cash zone access events
  - Each: message, severity, camera, timestamp
  - Rules displayed: "Monitor if cash goes to cashbox — employees should not pocket cash"

  **Stock Monitoring Tab:**
  - Exposed stock alerts
  - "Items should not be kept openly" — rule display
  - Inward/outward monitoring events

  **Lift Monitoring Tab:**
  - Lift entry/exit events with duration
  - Filter by camera
  - Shows: track_id, event_type (entry/exit/idle), floor_from, floor_to, duration, timestamp
  - Rule: "Lift monitoring for all 3 floors"

  **Packing Section Tab:**
  - Packing zone idle alerts
  - Rule: "For packing section, employee can be idle in position but hands should be moving"
  - Shows packing summary data

---

### 9. Camera Management (`/settings/cameras`)

**Purpose:** CRUD management for all IP cameras in the factory.

**Data Sources:**
- `GET /cameras/` — list all cameras
- `POST /cameras` — create new camera
- `PUT /cameras/{id}` — update camera config
- `DELETE /cameras/{id}` — delete camera
- `POST /cameras/{id}/restart` — restart camera reader thread
- `POST /cameras/discover` — LAN discovery scan
- `GET /cameras/{id}/snapshot` — latest frame
- `GET /cameras/{id}/zones` — zone polygons
- `POST /cameras/{id}/zones` — create/update zone
- `DELETE /cameras/{id}/zones/{zone_name}` — delete zone
- `GET /cameras/{id}/calibrations` — calibration version history
- `POST /cameras/{id}/calibrations/snapshot` — save calibration snapshot
- `POST /cameras/{id}/calibrations/{version}/restore` — restore calibration version

**Layout:**

**Camera List View:**
- Card grid of all cameras
- Each card shows: Name, Floor, Zone Type, Status (online/offline/error with colour), RTSP URL (masked), Camera Code, Department, Last Seen timestamp, Health Status, FPS, Capabilities icons (OCR, Pose, Tracking, Recording, Snapshot)
- Card actions: Edit, Restart, Delete, Open Zone Calibration
- "Add Camera" button → opens add form
- "Discover Cameras" button → triggers LAN scan

**Add/Edit Camera Form (modal or drawer):**
- Name (required)
- Floor (dropdown: ground/first/second/shop)
- Zone Type (text or preset dropdown: entrance, dough-mixing, oven, packing, shop-counter, etc.)
- RTSP URL (text input with helper text showing common URL patterns for Hikvision, Dahua, Generic)
- Status (online/offline)
- Camera Code, Department, Purpose
- Camera Type (IP/Analog/Wireless)
- Mount Height, View Direction, Resolution, FPS
- AI capabilities toggles: AI Enabled, Multi-Zone, OCR, Pose, Tracking, Recording, Snapshot
- Rule Profile selector, Workflow Profile selector
- Save → `POST /cameras` (create) or `PUT /cameras/{id}` (update)

**Zone Calibration View (per camera):**
This is a critical feature. When the admin clicks "Calibrate Zones" on a camera, show:
- Camera's latest frame as a static background image (from `GET /cameras/{id}/snapshot`)
- Canvas overlay for polygon drawing
- Zone name selector (preset list: entrance, exit, packing, dough_table, machine, oven, cash_counter, cashbox, lift, window, dispatch, loading, raw_material, finished_goods, stock, employee_area, supervisor_area, cleaning_area, waiting_area, vehicle_area, document_scan_area, shop_counter, payment_desk, camera_standing — or custom name input)
- Click canvas to place polygon vertices (minimum 3 points)
- Right-click to undo last point
- Drag existing vertices to reposition
- Show work-in-progress polygon with dashed lines
- "Save Zone" button → `POST /cameras/{id}/zones` with `{ zone_name, polygon_json: "[[x,y],[x,y],...]" }`
- Existing saved zones rendered as filled semi-transparent coloured polygons with labels
- Each saved zone has Edit, Copy, Delete buttons
- Calibration history: list of past calibration versions with "Restore" button → `POST /cameras/{id}/calibrations/{version}/restore`
- "Save Calibration Snapshot" button → `POST /cameras/{id}/calibrations/snapshot`

**Zone Configuration fields (per zone):**
- Zone type, Preset type, Display name, Colour
- Priority level
- Workflow stage
- Rule profile
- Expected objects, Allowed objects, Forbidden objects
- Time constraints (JSON)
- Alert thresholds
- Movement threshold, Idle threshold, Confidence threshold, Visibility threshold

---

### 10. Employee Management (`/settings/employees`)

**Purpose:** Manage factory employees being monitored (not login accounts).

**Data Sources:**
- Employee data is accessed via the faces and attendance system
- `GET /faces/` — list registered face encodings
- `POST /faces/register` → `{ label, image_b64 }` — register a new face
- `PUT /faces/{id}/label` → `{ label }` — rename
- `DELETE /faces/{id}` — delete face encoding

**Layout:**
- **Face registration section**:
  - "Register New Face" form: Name label + image upload (file picker or webcam capture → convert to base64 data URL)
  - Calls `POST /faces/register` with `{ label, image_b64 }`
  - Show success/error messages (e.g., "No face detected in image")
- **Registered faces grid**:
  - Card per face showing: thumbnail image (from `thumbnail_b64`), label name, created date
  - Edit button → rename label (inline edit or modal)
  - Delete button with confirmation
- **Employee list** (if separate employee management exists):
  - Name, Role (baker/packer/cashier), Department, Active status, Face linked indicator

---

### 11. Settings (`/settings`)

**Purpose:** System-wide configuration for detection thresholds, notifications, and admin parameters.

**Data Sources:**
- `GET /settings/` — list all SystemSettings key/value pairs
- `GET /settings/{key}` — single setting
- `PUT /settings/{key}` → `{ value }` — update setting
- `GET /settings/cylinder-logs?camera_id=X&event_type=X&limit=100` — cylinder usage history
- `GET /users/me/config` — user-specific settings
- `PUT /users/me/config` → update user settings
- `POST /cameras/{id}/baseline` — upload dirty-floor baseline image
- `DELETE /cameras/{id}/baseline` — remove baseline
- `POST /alerts/test-telegram` — test Telegram integration
- **Enterprise endpoints:**
  - `GET /platform/models` — AI model registry
  - `POST /platform/models` — register model
  - `GET /platform/health` — system health logs
  - `GET /platform/analytics/summary?days=7` — analytics
  - `GET /platform/workflow-profiles` / `POST` — workflow profile CRUD
  - `GET /platform/rule-profiles` / `POST` — rule profile CRUD
  - `GET /platform/rules` / `POST` — rule definition CRUD
  - `GET /platform/alert-cases` — enterprise alert cases
  - `POST /platform/alert-cases/{id}/acknowledge` or `/dismiss`

**Layout — Tabbed Interface:**

**Tab 1: Detection Thresholds**
- List of all system settings from `GET /settings/` displayed as editable key-value rows
- Each row: Key name (human-readable label), Current value, Description, Edit button
- Common settings:
  - `idle_limit_default` — Idle alert threshold (seconds)
  - `idle_limit_shop` — Shop idle limit
  - `shift_start_ground` — Ground floor shift start time (HH:MM)
  - `shift_start_first` — First floor shift start time
  - `shift_start_second` — Second floor shift start time
  - `dirty_floor_threshold` — Dirty floor pixel fraction
  - `detection_confidence` — Minimum detection confidence
  - `alert_cooldown` — Seconds between alerts
- Edit → inline editing → PUT /settings/{key} → show toast

**Tab 2: Notification Settings**
- Telegram configuration status: shows if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are configured
- "Test Telegram" button → `POST /alerts/test-telegram` → shows success/failure
- User notification preferences: `GET /users/me/config` → toggles for notify_sound, notify_ui
- Detection sensitivity slider
- Custom PPE items selection
- No-phone-zone toggle

**Tab 3: Dirty Floor Baselines**
- Per-camera baseline image management
- Select camera → shows current baseline image (if any)
- Upload new baseline: file picker → `POST /cameras/{id}/baseline` (multipart with zone_name)
- Delete baseline: `DELETE /cameras/{id}/baseline`

**Tab 4: Cylinder Logs**
- Paginated table of gas cylinder usage events
- Filter by camera, event type
- Columns: Camera, Event Type (detected/swapped), Usage Day Count, Timestamp

**Tab 5: AI Models and Rules (Enterprise)**
- **Model Registry**: Table of registered AI models with model_key, display_name, type, version, provider, status, capabilities. Add Model button
- **Rule Profiles**: List + create rule profiles (name, description, definition JSON)
- **Workflow Profiles**: List + create workflow profiles (name, description, floor, definition JSON)
- **Rules**: List all rule definitions with name, priority, enabled status, associated zone, workflow stage, cooldown. Create new rule with full field set
- **Alert Cases**: Enterprise alert case list with status (open/acknowledged/dismissed), severity, title, camera. Acknowledge/Dismiss buttons

**Tab 6: Platform Health**
- Health logs from `GET /platform/health` — component type, status, measured_at, metrics, message
- Analytics summary from `GET /platform/analytics/summary?days=7`

---

### 12. Profile (`/profile`)

**Purpose:** User profile management.

**Data Sources:**
- `GET /auth/me` → `{ id, email, name, role, created_at }`
- `PUT /users/me` → `{ name, role }` — update profile
- `GET /users/me/config` — user config
- `PUT /users/me/config` — update config

**Layout:**
- User avatar (initials in circle)
- Name (editable)
- Email (read-only)
- Role (dropdown — editable)
- Account created date
- Camera config section: Camera Type (webcam/rtsp/upload), RTSP URL, Detection Sensitivity slider
- Custom PPE items: multi-select checkboxes for PPE violation types to monitor
- No Phone Zone toggle
- Notification preferences: Sound on/off, UI notifications on/off
- "Save Changes" button

---

### 13. TV Display (`/tv`)

**Purpose:** Full-screen, chromeless monitoring wall for large displays / TV screens mounted on the factory floor.

**Behaviour:**
- Opens in a new tab
- No sidebar, no header — full-screen camera grid only
- Auto-rotates through all online cameras or shows a fixed grid (configurable)
- Shows large live feeds with AI annotations
- Alert banners flash across the bottom when violations are detected
- Designed for 1080p+ TV displays
- Auto-reconnect on stream loss
- Camera name + status overlay on each tile
- Optional: auto-cycle through cameras every 30 seconds in 1x1 mode

---

### 14. Video Upload and Analysis (accessible from Live Monitor or as sub-page)

**Purpose:** Upload a recorded video for offline AI analysis.

**Data Sources:**
- `POST /video/upload` (multipart file) → `{ job_id, status }`
- `GET /video/status/{job_id}` → `{ status, progress, frames_processed, total_violations, alerts[], violation_timestamps[], ppe_summary, annotated_video_url, video_duration_sec }`
- `GET /video/result/{job_id}` → download annotated MP4

**Layout:**
- Upload drop zone: drag-and-drop or file picker for MP4/AVI/MOV/MKV/WebM
- After upload: progress bar showing analysis progress percentage
- On completion:
  - Video player for annotated output video (from annotated_video_url)
  - Violation timeline: clickable timeline bar showing violation timestamps — click to seek video
  - PPE summary: bar chart of violation types and counts
  - Alert list: table of all detected violations with timestamp, severity, missing items, persons count, thumbnail
  - Download button for annotated video

---

## API Client Module

Create a centralized API client module (`src/api/api.js`) with Axios instance:

```
Base URL: from VITE_API_BASE_URL environment variable
Default headers: Authorization: Bearer <token>
Interceptors: 
  - Request: attach token from localStorage
  - Response: on 401, clear token and redirect to login
```

**API namespaces:**

- `authApi`: login, signup, me
- `usersApi`: updateProfile, getConfig, updateConfig
- `alertsApi`: list, getOne, delete, deleteAll, stats, pending, confirm, dismiss, testTelegram
- `camerasApi`: list, create, get, update, delete, restart, discover, status, snapshot, getZones, createZone, deleteZone, calibrationHistory, versionCalibration, restoreCalibration
- `detectionApi`: WebSocket connection helpers
- `videoApi`: upload, status, result
- `facesApi`: register, list, rename, delete
- `settingsApi`: list, get, update, cylinderLogs, uploadBaseline, deleteBaseline
- `attendanceApi`: list, stats, clockIn, clockOut, exportCSV
- `documentsApi`: list, stats, scan, approve, reject, delete
- `workflowApi`: summary, cashEvents, stockEvents, liftEvents, packingEvents
- `enterpriseApi`: models, registerModel, health, analyticsSummary, workflows, createWorkflow, ruleProfiles, createRuleProfile, rules, createRule, alertCases, actionAlertCase

---

## WebSocket Integration

Two WebSocket endpoints:

### `/ws/detect` — Webcam detection (browser webcam)
- Used for testing with laptop webcam
- Client sends base64 frames from getUserMedia
- Server returns annotated frames with detection results

### `/ws/detect-cctv` — IP Camera detection (primary)
- Used for all live monitoring
- Client sends handshake with `camera_id` (managed mode) or `camera_url` (legacy mode)
- Server continuously returns annotated frames at up to 20 FPS
- Client can send mid-stream config updates: `{ filters, no_phone_zone, enable_face }`

**Connection management:**
- Auto-reconnect with exponential backoff (1s, 2s, 4s, max 30s)
- Connection status indicator per camera tile
- Graceful close on component unmount

---

## Real-Time Features

- **Live alert count badge** on sidebar "Review Queue" item — polls every 30s
- **Dashboard auto-refresh** every 30s
- **Camera status polling** every 10s on the Live Monitor page
- **Toast notifications** for new critical alerts
- **Sound alert** option (configurable) when violations are detected

---

## Mobile Responsiveness (PWA)

- Responsive layout: sidebar becomes slide-out drawer on mobile
- Mobile top bar with hamburger menu, logo, theme toggle
- Camera grid adjusts: 1 column on mobile, 2 on tablet
- Touch-friendly: large tap targets, swipe to dismiss toasts
- PWA manifest + service worker for installable app experience
- Push notification readiness (service worker registration)
- Works on any device on the same LAN (WiFi) — important for factory floor use

---

## Key Interactions Summary

| Action | API Call | Method |
|--------|----------|--------|
| Login | `/auth/login` | POST (form) |
| Signup | `/auth/signup` | POST |
| Get user profile | `/auth/me` | GET |
| List cameras | `/cameras/` | GET |
| Add camera | `/cameras` | POST |
| Edit camera | `/cameras/{id}` | PUT |
| Delete camera | `/cameras/{id}` | DELETE |
| Restart camera | `/cameras/{id}/restart` | POST |
| Discover LAN cameras | `/cameras/discover` | POST |
| Get camera snapshot | `/cameras/{id}/snapshot` | GET |
| Get zones | `/cameras/{id}/zones` | GET |
| Create zone | `/cameras/{id}/zones` | POST |
| Delete zone | `/cameras/{id}/zones/{name}` | DELETE |
| Calibration history | `/cameras/{id}/calibrations` | GET |
| Save calibration | `/cameras/{id}/calibrations/snapshot` | POST |
| Restore calibration | `/cameras/{id}/calibrations/{ver}/restore` | POST |
| List alerts | `/alerts/?filters...` | GET |
| Get alert detail | `/alerts/{id}` | GET |
| Delete alert | `/alerts/{id}` | DELETE |
| Clear all alerts | `/alerts/` | DELETE |
| Alert stats | `/alerts/stats` | GET |
| Pending alerts | `/alerts/pending` | GET |
| Confirm alert | `/alerts/{id}/confirm` | PATCH |
| Dismiss alert | `/alerts/{id}/dismiss` | PATCH |
| Test Telegram | `/alerts/test-telegram` | POST |
| Attendance stats | `/attendance/stats` | GET |
| List attendance | `/attendance/` | GET |
| Clock in | `/attendance/clock-in` | POST |
| Clock out | `/attendance/clock-out/{id}` | POST |
| Export attendance CSV | `/attendance/export` | GET |
| Document stats | `/documents/stats` | GET |
| List documents | `/documents/` | GET |
| Scan document | `/documents/scan` | POST (multipart) |
| Approve document | `/documents/{id}/approve` | PATCH |
| Reject document | `/documents/{id}/reject` | PATCH |
| Delete document | `/documents/{id}` | DELETE |
| Workflow summary | `/workflow/summary` | GET |
| Cash events | `/workflow/cash-events` | GET |
| Stock events | `/workflow/stock-events` | GET |
| Lift events | `/workflow/lift-events` | GET |
| Packing events | `/workflow/packing-events` | GET |
| List settings | `/settings/` | GET |
| Update setting | `/settings/{key}` | PUT |
| Cylinder logs | `/settings/cylinder-logs` | GET |
| Upload baseline | `/cameras/{id}/baseline` | POST (multipart) |
| Delete baseline | `/cameras/{id}/baseline` | DELETE |
| Register face | `/faces/register` | POST |
| List faces | `/faces/` | GET |
| Rename face | `/faces/{id}/label` | PUT |
| Delete face | `/faces/{id}` | DELETE |
| Update profile | `/users/me` | PUT |
| Get user config | `/users/me/config` | GET |
| Update user config | `/users/me/config` | PUT |
| Upload video | `/video/upload` | POST (multipart) |
| Video job status | `/video/status/{id}` | GET |
| Download annotated video | `/video/result/{id}` | GET |
| List AI models | `/platform/models` | GET |
| Register model | `/platform/models` | POST |
| Platform health | `/platform/health` | GET |
| Analytics summary | `/platform/analytics/summary` | GET |
| Workflow profiles | `/platform/workflow-profiles` | GET/POST |
| Rule profiles | `/platform/rule-profiles` | GET/POST |
| Rules | `/platform/rules` | GET/POST |
| Alert cases | `/platform/alert-cases` | GET |
| Action alert case | `/platform/alert-cases/{id}/{action}` | POST |

---

## Important Notes for Implementation

1. **All API routes exist at both root level and under `/api` prefix** — use whichever is simpler. The backend mounts both `POST /cameras` and `POST /api/cameras` (same handler).

2. **Static files are served at `/uploads/`** — snapshot images and annotated videos are accessed at `/uploads/snapshots/...` and `/uploads/annotated_*.mp4`.

3. **WebSocket URLs** should be constructed relative to the API base URL, replacing `http` with `ws` (e.g., `ws://192.168.1.100:8000/ws/detect-cctv`).

4. **SQLite database** — on-premise, single-server deployment. No multi-tenant or cloud considerations.

5. **Camera RTSP URLs contain credentials** — mask the password in the UI (show `rtsp://admin:****@192.168.1.64/...`).

6. **The app name is "OccuSafe"** — use this consistently in the title, sidebar logo, PWA manifest, and meta tags.

7. **Floors are fixed**: Ground, First, Second, Shop — these are not user-configurable.

8. **Employee roles**: baker, packer, cashier, supervisor, etc. — free-text, not an enum.

9. **Zone polygon coordinates are in pixel space** of the camera's native frame resolution. The ZonePainter must scale between the canvas display size and the image's natural resolution when saving/loading polygons.

10. **Face registration** requires a clear, front-facing photo with good lighting. The backend returns an error if no face is detected in the uploaded image.
