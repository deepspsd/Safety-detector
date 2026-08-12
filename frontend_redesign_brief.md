# Frontend Redesign Brief

## Objective

Redesign the frontend as an operator-first bakery/factory monitoring platform that matches:

1. The actual backend capabilities already present in this repo
2. The client requirements in [req.md](file:///e:/dasari/safety-detection-tool/req.md)
3. A multi-floor, real-time control-room workflow

This brief is written so it can be used directly as:

- a product/UX redesign document
- a handoff to a frontend team
- a prompt source for an AI UI generation tool

---

## 1. What The Backend Already Supports

### Core platform entities

- `users`: admin/supervisor accounts
- `cameras`: multi-floor camera inventory with health, calibration, capabilities, drift status
- `zone_configs`: per-camera polygon zones with workflow/rule metadata
- `alerts`: confirmed, pending review, dismissed
- `attendance_records`: clock-in/clock-out
- `invoice_logs` and `order_form_logs`: inward/outward OCR document logs
- `lift_events`: lift entry, exit, idle events
- `cylinder_logs`: cylinder detection and usage history
- `dirty_floor_baselines`: cleanliness baseline images
- enterprise tables for rule profiles, workflow profiles, alert cases, analytics, health logs, model registry, calibration history

### Existing backend modules exposed by API

- Auth and profile
- Alert feed + review queue
- Camera CRUD + live status + restart + snapshot + discovery
- Zone calibration per camera
- Attendance
- Document OCR scanning and approval/rejection
- Workflow monitoring:
  - cash zone
  - stock exposure
  - lift monitoring
  - packing idle monitoring
- Settings / thresholds
- Dirty-floor baseline upload
- Enterprise pages that are not yet surfaced well in the UI:
  - model registry
  - analytics summary
  - health logs
  - alert cases
  - workflow profiles
  - rule profiles
  - rules
  - calibration history / restore

### Important backend-detected behaviors

Implemented or partially implemented:

- PPE / dress compliance
- bakery head cap detection
- mask / gloves / goggles / vest / hardhat / shoes
- bangles detection
- phone usage / no-phone-zone
- idle person detection
- packing-hand-movement idle logic
- lift idle logic
- shift start lateness
- dirty floor detection
- gas / oven idle monitoring
- shop unattended monitoring
- stock exposure
- unauthorized cashbox access
- document OCR for inward invoice and outward order form
- camera offline and camera drift alerts
- face recognition / unknown face alerts
- attendance auto-trigger via face match hook

### Important backend gaps or mismatches

These matter because the new UI should not pretend these are finished:

1. `employees` table exists in backend schema, but there is no dedicated employee CRUD API yet.
   Current frontend uses `faces` as a proxy for employee roster.

2. Some client requirements are only partially covered or not implemented:
   - clean shave
   - haircut length
   - chewing / eating
   - theft via window passing / throwing goods
   - stolen goods / pocketed cash evidence workflow
   - raw-material and finished-goods movement chain across all floors
   - “items should not be kept openly” is only partly mapped through stock exposure logic
   - full store authorization workflow is not modeled as a dedicated UI flow

3. Backend has enterprise/admin power features, but current frontend does not expose them clearly.

4. Current app navigation is feature-centric, not plant-operations-centric.

---

## 2. What Is Wrong With The Current Frontend

### Structural problems

- The app is organized like a generic monitoring demo, not like a bakery operations control system.
- Pages are split by technical feature (`Alerts`, `Workflow`, `Documents`, `Settings`) instead of how operators think (`Entrance Control`, `Production`, `Cleanliness`, `Attendance`, `Incident Review`).
- The floor overview only shows cameras, not floor operations, shift status, or required checks.
- There is no true command center page for admins.
- No strong distinction between:
  - live monitoring
  - exception management
  - audit/history
  - configuration

### UX problems

- Too many controls are buried inside `Settings`.
- The current dashboard is still biased toward the original PPE product, not this bakery workflow.
- Documents page is functional but not framed as an “entry gate approval” workflow.
- Camera setup and zone calibration are usable but not intuitive for non-technical operators.
- Review Queue exists, but the app still lacks a strong incident case-management experience.
- Employee management is confusing because “Employees” is really “Faces”.

### Domain mismatch with client expectations

The client thinks in terms of:

- floors
- sections
- entrances/exits
- shop
- cleanliness
- stock movement
- shift start rules
- admin alerts on phone

The UI should therefore feel like:

- a bakery operations console
- a compliance dashboard
- a floor-wise monitoring tool

not like a generic AI detection playground.

---

## 3. Recommended Product Positioning

The new UI should be presented as:

**AI Bakery Operations & Safety Command Center**

It should support 3 main user mindsets:

1. **Control Room / Admin**
   - wants plant-wide status, critical alerts, camera health, pending reviews

2. **Supervisor / Floor Manager**
   - wants floor-specific cameras, shift compliance, production flow, cleanliness, attendance

3. **Audit / Management**
   - wants logs, incidents, document proof, reports, approval history

---

## 4. Recommended Navigation / Information Architecture

### Primary navigation

1. Command Center
2. Floors
3. Entrance & Store Control
4. Operations Monitoring
5. Alerts & Incidents
6. Attendance & People
7. Reports & Audit
8. System Setup

### Secondary navigation inside each area

#### 1. Command Center

- Executive Overview
- Critical Alerts
- Camera Health
- Shift Status
- Pending Reviews

#### 2. Floors

- Ground Floor
- First Floor
- Second Floor
- Shop
- All Cameras Map

#### 3. Entrance & Store Control

- Inward Gate
- Outward Gate
- Store Entry Control
- Document Verification Log
- Authorized Entry Log

#### 4. Operations Monitoring

- Dough Mixing
- Biscuit Cutting
- Oven / Gas Monitoring
- Packing Monitoring
- Lift Monitoring
- Stock Monitoring
- Cleanliness Monitoring

#### 5. Alerts & Incidents

- Live Alerts
- Review Queue
- Incident Cases
- Critical Timeline
- Alert Rules by Severity

#### 6. Attendance & People

- Attendance Dashboard
- Employee Directory
- Face Registry
- Unknown Person Events
- Shift Punctuality

#### 7. Reports & Audit

- Daily Summary
- Compliance Reports
- Attendance Export
- Document Audit
- Stock / Lift / Packing Logs

#### 8. System Setup

- Camera Inventory
- Zone Calibration
- Floor Configuration
- Rule Profiles
- Workflow Profiles
- Threshold Settings
- Device / Model Health
- Notification Settings

---

## 5. Recommended Top-Level Screens

## A. Command Center

This should be the default landing page after login.

### Purpose

Give the admin a single-glance understanding of plant health.

### Must contain

- Global status banner:
  - cameras online/offline
  - active alerts
  - pending review count
  - current shift status by floor
- Floor cards:
  - Ground, First, Second, Shop
  - each card shows:
    - active cameras
    - current violations
    - cleanliness status
    - attendance status
    - production status
- Critical incident rail
- Camera health strip
- “Needs attention now” panel
- Quick actions:
  - open review queue
  - open ground floor
  - verify documents
  - restart offline camera

### Best layout

- Top KPI ribbon
- Left: floor summary grid
- Right: critical alerts + pending review queue
- Bottom: trend charts and camera health timeline

---

## B. Floor Command Page

Create one reusable page template for each floor:

- Ground Floor
- First Floor
- Second Floor
- Shop

### Purpose

Show everything relevant to that floor in one place.

### Each floor page should contain

- floor header
- shift start compliance
- live camera wall
- section status cards
- active people count
- alerts in this floor
- cleanliness status
- stock status where relevant
- recent document / movement / lift events where relevant

### Ground Floor page sections

- Main entrance inward/outward monitoring
- Attendance at glass door
- Dough mixing
- Biscuit cutting
- Oven section
- Packing entrance
- Packing near oven
- Main entrance loading/unloading
- Stock behind main entrance
- Cleanliness status

### First Floor page sections

- 4 working tables overview
- Dough mixing start by 6 AM
- Dress code status
- Lift movement
- Raw material stock
- Cylinder usage / days-in-use
- Cleanliness
- Idle monitoring

### Second Floor page sections

- Shift start by 5 AM
- Fire / stove supervision
- Gas / oil idle loss alerts
- Dress code
- Cleanliness
- Stock monitoring
- Theft / suspicious handling / window-side risk zone
- Camera obstruction alerts

### Shop page sections

- Cash handling monitoring
- Dress code
- Vendor payment proof
- Shop unattended alerts
- Idle duration
- Counter camera view

---

## C. Entrance & Store Control

This should be a dedicated workflow area, not just a generic documents page.

### 1. Inward Gate Approval

Purpose:
Monitor goods entering the premises.

UI sections:

- live camera feed
- document scan panel
- OCR result
- invoice status: approved / rejected / manual review
- person image / captured proof
- goods count / weight fields if extracted
- recent inward entries table

### 2. Outward Gate Approval

Purpose:
Monitor goods leaving the premises.

UI sections:

- live feed
- order form scan panel
- OCR result
- person holding document snapshot
- approved / rejected status
- linked dispatch notes
- outward movement history

### 3. Store Entry Control

Purpose:
Monitor authorized entry and stock movement in store areas.

UI sections:

- store camera list
- authorized person list
- unknown person events
- inward/outward store movements
- suspicious access alerts

### 4. Unified Document Verification Log

Purpose:
Audit all invoice and order-form scans.

UI sections:

- tabs for inward / outward / all
- approved / rejected / manual override
- OCR text preview
- image preview
- linked camera
- linked person if known
- export

---

## D. Operations Monitoring

This area should replace the current fragmented workflow pages with domain-based modules.

### 1. Dough Mixing Monitor

- live status by camera
- started on time or late
- operators present
- idle alerts
- movement to next section

### 2. Biscuit Cutting Monitor

- machine running status proxy
- operator present after machine start
- transition to packing section
- idle / delay alerts

### 3. Oven / Gas Monitor

- oven-area live feed
- supervision present / absent
- gas idle timer
- prolonged boiling / heating risk alerts
- second-floor priority indicator

### 4. Packing Monitor

- hand movement activity
- idle while standing still logic
- packing line view
- workers per station
- repeat idle offenders

### 5. Lift Monitoring

- lift event timeline
- current lift zone occupancy
- idle in lift zone
- movement between floors
- finished goods movement context

### 6. Stock Monitoring

- raw materials by floor
- exposed items alerts
- goods movement in/out
- finished goods dispatch visibility
- store and entrance stock exceptions

### 7. Cleanliness Monitoring

- cleanliness status by floor
- baseline image vs current image
- dirty floor alerts
- morning clean-state confirmation
- unresolved cleanliness issues

---

## E. Alerts & Incidents

The current alerts pages should be redesigned into a proper incident center.

### 1. Live Alerts

- real-time feed
- filters by floor, camera, severity, type, section
- snapshot preview
- grouped by newest / critical first

### 2. Review Queue

Use for low-confidence detections:

- dirty floor
- gas idle
- chewing / eating when implemented
- cash-in-pocket when implemented

Actions:

- confirm
- dismiss
- escalate
- assign note

### 3. Incident Cases

This should surface backend `alert_cases`.

UI sections:

- open / acknowledged / dismissed
- linked alerts
- timestamps
- notes
- assignee
- status trail

### 4. Alert Detail Drawer / Modal

Should show:

- image / frame
- floor
- camera
- issue
- severity
- confidence
- operator notes
- linked workflow or document if relevant

---

## F. Attendance & People

This area should be cleaned up heavily because current “Employees” and “Faces” overlap.

### 1. Attendance Dashboard

- present / absent / late / on-site
- shift-wise attendance
- floor-wise attendance
- manual clock-in / clock-out
- export CSV

### 2. Employee Directory

This is a new proper UX even if backend still needs dedicated employee CRUD later.

For now AI-generated UI should be designed for future support.

Fields:

- name
- role
- department
- assigned floor
- active/inactive
- face enrolled
- attendance status

### 3. Face Registry

- enroll face
- rename
- delete
- see match confidence examples
- recent unknown detections

### 4. Unknown Person Events

- unrecognized faces
- entry time
- camera
- snapshot
- resolved / unresolved

---

## G. Reports & Audit

### Pages

- Daily Operations Summary
- Compliance Report
- Attendance Export
- Documents Audit
- Alert Analytics
- Camera Health History

### Useful widgets

- alerts by floor
- alerts by type
- attendance by day
- top repeated violations
- document approval rate
- camera uptime
- pending vs confirmed alerts

---

## H. System Setup

The current settings area should become a structured admin console.

### 1. Camera Inventory

- table and cards
- online / offline / error / drift
- purpose, floor, department, zone coverage
- actions: edit, restart, disable AI, snapshot, calibration

### 2. Zone Calibration Studio

- live camera frame
- polygon drawing
- named zones
- zone type presets:
  - entrance
  - outward gate
  - packing
  - oven
  - dough table
  - biscuit cutting
  - lift
  - stock
  - cashbox
  - camera obstruction risk zone
- calibration history
- restore previous version

### 3. Rule Profiles

- list rule profiles
- attach to cameras
- toggle enabled rules
- thresholds and cooldowns

### 4. Workflow Profiles

- setup movement stages
- map floor sections
- define expected transitions

### 5. Threshold Settings

Surface the backend settings cleanly:

- idle limits
- shift start times
- dirty floor threshold
- gas idle threshold
- shop absence threshold
- cashbox allowed hours

### 6. Health & Models

- model registry
- AI model health
- camera health logs
- drift detection logs
- notification delivery health

---

## 6. Recommended Design Language

### Tone

- industrial, clean, premium, operational
- not playful
- easy to scan from distance

### Visual direction

- dark-first monitoring UI
- optional light mode for office use
- large KPI tiles
- strong color semantics:
  - green = normal
  - amber = needs attention
  - red = critical
  - blue = informational
- large camera tiles with clear status badges
- sticky alert rail / command bar

### Component library guidance

Use a modern admin/ops style similar to:

- Vercel-grade spacing and clarity
- Datadog / Grafana style monitoring density
- linear, crisp cards and tables
- polished drawer and modal interactions

### Component patterns

- command center tiles
- timeline rail
- split-pane live video + event list
- floor map cards
- status chips
- review queue cards with approve/dismiss actions
- right-side incident drawer

---

## 7. Page List To Build In The New Frontend

These are the exact pages I recommend:

1. Login
2. Command Center
3. Floor Overview Index
4. Ground Floor Command
5. First Floor Command
6. Second Floor Command
7. Shop Command
8. Inward Gate Control
9. Outward Gate Control
10. Store Entry Control
11. Document Verification Log
12. Dough Mixing Monitor
13. Biscuit Cutting Monitor
14. Oven & Gas Monitor
15. Packing Monitor
16. Lift Monitor
17. Stock Monitor
18. Cleanliness Monitor
19. Live Alerts
20. Review Queue
21. Incident Cases
22. Attendance Dashboard
23. Employee Directory
24. Face Registry
25. Unknown Person Events
26. Reports Dashboard
27. Camera Inventory
28. Zone Calibration Studio
29. Rule Profiles
30. Workflow Profiles
31. Threshold Settings
32. Health & Models

Optional:

33. TV Wall Display
34. Mobile Alert Inbox

---

## 8. Recommended Mapping From Current Pages To New Pages

### Keep but redesign heavily

- `Dashboard` -> `Command Center`
- `FloorOverview` -> `Floor Overview Index` + per-floor command pages
- `Documents` -> `Inward Gate`, `Outward Gate`, `Document Verification Log`
- `WorkflowMonitor` -> split into operations modules
- `Alerts` -> `Live Alerts`
- `AlertsReview` -> `Review Queue`
- `Attendance` -> keep, but expand into people module
- `Cameras` -> `Camera Inventory`
- `Settings` -> `System Setup`

### Replace or reorganize

- `Employees` should become `Employee Directory` and stop behaving like only a face list
- `LiveMonitor` should become a reusable live-monitor shell that can be embedded inside floor and operations pages
- hidden enterprise routes should become visible admin pages

---

## 9. Requirement Coverage Matrix

### Strongly supported now

- multi-floor camera monitoring
- entrance document OCR
- attendance logging
- idle detection
- packing idle
- lift monitoring
- stock exposure
- cleanliness baseline workflow
- gas/oven idle
- shop unattended
- cash zone alerts
- mobile notifications via Telegram
- on-prem data storage
- camera offline/drift alerts

### Partially supported

- employee identity flow
- section-to-section workflow transitions
- store authorization
- finished goods dispatch traceability
- suspicious item handling

### Not clearly implemented yet

- clean shave
- haircut length
- chewing / eating reliable production UI
- pocketing cash proof flow
- window theft / throwing goods logic
- exact goods weight extraction from invoices

The new UI should mark these as:

- `Active`
- `Beta / Needs Review`
- `Planned`

so the client sees transparency.

---

## 10. Suggested AI Tool Prompt

Use this as the prompt for an AI UI generation tool:

```text
Design a complete modern web app UI for an AI-powered bakery operations and safety monitoring platform called "Bakery Command Center".

This is a multi-floor factory monitoring system for a bakery with Ground Floor, First Floor, Second Floor, Shop, and store/entrance monitoring areas.

The UI should feel like a premium industrial operations dashboard: clean, dark-first, highly legible, enterprise-grade, and easy for admins and floor supervisors to scan quickly.

Main product goals:
- monitor live cameras across multiple floors
- track dress code and safety compliance
- monitor entrance inward and outward document verification
- monitor attendance and authorized entry
- track packing activity, lift activity, stock exposure, cleanliness, shop attendance, gas/oven supervision, and camera health
- support alert review workflows with snapshot evidence
- support reports, audit logs, and admin system configuration

Design the following pages:
1. Login
2. Command Center dashboard
3. Floor overview index
4. Ground Floor command page
5. First Floor command page
6. Second Floor command page
7. Shop command page
8. Inward Gate Control
9. Outward Gate Control
10. Store Entry Control
11. Document Verification Log
12. Dough Mixing Monitor
13. Biscuit Cutting Monitor
14. Oven and Gas Monitor
15. Packing Monitor
16. Lift Monitor
17. Stock Monitor
18. Cleanliness Monitor
19. Live Alerts
20. Review Queue
21. Incident Cases
22. Attendance Dashboard
23. Employee Directory
24. Face Registry
25. Unknown Person Events
26. Reports Dashboard
27. Camera Inventory
28. Zone Calibration Studio
29. Rule Profiles
30. Workflow Profiles
31. Threshold Settings
32. Health and Models

Important UX requirements:
- landing page must be a command center with floor cards, critical alerts, camera health, pending reviews, and shift status
- each floor page must show live camera wall, section status, alerts, cleanliness, and operations widgets specific to that floor
- create strong workflows for inward and outward gate control with OCR results, document approval status, and person snapshot proof
- alerts must have strong severity hierarchy and a review queue for low-confidence detections
- employee and face management should be separate but connected
- admin setup pages must support camera inventory, zone calibration, rules, workflows, thresholds, and health monitoring
- use polished cards, drawers, tables, timeline components, KPI strips, and split-screen monitoring layouts
- support desktop-first control room usage, with responsive tablet and mobile views

Visual style:
- dark-first enterprise monitoring UI
- premium, modern, crisp, minimal clutter
- green/amber/red/blue semantic status colors
- dense but readable layouts
- large camera cards and strong status chips
- modern typography, subtle gradients, glassy overlays only where useful
- no cartoon styling

Also generate:
- global sidebar navigation
- page headers
- reusable design system components
- alert cards
- camera cards
- incident drawer
- floor summary widgets
- timeline widgets
- tables and filters

The output should look production-ready for a real bakery factory monitoring system, not a generic PPE demo.
```

---

## 11. Final Recommendation

If you want the redesign to land well, the frontend should be built around:

- **Command Center first**
- **Floor-based navigation second**
- **Operations workflows third**
- **Admin configuration last**

That order matches how the client thinks about the product and also matches the backend much better than the current UI.
