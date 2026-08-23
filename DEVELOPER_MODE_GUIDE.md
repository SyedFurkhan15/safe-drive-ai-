# SafeDrive AI — Clean Camera Feed + Developer Mode (Implementation Notes)

## Goal
- Default (Developer Mode OFF): clean professional camera overlay.
- Developer Mode ON: show full MediaPipe landmarks + EAR/MAR + yaw/pitch/roll + gaze + calibration labels.

## Current debug overlays located in
- `driver_monitor.py`
  - Mesh/contours/irises rendering
  - `FACE LOCKED` / `CALIBRATING...`
  - `EAR ...`, `DROWSY ...`, `MAR ...`, yawn text
  - pose text `Yaw ... Pitch ... Roll ...`
  - gaze text `Gaze: ...`

## Required toggle UI
- Add to `app.py` sidebar:
  - `[ ] Show AI Landmarks`
  - `[ ] Developer Mode`
- Default OFF.

## Rendering rules
### Developer Mode OFF
- Only draw:
  - Face box / driver lock indicator (minimal)
  - Phone bounding box (handled by `phone_detector.py`)
  - Critical warnings (high risk / critical emergency)
- Suppress all text/tech overlays:
  - EAR/MAR/yawn counters
  - pose Euler angles
  - gaze direction
  - FACE LOCKED / CALIBRATING...
  - full MediaPipe mesh/contours/irises (unless Show AI Landmarks ON)

### Developer Mode ON
- Show full technical HUD:
  - Face mesh/contours/irises
  - EAR/MAR and blink/yawn info
  - yaw/pitch/roll and gaze direction
  - calibration/lock label

## Wiring approach (minimal structural change)
- In `DriverMonitor.__init__` add flags, e.g.:
  - `self.developer_mode: bool = False`
  - `self.show_ai_landmarks: bool = False`
- In `process_frame()` gate each draw_* and cv2.putText block behind those flags.
- In `app.py` inside the processing loop set:
  - `ctx.video_processor.driver_monitor.developer_mode = ...`
  - `ctx.video_processor.driver_monitor.show_ai_landmarks = ...`

## Notes
- This keeps existing detection logic intact; only overlay drawing is changed.
- Must not reduce FPS significantly: avoid extra copies when overlays are OFF.

