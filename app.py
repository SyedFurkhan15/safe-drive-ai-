"""
SafeDrive AI - Main Application
=================================
Professional ADAS-grade Driver Monitoring System Dashboard.

Real-Time Driver Risk Assessment and Accident Prevention System.
Inspired by Tesla DMS / NVIDIA DRIVE / Mercedes ADAS dashboards.

Features:
- Live webcam feed with AI overlays
- Real-time drowsiness detection with continuous alarm (eyes closed timer)
- Mobile phone detection with escalating alerts (yolov8s)
- Driver distraction monitoring (head pose estimation)
- Yawn count tracking (60s rolling window)
- Animated risk score gauge (0-100) with time-gated escalation
- Continuous alarm thread (beeps every 0.5s while active)
- Large animated popup notifications
- Auto screenshot capture (drowsy/phone/high-risk events)
- CSV event logging (logs.csv)
- FPS counter, detection latency, session statistics
- Risk trend mini-graph
- Session statistics

Usage:
    streamlit run app.py

Author: SafeDrive AI Team
"""


import mediapipe as mp

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import time
import threading
try:
    import av
except ModuleNotFoundError:
    av = None

try:
    from streamlit_webrtc import VideoProcessorBase
except ImportError:
    class VideoProcessorBase:
        pass
import os
import sys
from datetime import datetime

# ── Configure WebRTC for Stable Connection (STUN servers with safety guard) ──
try:
    import aiortc
    from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer
    _orig_pc_init = RTCPeerConnection.__init__
    def _safe_pc_init(self, configuration=None):
        if configuration is None:
            configuration = RTCConfiguration(iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            ])
        _orig_pc_init(self, configuration=configuration)
    RTCPeerConnection.__init__ = _safe_pc_init
except Exception:
    pass

# ── Environment Validation Guard ───────────────────────────────────────
if not hasattr(mp, "solutions"):
    st.set_page_config(
        page_title="SafeDrive AI | Environment Error",
        page_icon="⚠️",
        layout="wide"
    )
    _venv_py = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")
    st.error(f"""
    ### ⚠️ Wrong Python Environment Detected!

    SafeDrive AI is currently running on **Python {sys.version.split()[0]}** (`{sys.executable}`).
    
    This Python environment has MediaPipe {getattr(mp, '__version__', 'unknown')} without the legacy `mp.solutions` API.

    SafeDrive AI requires the dedicated **Python 3.11 virtual environment** at:
    `{_venv_py}`

    ---
    ### 🚀 How to start correctly:
    1. Stop this Streamlit server (press **Ctrl+C** in your terminal).
    2. Start the app with the virtual environment:
    ```cmd
    .\\venv\\Scripts\\python.exe -m streamlit run app.py
    ```
    or simply double-click **`START_SAFEDRIVE.bat`** or **`run_app.bat`**.
    """)
    st.stop()

# ── Page Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeDrive AI | Professional Driver Monitoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "# SafeDrive AI\n"
            "Professional ADAS-grade Driver Monitoring System\n\n"
            "Built with OpenCV, MediaPipe, YOLOv8s, and Streamlit.\n"
            "Inspired by Tesla DMS / NVIDIA DRIVE / Mercedes ADAS."
        )
    }
)

# ── Import Project Modules ─────────────────────────────────────────────
from driver_monitor import DriverMonitor
from phone_detector import PhoneDetector
from risk_engine import RiskEngine
from alert_system import AlertSystem
from event_logger import EventLogger

# ── Load Custom CSS ────────────────────────────────────────────────────
def load_css():
    """Load the premium custom CSS stylesheet."""
    css_path = os.path.join(os.path.dirname(__file__), "static", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
            html, body, [data-testid="stAppViewContainer"] {
                background-color: #080c14 !important;
                color: #e8f1ff !important;
                font-family: 'Inter', sans-serif !important;
            }
        </style>
        """, unsafe_allow_html=True)

load_css()

# ── Generate Alert Sound ───────────────────────────────────────────────
alert_sound_path = os.path.join(os.path.dirname(__file__), "assets", "alert_sound.wav")
if not os.path.exists(alert_sound_path):
    os.makedirs(os.path.dirname(alert_sound_path), exist_ok=True)
    AlertSystem.generate_alert_sound(alert_sound_path)

# ── Ensure screenshots directory exists ────────────────────────────────
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "monitoring_active": False,
        "alerts_enabled": True,
        "session_started": False,
        "driver_status": "Waiting",
        "risk_score": 0,
        "risk_level": "Safe",
        "risk_color": "#76b900",
        "ear_value": 0.0,
        "mar_value": 0.0,
        "yaw_value": 0.0,
        "pitch_value": 0.0,
        "gaze_direction": "—",
        "blink_rate": 0.0,
        "phone_detected": False,
        "phone_confidence": 0.0,
        "phone_seconds": 0.0,
        "drowsy_detected": False,
        "distracted_detected": False,
        "yawning_detected": False,
        "face_detected": False,
        "eyes_closed_seconds": 0.0,
        "yawn_count_60s": 0,
        "alert_log": [],
        "session_duration": "00:00:00",
        "total_alerts": 0,
        "max_risk": 0,
        "avg_risk": 0.0,
        "drowsy_events": 0,
        "phone_events": 0,
        "distraction_events": 0,
        "contributions": {},
        "processed_frame": None,
        "last_update_time": time.time(),
        "risk_trend": [],
        "screenshots_saved": 0,
        "fps_display": 0.0,
        "latency_ms": 0.0,
        "popup_html": "",
        "popup_expiry": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ── Shared Model Singletons for Instant WebRTC Handshake ──────────────
_SHARED_DRIVER_MONITOR = None
_SHARED_PHONE_DETECTOR = None

def get_shared_driver_monitor():
    global _SHARED_DRIVER_MONITOR
    if _SHARED_DRIVER_MONITOR is None:
        _SHARED_DRIVER_MONITOR = DriverMonitor()
    return _SHARED_DRIVER_MONITOR

def get_shared_phone_detector():
    global _SHARED_PHONE_DETECTOR
    if _SHARED_PHONE_DETECTOR is None:
        _SHARED_PHONE_DETECTOR = PhoneDetector()
    return _SHARED_PHONE_DETECTOR


# ══════════════════════════════════════════════════════════════════════
#  VIDEO PROCESSORS
# ══════════════════════════════════════════════════════════════════════

class TestVideoProcessor(VideoProcessorBase):
    """Minimal test video processor for debugging WebRTC stream."""
    def recv(self, frame):
        return frame

    def get_state(self):
        return {}


class SafeDriveVideoProcessor(VideoProcessorBase):
    """
    Video frame processor for streamlit-webrtc.

    Runs in a separate thread. Processes each video frame through:
    1. MediaPipe face analysis (drowsiness, yawn, head pose)
    2. YOLOv8s phone detection (with face bbox for overlap)
    3. Risk score calculation (time-gated escalation)
    4. Continuous alarm management
    5. Alert generation + popup triggers
    6. Auto screenshot capture
    7. CSV event logging
    8. Visual overlay rendering
    """

    def __init__(self):
        """Initialize all AI modules."""
        print("[SafeDrive AI] SafeDriveVideoProcessor.__init__() starting...")
        try:
            print("[SafeDrive AI] Initializing DriverMonitor...")
            self.driver_monitor = get_shared_driver_monitor()
            print("[SafeDrive AI] DriverMonitor initialized successfully.")
        except Exception as e:
            import traceback
            print(f"[SafeDrive AI] ERROR initializing DriverMonitor: {e}")
            traceback.print_exc()
            raise

        try:
            print("[SafeDrive AI] Initializing PhoneDetector...")
            self.phone_detector = get_shared_phone_detector()
            print("[SafeDrive AI] PhoneDetector initialized successfully.")
        except Exception as e:
            import traceback
            print(f"[SafeDrive AI] ERROR initializing PhoneDetector: {e}")
            traceback.print_exc()
            raise

        self.risk_engine = RiskEngine()
        self.alert_system = AlertSystem()
        self.event_logger = EventLogger()
        print("[SafeDrive AI] SafeDriveVideoProcessor.__init__() completed successfully.")

        # Thread-safe shared state
        self.lock = threading.Lock()
        self.shared_state = {
            "driver_status": "Initializing...",
            "risk_score": 0,
            "risk_level": "Safe",
            "risk_color": "#76b900",
            "ear": 0.0,
            "mar": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "gaze_direction": "—",
            "blink_rate": 0.0,
            "phone_detected": False,
            "phone_confidence": 0.0,
            "phone_seconds": 0.0,
            "drowsy": False,
            "distracted": False,
            "yawning": False,
            "face_detected": False,
            "eyes_closed_seconds": 0.0,
            "yawn_count_60s": 0,
            "new_alerts": [],
            "contributions": {},
            "session_stats": {},
            "risk_trend": [],
            "screenshots_saved": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "popup_type": None,
            "popup_message": "",
        }

        # Frame management
        self.frame_count = 0
        self.yolo_interval = 3             # Run YOLO every 3 frames
        self.last_phone_result = {
            "detected": False, "confidence": 0.0, "bbox": None,
            "phone_continuous_duration": 0.0, "near_face": False, "near_ear": False
        }

        self.alerts_enabled = True
        self.screenshots_saved = 0

        # FPS tracking
        self._fps_start = time.time()
        self._fps_frame_count = 0
        self._current_fps = 0.0

        # Screenshot throttling
        self._last_screenshot_times = {}   # event_type -> last save timestamp

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process a single video frame (called by streamlit-webrtc)."""
        t_start = time.time()
        try:
            img = frame.to_ndarray(format="bgr24")
            if img is None:
                return frame
            self.frame_count += 1

            # ── 1. Driver Monitoring (MediaPipe) ──────────────────────────
            try:
                monitor_result = self.driver_monitor.process_frame(img)
                processed_frame = monitor_result.get("processed_frame", img)
            except Exception as e:
                print(f"[SafeDrive AI] DriverMonitor error: {e}")
                monitor_result = {"face_detected": False}
                processed_frame = img

            # ── 2. Phone Detection (YOLOv8s) ─────────────────────────────
            try:
                face_bbox = monitor_result.get("face_bbox", None)
                if self.frame_count % self.yolo_interval == 0:
                    phone_result = self.phone_detector.detect(img, face_bbox=face_bbox)
                    self.last_phone_result = phone_result
                else:
                    phone_result = self.last_phone_result

                processed_frame = self.phone_detector.draw_detection(processed_frame, phone_result)
            except Exception as e:
                print(f"[SafeDrive AI] PhoneDetector error: {e}")
                phone_result = {"detected": False}

            # ── 3. Risk Score ──────────────────────────────────────────────
            try:
                risk_result = self.risk_engine.update(monitor_result, phone_result)
                final_status = risk_result.get("status", "Normal")
            except Exception as e:
                print(f"[SafeDrive AI] RiskEngine error: {e}")
                risk_result = {"score": 0, "level": "Safe", "level_color": "#76b900", "contributions": {}}
                final_status = "Normal"

            # ── 4. Alert Generation ────────────────────────────────────────
            try:
                self.alert_system.alerts_enabled = self.alerts_enabled
                new_alerts = self.alert_system.check_and_generate(
                    monitor_result, phone_result, risk_result
                )
            except Exception as e:
                print(f"[SafeDrive AI] AlertSystem error: {e}")
                new_alerts = []

            # Record alerts and build a combined popup for every detection event
            popup_type = None
            popup_messages = []
            popup_priority = {"critical": 3, "phone": 2, "drowsy": 2, "yawn": 1}

            for alert in new_alerts:
                try:
                    self.risk_engine.record_alert(alert["type"], alert["message"])
                    self.event_logger.log_event(
                        event_type=alert["type"],
                        severity=alert.get("severity", "medium"),
                        risk_score=risk_result.get("score", 0),
                        additional_info=alert.get("message", "")
                    )
                    popup_messages.append(alert.get("message", ""))
                except Exception:
                    pass

                atype = alert.get("type", "")
                if "DROWSINESS" in atype:
                    candidate_type = "drowsy"
                elif "PHONE" in atype:
                    candidate_type = "phone"
                elif "YAWNING" in atype:
                    candidate_type = "yawn"
                elif alert.get("severity", "medium") == "critical":
                    candidate_type = "critical"
                else:
                    candidate_type = None

                if candidate_type is not None:
                    current_priority = popup_priority.get(popup_type, 0)
                    candidate_priority = popup_priority.get(candidate_type, 0)
                    if candidate_priority >= current_priority:
                        popup_type = candidate_type

            popup_message = "<br>".join(popup_messages) if popup_messages else ""

            # ── 5. Auto Screenshot Capture ─────────────────────────────────
            try:
                self._maybe_save_screenshot(processed_frame, monitor_result, phone_result, risk_result)
            except Exception:
                pass

            # ── 6. Session Statistics ──────────────────────────────────────
            try:
                session_stats = self.risk_engine.get_session_stats()
            except Exception:
                session_stats = {}

            # ── 7. FPS Calculation ─────────────────────────────────────────
            self._fps_frame_count += 1
            elapsed = time.time() - self._fps_start
            if elapsed >= 1.0:
                self._current_fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._fps_start = time.time()

            latency_ms = (time.time() - t_start) * 1000

            # ── 8. Update Shared State (thread-safe) ──────────────────────
            with self.lock:
                self.shared_state.update({
                    "driver_status": final_status,
                    "risk_score": risk_result.get("score", 0),
                    "risk_level": risk_result.get("level", "Safe"),
                    "risk_color": risk_result.get("level_color", "#76b900"),
                    "ear": monitor_result.get("ear", 0.0),
                    "mar": monitor_result.get("mar", 0.0),
                    "yaw": monitor_result.get("yaw", 0.0),
                    "pitch": monitor_result.get("pitch", 0.0),
                    "gaze_direction": monitor_result.get("gaze_direction", "—"),
                    "blink_rate": monitor_result.get("blink_rate", 0.0),
                    "phone_detected": phone_result.get("detected", False),
                    "phone_confidence": phone_result.get("confidence", 0.0),
                    "phone_seconds": phone_result.get("phone_continuous_duration", 0.0),
                    "drowsy": monitor_result.get("drowsy", False),
                    "distracted": monitor_result.get("distracted", False),
                    "yawning": monitor_result.get("yawning", False),
                    "face_detected": monitor_result.get("face_detected", False),
                    "eyes_closed_seconds": monitor_result.get("eyes_closed_duration", 0.0),
                    "yawn_count_60s": monitor_result.get("yawn_count_60s", 0),
                    "new_alerts": new_alerts,
                    "contributions": risk_result.get("contributions", {}),
                    "session_stats": session_stats,
                    "risk_trend": session_stats.get("risk_trend", []),
                    "screenshots_saved": self.screenshots_saved,
                    "fps": round(self._current_fps, 1),
                    "latency_ms": round(latency_ms, 1),
                    "popup_type": popup_type,
                    "popup_message": popup_message,
                })

            if not isinstance(processed_frame, np.ndarray):
                processed_frame = img

            return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

        except Exception as e:
            import traceback
            print(f"[SafeDrive AI] Fatal Exception in recv(): {e}")
            traceback.print_exc()
            return frame

    def _maybe_save_screenshot(self, frame, monitor_result, phone_result, risk_result):
        """
        Auto-save screenshot on significant events.
        - Drowsiness detected (eyes closed > 2s)
        - Phone usage confirmed
        - Risk score > 80
        Throttled: max 1 per event type per 10 seconds.
        """
        now = time.time()
        throttle = 10.0  # seconds

        def save(prefix):
            last = self._last_screenshot_times.get(prefix, 0)
            if now - last < throttle:
                return
            self._last_screenshot_times[prefix] = now
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{ts}.jpg"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            try:
                cv2.imwrite(filepath, frame)
                self.screenshots_saved += 1
            except Exception as e:
                print(f"Screenshot save error: {e}")

        if monitor_result.get("drowsy") and monitor_result.get("eyes_closed_duration", 0) >= 2.0:
            save("drowsy")

        if phone_result.get("detected") and phone_result.get("phone_continuous_duration", 0) >= 2.0:
            save("phone")

        if risk_result.get("score", 0) >= 80:
            save("high_risk")

    def get_state(self):
        """Get a thread-safe copy of the current shared state."""
        with self.lock:
            return dict(self.shared_state)

    def reset(self):
        """Reset all modules for a new session."""
        self.driver_monitor.reset()
        self.phone_detector.reset()
        self.risk_engine.reset()
        self.alert_system.reset()
        self.frame_count = 0
        self.last_phone_result = {
            "detected": False, "confidence": 0.0, "bbox": None,
            "phone_continuous_duration": 0.0, "near_face": False, "near_ear": False
        }
        self.screenshots_saved = 0
        self._last_screenshot_times = {}
        self._fps_frame_count = 0
        self._fps_start = time.time()


# ══════════════════════════════════════════════════════════════════════
#  UI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def render_header(state=None):
    """Render the premium dashboard header with live session/FPS/latency badges."""
    fps = state.get("fps", 0.0) if state else 0.0
    latency = state.get("latency_ms", 0.0) if state else 0.0
    session = state.get("session_stats", {}) if state else {}
    duration = session.get("duration_formatted", "00:00:00")
    total_alerts = session.get("total_alerts", 0)

    fps_color = "#76b900" if fps >= 20 else ("#ff9800" if fps >= 10 else "#f44336")

    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1>🛡️ SafeDrive AI</h1>
            <p>Professional AI Driver Monitoring System &nbsp;·&nbsp; NVIDIA ADAS Demo</p>
        </div>
        <div class="header-badges">
            <div class="header-badge">
                <span class="badge-value" style="color:{fps_color};">{fps:.0f}</span>
                <span class="badge-label">FPS</span>
            </div>
            <div class="header-badge">
                <span class="badge-value" style="color:#94a3b8;">{latency:.0f}ms</span>
                <span class="badge-label">Latency</span>
            </div>
            <div class="header-badge">
                <span class="badge-value">{duration}</span>
                <span class="badge-label">Session</span>
            </div>
            <div class="header-badge">
                <span class="badge-value" style="color:{'#f44336' if total_alerts > 5 else '#76b900'};">{total_alerts}</span>
                <span class="badge-label">Alerts</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status):
    """Render a colored status badge based on driver status."""
    status_map = {
        "Attentive":               ("status-safe",     "✅"),
        "Safe":                    ("status-safe",     "✅"),
        "Drowsy":                  ("status-danger",   "😴"),
        "Distracted":              ("status-moderate", "👀"),
        "Phone Usage Detected":    ("status-moderate", "📱"),
        "Phone Distraction":       ("status-danger",   "📱"),
        "High Risk Driver":        ("status-danger",   "🚨"),
        "🚨 Critical Emergency":   ("status-critical", "🆘"),
        "No Face Detected":        ("status-moderate", "❓"),
        "Waiting":                 ("status-safe",     "⏳"),
        "Initializing...":         ("status-safe",     "⚙️"),
    }
    css_class, icon = status_map.get(status, ("status-moderate", "ℹ️"))

    st.markdown(f"""
    <div class="status-badge {css_class}">
        <span>{icon}</span>
        <span>{status}</span>
    </div>
    """, unsafe_allow_html=True)


def render_risk_gauge(score, level, color):
    """Render an animated risk score gauge with gradient fill."""
    score_class = ""
    if score >= 90:
        score_class = "critical"
    elif score > 60:
        score_class = "danger"
    elif score > 30:
        score_class = "moderate"

    if score <= 30:
        fill_color = "linear-gradient(90deg, #76b900, #8fd400)"
    elif score <= 60:
        fill_color = "linear-gradient(90deg, #ff9800, #ffc107)"
    elif score < 90:
        fill_color = "linear-gradient(90deg, #f44336, #ff5722)"
    else:
        fill_color = "linear-gradient(90deg, #ff0033, #ff3300, #ff6600)"

    st.markdown(f"""
    <div class="risk-gauge-container">
        <div style="font-size: 0.7rem; color: #8faabf; text-transform: uppercase;
                    letter-spacing: 0.12em; font-weight: 700;">Risk Score</div>
        <div class="risk-score-number {score_class}">{score}</div>
        <div style="font-size: 0.85rem; color: {color}; font-weight: 700;
                    margin-bottom: 0.25rem; letter-spacing: 0.03em;">{level}</div>
        <div class="risk-gauge-bar">
            <div class="risk-gauge-fill" style="width: {score}%; background: {fill_color};"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.6rem; color:#4d6280;">
            <span>0 Safe</span>
            <span>30 Moderate</span>
            <span>60 High</span>
            <span>90 Critical</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_eyes_closed_countdown(seconds):
    """Render flashing eyes-closed countdown banner when drowsy."""
    if seconds < 2.0:
        return

    risk_color = "#ff0033" if seconds >= 8 else "#ff4444" if seconds >= 6 else "#ff6666"

    st.markdown(f"""
    <div class="eyes-closed-countdown">
        <span class="eyes-closed-value" style="color:{risk_color};">{seconds:.1f}s</span>
        <span class="eyes-closed-label">👁 Eyes Closed — WAKE UP DRIVER!</span>
    </div>
    """, unsafe_allow_html=True)


def render_phone_timer(seconds):
    """Render phone usage timer when phone is active."""
    if seconds < 1.0:
        return

    level_color = "#ff0033" if seconds >= 8 else "#ff4400" if seconds >= 5 else "#ff6600"
    level_label = "EMERGENCY" if seconds >= 8 else "DANGER" if seconds >= 5 else "WARNING"

    st.markdown(f"""
    <div class="phone-timer">
        <span class="phone-timer-value" style="color:{level_color};">{seconds:.1f}s</span>
        <span style="font-size:0.6rem; color:#ff8844; display:block; text-transform:uppercase; letter-spacing:0.08em;">📱 Phone Duration — {level_label}</span>
    </div>
    """, unsafe_allow_html=True)


def render_detection_indicator(label, is_active, icon_active, icon_inactive, dot_color=None):
    """Render a detection status indicator with colored dot."""
    state_cls = "active" if is_active else "inactive"
    if dot_color:
        dot_cls = dot_color
    else:
        dot_cls = "red" if is_active else "green"
    icon = icon_active if is_active else icon_inactive
    text_color = "#f44336" if is_active else "#76b900"

    st.markdown(f"""
    <div class="detection-indicator {state_cls}">
        <span class="indicator-dot {dot_cls}"></span>
        <span style="font-size:0.84rem; color:{text_color}; font-weight:500;">{icon} {label}</span>
    </div>
    """, unsafe_allow_html=True)


def render_yawn_counter(count):
    """Render yawn count badge with color-coded severity."""
    if count == 0:
        color, bg = "#76b900", "rgba(118,185,0,0.1)"
        label = "Normal"
    elif count == 1:
        color, bg = "#76b900", "rgba(118,185,0,0.1)"
        label = "Slight"
    elif count == 2:
        color, bg = "#ffaa00", "rgba(255,170,0,0.12)"
        label = "Moderate"
    elif count == 3:
        color, bg = "#ff6600", "rgba(255,102,0,0.12)"
        label = "⚠ Fatigued"
    else:
        color, bg = "#ff0033", "rgba(255,0,51,0.15)"
        label = "🚨 Exhausted"

    st.markdown(f"""
    <div style="background:{bg}; border:1px solid {color}44; border-radius:10px;
                padding:0.6rem 1rem; margin-bottom:0.6rem; text-align:center;">
        <span style="font-family:'JetBrains Mono',monospace; font-size:1.6rem;
                     font-weight:800; color:{color};">{count}</span>
        <span style="font-size:0.65rem; color:{color}; display:block;
                     text-transform:uppercase; letter-spacing:0.1em;">Yawns/60s — {label}</span>
    </div>
    """, unsafe_allow_html=True)


def render_risk_trend(trend_data):
    """Render a mini risk trend sparkline bar graph."""
    if not trend_data:
        return

    # Use last 40 points for display
    data = trend_data[-40:] if len(trend_data) > 40 else trend_data
    max_val = max(data) if data else 1
    if max_val == 0:
        max_val = 1

    bars_html = ""
    for val in data:
        pct = val / 100.0
        height_px = max(2, int(pct * 48))
        if val >= 90:
            color = "#ff0033"
        elif val >= 60:
            color = "#f44336"
        elif val >= 30:
            color = "#ff9800"
        else:
            color = "#76b900"
        bars_html += f'<div class="trend-segment" style="height:{height_px}px; background:{color}; opacity:0.85;"></div>'

    st.markdown(f"""
    <div class="risk-trend-container">
        <div style="font-size:0.65rem; color:#4d6280; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom:0.5rem; font-weight:700;">
            📈 Risk Trend (last {len(data)} samples)
        </div>
        <div class="risk-trend-bar">{bars_html}</div>
        <div style="display:flex; justify-content:space-between; font-size:0.55rem;
                    color:#4d6280; margin-top:4px;">
            <span>← Earlier</span>
            <span>Now →</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with controls and information."""

    with st.sidebar:
        # Input Source selector (Live vs Dataset Analysis)
        st.markdown("##### 🧩 Input Source")
        operation_mode = st.radio(
            "Select mode",
            options=["Live Monitoring", "Dataset Analysis"],
            index=0,
            key="sidebar_operation_mode_radio",
            horizontal=False,
        )
        st.session_state["operation_mode"] = operation_mode

        st.markdown("---")

        st.markdown("""

        <div style="text-align:center; padding:1rem 0;">
            <h2 style="font-family:'Outfit',sans-serif; background:linear-gradient(135deg,#ffffff,#76b900,#00d4ff);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       font-size:1.3rem; margin:0; font-weight:800;">🛡️ SafeDrive AI</h2>
            <p style="color:#4d6280; font-size:0.68rem; margin:0.3rem 0 0 0;">
                v2.0 Professional · NVIDIA AI/ML Internship</p>
        </div>
        <hr style="border-color:rgba(0,212,255,0.08); margin:0.5rem 0 1rem 0;">
        """, unsafe_allow_html=True)


        # ── Alert Controls ─────────────────────────────────────────────


        st.markdown("##### 🔔 Alert Settings")
        alerts_enabled = st.toggle(
            "Enable Audio Alerts",
            value=st.session_state.get("alerts_enabled", True),
            key="alerts_toggle"
        )
        st.session_state["alerts_enabled"] = alerts_enabled

        st.markdown("---")

        # ── Session Control ────────────────────────────────────────────
        st.markdown("##### 🔄 Session Control")
        if st.button("🔄 Reset Session", use_container_width=True, key="reset_btn"):
            for key in list(st.session_state.keys()):
                if key not in ["alerts_enabled", "alerts_toggle", "reset_btn"]:
                    del st.session_state[key]
            init_session_state()
            st.rerun()

        st.markdown("---")

        # ── Developer Mode for Faculty Demonstration ───────────────────
        st.markdown("##### 🎓 Developer Mode (Faculty Presentation)")
        developer_mode = st.toggle(
            "Developer Mode (show technical HUD)",
            value=False,
            key="developer_mode_toggle",
        )
        show_ai_landmarks = st.toggle(
            "Show AI Landmarks",
            value=False,
            key="show_ai_landmarks_toggle",
        )

        st.session_state["developer_mode"] = developer_mode
        st.session_state["show_ai_landmarks"] = show_ai_landmarks

        st.markdown("---")

        # ── Technology Stack ───────────────────────────────────────────
        st.markdown("##### 🛠️ AI Stack")

        st.markdown("""
        <div style="font-size:0.78rem; color:#8faabf; line-height:2.0;">
            🧠 <b style="color:#76b900;">MediaPipe</b> Face Mesh 468pts<br>
            🎯 <b style="color:#76b900;">YOLOv8s</b> Object Detection<br>
            👁 <b style="color:#76b900;">OpenCV</b> Computer Vision<br>
            📊 <b style="color:#76b900;">Streamlit</b> Dashboard<br>
            🔥 <b style="color:#76b900;">PyTorch</b> Deep Learning<br>
            🔔 <b style="color:#00d4ff;">Thread-Safe</b> Alarm System
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Detection Parameters ───────────────────────────────────────
        with st.expander("⚙️ Detection Parameters", expanded=False):
            st.markdown("""
            <div style="font-size:0.74rem; color:#8faabf; line-height:1.9;">
                <b style="color:#00d4ff;">EAR Threshold:</b> 0.22<br>
                <b style="color:#00d4ff;">MAR Threshold:</b> 0.60<br>
                <b style="color:#00d4ff;">Yaw Threshold:</b> ±30°<br>
                <b style="color:#00d4ff;">Phone Confidence:</b> 0.30<br>
                <b style="color:#00d4ff;">Drowsy Trigger:</b> 2.0s<br>
                <b style="color:#00d4ff;">Phone Alarm:</b> 4.0s<br>
                <b style="color:#00d4ff;">Alarm Interval:</b> 0.5s<br>
                <b style="color:#00d4ff;">YOLO Model:</b> YOLOv8s
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Risk Score Guide ───────────────────────────────────────────
        with st.expander("📊 Risk Score Guide", expanded=False):
            st.markdown("""
            <div style="font-size:0.72rem; line-height:2.0;">
                <span style="color:#76b900;">■</span> <b>0-30</b> — Safe<br>
                <span style="color:#ff9800;">■</span> <b>31-60</b> — Moderate<br>
                <span style="color:#f44336;">■</span> <b>61-89</b> — High Risk<br>
                <span style="color:#ff0033;">■</span> <b>90-100</b> — Critical<br>
                <hr style="border-color:rgba(255,255,255,0.05); margin:0.4rem 0;">
                📱 Phone 2s: +40<br>
                📱 Phone 5s: +60<br>
                📱 Phone 8s: +90<br>
                😴 Eyes 2s: +50<br>
                😴 Eyes 4s: +70<br>
                😴 Eyes 6s: +90<br>
                😴 Eyes 8s: =100<br>
                👀 Distraction: +25<br>
                🥱 3 Yawns: +25<br>
                🥱 4+ Yawns: +40
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div style="text-align:center; padding:0.5rem 0; font-size:0.63rem; color:#4d6280;">
            Built for NVIDIA AI/ML Internship<br>
            © 2026 SafeDrive AI Team
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""

    render_sidebar()

    # Decide runtime operation mode
    operation_mode = st.session_state.get("operation_mode", "Live Monitoring")

    # ── Dataset Analysis Mode (MP4/AVI + image uploads) ─────────────────────────
    if operation_mode == "Dataset Analysis":
        import pandas as pd
        import tempfile
        import shutil
        import zipfile
        import io
        from PIL import Image

        st.markdown("""
        <div style="margin:0.5rem 0 1rem 0;">
            <h2 style="font-family:'Outfit',sans-serif;">📼 Dataset Analysis</h2>
            <p style="color:#4d6280; font-size:0.8rem;">Run SafeDrive AI on recorded media for validation and benchmarking.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.sidebar:
            pass

        # Disable continuous audio beeps in dataset mode by default.
        st.session_state["alerts_enabled"] = False

        col_left, col_right = st.columns([3, 2], gap="medium")
        with col_left:
            st.markdown("""
            <div class="section-label">🎥 Input Media</div>
            """, unsafe_allow_html=True)

            dataset_images = st.file_uploader(
                "Upload image(s) (.jpg/.jpeg/.png)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="dataset_images",
            )
            dataset_videos = st.file_uploader(
                "Upload video(s) (.mp4/.avi/.mov)",
                type=["mp4", "avi", "mov"],
                accept_multiple_files=True,
                key="dataset_videos",
            )
            dataset_zip = st.file_uploader(
                "Upload ZIP dataset (.zip)",
                type=["zip"],
                accept_multiple_files=False,
                key="dataset_zip",
            )
            dataset_folder_upload = st.file_uploader(
                "Upload folder dataset (optional)",
                type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
                accept_multiple_files=True,
                key="dataset_folder_upload",
            )
            dataset_folder_upload = dataset_folder_upload or []

            frame_limit_option = st.selectbox(
                "Max files to process",
                (50, 100, 200, 500, 1000, "ALL"),
                index=2,
            )
            frame_limit = None if frame_limit_option == "ALL" else frame_limit_option

            btn_col1, btn_col2 = st.columns([2, 1])
            with btn_col1:
                run_btn = st.button("▶ Run Dataset Analysis", type="primary", use_container_width=True)
            with btn_col2:
                if st.button("🗑️ Clear Results", use_container_width=True):
                    st.session_state["dataset_results"] = None
                    st.session_state["dataset_last_frame"] = None
                    st.rerun()

        with col_right:
            st.markdown("""
            <div class="section-label">📊 Results Summary</div>
            """, unsafe_allow_html=True)
            summary_placeholder = st.empty()

        if run_btn:
            processor = SafeDriveVideoProcessor()
            processor.alert_system.alerts_enabled = False
            processor.reset()

            debug_box = st.container()
            progress_bar = st.progress(0)
            status_text = st.empty()

            screenshots_before = processor.screenshots_saved
            start_time = time.time()

            def process_frame_bgr(frame_bgr):
                monitor_result = processor.driver_monitor.process_frame(frame_bgr)
                processed_frame = monitor_result.get("processed_frame", frame_bgr)
                face_bbox = monitor_result.get("face_bbox", None)

                phone_result = processor.phone_detector.detect(frame_bgr, face_bbox=face_bbox)
                processed_frame = processor.phone_detector.draw_detection(processed_frame, phone_result)

                risk_result = processor.risk_engine.update(monitor_result, phone_result)
                new_alerts = processor.alert_system.check_and_generate(monitor_result, phone_result, risk_result)

                for alert in new_alerts:
                    processor.risk_engine.record_alert(alert["type"], alert["message"])
                    processor.event_logger.log_event(
                        event_type=alert["type"],
                        severity=alert.get("severity", "medium"),
                        risk_score=risk_result["score"],
                        additional_info=alert["message"],
                    )

                processor._maybe_save_screenshot(processed_frame, monitor_result, phone_result, risk_result)
                return processed_frame, monitor_result, phone_result, risk_result

            last_frame = None
            processed_frames = 0
            per_file_rows = []
            max_risk_score_overall = 0

            img_ext = {".jpg", ".jpeg", ".png"}
            vid_ext = {".mp4", ".avi", ".mov"}

            def is_image_path(p):
                return os.path.splitext(p.lower())[1] in img_ext

            def is_video_path(p):
                return os.path.splitext(p.lower())[1] in vid_ext

            def collect_files_from_dir(root_dir):
                collected = []
                for dirpath, _, filenames in os.walk(root_dir):
                    for fn in filenames:
                        full = os.path.join(dirpath, fn)
                        if is_image_path(full) or is_video_path(full):
                            collected.append(full)
                collected.sort()
                return collected

            def grade_from_score(score):
                safety_grade_map = [(20, "A+"), (40, "A"), (60, "B"), (80, "C"), (100, "D")]
                for bound, grade in safety_grade_map:
                    if score <= bound:
                        return grade
                return "D"

            def status_from_score(score):
                grade = grade_from_score(score)
                if grade == "A+" or score <= 40:
                    return "Safe"
                if grade in ("B",):
                    return "Caution"
                if grade == "C":
                    return "Warning"
                return "Unsafe Driving"

            def process_image_file(path):
                nonlocal last_frame, processed_frames, max_risk_score_overall
                with open(path, "rb") as f:
                    img_bytes = f.read()
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (640, 480))
                last_frame, mon_res, ph_res, risk_res = process_frame_bgr(frame)
                processed_frames += 1

                score_now = int(round(risk_res["score"]))
                max_risk_score_overall = max(max_risk_score_overall, score_now)
                per_file_rows.append({
                    "File": os.path.basename(path),
                    "Driver Status": status_from_score(score_now),
                    "Risk Score": score_now,
                    "Safety Grade": grade_from_score(score_now),
                    "Face Detected": "Yes" if mon_res.get("face_detected") else "No",
                    "Drowsy": "Yes" if mon_res.get("drowsy") else "No",
                    "Phone Detected": "Yes" if ph_res.get("detected") else "No",
                    "Yawning": "Yes" if mon_res.get("yawning") else "No",
                    "Gaze Direction": mon_res.get("gaze_direction", "Forward"),
                    "Recommendation": "Safe Driving" if score_now <= 30 else ("Drowsiness Alert - Rest Required" if mon_res.get("drowsy") else "Keep Eyes on Road"),
                })

            def process_video_file(path):
                nonlocal last_frame, processed_frames, max_risk_score_overall
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    return
                while True:
                    if frame_limit and processed_frames >= frame_limit:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (640, 480))
                    last_frame, mon_res, ph_res, risk_res = process_frame_bgr(frame)
                    processed_frames += 1
                    score_now = int(round(risk_res["score"]))
                    max_risk_score_overall = max(max_risk_score_overall, score_now)
                cap.release()

            tmp_dir = None
            try:
                if dataset_zip is not None:
                    tmp_dir = tempfile.mkdtemp(prefix="safedrive_")
                    zip_path = os.path.join(tmp_dir, "dataset.zip")
                    with open(zip_path, "wb") as f:
                        f.write(dataset_zip.read())

                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(tmp_dir)

                    file_paths = collect_files_from_dir(tmp_dir)
                    total_files = len(file_paths)
                    if total_files == 0:
                        st.warning("No supported media files found inside the ZIP (.jpg/.jpeg/.png/.mp4/.avi/.mov).")

                    for idx, p in enumerate(file_paths):
                        if frame_limit and processed_frames >= frame_limit:
                            break
                        status_text.write(f"Processing ({idx+1}/{total_files}): `{os.path.basename(p)}`")
                        if total_files > 0:
                            progress_bar.progress(int(((idx + 1) / total_files) * 100))

                        if is_image_path(p):
                            process_image_file(p)
                        elif is_video_path(p):
                            process_video_file(p)
                else:
                    all_manual_files = []
                    if dataset_folder_upload:
                        all_manual_files.extend(sorted(dataset_folder_upload, key=lambda f: f.name))
                    if dataset_images:
                        all_manual_files.extend(sorted(dataset_images, key=lambda x: x.name))
                    if dataset_videos:
                        all_manual_files.extend(sorted(dataset_videos, key=lambda v: v.name))

                    total_files = len(all_manual_files)
                    if total_files == 0:
                        st.warning("Please upload a ZIP file, images, videos, or folder dataset.")

                    for idx, up in enumerate(all_manual_files):
                        if frame_limit and processed_frames >= frame_limit:
                            break
                        name_lower = up.name.lower()
                        status_text.write(f"Processing ({idx+1}/{total_files}): `{up.name}`")
                        if total_files > 0:
                            progress_bar.progress(int(((idx + 1) / total_files) * 100))

                        if os.path.splitext(name_lower)[1] in img_ext:
                            img_bytes = up.read()
                            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                            frame = cv2.resize(frame, (640, 480))
                            last_frame, mon_res, ph_res, risk_res = process_frame_bgr(frame)
                            processed_frames += 1
                            score_now = int(round(risk_res["score"]))
                            max_risk_score_overall = max(max_risk_score_overall, score_now)
                            per_file_rows.append({
                                "File": up.name,
                                "Driver Status": status_from_score(score_now),
                                "Risk Score": score_now,
                                "Safety Grade": grade_from_score(score_now),
                                "Face Detected": "Yes" if mon_res.get("face_detected") else "No",
                                "Drowsy": "Yes" if mon_res.get("drowsy") else "No",
                                "Phone Detected": "Yes" if ph_res.get("detected") else "No",
                                "Yawning": "Yes" if mon_res.get("yawning") else "No",
                                "Gaze Direction": mon_res.get("gaze_direction", "Forward"),
                                "Recommendation": "Safe Driving" if score_now <= 30 else ("Drowsiness Alert - Rest Required" if mon_res.get("drowsy") else "Keep Eyes on Road"),
                            })
                        elif os.path.splitext(name_lower)[1] in vid_ext:
                            suffix = os.path.splitext(name_lower)[1] or ".mp4"
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(up.read())
                                tmp_path = tmp.name
                            process_video_file(tmp_path)

            except Exception as e:
                debug_box.error(f"Dataset analysis error: {type(e).__name__}: {e}")
                raise
            finally:
                if tmp_dir and os.path.isdir(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            duration_s = time.time() - start_time
            session_stats = processor.risk_engine.get_session_stats()

            # Store results in st.session_state for persistence across reruns
            st.session_state["dataset_results"] = {
                "processed_frames": processed_frames,
                "duration_s": duration_s,
                "session_stats": session_stats,
                "per_file_rows": per_file_rows,
                "max_risk_score": max(max_risk_score_overall, session_stats.get("max_risk_score", 0)),
                "avg_risk_score": session_stats.get("avg_risk_score", 0),
                "drowsy_events": session_stats.get("drowsy_events", 0),
                "phone_events": session_stats.get("phone_events", 0),
                "distraction_events": session_stats.get("distraction_events", 0),
                "screenshots_saved": max(0, processor.screenshots_saved - screenshots_before),
                "risk_trend": list(session_stats.get("risk_trend", [])),
            }
            if last_frame is not None:
                st.session_state["dataset_last_frame"] = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)

            status_text.success("✅ Dataset analysis completed successfully!")
            st.rerun()

        # ── Persistent Dataset Results Display ─────────────────────────
        dataset_results = st.session_state.get("dataset_results")
        if dataset_results is not None:
            with col_left:
                if st.session_state.get("dataset_last_frame") is not None:
                    st.markdown("""<div style="font-size:0.84rem; font-weight:700; color:#e8f1ff; margin-bottom:0.4rem;">🖼️ Last Processed Frame (Overlay)</div>""", unsafe_allow_html=True)
                    st.image(st.session_state["dataset_last_frame"], channels="RGB", use_column_width=True)

            with col_right:
                with summary_placeholder.container():
                    st.markdown("""
                    <div class="metric-card" style="margin-bottom:0.8rem;">
                      <div class="section-label" style="margin-bottom:0.3rem;">✅ Session Summary</div>
                    </div>
                    """, unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Processed Files", dataset_results.get("processed_frames", 0))
                    m2.metric("Elapsed Time", f"{dataset_results.get('duration_s', 0):.2f}s")
                    m1.metric("Max Risk Score", dataset_results.get("max_risk_score", 0))
                    m2.metric("Average Risk", f"{dataset_results.get('avg_risk_score', 0):.1f}")
                    m1.metric("Drowsy Events", dataset_results.get("drowsy_events", 0))
                    m2.metric("Phone Detections", dataset_results.get("phone_events", 0))
                    m1.metric("Distraction Events", dataset_results.get("distraction_events", 0))
                    m2.metric("Screenshots Saved", dataset_results.get("screenshots_saved", 0))

            # Full Width Section: Interactive Risk Trend & Detailed Table
            st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

            # Risk Trend Chart
            risk_trend = dataset_results.get("risk_trend", [])
            if risk_trend:
                st.markdown("""<div class="section-label">📈 Risk Score Progression</div>""", unsafe_allow_html=True)
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=risk_trend,
                        mode="lines+markers",
                        name="Risk Score",
                        line=dict(color="#00ff9d", width=2),
                        marker=dict(size=4)
                    ))
                    fig.add_hline(y=70, line_dash="dash", line_color="#ff4757", annotation_text="Critical Threshold (70)")
                    fig.add_hline(y=40, line_dash="dot", line_color="#ffa502", annotation_text="Warning Threshold (40)")
                    fig.update_layout(
                        template="plotly_dark",
                        height=240,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title="Processed Frame Index",
                        yaxis_title="Risk Score (0-100)",
                        yaxis=dict(range=[0, 105]),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.line_chart(risk_trend)

            # Detailed Per-File Analysis Table
            rows = dataset_results.get("per_file_rows", [])
            if rows:
                st.markdown("""<div class="section-label">📋 Detailed File Analysis Results</div>""", unsafe_allow_html=True)
                df_results = pd.DataFrame(rows)
                st.dataframe(df_results, use_container_width=True)

                # Export CSV
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Analysis Results CSV",
                    data=csv_data,
                    file_name="safedrive_dataset_analysis.csv",
                    mime="text/csv",
                )

        return

    # ── Try to import streamlit-webrtc ──────────────────────────────

    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        webrtc_available = True
    except Exception as e:
        webrtc_available = False

    # Top header placeholder
    header_placeholder = st.empty()

    # Top layout: Video | Status Panel
    col_video, col_status = st.columns([3, 2], gap="medium")

    with col_video:
        # Live video label
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.5rem;">
            <span style="font-size:0.84rem; font-weight:700; color:#e8f1ff;">📹 Live Video Feed</span>
            <span class="live-indicator">● LIVE</span>
        </div>
        """, unsafe_allow_html=True)

        if webrtc_available:
            RTC_CONFIGURATION = {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
                ]
            }
            ctx = webrtc_streamer(
                key="safedrive-monitor",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=SafeDriveVideoProcessor,
                media_stream_constraints={
                    "video": True,
                    "audio": False,
                },
                rtc_configuration=RTC_CONFIGURATION,
                video_html_attrs={
                    "autoPlay": True,
                    "controls": True,
                    "style": {"width": "100%", "borderRadius": "12px"},
                },
                async_processing=False,
            )

    # Placeholders for right status panel, popups, and bottom section
    status_placeholder = col_status.empty()
    popup_placeholder = st.empty()
    bottom_placeholder = st.empty()

    if webrtc_available and ctx.state.playing:
        processor = ctx.video_processor
        if processor is None:
            for _ in range(25):
                time.sleep(0.1)
                processor = ctx.video_processor
                if processor is not None:
                    break

        if processor is not None:
            while ctx.state.playing:
                state = processor.get_state() if hasattr(processor, "get_state") else {}

                if hasattr(processor, "alerts_enabled"):
                    processor.alerts_enabled = st.session_state.get(
                        "alerts_enabled", True
                    )
                if hasattr(processor, "driver_monitor") and processor.driver_monitor is not None:
                    processor.driver_monitor.developer_mode = st.session_state.get(
                        "developer_mode", False
                    )
                    processor.driver_monitor.show_ai_landmarks = st.session_state.get(
                        "show_ai_landmarks", False
                    )

                # Update header with live stats
                with header_placeholder.container():
                    render_header(state)

                # Update status panel (right col)
                with status_placeholder.container():
                    render_right_panel(state)

                # Update bottom section
                with bottom_placeholder.container():
                    render_bottom_section(state)

                # Popup notifications
                popup_type = state.get("popup_type")
                popup_msg = state.get("popup_message", "")
                if popup_type:
                    popup_html = AlertSystem.get_popup_html(popup_type, popup_msg)
                    with popup_placeholder.container():
                        st.markdown(popup_html, unsafe_allow_html=True)
                else:
                    with popup_placeholder.container():
                        st.empty()

                # Browser audio alert
                new_alerts = state.get("new_alerts", [])
                if new_alerts and st.session_state.get("alerts_enabled", True):
                    sev = new_alerts[0].get("severity", "medium")
                    atype = new_alerts[0].get("type", "")
                    audio_html = AlertSystem.get_alert_audio_html(
                        "critical" if sev == "critical" else "warning",
                        alert_type_name=atype,
                    )
                    components.html(audio_html, height=0, width=0)

                time.sleep(0.1)
        else:
            with header_placeholder.container():
                render_header()
            with status_placeholder.container():
                render_right_panel_placeholder()
            with bottom_placeholder.container():
                render_bottom_section_placeholder()

    else:
        with header_placeholder.container():
            render_header()
        if not webrtc_available:
            st.error(
                "⚠️ `streamlit-webrtc` is required for live webcam access.\n"
                "Install it with:\n```\npip install streamlit-webrtc\n```"
            )
        with status_placeholder.container():
            render_right_panel_placeholder()
        with bottom_placeholder.container():
            render_bottom_section_placeholder()

    # ── Footer ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        <p>SafeDrive AI — Professional ADAS Driver Monitoring System</p>
        <p>Built with ❤️ for NVIDIA AI/ML Internship · OpenCV · MediaPipe · YOLOv8s · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PANEL RENDERING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def render_right_panel(state):
    """Render the right status panel with all live data."""

    # ── Driver Status ─────────────────────────────────────────────
    st.markdown("""
    <div class="section-label">🚗 Driver Status</div>
    """, unsafe_allow_html=True)
    render_status_badge(state.get("driver_status", "Waiting"))
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # ── Eyes Closed Countdown (only when drowsy) ──────────────────
    eyes_sec = state.get("eyes_closed_seconds", 0.0)
    if eyes_sec >= 2.0:
        render_eyes_closed_countdown(eyes_sec)

    # ── Phone Timer (only when phone present) ─────────────────────
    phone_sec = state.get("phone_seconds", 0.0)
    if phone_sec >= 1.0:
        render_phone_timer(phone_sec)

    # ── Risk Score Gauge ──────────────────────────────────────────
    st.markdown("""
    <div class="section-label">📊 Risk Score</div>
    """, unsafe_allow_html=True)
    render_risk_gauge(
        state.get("risk_score", 0),
        state.get("risk_level", "Safe"),
        state.get("risk_color", "#76b900")
    )
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # ── Detection Indicators ──────────────────────────────────────
    st.markdown("""
    <div class="section-label">🔍 Active Detections</div>
    """, unsafe_allow_html=True)

    is_drowsy = state.get("drowsy", False)
    render_detection_indicator(
        "DROWSINESS DETECTED" if is_drowsy else "Drowsiness",
        is_drowsy, "😴", "😊"
    )

    is_phone = state.get("phone_detected", False)
    render_detection_indicator(
        "PHONE DETECTED" if is_phone else "Phone Usage",
        is_phone, "📱", "✅"
    )

    is_dist = state.get("distracted", False)
    render_detection_indicator(
        f"DISTRACTED — {state.get('gaze_direction', '')}" if is_dist else "Distraction",
        is_dist, "👀", "🛣️"
    )

    is_yawn = state.get("yawning", False)
    render_detection_indicator(
        "YAWNING DETECTED" if is_yawn else "Yawning",
        is_yawn, "🥱", "😊",
        dot_color="amber" if is_yawn else "green"
    )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # ── Yawn Counter ──────────────────────────────────────────────
    st.markdown("""
    <div class="section-label">🥱 Fatigue Indicator</div>
    """, unsafe_allow_html=True)
    render_yawn_counter(state.get("yawn_count_60s", 0))

    # ── Face Metrics ──────────────────────────────────────────────
    ear = state.get("ear", 0)
    mar = state.get("mar", 0)
    yaw = state.get("yaw", 0)
    blink_rate = state.get("blink_rate", 0)

    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.65rem; color:#8faabf; text-transform:uppercase;
                    letter-spacing:0.1em; font-weight:700; margin-bottom:0.75rem;">
            👁 Facial Biometrics
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;">
            <div style="text-align:center; background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.5rem;">
                <div style="font-family:'JetBrains Mono'; font-size:1.1rem;
                            color:{'#f44336' if ear < 0.22 and ear > 0 else '#76b900'};
                            font-weight:700;">{ear:.2f}</div>
                <div style="font-size:0.6rem; color:#4d6280; margin-top:2px;">EAR · Eye</div>
            </div>
            <div style="text-align:center; background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.5rem;">
                <div style="font-family:'JetBrains Mono'; font-size:1.1rem;
                            color:{'#ff9800' if mar > 0.6 else '#76b900'};
                            font-weight:700;">{mar:.2f}</div>
                <div style="font-size:0.6rem; color:#4d6280; margin-top:2px;">MAR · Mouth</div>
            </div>
            <div style="text-align:center; background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.5rem;">
                <div style="font-family:'JetBrains Mono'; font-size:1.1rem;
                            color:{'#ff9800' if abs(yaw) > 30 else '#76b900'};
                            font-weight:700;">{yaw:.1f}°</div>
                <div style="font-size:0.6rem; color:#4d6280; margin-top:2px;">Yaw · Head</div>
            </div>
            <div style="text-align:center; background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.5rem;">
                <div style="font-family:'JetBrains Mono'; font-size:1.1rem;
                            color:{'#ff9800' if blink_rate > 25 else '#76b900'};
                            font-weight:700;">{blink_rate:.0f}</div>
                <div style="font-size:0.6rem; color:#4d6280; margin-top:2px;">Blinks/min</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_right_panel_placeholder():
    """Render the right panel in idle/waiting state."""
    st.markdown("""
    <div class="section-label">🚗 Driver Status</div>
    """, unsafe_allow_html=True)
    render_status_badge("Waiting")
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">📊 Risk Score</div>
    """, unsafe_allow_html=True)
    render_risk_gauge(0, "Safe", "#76b900")
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="text-align:center; padding:2.5rem 2rem;">
        <div style="font-size:2.5rem; margin-bottom:0.75rem; animation:float 3s ease-in-out infinite;">📹</div>
        <div style="color:#8faabf; font-size:0.9rem; font-weight:600; margin-bottom:0.4rem;">
            Monitoring Inactive
        </div>
        <div style="color:#4d6280; font-size:0.78rem; line-height:1.6;">
            Click <b style="color:#76b900;">START</b> on the webcam feed<br>
            to begin real-time monitoring
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_bottom_section(state):
    """Render the bottom section: stats, event history, risk trend."""
    col_stats, col_events = st.columns([1, 2], gap="medium")

    with col_stats:
        session = state.get("session_stats", {})
        duration = session.get("duration_formatted", "00:00:00")
        total_alerts = session.get("total_alerts", 0)
        max_risk = session.get("max_risk_score", 0)
        avg_risk = session.get("avg_risk_score", 0)
        drowsy_events = session.get("drowsy_events", 0)
        phone_events = session.get("phone_events", 0)
        distraction_events = session.get("distraction_events", 0)
        screenshots = state.get("screenshots_saved", 0)
        fps = state.get("fps", 0.0)

        st.markdown("""
        <div class="section-label">📊 Session Statistics</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{duration}</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color:{'#f44336' if total_alerts > 5 else '#76b900'};">
                    {total_alerts}</div>
                <div class="stat-label">Alerts</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color:{'#f44336' if max_risk > 60 else '#ff9800' if max_risk > 30 else '#76b900'};">
                    {max_risk}</div>
                <div class="stat-label">Max Risk</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{avg_risk:.0f}</div>
                <div class="stat-label">Avg Risk</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color:{'#ff9800' if drowsy_events > 0 else '#76b900'};">
                    {drowsy_events}</div>
                <div class="stat-label">Drowsy Events</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color:{'#ff6600' if phone_events > 0 else '#76b900'};">
                    {phone_events}</div>
                <div class="stat-label">Phone Events</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{distraction_events}</div>
                <div class="stat-label">Distractions</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color:#00d4ff;">{screenshots}</div>
                <div class="stat-label">Screenshots</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Risk Trend ─────────────────────────────────────────────
        risk_trend = state.get("risk_trend", [])
        if risk_trend:
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            render_risk_trend(risk_trend)

        # ── Risk Breakdown ─────────────────────────────────────────
        contributions = state.get("contributions", {})
        if contributions:
            st.markdown("""
            <div style="margin-top:0.75rem;">
            <div class="section-label">⚡ Risk Breakdown</div>
            """, unsafe_allow_html=True)
            for source, points in contributions.items():
                label = source.replace("_", " ").title()
                pct = min(100, points)
                bar_color = "#f44336" if points >= 70 else "#ff9800" if points >= 40 else "#76b900"
                st.markdown(f"""
                <div style="margin-bottom:0.4rem;">
                    <div style="display:flex; justify-content:space-between;
                                font-size:0.72rem; color:#8faabf; margin-bottom:3px;">
                        <span>{label}</span>
                        <span style="color:{bar_color}; font-weight:700;">+{points}</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.04); border-radius:4px; height:5px; overflow:hidden;">
                        <div style="width:{pct}%; height:100%; background:{bar_color}; border-radius:4px;
                                    transition:width 0.5s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_events:
        st.markdown("""
        <div class="section-label">📋 Event History</div>
        """, unsafe_allow_html=True)

        session = state.get("session_stats", {})
        alert_history = session.get("alert_history", [])

        if alert_history:
            st.markdown('<div class="event-history">', unsafe_allow_html=True)
            for alert in reversed(alert_history[-20:]):
                alert_time = time.strftime(
                    "%H:%M:%S", time.localtime(alert.get("timestamp", 0))
                )
                alert_type = alert.get("type", "INFO")
                message = alert.get("message", "")
                severity = alert.get("severity", "medium")

                if severity == "critical":
                    css_cls = ""
                elif severity == "high":
                    css_cls = "warning"
                else:
                    css_cls = "low"

                st.markdown(f"""
                <div class="alert-notification {css_cls}">
                    <span class="alert-time">{alert_time}</span>
                    <span class="alert-message">{message}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="event-history" style="text-align:center; padding:2.5rem 2rem;">
                <div style="font-size:2rem; margin-bottom:0.75rem; opacity:0.4;">📋</div>
                <div style="color:#4d6280; font-size:0.85rem;">
                    No alerts yet. Events will appear here when detected.
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_bottom_section_placeholder():
    """Render the bottom section in idle/waiting state."""
    col_stats, col_events = st.columns([1, 2], gap="medium")

    with col_stats:
        st.markdown("""
        <div class="section-label">📊 Session Statistics</div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-item"><div class="stat-value">00:00:00</div><div class="stat-label">Duration</div></div>
            <div class="stat-item"><div class="stat-value">0</div><div class="stat-label">Alerts</div></div>
            <div class="stat-item"><div class="stat-value">0</div><div class="stat-label">Max Risk</div></div>
            <div class="stat-item"><div class="stat-value">0</div><div class="stat-label">Avg Risk</div></div>
            <div class="stat-item"><div class="stat-value">0</div><div class="stat-label">Drowsy Events</div></div>
            <div class="stat-item"><div class="stat-value">0</div><div class="stat-label">Phone Events</div></div>
            <div class="stat-item"><div class="stat-value">0</div><div class="stat-label">Distractions</div></div>
            <div class="stat-item"><div class="stat-value" style="color:#00d4ff;">0</div><div class="stat-label">Screenshots</div></div>
        </div>
        """, unsafe_allow_html=True)

    with col_events:
        st.markdown("""
        <div class="section-label">📋 Event History</div>
        <div class="event-history" style="text-align:center; padding:2.5rem 2rem;">
            <div style="font-size:2rem; margin-bottom:0.75rem; opacity:0.4;">📋</div>
            <div style="color:#4d6280; font-size:0.85rem;">
                Start monitoring to see events here.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
