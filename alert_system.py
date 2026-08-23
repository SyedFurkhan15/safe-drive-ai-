"""
SafeDrive AI - Alert System Module
====================================
Manages alert generation, cooldown logic, continuous alarm threading,
and audio alert creation for the driver monitoring system.

Key upgrades:
- Continuous alarm thread: beeps every 0.5s while condition is active
- Drowsiness alarm: starts when eyes closed >2s, stops when eyes open
- Phone alarm: starts when phone detected >4s, stops when phone gone
- Yawn-count-aware alerts (popup, warning, fatigue levels)
- Popup HTML generator for animated overlay notifications
- Screenshot trigger signals

Author: SafeDrive AI Team
"""

import time
import numpy as np
import os
import struct
import wave
import threading


class AlertSystem:
    """
    Alert management system with cooldown logic and continuous alarm threading.

    Alert Types:
    - DROWSINESS: Eyes closed continuously — escalating beep alarm
    - PHONE: Mobile phone detected — escalating beep alarm
    - DISTRACTION: Looking away from road
    - HIGH_RISK: Risk score exceeds 60
    - YAWNING_2: Second yawn in 60s — popup only
    - YAWNING_3: Third yawn — warning sound
    - YAWNING_4: Fourth+ yawn — fatigue warning
    """

    ALERT_TYPES = {
        "DROWSINESS": {
            "message": "⚠ CRITICAL: Wake up! Driver appears drowsy.",
            "icon": "🔴",
            "severity": "critical",
            "cooldown": 0.5,
        },
        "PHONE": {
            "message": "📱 WARNING: Please put your phone away.",
            "icon": "🟠",
            "severity": "high",
            "cooldown": 0.5,
        },
        "DISTRACTION": {
            "message": "👀 WARNING: Driver Distracted — Eyes Off Road!",
            "icon": "🟡",
            "severity": "medium",
            "cooldown": 0.5,
        },
        "HIGH_RISK": {
            "message": "🚨 DANGER: Immediate attention required!",
            "icon": "🔴",
            "severity": "critical",
            "cooldown": 0.5,
        },
        "YAWNING_2": {
            "message": "😴 You seem tired. Consider taking a short break.",
            "icon": "🟡",
            "severity": "low",
            "cooldown": 30.0,
        },
        "YAWNING_3": {
            "message": "😴 WARNING: Signs of fatigue detected. Take a break soon!",
            "icon": "🟠",
            "severity": "medium",
            "cooldown": 20.0,
        },
        "YAWNING_4": {
            "message": "🚨 FATIGUE WARNING: Multiple yawns detected. Stop safely!",
            "icon": "🔴",
            "severity": "high",
            "cooldown": 15.0,
        },
        "PHONE_EMERGENCY": {
            "message": "🚨 EMERGENCY: Continuous phone usage — Immediate danger!",
            "icon": "🔴",
            "severity": "critical",
            "cooldown": 0.5,
        },
    }

    def __init__(self, alerts_enabled=True):
        """Initialize the alert system with continuous alarm support."""
        self.alerts_enabled = alerts_enabled
        self.last_alert_times = {k: 0 for k in self.ALERT_TYPES}
        self.alert_log = []
        self.total_alerts = 0

        # ── Continuous Alarm Thread ───────────────────────────────────
        self._alarm_lock = threading.Lock()
        self._alarm_active = False
        self._alarm_reason = None
        self._alarm_stop_event = threading.Event()
        self._alarm_thread = None
        self._alarm_beep_interval = 0.5   # seconds between beeps

    # ── Continuous Alarm Controls ─────────────────────────────────────

    def start_continuous_alarm(self, reason="drowsy"):
        """
        Start the continuous beeping alarm in a background thread.
        Beeps every 0.5 seconds until stop_continuous_alarm() is called.

        Args:
            reason: 'drowsy' (1500 Hz) | 'phone' (1200 Hz) | 'critical' (1800 Hz)
        """
        with self._alarm_lock:
            if self._alarm_active and self._alarm_reason == reason:
                return  # Already running for same reason
            self._alarm_active = True
            self._alarm_reason = reason
            self._alarm_stop_event.clear()

        # Start/restart thread
        if self._alarm_thread is not None and self._alarm_thread.is_alive():
            return  # Thread is already looping

        self._alarm_thread = threading.Thread(
            target=self._alarm_loop,
            daemon=True,
            name="SafeDrive-ContinuousAlarm"
        )
        self._alarm_thread.start()

    def stop_continuous_alarm(self):
        """Stop the continuous beeping alarm immediately."""
        with self._alarm_lock:
            self._alarm_active = False
            self._alarm_reason = None
        self._alarm_stop_event.set()

    def _alarm_loop(self):
        """Background thread: beeps repeatedly while alarm is active."""
        reason_freq = {
            "drowsy": 1500,
            "phone": 1200,
            "critical": 1800,
        }

        while True:
            with self._alarm_lock:
                active = self._alarm_active
                reason = self._alarm_reason

            if not active:
                break

            freq = reason_freq.get(reason, 1000)
            try:
                import winsound
                winsound.Beep(freq, 200)
            except Exception:
                pass

            # Wait 0.5s between beeps (or stop if event is set)
            self._alarm_stop_event.wait(timeout=self._alarm_beep_interval)
            if self._alarm_stop_event.is_set():
                break

    # ── Alert Generation ──────────────────────────────────────────────

    def check_and_generate(self, monitor_data, phone_data, risk_data):
        """Check all conditions and generate pending alerts.

        Uses continuous repeating alarm logic (0.5s cadence) with strict
        priority and immediate stop behavior when conditions clear.
        """
        if not self.alerts_enabled:
            if self._alarm_active:
                self.stop_continuous_alarm()
            return []

        current_time = time.time()
        new_alerts = []

        eyes_closed_sec = monitor_data.get("eyes_closed_duration", 0.0)
        phone_sec = phone_data.get("phone_continuous_duration", 0.0)
        yawn_count = monitor_data.get("yawn_count_60s", 0)
        risk_score = risk_data.get("score", 0)


        # ── Continuous repeating alarm state machine (priority based) ──
        # Priority order (only show highest priority state):
        # 1) CRITICAL (risk) / EMERGENCY
        # 2) HIGH RISK (risk)
        # 3) PHONE
        # 4) DROWSINESS
        # 5) DISTRACTION
        # 6) ATTENTIVE

        eyes_closed_active = monitor_data.get("drowsy", False) and eyes_closed_sec >= 2.0
        phone_active = phone_data.get("detected", False) and phone_sec >= 2.0

        # Distraction persistent away timer.
        now = current_time
        if not hasattr(self, "_distraction_start_time"):
            self._distraction_start_time = None
        if monitor_data.get("distracted", False):
            if self._distraction_start_time is None:
                self._distraction_start_time = now
        else:
            self._distraction_start_time = None
        distraction_active = ((now - self._distraction_start_time) if self._distraction_start_time is not None else 0.0) >= 2.0

        # HIGH RISK repeating emergency alarm:
        # Trigger when risk_score > 80, stop when risk_score < 60.
        if not hasattr(self, "_high_risk_emergency_latched"):
            self._high_risk_emergency_latched = False
        if self._high_risk_emergency_latched:
            if risk_score < 60:
                self._high_risk_emergency_latched = False
        else:
            if risk_score > 80:
                self._high_risk_emergency_latched = True
        high_risk_emergency_active = self._high_risk_emergency_latched

        # EMERGENCY MODE
        emergency_mode = phone_data.get("detected", False) or eyes_closed_active or risk_score > 90

        # Choose active alert state (priority).
        active_state = "ATTENTIVE"
        if high_risk_emergency_active or emergency_mode:
            active_state = "CRITICAL"
        elif risk_score >= 60:
            active_state = "HIGH_RISK"
        elif phone_active:
            active_state = "PHONE"
        elif eyes_closed_active:
            active_state = "DROWSINESS"
        elif distraction_active:
            active_state = "DISTRACTION"

        # Continuous alarm (repeat every 0.5s while active_state indicates condition)
        if active_state == "CRITICAL":
            self.start_continuous_alarm("critical")
        elif active_state == "DROWSINESS":
            self.start_continuous_alarm("drowsy")
        elif active_state == "DISTRACTION":
            self.start_continuous_alarm("drowsy")
        elif active_state == "PHONE":
            self.start_continuous_alarm("phone")
        elif active_state == "HIGH_RISK":
            # treat high-risk as critical tone but still obey stop threshold via latching
            self.start_continuous_alarm("critical")
        else:
            self.stop_continuous_alarm()

        # Discrete events (ensure they are not silently ignored):
        # We log at least once per activation using cooldowns.
        # Additionally, continuous beeping handles repeated audible alerts.

        # CRITICAL / HIGH RISK
        if active_state in ("CRITICAL", "HIGH_RISK"):
            alert = self._try_trigger("HIGH_RISK", current_time)
            if alert:
                alert["type"] = "HIGH_RISK" if active_state == "HIGH_RISK" else "EMERGENCY_MODE"
                alert["severity"] = "critical" if active_state == "CRITICAL" else "critical"
                alert["icon"] = "🆘" if active_state == "CRITICAL" else "🔴"
                alert["message"] = "🚨 EMERGENCY MODE ACTIVATED" if active_state == "CRITICAL" else f"🚨 DANGER: Risk Score {risk_score}/100 — Immediate Attention Required!"
                new_alerts.append(alert)

        # PHONE
        elif active_state == "PHONE":
            # PHONE_EMERGENCY when >=8s else PHONE
            if phone_sec >= 8.0:
                alert = self._try_trigger("PHONE_EMERGENCY", current_time)
                if alert:
                    alert["message"] = f"🚨 EMERGENCY: Phone used {phone_sec:.0f}s — Immediate danger!"
                    new_alerts.append(alert)
            else:
                alert = self._try_trigger("PHONE", current_time)
                if alert:
                    alert["message"] = "📱 Phone usage detected — please keep eyes on the road."
                    new_alerts.append(alert)

        # DROWSINESS
        elif active_state == "DROWSINESS":
            alert = self._try_trigger("DROWSINESS", current_time)
            if alert:
                alert["message"] = f"⚠ Wake up! Driver appears drowsy. Eyes closed {eyes_closed_sec:.1f}s."
                new_alerts.append(alert)

        # DISTRACTION
        elif active_state == "DISTRACTION":
            gaze = monitor_data.get("gaze_direction", "Away")
            alert = self._try_trigger("DISTRACTION", current_time)
            if alert:
                alert["message"] = f"👀 WARNING: Driver Distracted — {gaze}!"
                new_alerts.append(alert)

        # Yawning (may coexist but we keep only highest priority state for alarms; still log yawning events)
        if yawn_count >= 4:
            alert = self._try_trigger("YAWNING_4", current_time)
            if alert:
                alert["message"] = f"🚨 FATIGUE: {yawn_count} yawns detected! Stop safely and rest."
                new_alerts.append(alert)
                self._single_beep(900, 400)
        elif yawn_count == 3:
            alert = self._try_trigger("YAWNING_3", current_time)
            if alert:
                new_alerts.append(alert)
                self._single_beep(800, 300)
        elif yawn_count == 2:
            alert = self._try_trigger("YAWNING_2", current_time)
            if alert:
                new_alerts.append(alert)

        return new_alerts



    def _try_trigger(self, alert_type, current_time):
        """Attempt to trigger an alert, respecting cooldown."""
        alert_def = self.ALERT_TYPES.get(alert_type)
        if not alert_def:
            return None

        last_time = self.last_alert_times.get(alert_type, 0)
        cooldown = alert_def["cooldown"]

        if current_time - last_time < cooldown:
            return None

        self.last_alert_times[alert_type] = current_time
        self.total_alerts += 1

        alert = {
            "type": alert_type,
            "message": alert_def["message"],
            "icon": alert_def["icon"],
            "severity": alert_def["severity"],
            "timestamp": current_time,
            "time_str": time.strftime("%H:%M:%S", time.localtime(current_time)),
        }

        self.alert_log.append(alert)
        return alert

    def _single_beep(self, freq, duration_ms):
        """Fire a single non-blocking beep in a daemon thread."""
        def _beep():
            try:
                import winsound
                winsound.Beep(freq, duration_ms)
            except Exception:
                pass

        t = threading.Thread(target=_beep, daemon=True)
        t.start()

    def get_alert_log(self, max_entries=20):
        """Get the most recent alerts from the log."""
        return list(reversed(self.alert_log[-max_entries:]))

    def enable(self):
        """Enable alert generation."""
        self.alerts_enabled = True

    def disable(self):
        """Disable alert generation (stops alarm too)."""
        self.alerts_enabled = False
        self.stop_continuous_alarm()

    def reset(self):
        """Reset all alert state for a new session."""
        self.stop_continuous_alarm()
        for alert_type in self.ALERT_TYPES:
            self.last_alert_times[alert_type] = 0
        self.alert_log.clear()
        self.total_alerts = 0

    # ── Static Utilities ──────────────────────────────────────────────

    @staticmethod
    def generate_alert_sound(filepath="assets/alert_sound.wav",
                             frequency=880, duration_ms=500,
                             sample_rate=44100):
        """
        Generate an alert beep WAV file using numpy sine waves.
        Creates a triple-beep pattern, saved as 16-bit PCM WAV.
        """
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True
        )

        num_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)

        wave_data = np.sin(2 * np.pi * frequency * t)

        fade_in = int(sample_rate * 0.02)
        fade_out = int(sample_rate * 0.05)
        envelope = np.ones(num_samples)
        envelope[:fade_in] = np.linspace(0, 1, fade_in)
        envelope[-fade_out:] = np.linspace(1, 0, fade_out)

        wave_data *= envelope
        wave_data += 0.3 * np.sin(2 * np.pi * frequency * 2 * t) * envelope

        wave_data = wave_data / np.max(np.abs(wave_data))
        wave_data = (wave_data * 32767).astype(np.int16)

        silence = np.zeros(int(sample_rate * 0.15), dtype=np.int16)
        full_pattern = np.concatenate([wave_data, silence, wave_data, silence, wave_data])

        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(full_pattern.tobytes())

        return filepath

    @staticmethod
    def get_alert_audio_html(alert_type="critical", alert_type_name=""):
        """
        Generate HTML/JS for playing an alert sound in the browser
        using the Web Audio API (no audio file needed).

        Must be rendered with st.components.v1.html(), NOT
        st.markdown(unsafe_allow_html=True) -- the latter inserts HTML via
        innerHTML, and browsers do not execute <script> tags inserted that
        way, so the beep would silently never play.

        Args:
            alert_type: 'critical' for the urgent tone, anything else for
                        the standard warning tone (kept for compatibility).
            alert_type_name: the ALERT_TYPES key (e.g. 'DROWSINESS', 'PHONE',
                        'DISTRACTION', 'YAWNING_2') -- used to pick a
                        distinct pitch/pattern so each condition is
                        recognizable by ear, not just "a beep happened".
        """
        # Distinct tone per condition -- mirrors the frequencies used by
        # the server-side continuous alarm (drowsy=1500Hz, phone=1200Hz)
        # so the browser beep and the local alarm feel like the same system.
        tone_presets = {
            "DROWSINESS": (900, 3),     # low urgent triple-beep
            "PHONE": (700, 3),
            "PHONE_EMERGENCY": (900, 3),
            "DISTRACTION": (550, 2),    # lighter double-chime
            "YAWNING_2": (500, 1),
            "YAWNING_3": (600, 2),
            "YAWNING_4": (750, 3),
            "HIGH_RISK": (900, 3),
        }
        freq, beep_count = tone_presets.get(
            alert_type_name,
            (880 if alert_type == "critical" else 660, 3)
        )

        beep_calls = "\n".join(
            f"beep({freq}, 0.18, {i * 230});" for i in range(beep_count)
        )

        return f"""
        <script>
        (function() {{
            try {{
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                // Browsers suspend new AudioContexts until a user gesture
                // occurs on the page; resume() is a no-op if already running
                // but is required for the beep to be audible after the very
                // first click/tap on the Streamlit app.
                if (audioCtx.state === 'suspended') {{
                    audioCtx.resume();
                }}
                function beep(freq, duration, delay) {{
                    setTimeout(function() {{
                        const osc = audioCtx.createOscillator();
                        const gain = audioCtx.createGain();
                        osc.connect(gain);
                        gain.connect(audioCtx.destination);
                        osc.frequency.value = freq;
                        osc.type = 'sine';
                        gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
                        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
                        osc.start(audioCtx.currentTime);
                        osc.stop(audioCtx.currentTime + duration);
                    }}, delay);
                }}
                {beep_calls}
            }} catch(e) {{}}
        }})();
        </script>
        """

    @staticmethod
    def get_popup_html(popup_type, message="", duration=4):
        """
        Generate a large animated popup notification overlay HTML.

        Args:
            popup_type: 'drowsy' | 'phone' | 'yawn' | 'critical'
            message: Custom message override (otherwise uses default)
            duration: Seconds the popup stays visible (CSS animation)

        Returns:
            str: HTML string to inject via st.markdown()
        """
        defaults = {
            "drowsy": {
                "title": "😴 DRIVER DROWSY!",
                "msg": "Wake up! Keep your eyes on the road.",
                "color": "#ff2200",
                "bg": "rgba(255, 34, 0, 0.15)",
                "border": "#ff2200",
                "icon": "⚠️",
            },
            "phone": {
                "title": "📱 PHONE DETECTED",
                "msg": "Please put your phone away immediately.",
                "color": "#ff6600",
                "bg": "rgba(255, 102, 0, 0.15)",
                "border": "#ff6600",
                "icon": "🚫",
            },
            "yawn": {
                "title": "🥱 Feeling Tired?",
                "msg": "You seem tired. Consider taking a short break.",
                "color": "#ffaa00",
                "bg": "rgba(255, 170, 0, 0.12)",
                "border": "#ffaa00",
                "icon": "☕",
            },
            "critical": {
                "title": "🚨 CRITICAL ALERT",
                "msg": "Immediate attention required! Pull over safely.",
                "color": "#ff0000",
                "bg": "rgba(255, 0, 0, 0.2)",
                "border": "#ff0000",
                "icon": "🆘",
            },
        }

        cfg = defaults.get(popup_type, defaults["critical"])
        display_msg = message if message else cfg["msg"]

        return f"""
        <style>
        @keyframes popupSlideIn {{
            from {{ opacity: 0; transform: translateY(-30px) scale(0.95); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        @keyframes popupFadeOut {{
            0%, 80% {{ opacity: 1; }}
            100% {{ opacity: 0; pointer-events: none; }}
        }}
        .safedrive-popup {{
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 999999;
            min-width: 380px;
            max-width: 520px;
            background: {cfg["bg"]};
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 2px solid {cfg["border"]};
            border-radius: 16px;
            padding: 20px 28px;
            box-shadow: 0 8px 40px rgba(0,0,0,0.6), 0 0 60px {cfg["color"]}44;
            animation: popupSlideIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275),
                       popupFadeOut {duration}s ease-in-out forwards;
            font-family: 'Inter', sans-serif;
        }}
        .popup-icon {{ font-size: 2.5rem; margin-bottom: 6px; display: block; text-align: center; }}
        .popup-title {{
            font-size: 1.3rem;
            font-weight: 800;
            color: {cfg["color"]};
            text-align: center;
            margin: 0 0 8px 0;
            letter-spacing: -0.01em;
        }}
        .popup-msg {{
            font-size: 0.95rem;
            color: #f0f4f8;
            text-align: center;
            line-height: 1.5;
            margin: 0;
        }}
        </style>
        <div class="safedrive-popup">
            <span class="popup-icon">{cfg["icon"]}</span>
            <p class="popup-title">{cfg["title"]}</p>
            <p class="popup-msg">{display_msg}</p>
        </div>
        """
