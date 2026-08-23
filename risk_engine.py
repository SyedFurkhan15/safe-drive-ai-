"""
SafeDrive AI - Risk Score Engine
=================================
Calculates a composite driver risk score (0-100) by aggregating
signals from drowsiness detection, phone detection, distraction
detection, and yawn count analysis.

New scoring system with time-gated escalation:
- Phone detected: +40 base, escalating to +60 (>5s) and +90 (>8s)
- Eyes closed: +50 base, escalating to +70 (>4s), +90 (>6s), 100 (>8s)
- Distraction: +25
- Yawning: Progressive based on count in 60s window

Multiple simultaneous events apply cumulative scoring (capped at 100).

Author: SafeDrive AI Team
"""

import time
from collections import deque


class RiskEngine:
    """
    Computes a real-time driver risk score (0-100) from multiple signals.

    Risk escalates based on duration and severity of detected events.
    Multiple simultaneous events combine cumulatively.
    Score decays smoothly when conditions improve.
    """

    # ── Risk Level Boundaries ─────────────────────────────────────────
    SAFE_THRESHOLD = 30
    MODERATE_THRESHOLD = 60

    # ── Decay Settings ────────────────────────────────────────────────
    DECAY_RATE = 4.0       # Points per second
    MIN_SCORE = 0
    MAX_SCORE = 100

    def __init__(self):
        """Initialize the risk engine with clean state."""
        self.current_score = 0.0
        self.target_score = 0.0
        self.last_update_time = time.time()

        # ── Session Statistics ────────────────────────────────────────
        self.session_start_time = time.time()
        self.total_alerts = 0
        self.max_risk_score = 0
        self.risk_score_history = deque(maxlen=600)  # ~10 min at 1/sec
        self.alert_history = []

        # Event counters
        self.drowsy_events = 0
        self.phone_events = 0
        self.distraction_events = 0

        # State change tracking
        self._prev_drowsy = False
        self._prev_phone = False
        self._prev_distracted = False

    def _compute_drowsiness_score(self, monitor_data):
        """
        Compute time-gated drowsiness risk score.

        2s closed → +50
        4s closed → +70
        6s closed → +90
        8s closed → 100 (critical emergency)
        """
        if not monitor_data.get("drowsy", False):
            return 0, {}

        duration = monitor_data.get("eyes_closed_duration", 0.0)

        if duration >= 8.0:
            score = 100
            label = "eyes_closed_8s_critical"
        elif duration >= 6.0:
            score = 90
            label = "eyes_closed_6s"
        elif duration >= 4.0:
            score = 70
            label = "eyes_closed_4s"
        else:
            score = 50
            label = "eyes_closed_2s"

        return score, {label: score}

    def _compute_phone_score(self, phone_data):
        """
        Compute time-gated phone detection risk score.

        Detected: +40
        >5s continuous: +60
        >8s continuous: +90
        """
        if not phone_data.get("detected", False):
            return 0, {}

        duration = phone_data.get("phone_continuous_duration", 0.0)

        if duration >= 8.0:
            score = 90
            label = "phone_8s_emergency"
        elif duration >= 5.0:
            score = 60
            label = "phone_5s_warning"
        else:
            score = 40
            label = "phone_detected"

        context = {}
        if phone_data.get("near_face"):
            context["phone_near_face"] = 5
            score = min(100, score + 5)
        if phone_data.get("near_ear"):
            context["phone_near_ear"] = 5
            score = min(100, score + 5)

        context[label] = score
        return score, context

    def _compute_yawn_score(self, monitor_data):
        """
        Compute yawn-count-based risk score.

        1 yawn/60s: 0 (no action)
        2 yawns/60s: 0 (popup only — handled by alert system)
        3 yawns/60s: +25
        4+ yawns/60s: +40
        """
        yawn_count = monitor_data.get("yawn_count_60s", 0)

        if yawn_count >= 4:
            return 40, {"yawning_4plus": 40}
        elif yawn_count == 3:
            return 25, {"yawning_3": 25}
        elif yawn_count <= 1:
            return 0, {}
        else:
            # 2 yawns → only popup alert, no risk score penalty
            return 0, {}

    def update(self, monitor_data, phone_data):
        """
        Update the risk score based on current detection results.

        Computes target score from all active risk signals (cumulative),
        then smoothly interpolates current score toward target.

        Args:
            monitor_data: Dict from DriverMonitor.process_frame()
            phone_data: Dict from PhoneDetector.detect()

        Returns:
            dict: Risk assessment containing:
                - score (int): Current risk score 0-100
                - level (str): Risk level label
                - level_color (str): CSS color
                - contributions (dict): Score breakdown
                - status (str): Final driver status
                - eyes_closed_seconds (float): For UI countdown
                - phone_seconds (float): For UI phone timer
                - yawn_count_60s (int): Recent yawn count
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # ── Calculate Individual Scores ───────────────────────────────
        drowsy_score, drowsy_contrib = self._compute_drowsiness_score(monitor_data)
        phone_score, phone_contrib = self._compute_phone_score(phone_data)
        yawn_score, yawn_contrib = self._compute_yawn_score(monitor_data)

        # ── Distraction ───────────────────────────────────────────────
        distraction_score = 0
        distraction_contrib = {}
        if monitor_data.get("distracted", False):
            distraction_score = 25
            distraction_contrib = {"distraction": 25}

        # ── High blink rate ───────────────────────────────────────────
        blink_score = 0
        blink_contrib = {}
        blink_rate = monitor_data.get("blink_rate", 15)
        if blink_rate > 25:
            blink_score = 10
            blink_contrib = {"high_blink_rate": 10}

        # ── Cumulative Score ──────────────────────────────────────────
        # All signals combine, capped at 100
        target = drowsy_score + phone_score + yawn_score + distraction_score + blink_score
        target = min(self.MAX_SCORE, max(self.MIN_SCORE, target))

        contributions = {}
        contributions.update(drowsy_contrib)
        contributions.update(phone_contrib)
        contributions.update(yawn_contrib)
        contributions.update(distraction_contrib)
        contributions.update(blink_contrib)

        self.target_score = target

        # ── Smooth Score Update ───────────────────────────────────────
        if target > self.current_score:
            # Increase quickly
            delta = max(target - self.current_score, 15) * dt * 4
            self.current_score = min(target, self.current_score + delta)
        elif target < self.current_score:
            # Decay gradually
            self.current_score = max(target, self.current_score - self.DECAY_RATE * dt)

        # Emergency override — instant jump to 100 for 8s events
        eyes_closed_sec = monitor_data.get("eyes_closed_duration", 0.0)
        phone_sec = phone_data.get("phone_continuous_duration", 0.0)
        if eyes_closed_sec >= 8.0 or phone_sec >= 8.0:
            self.current_score = 100.0

        self.current_score = min(self.MAX_SCORE, max(self.MIN_SCORE, self.current_score))
        score_int = int(round(self.current_score))

        # ── Update Statistics ─────────────────────────────────────────
        self.max_risk_score = max(self.max_risk_score, score_int)
        self.risk_score_history.append(score_int)

        # Track distinct event transitions
        if monitor_data.get("drowsy", False):
            if not self._prev_drowsy:
                self.drowsy_events += 1
            self._prev_drowsy = True
        else:
            self._prev_drowsy = False

        if phone_data.get("detected", False):
            if not self._prev_phone:
                self.phone_events += 1
            self._prev_phone = True
        else:
            self._prev_phone = False

        if monitor_data.get("distracted", False):
            if not self._prev_distracted:
                self.distraction_events += 1
            self._prev_distracted = True
        else:
            self._prev_distracted = False

        # ── Determine Risk Level ──────────────────────────────────────
        if score_int <= self.SAFE_THRESHOLD:
            level = "Safe"
            level_color = "#76b900"
        elif score_int <= self.MODERATE_THRESHOLD:
            level = "Moderate Risk"
            level_color = "#ff9800"
        elif score_int < 90:
            level = "High Risk"
            level_color = "#f44336"
        else:
            level = "CRITICAL"
            level_color = "#ff0000"

        # ── Final Driver Status ───────────────────────────────────────
        monitor_status = monitor_data.get("status", "Attentive")
        phone_sec_val = phone_data.get("phone_continuous_duration", 0.0)

        if score_int >= 90:
            final_status = "🚨 Critical Emergency"
        elif phone_data.get("detected", False):
            if monitor_status in ["Drowsy", "High Risk Driver"]:
                final_status = "High Risk Driver"
            elif phone_sec_val >= 6.0:
                final_status = "Phone Distraction"
            else:
                final_status = "Phone Usage Detected"
        elif score_int > self.MODERATE_THRESHOLD:
            final_status = "High Risk Driver"
        else:
            final_status = monitor_status

        return {
            "score": score_int,
            "level": level,
            "level_color": level_color,
            "contributions": contributions,
            "status": final_status,
            "eyes_closed_seconds": eyes_closed_sec,
            "phone_seconds": phone_sec_val,
            "yawn_count_60s": monitor_data.get("yawn_count_60s", 0),
        }

    def record_alert(self, alert_type, message):
        """Record an alert event in the session history."""
        self.total_alerts += 1
        self.alert_history.append({
            "timestamp": time.time(),
            "type": alert_type,
            "message": message,
            "risk_score": int(round(self.current_score))
        })

    def get_session_stats(self):
        """Get comprehensive session statistics."""
        duration = time.time() - self.session_start_time

        if self.risk_score_history:
            avg_risk = sum(self.risk_score_history) / len(self.risk_score_history)
        else:
            avg_risk = 0.0

        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Risk trend (last 60 samples)
        trend = list(self.risk_score_history)[-60:] if self.risk_score_history else []

        return {
            "duration_seconds": duration,
            "duration_formatted": duration_formatted,
            "total_alerts": self.total_alerts,
            "max_risk_score": self.max_risk_score,
            "avg_risk_score": round(avg_risk, 1),
            "drowsy_events": self.drowsy_events,
            "phone_events": self.phone_events,
            "distraction_events": self.distraction_events,
            "alert_history": list(self.alert_history[-50:]),
            "risk_trend": trend,
        }

    def reset(self):
        """Reset all risk engine state for a new session."""
        self.current_score = 0.0
        self.target_score = 0.0
        self.last_update_time = time.time()
        self.session_start_time = time.time()
        self.total_alerts = 0
        self.max_risk_score = 0
        self.risk_score_history.clear()
        self.alert_history.clear()
        self.drowsy_events = 0
        self.phone_events = 0
        self.distraction_events = 0
        self._prev_drowsy = False
        self._prev_phone = False
        self._prev_distracted = False
