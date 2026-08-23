"""
SafeDrive AI - Phone Detector Module
======================================
Real-time mobile phone detection using YOLOv8s pretrained on COCO.

Upgraded from yolov8n → yolov8s for significantly better accuracy.
Tracks continuous phone detection duration for time-based risk escalation.
Detects phone in hand, near face, near ear, and covering face.

Author: SafeDrive AI Team
"""

import numpy as np
import cv2
from collections import deque
import time


class PhoneDetector:
    """
    Detects mobile phone usage using YOLOv8s pretrained COCO model.

    - Uses yolov8s for improved accuracy over yolov8n
    - Filters to 'cell phone' class (COCO ID 67)
    - Temporal smoothing to reduce false positives
    - Tracks continuous detection duration for escalating risk
    - Detects phone-to-face proximity (near ear, covering face)
    """

    CELL_PHONE_CLASS_ID = 67      # COCO class ID for 'cell phone'
    CONFIDENCE_THRESHOLD = 0.30   # Lowered slightly for yolov8s (better model)
    TEMPORAL_WINDOW = 7           # Number of recent frames to consider
    TEMPORAL_THRESHOLD = 3        # Detections needed in window to confirm

    # Duration thresholds (seconds) for escalating alerts
    DURATION_LEVEL_1 = 2.0        # Show popup
    DURATION_LEVEL_2 = 4.0        # Repeated warning beep
    DURATION_LEVEL_3 = 6.0        # PHONE DISTRACTION banner
    DURATION_LEVEL_4 = 8.0        # Emergency alarm

    def __init__(self, model_path="yolov8s.pt"):
        """
        Initialize the phone detector with YOLOv8s.

        Args:
            model_path: Path to YOLOv8 weights. Defaults to 'yolov8s.pt'
                        (auto-downloads ~22MB from Ultralytics on first run).
        """
        from ultralytics import YOLO
        import torch

        self.model = YOLO(model_path)
        self.model_path = model_path

        # Use GPU automatically when available — same weights, faster and
        # more consistent inference (fp32 stays fp32, no accuracy trade-off).
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Temporal smoothing
        self.detection_history = deque(maxlen=self.TEMPORAL_WINDOW)

        # State
        self.phone_detected = False
        self.last_detection_time = 0
        self.last_bbox = None
        self.last_confidence = 0.0

        # ── Continuous Duration Tracking ──────────────────────────────
        self.phone_detected_start_time = None   # When phone was first confirmed
        self.phone_continuous_duration = 0.0    # Seconds phone has been present

        # Detection context flags
        self.near_face = False
        self.near_ear = False

    def detect(self, frame, face_bbox=None):
        """
        Run phone detection on a single video frame.

        Args:
            frame: BGR image (numpy array) from webcam
            face_bbox: Optional (x1, y1, x2, y2) of driver face for overlap check

        Returns:
            dict: Detection results containing:
                - detected (bool): Confirmed phone detection
                - raw_detected (bool): Raw single-frame detection
                - confidence (float): Detection confidence (0-1)
                - bbox (tuple|None): Bounding box (x1, y1, x2, y2)
                - detection_count (int): Detections in temporal window
                - phone_continuous_duration (float): Seconds phone present
                - near_face (bool): Phone bbox overlaps face region
                - near_ear (bool): Phone is in upper quadrant near ear
        """
        img_h, img_w = frame.shape[:2]

        # Run YOLOv8 inference — only cell phones
        results = self.model.predict(
            source=frame,
            classes=[self.CELL_PHONE_CLASS_ID],
            conf=self.CONFIDENCE_THRESHOLD,
            iou=0.45,           # standard NMS overlap threshold — merges duplicate boxes on the same phone
            device=self.device,
            verbose=False,
            imgsz=640
        )

        raw_detected = False
        best_confidence = 0.0
        best_bbox = None

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > best_confidence:
                        best_confidence = conf
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        best_bbox = (int(x1), int(y1), int(x2), int(y2))
                        raw_detected = True

        # Update temporal history
        self.detection_history.append(1 if raw_detected else 0)
        detection_count = sum(self.detection_history)
        confirmed = detection_count >= self.TEMPORAL_THRESHOLD

        current_time = time.time()

        if confirmed:
            self.phone_detected = True
            self.last_detection_time = current_time
            if best_bbox:
                self.last_bbox = best_bbox
                self.last_confidence = best_confidence

            # Start/update continuous duration timer
            if self.phone_detected_start_time is None:
                self.phone_detected_start_time = current_time
            self.phone_continuous_duration = current_time - self.phone_detected_start_time

        else:
            # Keep showing detection briefly after disappearance
            if current_time - self.last_detection_time > 2.0:
                self.phone_detected = False
                self.last_bbox = None
                self.last_confidence = 0.0
                self.phone_detected_start_time = None
                self.phone_continuous_duration = 0.0

        # ── Proximity Analysis ────────────────────────────────────────
        active_bbox = best_bbox if raw_detected else self.last_bbox
        self.near_face = False
        self.near_ear = False

        if active_bbox and self.phone_detected:
            px1, py1, px2, py2 = active_bbox
            phone_cx = (px1 + px2) / 2
            phone_cy = (py1 + py2) / 2

            # Near-face: phone bbox overlaps with face region
            if face_bbox is not None:
                fx1, fy1, fx2, fy2 = face_bbox
                overlap_x = max(0, min(px2, fx2) - max(px1, fx1))
                overlap_y = max(0, min(py2, fy2) - max(py1, fy1))
                if overlap_x > 0 and overlap_y > 0:
                    self.near_face = True

            # Near-ear heuristic: phone is in upper 40% of frame, right or left side
            if phone_cy < img_h * 0.4 and (phone_cx < img_w * 0.35 or phone_cx > img_w * 0.65):
                self.near_ear = True

        return {
            "detected": self.phone_detected,
            "raw_detected": raw_detected,
            "confidence": best_confidence if raw_detected else self.last_confidence,
            "bbox": best_bbox if raw_detected else self.last_bbox,
            "detection_count": detection_count,
            "phone_continuous_duration": self.phone_continuous_duration,
            "near_face": self.near_face,
            "near_ear": self.near_ear,
        }

    def draw_detection(self, frame, detection_result):
        """
        Draw phone detection bounding box and label on the frame.

        Includes escalating visual indicators based on detection duration.
        """
        if not (detection_result["detected"] and detection_result["bbox"]):
            return frame

        x1, y1, x2, y2 = detection_result["bbox"]
        conf = detection_result["confidence"]
        duration = detection_result.get("phone_continuous_duration", 0)

        # Color escalates with duration: orange → red
        if duration >= self.DURATION_LEVEL_3:
            color = (0, 0, 255)      # Full red
            border_thickness = 4
        elif duration >= self.DURATION_LEVEL_2:
            color = (0, 30, 255)
            border_thickness = 3
        elif duration >= self.DURATION_LEVEL_1:
            color = (0, 69, 255)     # Orange-red
            border_thickness = 3
        else:
            color = (0, 140, 255)    # Orange
            border_thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, border_thickness)

        # Outer pulsing border
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 1)

        # Label
        context_flags = []
        if detection_result.get("near_face"):
            context_flags.append("NEAR FACE")
        if detection_result.get("near_ear"):
            context_flags.append("NEAR EAR")

        context = " | " + " | ".join(context_flags) if context_flags else ""
        label = f"📱 PHONE {conf:.0%}{context} [{duration:.1f}s]"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 14),
            (x1 + label_size[0] + 10, y1),
            color, -1
        )
        cv2.putText(
            frame, label,
            (x1 + 5, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        # Show phone distraction banner for level 3+
        if duration >= self.DURATION_LEVEL_3:
            img_h, img_w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, img_h - 80), (img_w, img_h - 40), (0, 0, 180), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
            cv2.putText(
                frame, "!! PHONE DISTRACTION DETECTED !!",
                (10, img_h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2
            )

        return frame

    def reset(self):
        """Reset detection state for a new session."""
        self.detection_history.clear()
        self.phone_detected = False
        self.last_detection_time = 0
        self.last_bbox = None
        self.last_confidence = 0.0
        self.phone_detected_start_time = None
        self.phone_continuous_duration = 0.0
        self.near_face = False
        self.near_ear = False
