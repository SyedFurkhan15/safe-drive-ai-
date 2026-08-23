"""
SafeDrive AI - Driver Monitor Module
=====================================
Real-time driver behavior analysis using MediaPipe Face Mesh.

Detects:
- Drowsiness (Eye Aspect Ratio / prolonged eye closure with wall-clock timing)
- Yawning (Mouth Aspect Ratio) with 60-second yawn count history
- Head pose / distraction (solvePnP → Euler angles)
- Blink rate anomalies

Uses MediaPipe's 468-landmark Face Mesh model for lightweight,
real-time facial landmark detection without custom training.

Author: SafeDrive AI Team
"""

import cv2
import numpy as np
import mediapipe as mp


import time
from collections import deque


class DriverMonitor:
    """
    Monitors driver facial behavior in real time using MediaPipe Face Mesh.

    Provides drowsiness detection (EAR), yawn detection (MAR), head pose
    estimation (solvePnP), and blink rate tracking. Designed to be
    instantiated once and called per-frame from a video processing thread.
    """

    # ── MediaPipe Face Mesh Landmark Indices ─────────────────────────────────

    # Left eye landmarks (from the driver's perspective)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    # Right eye landmarks
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Mouth landmarks for yawn detection
    UPPER_LIP = [13]
    LOWER_LIP = [14]
    LEFT_LIP_CORNER = [78]
    RIGHT_LIP_CORNER = [308]
    UPPER_LIP_INNER = [82, 312]
    LOWER_LIP_INNER = [87, 317]

    # Head pose estimation landmarks (6-point model)
    NOSE_TIP = 4
    CHIN = 152
    LEFT_EYE_CORNER = 263
    RIGHT_EYE_CORNER = 33
    LEFT_MOUTH_CORNER = 287
    RIGHT_MOUTH_CORNER = 57

    # ── Thresholds ───────────────────────────────────────────────────────

    EAR_THRESHOLD = 0.22          # Below this → eyes considered closed
    EAR_CONSEC_FRAMES = 15        # Kept for blink detection fallback
    MAR_THRESHOLD = 0.60          # Above this → yawning detected
    MAR_CONSEC_FRAMES = 10        # ~0.33s at 30fps
    YAW_THRESHOLD = 30.0          # Degrees → looking left/right
    PITCH_DOWN_THRESHOLD = 20.0   # Degrees → looking down
    PITCH_UP_THRESHOLD = -15.0    # Degrees → looking up
    BLINK_EAR_THRESHOLD = 0.22    # Same as EAR threshold for blinks
    HIGH_BLINK_RATE = 25          # Blinks per minute → fatigue indicator

    # ── Eye closure duration thresholds (seconds) ─────────────────────────────
    DROWSY_THRESHOLD_SEC = 2.0    # Trigger DROWSINESS state
    RISK_LEVEL_2_SEC = 2.0        # +50 risk
    RISK_LEVEL_4_SEC = 4.0        # +70 risk
    RISK_LEVEL_6_SEC = 6.0        # +90 risk
    RISK_LEVEL_8_SEC = 8.0        # Critical emergency = 100

    # ── 3D Model Points for Head Pose ─────────────────────────────────────────
    FACE_3D_MODEL = np.array([
        [0.0, 0.0, 0.0],           # Nose tip
        [0.0, -330.0, -65.0],      # Chin
        [-225.0, 170.0, -135.0],   # Left eye corner
        [225.0, 170.0, -135.0],    # Right eye corner
        [-150.0, -150.0, -125.0],  # Left mouth corner
        [150.0, -150.0, -125.0],   # Right mouth corner
    ], dtype=np.float64)

    def __init__(self):
        """Initialize the driver monitoring system with MediaPipe Face Mesh."""

        # Developer/UI overlay controls (wired from app.py)
        # - developer_mode: show full technical HUD
        # - show_ai_landmarks: show MediaPipe mesh overlay (can be enabled
        #   independently while developer_mode remains OFF)
        self.developer_mode: bool = False
        self.show_ai_landmarks: bool = False


        import mediapipe as mp
        import sys
        
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                f"\n[SafeDrive AI Error] MediaPipe in current Python ({sys.executable}) has no 'solutions' attribute.\n"
                f"You are running Python {sys.version.split()[0]} instead of the project's Python 3.11 virtual environment.\n"
                f"Please start SafeDrive AI using: .\\venv\\Scripts\\python.exe -m streamlit run app.py\n"
                f"or double-click START_SAFEDRIVE.bat / run_app.bat"
            )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6     # stricter frame-to-frame tracking -> less landmark jitter
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        # ── State Tracking ────────────────────────────────────────────
        self.ear_counter = 0              # Consecutive frames with low EAR
        self.mar_counter = 0              # Consecutive frames with high MAR
        self.blink_counter = 0            # Total blinks in current window
        self.blink_timestamps = deque(maxlen=100)
        self.was_eye_closed = False
        self.drowsy_detected = False
        self.yawn_detected = False
        self.distracted = False
        self.gaze_direction = "Forward"

        # ── Eye Closure Wall-Clock Timing ─────────────────────────────
        self.eyes_closed_start_time = None   # Wall-clock when eyes first closed
        self.eyes_closed_duration = 0.0      # Current continuous closure duration (s)
        self._both_eyes_closed = False       # Require BOTH eyes below threshold

        # ── Yawn Tracking ──────────────────────────────────────────────
        self.yawn_timestamps = deque(maxlen=100)  # Timestamps of detected yawns
        self.yawn_count_60s = 0                   # Yawns in last 60 seconds
        self._yawn_in_progress = False            # Debounce for yawn detection

        # ── Adaptive Per-Driver Calibration ─────────────────────────────
        # A fixed EAR threshold (0.22) doesn't fit every eye shape, camera
        # angle, or lighting condition. Instead we learn each driver's own
        # "eyes open" baseline during the first ~1.5s of tracking and derive
        # a personalized threshold from it. This materially cuts both missed
        # drowsiness events and false alarms versus a one-size-fits-all cutoff.
        self.CALIBRATION_FRAMES = 45              # ~1.5s @ 30fps
        self._calib_ear_samples = []
        self._calibrated = False
        self.ear_threshold = self.EAR_THRESHOLD    # falls back to default until calibrated

        # ── Exponential Moving Average smoothing ────────────────────────
        # Raw per-frame EAR/MAR/head-pose values are noisy (landmark jitter,
        # compression artifacts). Smoothing them reduces flicker in both the
        # readouts and the alert triggers without adding real latency.
        self.SMOOTHING_ALPHA = 0.4                 # higher = more responsive, lower = smoother
        self._ear_smooth = None
        self._mar_smooth = None
        self._yaw_smooth = None
        self._pitch_smooth = None
        self._roll_smooth = None

        # Cached solvePnP outputs (used to draw the 3D head-pose axis)
        self._last_rotation_vector = None
        self._last_translation_vector = None
        self._last_camera_matrix = None
        self._last_dist_coeffs = None

        # ── HUD color palette (BGR) — matches the dashboard's NVIDIA theme ──
        self.COLOR_SAFE = (0, 185, 118)      # NVIDIA green   #76b900
        self.COLOR_INFO = (255, 212, 0)      # Cyan           #00d4ff
        self.COLOR_WARN = (0, 152, 255)      # Amber          #ff9800
        self.COLOR_DANGER = (54, 67, 244)    # Red            #f44336

    @staticmethod
    def _smooth(previous, new_value, alpha):
        """Apply exponential moving average smoothing to a scalar signal."""
        if previous is None:
            return new_value
        return alpha * new_value + (1 - alpha) * previous

    @staticmethod
    def eye_aspect_ratio(eye_landmarks):
        """
        Calculate the Eye Aspect Ratio (EAR) to detect eye closure.

        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

        Args:
            eye_landmarks: List of 6 (x, y) tuples representing eye landmarks
                          [p1, p2, p3, p4, p5, p6]

        Returns:
            float: Eye aspect ratio value
        """
        # Convert landmarks to numpy arrays
        p = np.array(eye_landmarks, dtype=np.float32)

        # Calculate distances
        d1 = np.linalg.norm(p[1] - p[5])  # Vertical 1
        d2 = np.linalg.norm(p[2] - p[4])  # Vertical 2
        d3 = np.linalg.norm(p[0] - p[3])  # Horizontal

        # Avoid division by zero
        if d3 < 1e-6:
            return 0.0

        return (d1 + d2) / (2.0 * d3)

    @staticmethod
    def mouth_aspect_ratio(mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR) to detect yawning.

        MAR = (||upper_lip - lower_lip||) / (||left_corner - right_corner||)

        Args:
            mouth_landmarks: Dictionary with 'upper', 'lower', 'left', 'right' keys
                           containing (x, y) tuples

        Returns:
            float: Mouth aspect ratio value
        """
        upper = np.array(mouth_landmarks['upper'], dtype=np.float32)
        lower = np.array(mouth_landmarks['lower'], dtype=np.float32)
        left = np.array(mouth_landmarks['left'], dtype=np.float32)
        right = np.array(mouth_landmarks['right'], dtype=np.float32)

        vertical = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left - right)

        if horizontal < 1e-6:
            return 0.0

        return vertical / horizontal

    def estimate_head_pose(self, landmarks, image_shape):
        """
        Estimate head pose (yaw, pitch, roll) using solvePnP.

        Args:
            landmarks: MediaPipe face mesh landmarks
            image_shape: Tuple (height, width) of input image

        Returns:
            tuple: (yaw, pitch, roll) in degrees
        """
        h, w = image_shape[:2]
        focal_length = w
        center = (w / 2, h / 2)

        # Camera matrix
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Get 2D image points from landmarks
        image_points = np.array([
            (landmarks[self.NOSE_TIP].x * w, landmarks[self.NOSE_TIP].y * h),
            (landmarks[self.CHIN].x * w, landmarks[self.CHIN].y * h),
            (landmarks[self.LEFT_EYE_CORNER].x * w, landmarks[self.LEFT_EYE_CORNER].y * h),
            (landmarks[self.RIGHT_EYE_CORNER].x * w, landmarks[self.RIGHT_EYE_CORNER].y * h),
            (landmarks[self.LEFT_MOUTH_CORNER].x * w, landmarks[self.LEFT_MOUTH_CORNER].y * h),
            (landmarks[self.RIGHT_MOUTH_CORNER].x * w, landmarks[self.RIGHT_MOUTH_CORNER].y * h),
        ], dtype=np.float64)

        # Solve for rotation and translation vectors
        # SOLVEPNP_ITERATIVE with a prior guess is more stable frame-to-frame
        # than a cold solve every time, which reduces head-pose jitter.
        use_extrinsic_guess = self._last_rotation_vector is not None
        rvec_guess = self._last_rotation_vector if use_extrinsic_guess else np.zeros((3, 1))
        tvec_guess = self._last_translation_vector if use_extrinsic_guess else np.zeros((3, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
            rvec=rvec_guess.copy(), tvec=tvec_guess.copy(),
            useExtrinsicGuess=use_extrinsic_guess,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Cache for axis drawing and next frame's extrinsic guess
        self._last_rotation_vector = rotation_vector
        self._last_translation_vector = translation_vector
        self._last_camera_matrix = camera_matrix
        self._last_dist_coeffs = dist_coeffs

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Calculate Euler angles (yaw, pitch, roll)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0.0

        # Convert to degrees
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)

        return yaw, pitch, roll

    @staticmethod
    def _draw_corner_box(frame, bbox, color, thickness=2, corner_len_ratio=0.18):
        """
        Draw a sci-fi HUD style bounding box: short corner brackets instead
        of a full rectangle. Reads as a tracking lock, not just a box.
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cl = int(min(w, h) * corner_len_ratio)
        cl = max(cl, 12)

        corners = [
            ((x1, y1), (1, 0), (0, 1)),    # top-left
            ((x2, y1), (-1, 0), (0, 1)),   # top-right
            ((x1, y2), (1, 0), (0, -1)),   # bottom-left
            ((x2, y2), (-1, 0), (0, -1)),  # bottom-right
        ]
        for (cx, cy), (dx, dy), (ex, ey) in corners:
            cv2.line(frame, (cx, cy), (cx + dx * cl, cy), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + ey * cl), color, thickness, cv2.LINE_AA)

    def _draw_pose_axis(self, frame, origin_2d, length=60):
        """
        Project a 3D axis (X=red, Y=green, Z=blue) from the nose tip using
        the cached solvePnP extrinsics. This is the standard way ADAS/DMS
        systems visualize head orientation — far clearer at a glance than
        text alone.
        """
        if self._last_rotation_vector is None:
            return

        axis_3d = np.array([
            [length, 0, 0],
            [0, length, 0],
            [0, 0, length],
        ], dtype=np.float64)

        try:
            image_points, _ = cv2.projectPoints(
                axis_3d, self._last_rotation_vector, self._last_translation_vector,
                self._last_camera_matrix, self._last_dist_coeffs
            )
        except cv2.error:
            return

        origin = tuple(np.int32(origin_2d))
        x_pt = tuple(np.int32(image_points[0].ravel()))
        y_pt = tuple(np.int32(image_points[1].ravel()))
        z_pt = tuple(np.int32(image_points[2].ravel()))

        cv2.line(frame, origin, x_pt, (0, 0, 255), 2, cv2.LINE_AA)    # X - red
        cv2.line(frame, origin, y_pt, (0, 255, 0), 2, cv2.LINE_AA)    # Y - green
        cv2.line(frame, origin, z_pt, (255, 0, 0), 2, cv2.LINE_AA)    # Z - blue

    @staticmethod
    def _draw_landmark_marker(frame, point, color, radius=3, label=None):
        """Draw a single tracked anchor point (e.g. nose tip, eye corner)."""
        pt = (int(point[0]), int(point[1]))
        cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, radius + 2, color, 1, cv2.LINE_AA)
        if label:
            cv2.putText(frame, label, (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def process_frame(self, frame):
        """
        Process a single video frame and detect driver state.

        Args:
            frame: BGR video frame as numpy array

        Returns:
            dict: Dictionary with detection results:
                - 'drowsy': bool, True if drowsy
                - 'eyes_closed_duration': float, seconds eyes have been closed
                - 'distracted': bool, True if distracted
                - 'gaze_direction': str, 'Forward', 'Left', 'Right', 'Up', 'Down'
                - 'yawning': bool, True if yawning
                - 'yawn_count_60s': int, number of yawns in last 60 seconds
                - 'blink_rate': float, blinks per minute
                - 'face_bbox': tuple or None, (x1, y1, x2, y2) face bounding box
                - 'annotated_frame': numpy array, frame with annotations drawn
        """
        # Create a copy for annotation
        annotated_frame = frame.copy()

        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process with Face Mesh
        results = self.face_mesh.process(rgb_frame)

        # Initialize result values
        drowsy = False
        eyes_closed_duration = 0.0
        distracted = False
        gaze_direction = "Forward"
        yawning = False
        yawn_count_60s = 0
        blink_rate = 0.0
        face_bbox = None
        face_detected = False
        avg_ear = 0.0
        mar = 0.0
        yaw = 0.0
        pitch = 0.0
        roll = 0.0

        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]

                # ── Optional MediaPipe Face Mesh (Developer/UI overlay) ──
                if self.developer_mode or self.show_ai_landmarks:
                    # Drawing straight onto annotated_frame makes a dense, fully
                    # opaque mesh that obscures the driver's face. Instead we draw
                    # onto a transparent-style overlay copy and alpha-blend it
                    # back in, which reads as a proper HUD diagnostic overlay
                    # rather than a solid mask.
                    mesh_overlay = annotated_frame.copy()
                    self.mp_drawing.draw_landmarks(
                        mesh_overlay, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=self.COLOR_INFO, thickness=1, circle_radius=0)
                    )
                    cv2.addWeighted(mesh_overlay, 0.35, annotated_frame, 0.65, 0, dst=annotated_frame)

                    # Contours + irises drawn at full strength.
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=self.COLOR_SAFE, thickness=1, circle_radius=0)
                    )
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, face_landmarks, self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=self.COLOR_INFO, thickness=1, circle_radius=0)
                    )


                # Get all landmarks as (x, y) tuples (pixel space kept separately below)
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.append((lm.x, lm.y))

                # Calculate face bounding box
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                face_bbox = (x1, y1, x2, y2)

                # ── Eye Aspect Ratio (smoothed) ──────────────────────────────
                left_eye = [landmarks[i] for i in self.LEFT_EYE]
                right_eye = [landmarks[i] for i in self.RIGHT_EYE]
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                raw_avg_ear = (left_ear + right_ear) / 2.0
                self._ear_smooth = self._smooth(self._ear_smooth, raw_avg_ear, self.SMOOTHING_ALPHA)
                avg_ear = self._ear_smooth

                # ── Personalized calibration ─────────────────────────────────
                # Learn this driver's natural "eyes open" EAR for the first
                # CALIBRATION_FRAMES frames, then derive a threshold at 75% of
                # baseline. This adapts to face shape/camera angle instead of
                # relying on one global constant.
                if not self._calibrated:
                    self._calib_ear_samples.append(raw_avg_ear)
                    if len(self._calib_ear_samples) >= self.CALIBRATION_FRAMES:
                        baseline = float(np.median(self._calib_ear_samples))
                        self.ear_threshold = float(np.clip(baseline * 0.75, 0.15, 0.25))
                        self._calibrated = True

                # Check if both eyes are closed against the calibrated threshold
                both_closed = avg_ear < self.ear_threshold

                # Update eye closure timing
                current_time = time.time()
                if both_closed:
                    if self.eyes_closed_start_time is None:
                        self.eyes_closed_start_time = current_time
                    eyes_closed_duration = current_time - self.eyes_closed_start_time
                else:
                    self.eyes_closed_start_time = None
                    eyes_closed_duration = 0.0

                # Blink detection
                if both_closed and not self.was_eye_closed:
                    self.blink_timestamps.append(current_time)
                    self.was_eye_closed = True
                elif not both_closed:
                    self.was_eye_closed = False

                # Calculate blink rate (blinks per minute)
                if len(self.blink_timestamps) >= 2:
                    time_window = max(60, current_time - self.blink_timestamps[0])
                    blink_rate = (len(self.blink_timestamps) / time_window) * 60
                else:
                    blink_rate = 0.0

                # Drowsiness detection
                drowsy = eyes_closed_duration >= self.DROWSY_THRESHOLD_SEC

                # ── Mouth Aspect Ratio (smoothed) ────────────────────────────
                mouth_landmarks = {
                    'upper': landmarks[self.UPPER_LIP[0]],
                    'lower': landmarks[self.LOWER_LIP[0]],
                    'left': landmarks[self.LEFT_LIP_CORNER[0]],
                    'right': landmarks[self.RIGHT_LIP_CORNER[0]]
                }
                raw_mar = self.mouth_aspect_ratio(mouth_landmarks)
                self._mar_smooth = self._smooth(self._mar_smooth, raw_mar, self.SMOOTHING_ALPHA)
                mar = self._mar_smooth

                # Yawn detection with debounce
                if mar > self.MAR_THRESHOLD:
                    self.mar_counter += 1
                    if self.mar_counter >= self.MAR_CONSEC_FRAMES and not self._yawn_in_progress:
                        self.yawn_timestamps.append(current_time)
                        self._yawn_in_progress = True
                else:
                    self.mar_counter = 0
                    self._yawn_in_progress = False

                # Count yawns in last 60 seconds
                yawn_count_60s = sum(1 for t in self.yawn_timestamps if current_time - t <= 60)
                yawning = self._yawn_in_progress

                # ── Head Pose Estimation (smoothed) ──────────────────────────
                raw_yaw, raw_pitch, raw_roll = self.estimate_head_pose(face_landmarks.landmark, frame.shape)
                self._yaw_smooth = self._smooth(self._yaw_smooth, raw_yaw, self.SMOOTHING_ALPHA)
                self._pitch_smooth = self._smooth(self._pitch_smooth, raw_pitch, self.SMOOTHING_ALPHA)
                self._roll_smooth = self._smooth(self._roll_smooth, raw_roll, self.SMOOTHING_ALPHA)
                yaw, pitch, roll = self._yaw_smooth, self._pitch_smooth, self._roll_smooth

                # Determine gaze direction
                if abs(yaw) > self.YAW_THRESHOLD:
                    gaze_direction = "Right" if yaw > 0 else "Left"
                elif pitch > self.PITCH_DOWN_THRESHOLD:
                    gaze_direction = "Down"
                elif pitch < self.PITCH_UP_THRESHOLD:
                    gaze_direction = "Up"
                else:
                    gaze_direction = "Forward"

                distracted = gaze_direction != "Forward"

                # ── HUD state color: reflects the worst active condition ────
                if drowsy or distracted:
                    hud_color = self.COLOR_DANGER
                elif yawning:
                    hud_color = self.COLOR_WARN
                else:
                    hud_color = self.COLOR_SAFE

                # ── Clean professional HUD vs Developer HUD ─────────────────
                # Always draw a minimal face tracking box (corner brackets).
                self._draw_corner_box(annotated_frame, face_bbox, hud_color, thickness=2)

                # Developer mode overlays (text + markers + pose axis)
                if self.developer_mode or self.show_ai_landmarks:
                    lock_label = "FACE LOCKED" if self._calibrated else "CALIBRATING..."
                    cv2.putText(
                        annotated_frame,
                        lock_label,
                        (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        hud_color,
                        1,
                        cv2.LINE_AA,
                    )

                # Anchor points and pose axis only in developer mode
                if self.developer_mode:
                    nose_px = (landmarks[self.NOSE_TIP][0] * w, landmarks[self.NOSE_TIP][1] * h)
                    anchor_points = {
                        "Nose": (self.NOSE_TIP, self.COLOR_INFO),
                        "Chin": (self.CHIN, self.COLOR_INFO),
                        "L-Eye": (self.LEFT_EYE_CORNER, self.COLOR_SAFE),
                        "R-Eye": (self.RIGHT_EYE_CORNER, self.COLOR_SAFE),
                    }
                    for label, (idx, color) in anchor_points.items():
                        px = (landmarks[idx][0] * w, landmarks[idx][1] * h)
                        self._draw_landmark_marker(annotated_frame, px, color, radius=3)

                    self._draw_pose_axis(
                        annotated_frame,
                        nose_px,
                        length=max(40, (x2 - x1) // 3),
                    )

                # Telemetry text only in developer mode
                if self.developer_mode:
                    cv2.putText(
                        annotated_frame,
                        f"EAR {avg_ear:.2f} / thr {self.ear_threshold:.2f}",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.COLOR_INFO,
                        1,
                        cv2.LINE_AA,
                    )
                    if drowsy:
                        cv2.putText(
                            annotated_frame,
                            f"DROWSY {eyes_closed_duration:.1f}s",
                            (x1, y1 - 48),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            self.COLOR_DANGER,
                            2,
                            cv2.LINE_AA,
                        )

                    cv2.putText(
                        annotated_frame,
                        f"MAR {mar:.2f}",
                        (x1, y2 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.COLOR_INFO,
                        1,
                        cv2.LINE_AA,
                    )
                    if yawn_count_60s > 0:
                        cv2.putText(
                            annotated_frame,
                            f"Yawns (60s): {yawn_count_60s}",
                            (x1, y2 + 44),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            self.COLOR_WARN,
                            1,
                            cv2.LINE_AA,
                        )

                    pose_text = f"Yaw {yaw:.0f} deg  Pitch {pitch:.0f} deg  Roll {roll:.0f} deg"
                    cv2.putText(
                        annotated_frame,
                        pose_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        self.COLOR_INFO,
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Gaze: {gaze_direction}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        hud_color,
                        2,
                        cv2.LINE_AA,
                    )


        return {
            'drowsy': drowsy,
            'eyes_closed_duration': eyes_closed_duration,
            'distracted': distracted,
            'gaze_direction': gaze_direction,
            'yawning': yawning,
            'yawn_count_60s': yawn_count_60s,
            'blink_rate': blink_rate,
            'face_bbox': face_bbox,
            'face_detected': face_detected,
            'ear': avg_ear,
            'mar': mar,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'annotated_frame': annotated_frame,
            'processed_frame': annotated_frame,  # alias — app.py reads this key for the video feed
        }

    def reset(self):
        """Reset all state tracking for a new session."""
        self.ear_counter = 0
        self.mar_counter = 0
        self.blink_counter = 0
        self.blink_timestamps.clear()
        self.was_eye_closed = False
        self.drowsy_detected = False
        self.yawn_detected = False
        self.distracted = False
        self.gaze_direction = "Forward"
        self.eyes_closed_start_time = None
        self.eyes_closed_duration = 0.0
        self.yawn_timestamps.clear()
        self.yawn_count_60s = 0
        self._yawn_in_progress = False

        # Reset calibration so a new driver gets a fresh personalized baseline
        self._calib_ear_samples = []
        self._calibrated = False
        self.ear_threshold = self.EAR_THRESHOLD

        # Reset smoothing so stale values don't bleed into the new session
        self._ear_smooth = None
        self._mar_smooth = None
        self._yaw_smooth = None
        self._pitch_smooth = None
        self._roll_smooth = None
        self._last_rotation_vector = None
        self._last_translation_vector = None
