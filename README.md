# 🚗 SafeDrive AI
## AI-Based Driver Monitoring and Safety Assistance System

SafeDrive AI is an AI-powered driver monitoring prototype designed to analyze visible driver behaviour and identify potential safety risks such as **drowsiness, yawning, distraction, and mobile phone usage**.

The system combines **computer vision, facial landmark analysis, object detection, and risk-based alert logic** to monitor the driver through a live camera feed or uploaded images, videos, and ZIP-based datasets.

---

# 📌 Project Overview

Driver fatigue and distraction are major factors that can contribute to unsafe driving. SafeDrive AI demonstrates how Artificial Intelligence and Computer Vision can be used to monitor visible driver behaviour and generate safety-related alerts.

## 🎯 Objectives

- Detect possible drowsiness using eye-related facial features.
- Identify yawning using mouth landmark analysis.
- Analyze head orientation for possible distraction.
- Detect mobile phone usage using YOLOv8.
- Combine multiple detected behaviours into a driver risk level.
- Provide live monitoring through an interactive dashboard.
- Support images, videos, and ZIP-based datasets for testing.
- Generate alerts and maintain event information.

---

# 🧠 System Architecture

```text
Camera / Image / Video / Dataset
              │
              ▼
        Frame Processing
            OpenCV
              │
       ┌──────┴──────┐
       ▼             ▼
MediaPipe        YOLOv8
Face Mesh     Object Detection
       │             │
       ▼             ▼
Eye / Mouth /      Mobile Phone
Head Analysis       Detection
       │             │
       └──────┬──────┘
              ▼
      Risk & Alert Engine
              │
              ▼
      Streamlit Dashboard
              │
              ▼
   Alerts • Events • Results
