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

# 🤖 AI Models and Algorithms

| Component | Algorithm / Model | Purpose |
|---|---|---|
| Facial Landmark Detection | MediaPipe Face Mesh | Detect facial landmarks |
| Eye Analysis | Landmark-based geometric analysis | Detect prolonged eye closure |
| Mouth Analysis | Landmark-based analysis | Detect yawning indicators |
| Head Analysis | Face orientation analysis | Detect possible distraction |
| Object Detection | YOLOv8 | Detect mobile phone usage |
| Image Processing | OpenCV | Process images and video frames |
| Risk Evaluation | Risk and Alert Logic | Combine unsafe behaviour signals |
| Dashboard | Streamlit | Display monitoring results |

---

# ⚙️ How the System Works

### 1. Input Acquisition

The system accepts:

- Live webcam
- Uploaded images
- Video files
- ZIP-based datasets

### 2. Frame Processing

OpenCV processes images and videos frame by frame.

### 3. Driver Face Analysis

MediaPipe Face Mesh extracts facial landmarks for analyzing:

- Eyes
- Mouth
- Face orientation

### 4. Behaviour Detection

The system identifies:

- Possible drowsiness
- Yawning
- Possible distraction

### 5. Mobile Phone Detection

YOLOv8 detects mobile phones and provides:

- Object class
- Bounding box
- Detection confidence

### 6. Risk Analysis

```text
Drowsiness
     +
Yawning
     +
Distraction
     +
Phone Usage
     ↓
Risk Analysis
     ↓
Driver Safety Status



