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
````

---

# 🤖 AI Models and Algorithms

| Component                 | Algorithm / Model                 | Purpose                          |
| ------------------------- | --------------------------------- | -------------------------------- |
| Facial Landmark Detection | MediaPipe Face Mesh               | Detect facial landmarks          |
| Eye Analysis              | Landmark-based geometric analysis | Detect prolonged eye closure     |
| Mouth Analysis            | Landmark-based analysis           | Detect yawning indicators        |
| Head Analysis             | Face orientation analysis         | Detect possible distraction      |
| Object Detection          | YOLOv8                            | Detect mobile phone usage        |
| Image Processing          | OpenCV                            | Process images and video frames  |
| Risk Evaluation           | Risk and Alert Logic              | Combine unsafe behaviour signals |
| Dashboard                 | Streamlit                         | Display monitoring results       |

---

# ⚙️ How the System Works

### 1. Input Acquisition

The system accepts:

* Live webcam
* Uploaded images
* Video files
* ZIP-based datasets

### 2. Frame Processing

OpenCV processes images and videos frame by frame.

### 3. Driver Face Analysis

MediaPipe Face Mesh extracts facial landmarks for analyzing:

* Eyes
* Mouth
* Face orientation

### 4. Behaviour Detection

The system identifies:

* Possible drowsiness
* Yawning
* Possible distraction

### 5. Mobile Phone Detection

YOLOv8 detects mobile phones and provides:

* Object class
* Bounding box
* Detection confidence

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
```

### 7. Dashboard and Alerts

The dashboard displays:

* Live video feed
* Driver status
* Risk score and risk level
* Phone detection status
* Alerts
* Event information

---

# 📊 Dataset and Testing

SafeDrive AI supports:

* Images
* Videos
* ZIP-based datasets

```text
Dataset / ZIP Upload
        ↓
File Extraction
        ↓
Image / Video Detection
        ↓
Frame Processing
        ↓
MediaPipe + YOLOv8 Analysis
        ↓
Behaviour Detection
        ↓
Risk Analysis
        ↓
Results Dashboard
```

---

# ✨ Key Features

* 🚗 Live Driver Monitoring
* 😴 Drowsiness Monitoring
* 🥱 Yawning Detection
* 👀 Distraction Detection
* 📱 Mobile Phone Detection
* ⚠️ Risk Analysis
* 🚨 Alert System
* 📊 Interactive Dashboard
* 📁 Dataset Processing
* 📝 Event Logging

---

# 🛠️ Technologies Used

* Python
* Streamlit
* Streamlit WebRTC
* OpenCV
* MediaPipe
* YOLOv8
* Ultralytics
* NumPy
* Pandas
* PyTorch
* Torchvision
* AV

---

# 📂 Project Structure

```text
SafeDrive-AI/
│
├── app.py
├── driver_monitor.py
├── phone_detector.py
├── risk_engine.py
├── alert_system.py
├── event_logger.py
│
├── assets/
├── screenshots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/SafeDrive-AI.git
cd SafeDrive-AI
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```cmd
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 📈 Model Evaluation

Different components can be evaluated using:

### YOLOv8 Object Detection

* Precision
* Recall
* mAP@50
* mAP@50:95

### Driver Behaviour Detection

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Real-Time Performance

* FPS
* Processing latency
* Frame processing time

---

# ⚠️ Limitations

* Performance may depend on lighting conditions.
* Face detection can be affected by occlusion.
* Extreme head angles may reduce landmark reliability.
* Camera quality can affect detection performance.
* Mobile phones may not be detected if heavily occluded.
* Real-time performance depends on available hardware.
* This system does not directly control a vehicle.
* The system does not guarantee accident prevention.

---

# 🔮 Future Scope

* Improved real-time performance
* Custom-trained driver behaviour models
* Better evaluation using labelled datasets
* Night-time monitoring
* Infrared camera support
* Advanced gaze tracking
* Mobile application integration
* GPS and vehicle sensor integration
* Personalized driver behaviour models
* Advanced deep-learning-based risk prediction

---

# 🎓 Academic Purpose

SafeDrive AI was developed as an academic project to explore practical applications of:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Object Detection
* Facial Landmark Analysis
* Real-Time Video Processing

---

# 👨‍💻 Author

**Syed Furkhan**

B.Tech – Computer Science and Engineering
 (Artificial Intelligence and Machine Learning)

---

# 📜 Disclaimer

SafeDrive AI is an academic and experimental prototype designed to demonstrate AI-based driver monitoring using computer vision techniques. It should not be considered a certified automotive safety system and must not be relied upon as the sole mechanism for preventing accidents or ensuring driver safety.

---

⭐ **If you found this project interesting, consider giving it a star!**

```
```
