 **SafeDrive AI GitHub repository**.

````markdown
# 🚗 SafeDrive AI
## AI-Based Driver Monitoring and Safety Assistance System

SafeDrive AI is an AI-powered driver monitoring prototype designed to analyze visible driver behaviour and identify potential safety risks such as **drowsiness, yawning, distraction, and mobile phone usage**.

The system combines **computer vision, facial landmark analysis, object detection, and risk-based alert logic** to monitor the driver through a live camera feed or uploaded image/video datasets.

---

# 📌 Project Overview

Driver distraction and fatigue are important factors that can contribute to unsafe driving. SafeDrive AI aims to demonstrate how Artificial Intelligence and Computer Vision can be used to monitor visible driver behaviour and generate safety-related alerts.

The system analyzes video frames and identifies multiple indicators, including:

- 😴 Eye closure and possible drowsiness
- 🥱 Yawning behaviour
- 👀 Head movement and possible distraction
- 📱 Mobile phone detection
- ⚠️ Risk level estimation
- 🚨 Safety alerts and event logging

SafeDrive AI is developed as an **academic and research prototype** and is not intended to replace professional automotive safety systems.

---

# 🎯 Objectives

The main objectives of SafeDrive AI are:

1. To develop an AI-based driver monitoring system.
2. To analyze facial landmarks for driver behaviour monitoring.
3. To detect possible drowsiness using eye-related facial features.
4. To identify yawning using mouth landmark analysis.
5. To analyze head orientation for possible distraction.
6. To detect mobile phone usage using YOLOv8.
7. To combine multiple detected behaviours into a driver risk level.
8. To provide live monitoring through an interactive dashboard.
9. To support uploaded images, videos, and ZIP-based datasets for testing and analysis.
10. To generate alerts and maintain event information during monitoring.

---

# 🧠 System Architecture

The overall workflow of SafeDrive AI is:

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

# 🤖 AI Models and Technologies

## 1. MediaPipe Face Mesh

MediaPipe Face Mesh is used to extract facial landmarks from the driver's face.

These landmarks are used to analyze:

* Eye behaviour
* Eye closure
* Mouth opening
* Yawning indicators
* Face orientation
* Head movement
* Possible distraction

The project uses a pretrained facial landmark solution rather than training a facial landmark model from scratch.

---

## 2. YOLOv8

YOLOv8 is used for object detection.

In SafeDrive AI, it is used primarily to detect:

* 📱 Mobile phones

YOLOv8 provides object detection information such as:

* Object class
* Bounding box location
* Detection confidence

The detected phone information is passed to the monitoring and risk analysis system.

---

## 3. Landmark-Based Behaviour Analysis

Facial landmark coordinates are analyzed to derive behavioural indicators.

### Eye Analysis

Eye-related facial landmarks are used to monitor eye openness and prolonged eye closure.

Possible prolonged closure can be used as an indicator of:

> **Potential driver drowsiness**

---

### Mouth Analysis

Mouth landmarks are analyzed to determine significant mouth opening.

This can be used as an indicator of:

> **Possible yawning behaviour**

---

### Head Orientation Analysis

Facial landmarks are used to estimate the orientation of the driver's face.

If the driver's face remains significantly away from the expected forward direction, the system can identify:

> **Potential distraction**

---

# ⚙️ Algorithms and Methods Used

| Component                 | Algorithm / Method                   | Purpose                                      |
| ------------------------- | ------------------------------------ | -------------------------------------------- |
| Facial Landmark Detection | MediaPipe Face Mesh                  | Detect facial landmark points                |
| Eye Analysis              | Landmark-based geometric analysis    | Detect eye closure and drowsiness indicators |
| Mouth Analysis            | Landmark-based analysis              | Detect yawning indicators                    |
| Head Analysis             | Face orientation / landmark analysis | Detect possible distraction                  |
| Object Detection          | YOLOv8                               | Detect mobile phone usage                    |
| Image Processing          | OpenCV                               | Process images and video frames              |
| Risk Evaluation           | Risk and alert logic                 | Combine unsafe behaviour signals             |
| User Interface            | Streamlit                            | Interactive monitoring dashboard             |

---

# 🔄 How the System Works

## Step 1: Input Acquisition

The system accepts input from:

* Live webcam
* Uploaded images
* Video files
* ZIP-based datasets

---

## Step 2: Frame Processing

The input image or video is processed frame by frame using OpenCV.

Each frame is prepared for analysis by the AI modules.

---

## Step 3: Driver Face Analysis

MediaPipe Face Mesh extracts facial landmark coordinates.

The landmarks are used to analyze:

* Eyes
* Mouth
* Face orientation

---

## Step 4: Behaviour Detection

The extracted facial information is used to identify visible behavioural indicators such as:

* Eye closure
* Possible drowsiness
* Yawning
* Possible distraction

---

## Step 5: Mobile Phone Detection

YOLOv8 analyzes the frame and detects mobile phones.

The detection includes:

* Phone location
* Bounding box
* Detection confidence

---

## Step 6: Risk Analysis

The system combines the detected behavioural indicators.

For example:

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

The system can classify the driver's current state into different safety levels.

---

## Step 7: Alerts and Dashboard

The final information is displayed on the SafeDrive AI dashboard.

The dashboard can show:

* Live video feed
* Driver status
* Risk score
* Risk level
* Eye-related information
* Mouth-related information
* Head orientation
* Phone detection status
* Alerts
* Event information

---

# 📊 Dataset and Testing Support

SafeDrive AI supports offline testing using uploaded:

* Images
* Videos
* ZIP files containing supported media

The dataset processing pipeline allows the project to analyze multiple files and display results through the dashboard.

The general dataset workflow is:

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

# 🖥️ Features

### 🚗 Live Driver Monitoring

Monitor the driver using a webcam and process video frames in real time or near-real time.

### 😴 Drowsiness Monitoring

Analyze eye-related facial landmarks to identify prolonged eye closure.

### 🥱 Yawning Detection

Analyze mouth landmarks to identify possible yawning behaviour.

### 👀 Distraction Detection

Analyze facial orientation and head movement to identify possible driver distraction.

### 📱 Mobile Phone Detection

Use YOLOv8 object detection to identify mobile phones.

### ⚠️ Risk Analysis

Combine multiple detected behaviours to estimate the current driver safety level.

### 🚨 Alert System

Generate alerts when unsafe behaviour is detected.

### 📊 Interactive Dashboard

Display monitoring information using a Streamlit-based dashboard.

### 📁 Dataset Processing

Support images, videos, and ZIP-based datasets for testing.

### 📝 Event Logging

Record important detected events during the monitoring session.

---

# 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **OpenCV**
* **MediaPipe**
* **YOLOv8**
* **Ultralytics**
* **NumPy**
* **Pandas**
* **Streamlit WebRTC**

---

# 📂 Project Structure

```text
SafeDrive-AI/
│
├── app.py                  # Main Streamlit application
├── driver_monitor.py       # Facial and driver behaviour analysis
├── phone_detector.py       # YOLOv8 phone detection
├── risk_engine.py          # Risk calculation and analysis
├── alert_system.py         # Alert management
├── event_logger.py         # Event logging
│
├── assets/                 # Images and project assets
├── screenshots/            # Application screenshots
│
├── requirements.txt        # Required Python libraries
├── README.md               # Project documentation
└── .gitignore              # Ignored files and folders
```

> The exact project structure may vary depending on the files included in the repository.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/SafeDrive-AI.git
```

Navigate to the project directory:

```bash
cd SafeDrive-AI
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

### Windows

```cmd
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

After installing all dependencies, run:

```bash
python -m streamlit run app.py
```

The application will start and provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 📦 Requirements

The project may require libraries such as:

```text
streamlit
streamlit-webrtc
opencv-python
mediapipe
ultralytics
numpy
pandas
av
torch
torchvision
```

The exact versions should be defined in:

```text
requirements.txt
```

---

# 📈 Model Evaluation

SafeDrive AI is an integrated prototype containing multiple components.

Therefore, it is important not to represent the complete system using a single accuracy value without proper evaluation.

Different components require different evaluation metrics.

### Object Detection

YOLOv8 can be evaluated using:

* Precision
* Recall
* mAP@50
* mAP@50:95

### Behaviour Detection

Driver behaviour classification can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Real-Time Performance

The system can also be evaluated using:

* Frames Per Second (FPS)
* Processing latency
* Frame processing time

---

# ⚠️ Limitations

SafeDrive AI is a prototype and has several limitations:

* Performance may depend on lighting conditions.
* Face detection can be affected by occlusion.
* Extreme head angles may reduce landmark accuracy.
* Camera quality can affect detection performance.
* Mobile phones may not be detected if they are heavily occluded.
* Real-time performance depends on the available hardware.
* The system analyzes selected visible behaviours only.
* The system does not directly control a vehicle.
* The system does not guarantee accident prevention.

---

# 🔮 Future Scope

Future improvements may include:

* Improved real-time performance
* Custom-trained driver behaviour models
* Better evaluation using labelled datasets
* Improved night-time monitoring
* Infrared camera support
* Advanced gaze tracking
* Multi-driver support
* Cloud-based monitoring
* Mobile application integration
* GPS and vehicle sensor integration
* Personalized driver behaviour models
* Advanced deep-learning-based risk prediction

---

# 🎓 Academic Purpose

SafeDrive AI was developed as an academic project to explore the practical application of:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Object Detection
* Facial Landmark Analysis
* Real-Time Video Processing

The project demonstrates how multiple AI and computer vision components can be integrated into a single driver monitoring prototype.

---

# 👨‍💻 Author

**Syed Furkhan**

B.Tech – Computer Science and Engineering
Artificial Intelligence and Machine Learning

---

# 📜 Disclaimer

SafeDrive AI is an academic and experimental prototype.

It is designed to demonstrate AI-based driver monitoring using computer vision techniques. It should not be considered a certified automotive safety system and must not be relied upon as the sole mechanism for preventing accidents or ensuring driver safety.

---

## ⭐ If you found this project interesting, consider giving it a star!

```

**One thing before you upload:** make sure the filenames in the **Project Structure** section exactly match the files you actually upload. Don't leave fake filenames in the README if your project doesn't contain them.
```
