
# 🖼️ Object Detection Projects

## 📌 Overview
This repository contains multiple **Computer Vision projects** using YOLO, OpenCV, and cvzone:
- **Model Loading & Testing** (root level scripts)
- **Webcam Detection**
- **Video File Detection**
- **Car Counter Project** (dedicated folder)

Each project demonstrates real-time object detection and tracking with bounding boxes and overlays.

---

## ⚙️ Environment Setup

### 1. Create Environment (Python 3.10)

#### Conda
```bash
conda create -n computerVission python=3.10
conda activate computerVission
```

#### venv
```bash
python3.10 -m venv computerVission
# Activate
source computerVission/bin/activate   # Linux/Mac
computerVission\Scripts\activate      # Windows
```

---

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Model Load Test
```bash
python videos/model_load.py
```

### 🔹 Webcam Detection
```bash
python videos/live_Webcam.py
```

### 🔹 Video File Detection
```bash
python videos/video_webcam.py
```

---

## 🚗 Car Counter Project

### Run Car Counter Script
```bash
python CarCounterproject/carCounter.py
```

- Plays the sample video `carcounter.mp4`  
- Detects vehicles (cars, buses, trucks, motorbikes)  
- Displays bounding boxes + live counter overlay  

👉 Press **Q** to quit video window.

---

## 📂 Project Structure
```
OBJECTDETECTION/
│── .vscode/
│   └── settings.json
│── CarCounterproject/
│   ├── carCounter.py
│   ├── carcounter.mp4
│   ├── graphics.png
│   ├── mask.png
│   ├── sort.py
│   ├── images/
│   ├── results/
│   └── videos/
│── videos/
│   ├── model_load.py
│   ├── live_Webcam.py
│   ├── video_webcam.py
│   ├── yolo26n.pt
│   └── yolo26l.pt
│── requirements.txt
│── README.md
```

---

## 🎬 Output Demo
Sample output video of the **Car Counter Project**:

```markdown
![Car Counter Demo](results\carcounter.gif)
```

---
