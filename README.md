# 🚧 Road AI – Pothole Detection and Road Monitoring System

**An AI-powered road monitoring application that detects potholes, tracks road damage, evaluates road health, and helps identify appropriate road repair actions.**

## 📌 Overview

Road AI is a computer vision-based road monitoring system built using Python, YOLO11, OpenCV, and Streamlit.

The application detects potholes from uploaded images and videos, assigns unique IDs to detected potholes, classifies their severity, and evaluates the overall road condition.

It also generates a Road Damage Heatmap to visualize the spatial distribution of detected potholes and provides repair recommendations based on road damage assessment.

## ✨ Features

- **AI Pothole Detection:** Detect potholes using a custom-trained YOLO11 model.
- **Image Detection:** Upload road images and identify potholes with bounding boxes.
- **Video Detection:** Analyze road videos and detect potholes across video frames.
- **Unique Pothole Tracking:** Assign unique IDs to tracked potholes to support more accurate counting.
- **Severity Classification:** Categorize potholes into Small, Medium, and Large.
- **Color-Coded Bounding Boxes:**
  - 🟢 Green – Small potholes
  - 🟡 Yellow – Medium potholes
  - 🔴 Red – Large potholes
- **Road Health Assessment:** Evaluate road conditions as GOOD, MODERATE, or BAD.
- **Road Damage Heatmap:** Visualize areas with concentrated detected road damage.
- **Repair Recommendations:** Provide guidance for minor maintenance, patch repair, or full repair assessment.
- **Dashboard Analytics:** Display detection statistics and road condition information.
- **Download Results:** Save processed detection outputs.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO11 | Pothole detection |
| OpenCV | Image and video processing |
| Streamlit | Interactive web application |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Plotly | Data visualization |

## 📂 Project Structure

```text
Pothole-Detection-System/
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   └── style.css
│
├── models/
│   └── best.pt
│
├── outputs/
│   └── .gitkeep
│
├── utils/
│   ├── detection.py
│   ├── heatmap.py
│   ├── road_health.py
│   ├── severity.py
│   └── video_processing.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ASWINJ1710/Pothole-Detection-Road-Monitoring-System.git
```

### 2. Navigate to the project directory

```bash
cd Pothole-Detection-Road-Monitoring-System
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🧠 Model Information

Road AI uses a custom-trained YOLO11 model for pothole detection.

- Model file: `models/best.pt`
- Framework: Ultralytics
- Model architecture: YOLO11
- Detection target: Road potholes

The trained model is included in the repository.

## 🚦 Road Health Assessment

The system evaluates road conditions using detected pothole severity and damage information.

| Road Condition | Description |
|---|---|
| 🟢 GOOD | Relatively low detected road damage |
| 🟡 MODERATE | Noticeable road damage requiring maintenance |
| 🔴 BAD | Significant road damage requiring repair assessment |

### Repair Recommendations

- Minor Maintenance – For relatively small damage.
- Patch Repair – For moderate pothole damage.
- Full Repair Assessment – For roads with significant damage.

*Note: These are AI-assisted estimates based on image/video detections, not certified engineering or structural road assessments.*

## 🔥 Road Damage Heatmap

The Road Damage Heatmap provides a visual representation of where detected potholes are concentrated.

It helps users identify damage-prone areas within an analyzed road image.

## 🎯 Project Objectives

- Automate pothole detection using deep learning.
- Reduce manual road inspection efforts.
- Improve road damage monitoring.
- Provide visual road condition assessment.
- Support maintenance and repair planning.

## 🔮 Future Improvements

- GPS-based pothole location tracking.
- Geographic road damage mapping.
- Real-time camera-based detection.
- Improved road damage severity estimation.
- Road condition monitoring reports.
- Integration with municipal road maintenance systems.

## 👨‍💻 Author

**ASWIN J**

GitHub: [ASWINJ1710](https://github.com/ASWINJ1710)

## 📜 License

This project is intended for educational, research, and demonstration purposes.
