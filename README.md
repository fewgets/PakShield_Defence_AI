# PakShield Defence AI

An AI-powered national security and surveillance platform designed to support intelligent defence operations. PakShield Defence AI integrates multiple computer vision and deep learning modules for real-time monitoring, threat detection, anomaly identification, and strategic situational awareness.

---

## Overview

PakShield Defence AI is a comprehensive defence intelligence system built to assist in monitoring and securing critical environments. The platform leverages artificial intelligence, computer vision, and machine learning to analyze visual data from multiple sources and detect potential threats automatically.

### Core Capabilities

* Real-time surveillance and monitoring
* Border anomaly detection
* Drone detection and tracking
* Weapon detection
* Intrusion and suspicious activity identification
* AI-assisted threat analysis
* Centralized security dashboard
* Automated alert generation

---

## Key Features

### Border Anomaly Detection

Detects unusual movement and suspicious activities near restricted border zones.

### Drone Detection System

Identifies unauthorized drones in restricted airspace using deep learning models.

### Weapon Detection

Recognizes firearms and dangerous objects in surveillance footage.

### Intelligent Monitoring Dashboard

Provides a unified interface for monitoring multiple AI modules in real time.

### Automated Alerts

Instantly generates alerts when suspicious or high-risk events are detected.

---

## Technology Stack

* **Frontend:** Python CustomTkinter
* **Backend:** Python
* **Computer Vision:** OpenCV
* **Deep Learning Framework:** PyTorch / YOLO
* **Machine Learning:** NumPy, Pandas
* **Visualization:** Matplotlib
* **Database (optional):** SQLite / Supabase

---

## Project Structure

```text
PakShield_Defence_AI/
├── Backend/
│   ├── BorderAnomly/
│   │   └── drones/
│   │       └── best.pt   # Download separately
│   ├── WeaponDetection/
│   └── ...
├── Frontend/
├── Assets/
├── README.md
└── requirements.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/fewgets/PakShield_Defence_AI.git
cd PakShield_Defence_AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Model File Download

Due to GitHub's file size limitations, the trained drone detection model is not included in this repository.

### Download the Model

Download the `best.pt` file from the following Google Drive folder:

**Google Drive Link:**
[https://drive.google.com/drive/folders/1ij55ZdL8atJGVm1cakVKcxjZ3jSrcuab?usp=sharing](https://drive.google.com/drive/folders/1ij55ZdL8atJGVm1cakVKcxjZ3jSrcuab?usp=sharing)

### Placement Instructions

After downloading, place the `best.pt` file in the following directory:

```text
Backend/BorderAnomly/drones/best.pt
```

If the `drones` folder does not exist, create it manually.

---

## How to Run

```bash
python main.py
```

Or run the appropriate launcher file for your application.

---

## Use Cases

* Border surveillance and intrusion detection
* Military base perimeter security
* Restricted airspace drone monitoring
* Weapon threat detection in sensitive zones
* Smart city security and public safety
* Critical infrastructure protection

---

## Future Enhancements

* Multi-camera distributed monitoring
* Thermal imaging integration
* Facial recognition for watchlist detection
* Predictive threat analytics
* Cloud-based deployment
* Mobile monitoring application
* Real-time incident reporting

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## License

This project is intended for educational, research, and innovation purposes.

Please contact the author for commercial or defense-related deployment permissions.

---

## Author

**Usama Shahid**

* GitHub: [https://github.com/fewgets](https://github.com/fewgets)
* Email: [shaikhusama541@gmail.com](mailto:shaikhusama541@gmail.com)

---

## Important Note

This repository does not include large trained model files (`.pt`, `.pth`, etc.) due to GitHub storage restrictions.

Please download the required model files from the provided Google Drive link and place them in their designated directories before running the project.

---

## Acknowledgments

Special thanks to the open-source community and the developers of:

* PyTorch
* OpenCV
* YOLO
* CustomTkinter

Their tools and frameworks make projects like this possible.
