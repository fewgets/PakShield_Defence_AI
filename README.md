<!-- PakShield Defence AI · README by Usama Shahid (fewgets) -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,55:0F3D3E,100:36FFC7&height=230&section=header&text=PakShield%20Defence%20AI&fontSize=52&fontColor=FFFFFF&fontAlignY=36&desc=AI%20cyber%20defence%20and%20surveillance%20%C2%B7%20Uraan%20Pakistan%20Hackathon&descSize=17&descAlignY=58&animation=fadeIn" width="100%" alt="PakShield Defence AI"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=700&color=36FFC7&center=true&vCenter=true&width=720&height=60&lines=Threat+Intelligence+and+Intrusion+Detection;Weapon%2C+Face+and+Anomaly+Detection;Drone+and+Border+Surveillance" alt="Typing SVG"/>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLO"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
</p>

<p>
  <a href="https://pak-shield-ai.vercel.app"><b>🌐 Live Demo</b></a> · <a href="https://github.com/fewgets/PakShieldAI-Website"><b>🖥️ Web Dashboard</b></a>
</p>

</div>

---
## 📌 Overview

**PakShield Defence AI** is an AI-powered **cyber-defence and surveillance platform** built for the
**Uraan Pakistan National Hackathon (Phase 2)**. It brings three intelligence modules together behind
one **FastAPI** backend:

| | Module | What it protects against |
|:-:|---|---|
| 🔍 | **AI Threat Intelligence** | Phishing emails and network intrusions |
| 🛡️ | **Border Anomaly Detection** | Unauthorised drones, human intrusion, suspicious movement |
| 📹 | **Video Surveillance AI** | Weapons, known faces on a watch-list, abnormal activity |

A companion **[Next.js web dashboard](https://github.com/fewgets/PakShieldAI-Website)** gives operators
a single command centre for all modules.

> 🌐 **Live dashboard:** [pak-shield-ai.vercel.app](https://pak-shield-ai.vercel.app)

---

## 🏗️ Architecture

```mermaid
flowchart TB
    UI[🖥️ Web dashboard<br/>Next.js] -->|REST| API[⚙️ FastAPI · api.py · :8000]

    subgraph TI[🔍 AI Threat Intelligence]
        IDS[Network IDS<br/>ML classifier on traffic CSVs]
        PH[Phishing detector<br/>TF-IDF + ML]
    end
    subgraph BA[🛡️ Border Anomaly Detection]
        DR[Drone detection<br/>YOLO]
        HU[Human detection<br/>YOLO]
        SA[Suspicious activity<br/>video analysis]
    end
    subgraph SV[📹 Video Surveillance]
        WP[Weapon detection]
        FR[Face recognition<br/>watch-list matching]
        AN[Anomaly detection]
    end

    API --> TI
    API --> BA
    API --> SV
```

---

## ✨ Capabilities

### 🔍 AI Threat Intelligence
- **Network Intrusion Detection (IDS):** upload network-traffic CSVs and get each flow classified as normal or attack, using a trained model with its scaler and label encoder.
- **Phishing Email Detection:** a TF-IDF + machine-learning classifier flags phishing emails.
- Training notebooks are included: `Network_Intrusion_Detection.ipynb` and `Phishing_Model_Training.ipynb`.

### 🛡️ Border Anomaly Detection
- **Drone detection:** a YOLO model spots drones (and tells them apart from birds) in images and video.
- **Human detection:** flags people entering restricted border zones.
- **Suspicious activity:** analyses video for unusual movement patterns and saves annotated output.

### 📹 Video Surveillance AI
- **Weapon detection:** recognises firearms in images and video.
- **Face recognition:** matches faces against a watch-list of known people.
- **Anomaly detection:** detects abnormal behaviour in surveillance footage.

Every detection is saved with an **annotated image or video** and a **JSON log**, ready for review.

---

## 🔌 API Reference

| Method | Endpoint | Module |
|---|---|---|
| `GET` | `/health` | Service health |
| `GET` | `/email-classify` | Phishing email classification |
| `POST` | `/ids-predict` | Network intrusion detection (CSV upload) |
| `POST` | `/border/drones/detect` | Drone detection |
| `POST` | `/border/humans/detect` | Human intrusion detection |
| `POST` | `/border/suspicious/detect` | Suspicious activity in video |
| `POST` | `/surveillance/weapon/detect` | Weapon detection |
| `POST` | `/surveillance/face/recognize` | Face recognition |
| `POST` | `/surveillance/anomaly/detect` | Anomaly detection |
| `GET` | `/border/files/{category}/{filename}` | Download annotated results |

Interactive docs: **http://localhost:8000/docs**

---

## 🚀 Run Locally

```bash
git clone https://github.com/fewgets/PakShield_Defence_AI.git
cd PakShield_Defence_AI

python -m venv venv
venv\Scripts\activate            # macOS / Linux: source venv/bin/activate
```

Install the dependencies for the modules you need:

```bash
pip install -r Backend/BorderAnomly/requirements.txt
pip install -r Backend/Survilleance/requirements.txt
pip install fastapi uvicorn python-multipart scikit-learn pandas
```

**Download the drone model.** `best.pt` is too large for GitHub:
📥 [Google Drive folder](https://drive.google.com/drive/folders/1ij55ZdL8atJGVm1cakVKcxjZ3jSrcuab?usp=sharing) → place it at `Backend/BorderAnomly/drones/best.pt`.

Start the API:

```bash
python api.py          # serves on http://localhost:8000
```

Then run the **[web dashboard](https://github.com/fewgets/PakShieldAI-Website)** and point it at `http://localhost:8000`.

---

## 📁 Project Structure

```text
PakShield_Defence_AI/
├── api.py                         # FastAPI gateway for all modules
├── Backend/
│   ├── AIThreatIntelligence/      # IDS + phishing models, training notebooks
│   ├── BorderAnomly/              # drones/ · HUMAN_DETECTION/ · suspicious activity
│   └── Survilleance/              # weapon · face recognition · anomaly detection
└── Frontend/                      # Web dashboard (see PakShieldAI-Website)
```

---

## 🧰 Tech Stack

| Category | Technologies |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Computer Vision | OpenCV, YOLO (Ultralytics), PyTorch |
| Machine Learning | scikit-learn, Pandas, NumPy, TF-IDF |
| Frontend | Next.js 15, React 19, Tailwind CSS ([separate repo](https://github.com/fewgets/PakShieldAI-Website)) |

---

## 🎯 Use Cases

🛂 Border surveillance · 🪖 Military base perimeter security · 🚁 Restricted airspace monitoring ·
🔫 Weapon threat detection · 🏙️ Smart-city public safety · 🏭 Critical infrastructure protection

## 🔭 Roadmap

- [ ] Multi-camera distributed monitoring
- [ ] Thermal imaging and ANPR (number-plate recognition)
- [ ] Predictive threat analytics
- [ ] Mobile alerts for field teams

> ⚠️ Built for research, education and innovation. Contact me before any commercial or operational defence deployment.

---

## 🤝 Let's Work Together

<div align="center">

**Have an idea, a freelance project, or want to collaborate on something in AI?**
I'm always open to collaborations, freelance work and full-time opportunities.

<a href="mailto:shaikhusama541@gmail.com?subject=Collaboration%20-%20PakShield%20Defence%20AI">
  <img src="https://img.shields.io/badge/Email-shaikhusama541%40gmail.com-36FFC7?style=for-the-badge&logo=gmail&logoColor=white&labelColor=0D1117" alt="Email"/>
</a>
<a href="https://www.linkedin.com/in/-usamashahid/">
  <img src="https://img.shields.io/badge/LinkedIn-Usama%20Shahid-36FFC7?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=0D1117" alt="LinkedIn"/>
</a>
<a href="https://github.com/fewgets">
  <img src="https://img.shields.io/badge/GitHub-fewgets-36FFC7?style=for-the-badge&logo=github&logoColor=white&labelColor=0D1117" alt="GitHub"/>
</a>

<br/><br/>

⭐ **If you like this project, please give it a star. It really helps!** ⭐

<br/>

<h3>"Defending a nation starts with seeing threats before they arrive."</h3>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:36FFC7,55:0F3D3E,100:0D1117&height=150&section=footer&animation=twinkling" width="100%" alt="footer"/>

</div>
