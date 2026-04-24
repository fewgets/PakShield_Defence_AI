from fastapi import FastAPI, UploadFile, File
import os

from drones.detector import detect_drones
from HUMAN_DETECTION.detector import detect_humans
from Suspicious_Activity_Detection_master.detection import detect_shoplifting

app = FastAPI(title="URAAN API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"msg": "URAAN API is running 🚀"}

@app.post("/detect/drone")
async def drone_detection(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = detect_drones(file_path)
    return {"detections": result}

@app.post("/detect/human")
async def human_detection(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = detect_humans(file_path, conf_threshold=0.6, play_alarm_flag=False)
    return {"detections": result}

@app.post("/detect/suspicious")
async def suspicious_detection(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    output = detect_shoplifting(file_path)
    return {"output_video": output}
