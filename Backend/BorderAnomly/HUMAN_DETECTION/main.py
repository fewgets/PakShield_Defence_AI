from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import winsound
import csv
import time
import os

# Initialize FastAPI
app = FastAPI()

# Load YOLO model once (not every request)
model = YOLO("model.pt")

def play_alarm():
    """Play a short beep when a person is detected."""
    duration = 500  # milliseconds
    freq = 1000     # Hz
    winsound.Beep(freq, duration)

@app.get("/")
def root():
    return {"msg": "FastAPI is working 🎉"}

@app.post("/detect_video/")
async def detect_video(file: UploadFile = File(...)):
    # Save uploaded video temporarily
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Prepare CSV file for logging
    log_filename = f"detections_{file.filename}.csv"
    log_file = open(log_filename, mode="w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["frame_number", "timestamp_sec", "label", "confidence"])

    frame_count = 0
    start_time = time.time()
    human_detected_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = time.time() - start_time

        # Run YOLO detection
        results = model(frame, conf=0.6, verbose=False)

        # Check detections
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())   # class id
            conf = float(box.conf[0].item())  # confidence
            label = model.names[cls_id]       # class name (like "person")

            # Log to CSV
            csv_writer.writerow([frame_count, round(timestamp, 2), label, round(conf, 2)])

            # Trigger alarm only if class is "person"
            if label.lower() == "person":   # ⚠️ note: YOLO calls it "person", not "human"
                play_alarm()
                human_detected_frames += 1

    cap.release()
    log_file.close()
    os.remove(video_path)  # delete uploaded video after processing

    return {
        "filename": file.filename,
        "total_frames": frame_count,
        "frames_with_humans": human_detected_frames,
        "log_file": log_filename
    }
