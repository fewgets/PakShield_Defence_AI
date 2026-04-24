from pathlib import Path

from ultralytics import YOLO
import cv2
import winsound
import time

# Load YOLO model
_BASE_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _BASE_DIR / "model.pt"

if not _MODEL_PATH.exists():
    raise FileNotFoundError(f"Human detection model not found at {_MODEL_PATH}")

model = YOLO(str(_MODEL_PATH))  # replace with your trained model path

def detect_humans(video_path: str, conf_threshold: float = 0.6, play_alarm_flag: bool = False):
    """
    Runs YOLO detection on a video file and returns a list of detections.

    Args:
        video_path (str): Path to the video file.
        conf_threshold (float): Confidence threshold for detection.
        play_alarm_flag (bool): Whether to play alarm on human detection.

    Returns:
        List of dicts with frame_number, timestamp_sec, label, confidence.
    """
    cap = cv2.VideoCapture(video_path)
    detections = []
    frame_count = 0
    start_time = time.time()

    def play_alarm():
        duration = 500  # milliseconds
        freq = 1000     # Hz
        winsound.Beep(freq, duration)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = time.time() - start_time

        # Run YOLO detection
        results = model(frame, conf=conf_threshold, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = model.names[cls_id]

                if label.lower() in ["person", "human"]:
                    if play_alarm_flag:
                        play_alarm()
                    detections.append({
                        "frame_number": frame_count,
                        "timestamp_sec": round(timestamp, 2),
                        "label": label,
                        "confidence": round(conf, 2)
                    })

    cap.release()
    cv2.destroyAllWindows()
    return detections
