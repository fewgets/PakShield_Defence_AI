from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from ultralytics import YOLO

# ------------------------------
# Global Model Load (lazy)
# ------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "best.pt"
FALLBACK_MODEL_PATH = Path(__file__).resolve().parent / "yolov8n.pt"
_MODEL: YOLO | None = None
WEAPON_KEYWORDS = ("weapon", "gun", "pistol", "knife", "rifle", "shotgun")


def get_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = YOLO(str(MODEL_PATH))
            print(f"✅ Weapon detection model loaded successfully: {MODEL_PATH}")
        except Exception as exc:  # pragma: no cover - depends on environment
            print(f"❌ Error loading weapon detection model: {exc}")
            print("⚠️ Falling back to YOLOv8n...")
            _MODEL = YOLO(str(FALLBACK_MODEL_PATH))
    return _MODEL


class ObjectDetection:
    def __init__(self, mode="image", path=None, cam_index=0, conf_thresh=0.3):
        """
        mode: "image", "video", "live"
        path: image or video path
        cam_index: for live mode
        """

        self.model = get_model()  # global model use ho raha hai
        self.mode = mode
        self.path = path
        self.cam_index = cam_index
        self.conf_thresh = conf_thresh

        # Auto run based on mode
        if self.mode == "image" and self.path:
            self.test_image(self.path)
        elif self.mode == "video" and self.path:
            self.test_video(self.path)
        elif self.mode == "live":
            self.test_live(self.cam_index)
        else:
            print("⚠️ Invalid mode or missing path")

    # ------------------------------
    # Detection Function
    # ------------------------------
    def detect_and_draw(self, frame):
        frame, _ = detect_frame(frame, conf_thresh=self.conf_thresh)
        return frame

    # ------------------------------
    # Image Mode
    # ------------------------------
    def test_image(self, path):
        frame = cv2.imread(path)
        if frame is None:
            print(f"❌ Could not load image {path}")
            return
        frame = self.detect_and_draw(frame)
        cv2.imshow("Weapon Detection - Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ------------------------------
    # Video Mode
    # ------------------------------
    def test_video(self, path):
        print(f"🎬 Processing video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"❌ Could not open video {path}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect_and_draw(frame)
            cv2.imshow("Weapon Detection - Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ------------------------------
    # Live Camera Mode
    # ------------------------------
    def test_live(self, cam_index=0):
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"❌ Could not open camera {cam_index}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect_and_draw(frame)
            cv2.imshow("Weapon Detection - Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# ------------------------------
# Reusable helpers for API integration
# ------------------------------
def detect_frame(frame, conf_thresh: float = 0.3) -> Tuple[Any, List[Dict]]:
    """Annotate a frame with weapon detections and return metadata."""

    model = get_model()
    detections: List[Dict] = []

    results = model(frame, conf=conf_thresh)
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = model.names[class_id]
            confidence = float(conf)

            is_weapon = any(keyword in class_name.lower() for keyword in WEAPON_KEYWORDS)

            bbox_color = (0, 0, 255) if is_weapon else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            label_prefix = "🚨" if is_weapon else ""
            cv2.putText(
                frame,
                f"{label_prefix} {class_name} {confidence:.2f}".strip(),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                bbox_color,
                2,
            )

            detection_record = {
                "label": class_name,
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2],
                "is_weapon": is_weapon,
            }

            if is_weapon:
                detection_record["timestamp"] = datetime.datetime.now().strftime("%H:%M:%S")

            detections.append(detection_record)

    return frame, detections


def analyze_image(
    image_path: str | Path,
    output_path: str | Path | None = None,
    conf_thresh: float = 0.3,
) -> Dict:
    """Run weapon detection on a still image."""

    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    annotated, detections = detect_frame(frame, conf_thresh=conf_thresh)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)

    return {
        "detections_count": len(detections),
        "weapon_alerts": [det for det in detections if det.get("is_weapon")],
        "detections": detections,
    }


def analyze_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    conf_thresh: float = 0.3,
    max_logged_events: int = 50,
) -> Dict:
    """Run weapon detection on a video and optionally persist an annotated copy."""

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    writer = None
    if output_path is not None and width > 0 and height > 0:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )

    frames_processed = 0
    detections_total = 0
    weapon_events: List[Dict] = []
    sample_detections: List[Dict] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, detections = detect_frame(frame, conf_thresh=conf_thresh)
            frames_processed += 1
            detections_total += len(detections)

            weapons = [det for det in detections if det.get("is_weapon")]
            if weapons and len(weapon_events) < max_logged_events:
                for event in weapons:
                    enriched = event.copy()
                    enriched.setdefault("frame_index", frames_processed - 1)
                    weapon_events.append(enriched)

            if detections and len(sample_detections) < max_logged_events:
                sample_detections.append({
                    "frame_index": frames_processed - 1,
                    "items": detections,
                })

            if writer is not None:
                writer.write(annotated)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return {
        "frames_processed": frames_processed,
        "detections_total": detections_total,
        "weapon_event_total": len(weapon_events),
        "weapon_events": weapon_events,
        "sample_detections": sample_detections,
    }


# ------------------------------
# Usage
# ------------------------------
if __name__ == "__main__":
    # Run image
    obj = ObjectDetection(mode="image", path="../weapon_test_images/pistollll.png")
    # Run video (uncomment to test)
    obj = ObjectDetection(mode="video", path="../weapon_video_testing/pistolrec.mp4")

    # Run live (uncomment to test)
    obj = ObjectDetection(mode="live", cam_index=0)