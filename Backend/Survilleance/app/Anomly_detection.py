from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from ultralytics import YOLO


class ObjectDetection:
    """Legacy interactive interface for restricted-area anomaly detection."""

    MODEL_PATH = Path(__file__).resolve().parent / "yolov8n.pt"
    RESTRICTED_AREA = [150, 150, 400, 350]   # [x1, y1, x2, y2]
    RESTRICTED_OBJECTS = ["person", "car", "bag", "suitcase", "knife", "pistol"]
    ALERT_COLOR = (0, 0, 255)   # Red
    NORMAL_COLOR = (245, 255, 23)  # Green
    model: YOLO | None = None

    @classmethod
    def get_model(cls) -> YOLO:
        if cls.model is None:
            cls.model = YOLO(str(cls.MODEL_PATH))
        return cls.model

    def __init__(self, mode: str = "image", path: str | None = None, cam_index: int = 0):
        """
        mode: "image", "video", "live"
        path: file path for image or video
        cam_index: camera index for live mode
        """

        self.mode = mode
        self.path = path
        self.cam_index = cam_index
        self.model = self.get_model()

        # Run automatically on initialization
        if self.mode == "image" and self.path:
            self.test_image(self.path)
        elif self.mode == "video" and self.path:
            for _ in self.test_video(self.path):
                pass  # iterate through frames
        elif self.mode == "live":
            for _ in self.test_live(self.cam_index):
                pass
        else:
            print("⚠️ Invalid mode or path not provided.")

    # -------------------------
    # DETECTION FUNCTION
    # -------------------------
    def detect_and_draw(self, frame):
        frame, _, _ = analyze_frame(frame, conf=0.5)
        return frame

    # -------------------------
    # IMAGE TEST
    # -------------------------
    def test_image(self, path):
        frame = cv2.imread(path)
        frame = self.detect_and_draw(frame)
        cv2.imshow("Image Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return frame

    # -------------------------
    # VIDEO TEST
    # -------------------------
    def test_video(self, path):
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect_and_draw(frame)
            cv2.imshow("Video Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yield frame
        cap.release()
        cv2.destroyAllWindows()

    # -------------------------
    # LIVE TEST
    # -------------------------
    def test_live(self, cam_index: int = 0):
        cap = cv2.VideoCapture(cam_index)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect_and_draw(frame)
            cv2.imshow("Live Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yield frame
        cap.release()
        cv2.destroyAllWindows()


def analyze_frame(frame, conf: float = 0.5) -> Tuple[Any, List[Dict], List[Dict]]:
    """Run detection on a frame and return the annotated frame with metadata."""

    model = ObjectDetection.get_model()
    x1, y1, x2, y2 = ObjectDetection.RESTRICTED_AREA

    detections: List[Dict] = []
    restricted_events: List[Dict] = []

    # Draw restricted area boundary
    cv2.rectangle(frame, (x1, y1), (x2, y2), ObjectDetection.ALERT_COLOR, 2)
    cv2.putText(
        frame,
        "Restricted Area",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        ObjectDetection.ALERT_COLOR,
        2,
    )

    results = model(frame, conf=conf)
    if not results:
        return frame, detections, restricted_events

    for box in results[0].boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        cls_name = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), ObjectDetection.NORMAL_COLOR, 2)
        cv2.putText(
            frame,
            f"{cls_name} {confidence:.2f}",
            (bx1, by1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            ObjectDetection.NORMAL_COLOR,
            2,
        )

        center_x, center_y = (bx1 + bx2) // 2, (by1 + by2) // 2
        restricted = (
            cls_name in ObjectDetection.RESTRICTED_OBJECTS
            and x1 <= center_x <= x2
            and y1 <= center_y <= y2
        )

        detection_record = {
            "label": cls_name,
            "confidence": round(confidence, 4),
            "bbox": [bx1, by1, bx2, by2],
            "center": [center_x, center_y],
            "restricted": restricted,
        }
        detections.append(detection_record)

        if restricted:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), ObjectDetection.ALERT_COLOR, 3)
            cv2.putText(
                frame,
                f"🚨 {cls_name} {confidence:.2f} {timestamp}",
                (bx1, by1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                ObjectDetection.ALERT_COLOR,
                2,
            )
            restricted_events.append({
                "label": cls_name,
                "confidence": round(confidence, 4),
                "bbox": [bx1, by1, bx2, by2],
                "center": [center_x, center_y],
                "timestamp": timestamp,
            })

    return frame, detections, restricted_events


def analyze_image(
    image_path: str | Path,
    output_path: str | Path | None = None,
    conf: float = 0.5,
) -> Dict:
    """Run restricted-area detection on a still image."""

    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    annotated, detections, restricted_events = analyze_frame(frame, conf=conf)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)

    return {
        "detections_count": len(detections),
        "restricted_event_count": len(restricted_events),
        "detections": detections,
        "restricted_events": restricted_events,
    }


def analyze_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    conf: float = 0.5,
    max_logged_events: int = 50,
) -> Dict:
    """Run restricted-area detection on a video and optionally persist an annotated copy."""

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
    restricted_total = 0
    sample_detections: List[Dict] = []
    restricted_events: List[Dict] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, detections, restricted = analyze_frame(frame, conf=conf)
            frames_processed += 1
            detections_total += len(detections)
            restricted_total += len(restricted)

            if detections and len(sample_detections) < max_logged_events:
                sample_detections.append({
                    "frame_index": frames_processed - 1,
                    "items": detections,
                })

            if restricted and len(restricted_events) < max_logged_events:
                restricted_events.extend(
                    {
                        **event,
                        "frame_index": frames_processed - 1,
                    }
                    for event in restricted
                )

            if writer is not None:
                writer.write(annotated)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return {
        "frames_processed": frames_processed,
        "detections_total": detections_total,
        "restricted_event_total": restricted_total,
        "sample_detections": sample_detections,
        "restricted_events": restricted_events,
    }


if __name__ == "__main__":
    # Example usages:
    obj = ObjectDetection(mode="image", path="../test_anomly_images/test.jpg")
    obj = ObjectDetection(mode="video", path="../test_anomly_videos/anomly.mp4")
    obj = ObjectDetection(mode="live", cam_index=0)
