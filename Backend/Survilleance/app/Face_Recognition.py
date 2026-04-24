from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# ------------------------------
# GLOBAL MODEL LOADING (1 Dafa)
# ------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

# Models load once
GLOBAL_MTCNN = MTCNN(keep_all=True, device=DEVICE)
GLOBAL_RESNET = InceptionResnetV1(pretrained='casia-webface').eval().to(DEVICE)

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_KNOWN_FACES_DIR = MODULE_DIR.parent / "known_faces"


class FaceRecognitionSystem:
    def __init__(
        self,
        known_faces_folder: str | Path = DEFAULT_KNOWN_FACES_DIR,
        image_path: str | None = None,
        video_path: str | None = None,
        cam_index: int | None = None,
    ) -> None:
        # Use globally loaded models
        self.mtcnn = GLOBAL_MTCNN
        self.resnet = GLOBAL_RESNET
        self.device = DEVICE
        self.face_db: Dict[str, np.ndarray] = {}
        self.latest_detections: List[Dict[str, Any]] = []
        self.known_faces_folder = Path(known_faces_folder).resolve()

        # Step 1: Register known faces
        self.register_faces(self.known_faces_folder)

        # Step 2: Run tests automatically
        if image_path:
            self.test_image(image_path)

        if video_path:
            self.test_video(video_path)

        if cam_index is not None:
            self.test_live(cam_index)

    # ------------------------------
    # Register Faces
    # ------------------------------
    def register_faces(self, folder: str | Path | None = None) -> None:
        folder_path = Path(folder) if folder is not None else self.known_faces_folder

        if not folder_path.exists():
            print(f"[ERROR] Folder '{folder_path}' not found!")
            self.face_db = {}
            return

        self.face_db = {}
        for file_path in folder_path.iterdir():
            if not file_path.is_file():
                continue

            img = cv2.imread(str(file_path))

            if img is None:
                print(f"[ERROR] Could not read image: {file_path}")
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.mtcnn(rgb)

            if faces is not None:
                emb = self.resnet(faces[0].unsqueeze(0).to(self.device)).detach().cpu().numpy()
                name = file_path.stem
                self.face_db[name] = emb
                print(f"[INFO] Registered {name}")
            else:
                print(f"[WARNING] No face found in {file_path.name}")

    # ------------------------------
    # Recognition Function
    # ------------------------------
    def recognize(self, frame, distance_threshold: float = 0.9):
        self.latest_detections = []

        if frame is None:
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(rgb)
        faces = self.mtcnn(rgb)

        if faces is None or boxes is None:
            return frame

        embeddings = self.resnet(faces.to(self.device)).detach().cpu().numpy()

        for (box, emb) in zip(boxes, embeddings):
            if box is None:
                continue

            x1, y1, x2, y2 = [int(b) for b in box]

            min_dist, identity = float("inf"), "Unknown"
            for name, db_emb in self.face_db.items():
                dist = np.linalg.norm(emb - db_emb)
                if dist < min_dist:
                    min_dist, identity = dist, name

            authorized = bool(min_dist < distance_threshold)
            if authorized:
                label = f"{identity} - Access Granted"
                color = (0, 255, 0)
                print(f"[GRANTED] {identity} recognized. Distance={min_dist:.2f}")
            else:
                label = "Access Denied"
                color = (0, 0, 255)
                print(f"[DENIED] Unknown person detected. Distance={min_dist:.2f}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            self.latest_detections.append({
                "label": identity if authorized else "Unknown",
                "distance": round(float(min_dist), 4) if min_dist != float("inf") else None,
                "authorized": authorized,
                "bbox": [x1, y1, x2, y2],
            })

        return frame

    # ------------------------------
    # Image Test
    # ------------------------------
    def test_image(self, path):
        img = cv2.imread(path)
        frame = self.recognize(img)
        cv2.imshow("Image Test", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return frame

    # ------------------------------
    # Video Test
    # ------------------------------
    def test_video(self, path):
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.recognize(frame)
            cv2.imshow("Video Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------------
    # Live Test
    # ------------------------------
    def test_live(self, source=0):
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.recognize(frame)
            cv2.imshow("Live Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


FACE_SYSTEM_CACHE: Dict[Path, FaceRecognitionSystem] = {}


def get_face_system(known_faces_folder: str | Path | None = None) -> FaceRecognitionSystem:
    folder = Path(known_faces_folder) if known_faces_folder is not None else DEFAULT_KNOWN_FACES_DIR
    folder = folder.resolve()

    system = FACE_SYSTEM_CACHE.get(folder)
    if system is None:
        system = FaceRecognitionSystem(known_faces_folder=folder)
        FACE_SYSTEM_CACHE[folder] = system
    return system


def recognize_image(
    image_path: str | Path,
    output_path: str | Path | None = None,
    known_faces_folder: str | Path | None = None,
    distance_threshold: float = 0.9,
) -> Dict[str, Any]:
    system = get_face_system(known_faces_folder)
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    annotated = system.recognize(frame, distance_threshold=distance_threshold)
    detections = [det.copy() for det in getattr(system, "latest_detections", [])]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)

    authorized = [det for det in detections if det.get("authorized")]

    return {
        "detections_count": len(detections),
        "authorized_count": len(authorized),
        "detections": detections,
    }


def recognize_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    known_faces_folder: str | Path | None = None,
    distance_threshold: float = 0.9,
    max_logged_events: int = 50,
) -> Dict[str, Any]:
    system = get_face_system(known_faces_folder)
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
    authorized_events: List[Dict[str, Any]] = []
    sample_detections: List[Dict[str, Any]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = system.recognize(frame, distance_threshold=distance_threshold)
            detections = [det.copy() for det in getattr(system, "latest_detections", [])]
            frames_processed += 1
            detections_total += len(detections)

            authorized = [det for det in detections if det.get("authorized")]
            if authorized and len(authorized_events) < max_logged_events:
                for det in authorized:
                    enriched = det.copy()
                    enriched.setdefault("frame_index", frames_processed - 1)
                    authorized_events.append(enriched)

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
        "authorized_events_total": len(authorized_events),
        "authorized_events": authorized_events,
        "sample_detections": sample_detections,
    }


# ------------------------------
# Usage
# ------------------------------
if __name__ == "__main__":
    obj = FaceRecognitionSystem(
        known_faces_folder="../known_faces",
        image_path="../test_Face_images/images.jpeg",
        #video_path="../test_video.mp4",
        cam_index=0
    )
