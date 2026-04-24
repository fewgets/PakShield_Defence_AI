import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
from ultralytics import YOLO


def _resolve_model_path() -> str:
  """Return the first existing best.pt path for the drone detector."""

  search_paths = [
    Path(__file__).with_name("best.pt"),
    Path(__file__).resolve().parent.parent / "best.pt",
  ]

  for candidate in search_paths:
    if candidate.is_file():
      return str(candidate)

  raise FileNotFoundError(
    "Drone detection model best.pt not found. Checked: "
    + ", ".join(str(path) for path in search_paths)
  )


MODEL_PATH = _resolve_model_path()
model = YOLO(MODEL_PATH)


def detect_drones(
  image_path: str,
  output_path: Optional[str] = None,
) -> Union[List[Dict], Dict[str, Optional[Union[str, List[Dict]]]]]:
  """Run drone detection and optionally persist an annotated image."""

  results = model(image_path)
  detections: List[Dict] = []
  annotated_frame = None

  for r in results:
    if annotated_frame is None:
      annotated_frame = r.plot()  # returns BGR frame with drawn predictions

    for box in r.boxes:
      coords = box.xyxy[0].tolist()
      cls_id = int(box.cls[0].item())
      conf = float(box.conf[0].item())
      label = model.names[cls_id]
      detections.append(
        {
          "label": label,
          "confidence": round(conf, 2),
          "bbox": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
        }
      )

  if output_path:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if annotated_frame is None:
      # No detections were drawn; persist the original frame so the frontend still receives an output.
      original_frame = cv2.imread(image_path)
      if original_frame is None:
        annotated_frame = None
      else:
        annotated_frame = original_frame
    if annotated_frame is not None:
      cv2.imwrite(output_path, annotated_frame)

  if output_path:
    return {"detections": detections, "annotated_path": output_path if annotated_frame is not None else None}

  return detections
