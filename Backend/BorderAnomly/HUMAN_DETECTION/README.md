---
license: agpl-3.0
library: ultralytics
tags:
- object-detection
- pytorch
- ultralytics
- roboflow-universe
- human-detection
- yolov8
---
# Human Detection using Thermal Camera

## Use Case

This model is can be used for detecting humans from thermal images. This should work on both Pseudo-color and Grayscale thermal images. The model was fine tuned for humans only but can be finetuned further fort detecting other objects using Thermal images. 

To deploy this model use the following code:

- Install dependencies:
```bash
$ python -m pip install ultralytics supervision huggingface_hub 
```

- Python code
```python
# import libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import cv

# download model
model_path = hf_hub_download(
    repo_id = "pitangent-ds/YOLOv8-human-detection-thermal",
    filename = "model.pt"
)

# load model
model = YOLO(model_path)

# method for inference
def inference(image_path):
    cv_image = cv.imread(image_path, cv2.IMREAD_ANYCOLOR)
    model_output = model(cv_image, conf=0.6, verbose=False)
    detections = Detections.from_ultralytics(model_output[0])
    return detections
```

## Training Code

- Dataset Link: [Roboflow Universe](https://universe.roboflow.com/smart2/persondection-61bc2)

```python
from ultralytics import YOLO
import torch

# load model
model = YOLO("yolov8n.pt")

# hyper parameters
hyperparams = {
    "batch": 32,
    "epochs": 30,
    "imgsz": [640, 480],
    "optimizer": "AdamW",
    "cos_lr": True,
    "lr0": 3e-5,
    "warmup_epochs": 10
}

# start training
model.train(
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    data = "data.yaml",
    **hyperparams
)
```

- Click here for: [Training Arguments](./training_artifacts/args.yaml)

## Libraries

```yaml
python: 3.10.13
ultralytics: 8.0.206
torch: "2.1.0+cu118"
roboflow: 1.1.9
```