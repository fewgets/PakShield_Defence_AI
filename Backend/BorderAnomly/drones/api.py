import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO

# FastAPI app
app = FastAPI(title="YOLO Image Detection API")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# apna YOLO model load karo (path apna rakhna)
model = YOLO("best.pt")

@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}

@app.post("/detect-image/")
async def detect_image(file: UploadFile = File(...)):
    try:
        # save input image
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # run detection
        results = model(input_path)

        # output path
        output_path = os.path.join(OUTPUT_DIR, f"detected_{file.filename}")
        results[0].save(filename=output_path)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Output image not created")

        return FileResponse(output_path, media_type="image/jpeg", filename=f"detected_{file.filename}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
