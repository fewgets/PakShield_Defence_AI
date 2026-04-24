import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
from detection import detect_shoplifting  # tumhara function import ho raha hai

# FastAPI app
app = FastAPI(title="Suspicious Activity Detection API")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = r"C:\Users\dell\Desktop\Suspicious-Activity-Detection-master\output video"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}


@app.post("/detect")
async def detect_video(video: UploadFile = File(...)):
    try:
        # save input video
        input_path = os.path.join(UPLOAD_DIR, video.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # run detection (tumhara function apne aap output video bana lega)
        detect_shoplifting(input_path)

        # output path (tumhare code ke hisaab se)
        video_name = os.path.splitext(os.path.basename(video.filename))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_output.mp4")

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Output video not created")

        return FileResponse(output_path, media_type="video/mp4", filename=f"{video_name}_output.mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
