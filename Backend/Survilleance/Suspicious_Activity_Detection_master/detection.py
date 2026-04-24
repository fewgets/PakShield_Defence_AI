import cv2
from ultralytics import YOLO

class ShopliftingDetectionBackend:
    def __init__(self, model_path="best.pt", video_path="test.mp4", output_path="detected.mp4"):
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path

        # Run the full pipeline
        self.load_model()
        self.open_video()
        self.setup_writer()
        self.run_detection()
        self.cleanup()

    def load_model(self):
        print(f"[INFO] Loading YOLO model from {self.model_path} ...")
        self.model = YOLO(self.model_path)

    def open_video(self):
        print(f"[INFO] Opening video: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception("Error: Cannot open video file!")

        # get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def setup_writer(self):
        print(f"[INFO] Preparing output file: {self.output_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

    def run_detection(self):
        print("[INFO] Starting detection...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # run YOLO detection
            results = self.model(frame)
            annotated_frame = results[0].plot()

            # save annotated frame
            self.out.write(annotated_frame)

        print("[INFO] Detection finished.")

    def cleanup(self):
        print("[INFO] Releasing resources...")
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print(f"✅ Detection complete, saved at {self.output_path}")


if __name__ == "__main__":
    ShopliftingDetectionBackend(
        model_path="C:/Users/dell/Desktop/Uraan/Suspicious_Activity_Detection_master/best.pt", 
        video_path="C:/Users/dell/Desktop/Uraan/Suspicious_Activity_Detection_master/istockphoto-1391833001-640_adpp_is.mp4",
        output_path="C:/Users/dell/Desktop/Uraan/Suspicious_Activity_Detection_master/detected.mp4"
    )
