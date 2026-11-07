
from ultralytics import YOLO
import json

DEVICE = "cpu"
# DETECTOR_MODEL = "yolo26n.pt"
with open("detector_models/models.json", "r") as f:
    models = json.load(f)

class Detector:
    model = None
    results = list()

    def __init__(self, detector_type):
        if detector_type:
            if detector_type == "person":
                self.model = self.load_detector_model(detector_type)

    def load_results(self, frame, conf):
        model = self.get_model()
        if model:
            self.results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[0],          # COCO class 0 = person
                        conf=conf,
                        device=DEVICE,
                        verbose=False,
                    )

            return self.get_result()

        return None

    def load_detector_model(self, detector_type):
        if detector_type == "person":
            return YOLO(models["person"])


    def get_model(self):
        return self.model

    def get_result(self):
        return self.results[0] if self.results.__len__() > 0 else None

    def get_annotated_result(self):
        result = self.get_result()
        if result:
            return result.plot()
        else:
            return None

    def get_detections(self, frame, conf=0.35):
        detections = []
        # result = self.get_result()
        result = self.load_results(frame, conf)
        
        if result.boxes is None or result.boxes.id is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()

        height, width = frame.shape[:2]

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            # Reject tiny detections that are unlikely to yield
            # useful appearance features.
            if x2 - x1 < 32 or y2 - y1 < 64:
                continue

            detections.append({
                "track_id": int(track_id),
                "box": (x1, y1, x2, y2),
                "crop": frame[y1:y2, x1:x2].copy(),
            })
    
        return detections