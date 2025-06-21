from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to x, y, w, h
            detections.append((bbox, conf, str(int(cls))))

        return detections
