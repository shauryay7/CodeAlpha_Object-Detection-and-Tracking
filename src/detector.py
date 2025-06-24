import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="yolov8n.onnx", input_size=(640, 640), conf_threshold=0.4, iou_threshold=0.45):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        detections = []

        rows = outputs.shape[1]
        for i in range(rows):
            detection = outputs[0, i]
            conf = detection[4]
            if conf >= self.conf_threshold:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if scores[class_id] > self.conf_threshold:
                    cx, cy, w, h = detection[0:4]
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    detections.append(([x, y, int(w), int(h)], float(conf), str(class_id)))
        return detections
