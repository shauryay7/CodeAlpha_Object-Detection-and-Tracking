from detector import ObjectDetector
from tracker import ObjectTracker
import cv2
import os

# Initialize
detector = ObjectDetector("../models/yolov8n.pt")
tracker = ObjectTracker()

cap = cv2.VideoCapture(0)  # Or use a file like "../videos/input.mp4"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Object Detection and Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
