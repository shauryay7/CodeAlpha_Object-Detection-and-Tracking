from detector import ObjectDetector
from tracker import ObjectTracker
import cv2
import threading
import time

class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# Initialize
detector = ObjectDetector("../models/yolov8n.onnx")
tracker = ObjectTracker()
stream = VideoStream(1, width=640, height=480)

time.sleep(1)
frame_count = 0
process_every_n_frames = 2

while True:
    ret, frame = stream.read()
    if not ret:
        break

    if frame_count % process_every_n_frames == 0:
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        latest_tracks = tracks
    frame_count += 1

    for track in latest_tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("ONNX Object Detection & Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()
