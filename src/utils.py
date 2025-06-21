import cv2

def draw_text(frame, text, pos, color=(255, 255, 255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
