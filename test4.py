import cv2

capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

while True:
    _, img = capture.read()
    cv2.imshow('Video', img)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:   # ESC
        break
capture.release()
