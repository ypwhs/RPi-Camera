import cv2
import cv2.cv as cv

capture = cv2.VideoCapture(0)
capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv.CV_CAP_PROP_FRAME_COUNT, 100)

while True:
    _, img = capture.read()
    cv2.imshow('Video', img)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:   # ESC
        break
capture.release()
