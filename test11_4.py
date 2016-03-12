__author__ = 'ypw'
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

lasts = 0
last = 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray.copy()
    mask[mask != 255] = 0
    s = int(np.sum(mask)/100000)
    delta = s - last
    last = s
    if abs(delta) > 1:
        print delta
    cv2.imshow('raw', frame)
    cv2.imshow('cut', mask)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

