import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, img = cap.read()
    cv2.imshow('raw', img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == 32:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
        corners = np.int0(corners)
        cv2.drawContours(img, corners, -1, (0, 0, 255), 3)
        cv2.imshow("img", img)
