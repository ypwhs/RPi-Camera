__author__ = 'ypw'
import cv2
import numpy as np
from matplotlib import pyplot as plt

capture = cv2.VideoCapture(0)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, img = capture.read()
    cv2.imshow('Video', img)
    key = cv2.waitKey(30)
    key &= 0xFF
    if key == 27:
        break
    elif key == 32:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

capture.release()
