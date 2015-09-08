import cv2
import numpy as np
from matplotlib import pylab as pl

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, im = cap.read()
    cv2.imshow('raw', im)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == 32:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print contours
        cnt = contours[4]
        im = cv2.drawContours(im, [cnt], 0, (0, 255, 0), 3)
        print im
        cv2.imshow('con', im)
