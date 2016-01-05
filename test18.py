__author__ = 'ypw'

import os
import cv2
import numpy as np

while True:
    os.system("screencapture screen.png")
    img = cv2.imread("screen.png")
    img = img[300:1800:2, 100:1400:2]
    color = img[200, 200]
    img[np.where((img == color).all(axis=2))] = [0, 0, 0]
    cv2.imshow("screen", img)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break
