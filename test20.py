__author__ = 'ypw'

import cv2
import numpy as np
from matplotlib import pylab as pl


def lmt(val):
    if val > 255:
        return 255
    else:
        return val

img = cv2.imread("yy.png")
pl.subplot(231), pl.title('raw'), pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

blur = cv2.medianBlur(img, 3)
pl.subplot(232), pl.title('blur'), pl.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))

ret, bw = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
pl.subplot(233), pl.title('bw'), pl.imshow(bw)

hsv = cv2.cvtColor(bw, cv2.COLOR_BGR2HSV)
arange = 2
point = hsv[5][3]
lower = np.array([lmt(point[0] - arange), lmt(point[1] - arange), lmt(point[2] - arange)])
upper = np.array([lmt(point[0] + arange), lmt(point[1] + arange), lmt(point[2] + arange)])
mask = cv2.inRange(hsv, lower, upper)
pl.subplot(234), pl.title('hsv'), pl.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))

canny = cv2.Canny(blur, 50, 150)
pl.subplot(235), pl.title('canny'), pl.imshow(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))

pl.show()
