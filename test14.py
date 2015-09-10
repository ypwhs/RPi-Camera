__author__ = 'ypw'

import cv2
import numpy as np
from matplotlib import pylab as pl

img = cv2.imread('n.jpg')
rawimg = img.copy()

pl.subplot(231), pl.title('raw'), pl.imshow(cv2.cvtColor(img, cv2.cv.CV_BGR2RGB))

hsv = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
lower = np.array([0, 0, 0])
upper = np.array([255, 127, 127])
mask = cv2.inRange(hsv, lower, upper)
# pl.subplot(232), pl.title('HSV'), pl.imshow(cv2.cvtColor(mask, cv2.cv.CV_GRAY2RGB))

kernel = np.ones((10, 40), np.uint8)
img_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # kill the words
kernel = np.ones((5, 5), np.uint8)
img_close = cv2.morphologyEx(img_close, cv2.MORPH_DILATE, kernel)
# pl.subplot(233), pl.title('OPEN'), pl.imshow(cv2.cvtColor(img_close, cv2.cv.CV_GRAY2RGB))

img = cv2.bitwise_not(rawimg)
img = cv2.bitwise_and(img, cv2.cvtColor(img_close, cv2.cv.CV_GRAY2RGB))
pl.subplot(232), pl.title('mask'), pl.imshow(cv2.cvtColor(img, cv2.cv.CV_BGR2RGB))
cv2.imwrite('o.png', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pl.subplot(233), pl.title('gray'), pl.imshow(cv2.cvtColor(gray, cv2.cv.CV_GRAY2RGB))
cv2.imwrite('p.png', gray)

# img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
# img = cv2.bitwise_not(img)
# pl.subplot(236), pl.title('bw'), pl.imshow(cv2.cvtColor(img, cv2.cv.CV_GRAY2RGB))
# cv2.imwrite('q.png', img)

img = cv2.bitwise_not(gray)
pl.subplot(234), pl.title('not'), pl.imshow(img, cmap='gray')

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist[255] = 0
pl.subplot(235), pl.title('hist'), pl.plot(hist)
print hist


pl.show()
