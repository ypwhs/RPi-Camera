__author__ = 'ypw'

import cv2
import numpy as np
from matplotlib import pylab as pl

img = cv2.imread('m.jpg')
rawimg = img.copy()
# cv2.imshow('raw', img)
pl.subplot(221), pl.title('raw'), pl.imshow(cv2.cvtColor(img, cv2.cv.CV_BGR2RGB))

kernel = np.ones((20, 20), np.uint8)
img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # kill the words
pl.subplot(222), pl.title('Close'), pl.imshow(cv2.cvtColor(img_close, cv2.cv.CV_BGR2RGB))

img = rawimg.copy()
gray = cv2.cvtColor(img_close, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

cnt = contours[0]
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
cv2.circle(img, leftmost, 8, [255, 0, 0], -1)  # Blue
cv2.circle(img, rightmost, 8, [0, 255, 0], -1)  # Green
cv2.circle(img, topmost, 8, [0, 0, 255], -1)  # Red
cv2.circle(img, bottommost, 8, [0, 255, 255], -1)  # Yellow
pl.subplot(223), pl.title('Contours'), pl.imshow(cv2.cvtColor(img, cv2.cv.CV_BGR2RGB))
print leftmost, rightmost, topmost, bottommost


def dis(p1, p2):
    return np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

img = rawimg.copy()
width = int(dis(leftmost, topmost))
height = int(dis(leftmost, bottommost))

pts1 = np.float32([leftmost, topmost, bottommost, rightmost])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

M = cv2.getPerspectiveTransform(pts1, pts2)
print M
img = cv2.warpPerspective(img, M, (width, height))
pl.subplot(224), pl.title('Transform'), pl.imshow(cv2.cvtColor(img, cv2.cv.CV_BGR2RGB))
cv2.imwrite('n.jpg', img)

pl.show()
