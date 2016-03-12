__author__ = 'ypw'

import cv2
import numpy as np

img = cv2.imread("face/cat.jpg")
# 0:228,39
# 1:235,157
# 2:293,91
roi = img[39:157, 228:293]

pts1 = np.float32([[228, 39], [235, 157], [293, 91], [293, 91]])
pts2 = np.float32([[0, 0], [0, 200], [100, 50], [100, 50]])

M = cv2.getPerspectiveTransform(pts1, pts2)
print M
# roi = cv2.warpPerspective(roi, M, (70, 110))

cv2.namedWindow('raw')

while True:
    cv2.imshow('raw', roi)
    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
