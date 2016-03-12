__author__ = 'ypw'

import cv2
import numpy as np
from matplotlib import pylab as pl
import time

eye = cv2.imread("face/eye2.png", -1)
nose = cv2.imread("face/nose1.png", -1)
mouth = cv2.imread("face/mouth1.png", -1)
eye = cv2.resize(eye, None, fx=0.18, fy=0.16)
nose = cv2.resize(nose, None, fx=0.14, fy=0.14)
# mouth = cv2.resize(mouth, None, fx=0.18, fy=0.12)


def draw(y, x, img):
    mask = img[:, :, 3]
    mask_inv = cv2.bitwise_not(mask)
    roi = face[y:y + img.shape[0], x:x + img.shape[1]]
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_bg = cv2.bitwise_and(img, img, mask=mask)
    dst = cv2.add(img1_bg, img2_bg)
    face[y:y + img.shape[0], x:x + img.shape[1]] = dst


def seteye(y, x):
    draw(y, 160 - x - eye.shape[1], eye)
    draw(y, 160 + x, eye)

i = 0
while i < 50:
    face = np.zeros((240, 320, 4), np.uint8)
    face[:] = 255
    mouth2 = cv2.resize(mouth, None, fx=0.18+i/100.0*0.18, fy=0.12+i/200.0*0.12, interpolation=cv2.INTER_CUBIC)
    seteye(10, 20)
    # draw(80, 160 - nose.shape[1] / 2, nose)
    draw(160, 160 - mouth2.shape[1] / 2, mouth2)
    cv2.imshow("Face", face)
    cv2.waitKey(100)
    i += 1


# cv2.imwrite("face/face.png", face)


# pl.imshow(face)
# pl.show()

