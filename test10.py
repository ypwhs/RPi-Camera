import cv2
import numpy as np

capture = cv2.VideoCapture(0)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

_, bimg = capture.read()
kernel = np.ones((5, 5), np.uint8)

while True:
    ret, img = capture.read()
    img2gray = cv2.cvtColor(cv2.absdiff(bimg, img), cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 50, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask_inv = cv2.bitwise_not(mask)
    img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('Video', img)
    key = cv2.waitKey(30)
    key &= 0xFF
    if key == 27:
        break
    elif key == 32:
        ret, img = capture.read()
        bimg = img

capture.release()
