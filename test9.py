import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while True:
    rat, img = capture.read()
    cv2.imshow('Blue', img[:, :, 0])
    cv2.imshow('Green', img[:, :, 1])
    cv2.imshow('Red', img[:, :, 2])
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

capture.release()
