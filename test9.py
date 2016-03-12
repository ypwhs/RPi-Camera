import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)


while True:
    rat, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    red = img[:, :, 2]
    bg = img[:, :, 0] + img[:, :, 1]
    red = red-bg
    print bg
    cv2.imshow('Red', red)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

capture.release()
