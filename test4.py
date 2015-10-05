import cv2

capture = cv2.VideoCapture(0)

while True:
    _, img = capture.read()
    cv2.imshow('Video', img)
    key = cv2.waitKey(1)
    # cv2.imwrite('a.jpg', img)
capture.release()