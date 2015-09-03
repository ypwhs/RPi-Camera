import cv2

capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    print img
    cv2.imshow('Video', img)
    key = cv2.waitKey(1)
    # cv2.imwrite('a.jpg', img)
capture.release()