import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while True:
    flag, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 500, 2000, apertureSize=5)
    # vis = img.copy()
    # vis[edge != 0] = (0, 255, 0)
    cv2.imshow('edge', edge)
    ch = cv2.waitKey(5)
    if ch == 27:
        break
cv2.destroyAllWindows()
