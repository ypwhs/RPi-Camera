import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

color = np.uint8([[[111, 36, 0]]])
hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
lower = np.array([0, 128, 128])
upper = np.array([5, 255, 255])
print color, lower


def click(event, x, y, flags, param):
    if event == 1:
        global color, hsv_color, lower, upper
        color = np.uint8([[frame[y, x]]])
        hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        lower = np.array([hsv_color[0][0][0] - 10, 128, 128])
        upper = np.array([hsv_color[0][0][0] + 10, 255, 255])
        print color, lower

cv2.namedWindow('raw')
cv2.setMouseCallback('raw', click)
last = False
i = 1

while True:
    global frame
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # kill the noise
    # mask = cv2.medianBlur(mask, 7)  # smooth
    # kernel = np.ones((25, 25), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill the black
    s = int(np.sum(mask)/100000)
    if (s > 0) != last:
        last = s > 0
        if last:
            cv2.imwrite("tmp/" + str(i) + ".jpg", frame)
            i += 1
            print s
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('raw', frame)
    cv2.imshow('cut', res)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
