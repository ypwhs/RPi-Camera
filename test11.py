import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

color = np.uint8([[[70, 100, 130]]])
hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower = np.array([hsv_color[0][0][0] - 10, 80, 50])
upper = np.array([hsv_color[0][0][0] + 10, 255, 255])
print lower

while True:

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # kill the noise
    mask = cv2.medianBlur(mask, 7)  # smooth
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill the black
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('raw', frame)
    cv2.imshow('cut', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
