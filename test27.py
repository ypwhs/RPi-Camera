import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

img1 = cv2.imread('face.png', 0)  # queryImage
# img1 = cv2.resize(img1, None, fx=0.5, fy=0.5)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)

bf = cv2.BFMatcher()

while True:
    ret, img = capture.read()
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray, None)
    if des2 is None:
        continue
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img3 = cv2.drawMatchesKnn(img1, kp1, gray, kp2, good, img1, flags=2)

    cv2.imshow('Video', img3)
    key = cv2.waitKey(30)
    key &= 0xFF
    if key == 27:
        break
    elif key == 32:
        cv2.imwrite('SIFT.png', img3)

capture.release()
