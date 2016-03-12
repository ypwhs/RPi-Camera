import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, img = capture.read()
    gray = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # roi = img[y:y + h, x:x + w]
    cv2.imshow('Video', img)
    key = cv2.waitKey(30)
    key &= 0xFF
    if key == 27:
        break
    elif key == 32:
        cv2.imwrite('full.png', img)
        for (x, y, w, h) in faces:
            cv2.imwrite('face.png', img[y+1:y+h, x+1:x+w])
            break

capture.release()
