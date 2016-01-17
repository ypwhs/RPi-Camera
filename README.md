# 树莓派远程摄像头

## 在树莓派上安装opencv

[installopencv.sh](installopencv.sh)

``` shell
sudo rpi-update
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y install libcv-dev libopencv-dev python-dev python-opencv python-imaging python-pip
sudo pip install numpy
```

## 启动摄像头（python）

### 排线摄像头

[test3.py](test3.py)

``` python
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        print image
        rawCapture.truncate(0)

```

### USB摄像头

[test4.py](test4.py)

``` python
import cv2

capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    print img
    cv2.imshow('Video', img)
    key = cv2.waitKey(1)
    # cv2.imwrite('a.jpg', img)
capture.release()

```

## 启动摄像头（C++）

[test.cpp](test.cpp)

``` cpp
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

int main()
{
	VideoCapture cap(0);
	
	Mat frame;
	while(1)
	{
		cap.read(frame);
		imshow("当前视频",frame);
		waitKey(30);
	}
	return 0;
}
```



## 简单运用opencv做图像处理

### 边缘检测

[edge.py](edge.py)

``` python
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

```



## 使用PIL库进行图像编码

PIL（Python Image Library）是一个基于python的图像处理库。

[test5.py](test5.py)

``` python
# coding: utf-8

import cv2
import cv2.cv as cv
import PIL.Image as Image
import StringIO

capture = cv2.VideoCapture(-1)
capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, img = capture.read()   # 从摄像头读取图片
    cv2.imshow('Video', img)    # 显示图片
    key = cv2.waitKey(30)&0xFF
    if key == 27:   # ESC
        break
    elif key == 32:  # Space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图片从BGR转为RGB
        image = Image.fromarray(img)    # 将图片转为Image对象
        buf = StringIO.StringIO()   # 生成StringIO对象
        image.save(buf, format="JPEG")  # 将图片以JPEG格式写入StringIO
        jpeg = buf.getvalue()   # 获取JPEG图片内容
        buf.close()     # 关闭StringIO
        print '文件大小：', len(jpeg)/1024, 'KB'
        f = open('a.jpg', 'w')
        f.write(jpeg)
        f.close()

capture.release()


```


##利用socket传输图像

[test6.py](test6.py)

```python
# coding: utf-8

import cv2
import cv2.cv as cv
import PIL.Image as Image
import StringIO
import socket
import threading
import time

capture = cv2.VideoCapture(0)
capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

frame = 0
lastframe = 0
newimg = 0


class Reader(threading.Thread):
    def __init__(self, client):
        threading.Thread.__init__(self)
        self.client = client
        global frame
        frame += 1

    def run(self):
        global newimg
        while newimg == 0:
            time.sleep(0.001)
        newimg = 0
        self.client.sendall(jpeg)
        self.client.close()


class Listener(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', 8080))
        self.sock.listen(0)

    def run(self):
        while True:
            client = self.sock.accept()
            reader = Reader(client[0])
            reader.setDaemon(True)
            reader.start()

listener = Listener()
listener.setDaemon(True)
listener.start()


def displayfps():
    while True:
        global frame, lastframe
        print "FPS:", frame - lastframe
        lastframe = frame
        time.sleep(1)

fpsthread = threading.Thread(target=displayfps)
fpsthread.setDaemon(True)
fpsthread.start()

while True:
    ret, img = capture.read()  # 从摄像头读取图片
    cv2.imshow('Video', img)  # 显示图片
    key = cv2.waitKey(30) & 0xFF
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图片从BGR转为RGB
    image = Image.fromarray(img)  # 将图片转为Image对象
    buf = StringIO.StringIO()  # 生成StringIO对象
    image.save(buf, format="JPEG")  # 将图片以JPEG格式写入StringIO
    jpeg = buf.getvalue()  # 获取JPEG图片内容
    buf.close()  # 关闭StringIO
    newimg = 1
    if key == 27:  # ESC
        break
    elif key == 32:  # Space
        print 'File Size:', len(jpeg) / 1024, 'KB'
        f = open('a.jpg', 'w')
        f.write(jpeg)
        f.close()

capture.release()

```

##利用opencv实现FaceDetect

[test7.py](test7.py)

```python
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
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi = img[y:y + h, x:x + w]
    cv2.imshow('Video', img)
    key = cv2.waitKey(30)
    key &= 0xFF
    if key == 27:
        break
    elif key == 32:
        print img

capture.release()

```

##光流跟踪（OpticalFlow）

[test8.py](test8.py)

```python 
import numpy as np
import cv2

lk_params = {'winSize': (15, 15), 'maxLevel': 2,
             'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

feature_params = {'maxCorners': 500, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7}

track_len = 40
detect_interval = 1
tracks = []
cam = cv2.VideoCapture(0)
cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
frame_idx = 0

while True:
    ret, frame = cam.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv2.circle(vis, (x, y), 3, (0, 255, 0), 1)
        tracks = new_tracks
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
        # cv2.imshow('track count: %d' % len(tracks), vis)

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray
    cv2.imshow('lk_track', vis)

    ch = 0xFF & cv2.waitKey(50)
    if ch == 27:
        break
```

##颜色追踪（HSV）

[test11.py](test11.py)

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

color = np.uint8([[[111, 36, 0]]])
hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower = np.array([hsv_color[0][0][0] - 10, 100, 80])
upper = np.array([hsv_color[0][0][0] + 10, 255, 255])
print lower

while True:

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # kill the noise
    # mask = cv2.medianBlur(mask, 7)  # smooth
    kernel = np.ones((25, 25), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill the black
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('raw', frame)
    cv2.imshow('cut', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

```

##寻找轮廓（findContours）

空格键开始寻找

[test12.py](test12.py)

```python
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, img = cap.read()
    cv2.imshow('raw', img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == 32:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        cv2.imshow("img", img)

```
##名片矫正

[test13.py](test13.py)

MORPH_CLOSE, Contours, Transform

```python
__author__ = 'ypw'

import cv2
import numpy as np
from matplotlib import pylab as pl

img = cv2.imread('m.jpg')
rawimg = img.copy()
# cv2.imshow('raw', img)
pl.subplot(221), pl.title('raw'), pl.imshow(cv2.cvtColor(img, cv2.cv.CV_BGR2RGB))

kernel = np.ones((20, 20), np.uint8)
img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # kill the words
pl.subplot(222), pl.title('Close'), pl.imshow(cv2.cvtColor(img_close, cv2.cv.CV_BGR2RGB))

img = rawimg.copy()
gray = cv2.cvtColor(img_close, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

cnt = contours[0]
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
cv2.circle(img, leftmost, 8, [255, 0, 0], -1)  # Blue
cv2.circle(img, rightmost, 8, [0, 255, 0], -1)  # Green
cv2.circle(img, topmost, 8, [0, 0, 255], -1)  # Red
cv2.circle(img, bottommost, 8, [0, 255, 255], -1)  # Yellow
pl.subplot(223), pl.title('Contours'), pl.imshow(cv2.cvtColor(img, cv2.cv.CV_BGR2RGB))
print leftmost, rightmost, topmost, bottommost


def dis(p1, p2):
    return np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

img = rawimg.copy()
width = int(dis(leftmost, topmost))
height = int(dis(leftmost, bottommost))

pts1 = np.float32([leftmost, topmost, bottommost, rightmost])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

M = cv2.getPerspectiveTransform(pts1, pts2)
print M
img = cv2.warpPerspective(img, M, (width, height))
pl.subplot(224), pl.title('Transform'), pl.imshow(cv2.cvtColor(img, cv2.cv.CV_BGR2RGB))
cv2.imwrite('n.jpg', img)

pl.show()

```

##直线检测（HoughLines）

[test15.py](test15.py)

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    print lines
    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('cut', img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

```
