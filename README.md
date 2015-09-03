# 树莓派远程摄像头

## 在树莓派上安装opencv

[installopencv.sh](installopencv.sh)

``` shell
sudo apt-get update
sudo apt-get upgrade
sudo rpi-update
sudo apt-get install libcv-dev libopencv-dev python-dev python-opencv python-imaging python-pip
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图片从BGR转为RGB
    image = Image.fromarray(img)    # 将图片转为Image对象
    buf = StringIO.StringIO()   # 生成StringIO对象
    image.save(buf, format="JPEG")  # 将图片以JPEG格式写入StringIO
    jpeg = buf.getvalue()   # 获取JPEG图片内容
    buf.close()     # 关闭StringIO
    print len(jpeg)
    key = cv2.waitKey(30)
    if key == 27:   # ESC
        break
    elif key == 32:  # Space
        f = open('a.jpg', 'w')
        f.write(jpeg)
        f.close()

capture.release()

```

