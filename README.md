#树莓派远程摄像头

##在树莓派上安装opencv

[installopencv.sh](installopencv.sh)


```shell
sudo apt-get update
sudo apt-get upgrade
sudo rpi-update
sudo apt-get install libcv-dev libopencv-dev python-dev python-opencv python-imaging python-pip
sudo pip install numpy
```

##启动摄像头（python）

###排线摄像头

[test3.py](test3.py)

```python
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

###USB摄像头

[test4.py](test4.py)

```python
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

##启动摄像头（C++）

[test.cpp](test.cpp)

```cpp
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