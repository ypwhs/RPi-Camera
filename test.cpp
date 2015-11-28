#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <cstdio>

using namespace cv;

int main()
{
	VideoCapture cap(0);
	if(!cap.isOpened())  // check if we succeeded
		return -1;
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	Mat frame;
	while(1)
	{
		cap.read(frame);
		Mat hsv, mask, res;
		cvtColor(frame, hsv, COLOR_BGR2HSV);
		inRange(hsv, Scalar(110, 100, 75), Scalar(115, 255, 255), mask);
		bitwise_or(frame, frame, res, mask);
		imshow("Video", res);

		int key = waitKey(30);
		if((key&0xFF) == 27)break;
	}
	return 0;
}