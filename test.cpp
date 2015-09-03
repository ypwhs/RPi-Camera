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