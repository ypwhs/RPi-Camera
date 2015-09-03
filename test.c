#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <stdio.h>
IplImage *pFrame = NULL;
int main()
{
	CvCapture* pCapture = cvCreateCameraCapture(-1);
	//打开摄像头
	while(1)  
	{  
		pFrame=cvQueryFrame(pCapture);

	}
}