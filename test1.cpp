#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;

int main()
{
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	Mat frame;
	Mat frame_gray;
	std::vector<Rect> faces;
	face_cascade.load( "haarcascade_lowerbody.xml" );
	
	while(1)
	{
		cap.read(frame);
		cvtColor( frame, frame_gray, CV_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );
		// face_cascade.detectMultiScale( frame_gray, faces, 1.3, 5, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 18|9, Size(3,7));
		// face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		for( size_t i = 0; i < faces.size(); i++ )
		{
		    Rect face_i = faces[i];
			rectangle(frame,face_i,CV_RGB(255,0,0),2);
		    Mat faceROI = frame_gray( faces[i] );
		}
		imshow("当前视频",frame);
		waitKey(30);
	}
	return 0;
}