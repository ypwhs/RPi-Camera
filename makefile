test:test.cpp
	g++ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lpthread -lstdc++ -o test test.cpp