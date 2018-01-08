//standard libraries
#include <iostream>
#include <ctime>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>

//use standard namespace
using namespace std;
using namespace cv;

int main(){
	ofstream imgdata("imgdata.csv");
	for ( int i=1 ; i < 845 ; i++ )
	{
		char file_name[255];
		sprintf(file_name,"image/img(%d).png",i);
		Mat rgb = imread(file_name);
		resize(rgb,rgb,Size(28,28));

 		/*Mat src,mask;
		cvtColor(rgb, src, COLOR_BGR2GRAY);
		Canny(src, mask, 80, 160, 3);
		namedWindow("canny",0);
		for (int h=0; h<mask.rows; h++) for (int w=0; w<mask.cols; w++) imgdata << (int)mask.data[h*mask.cols+w]<< ",";
		*/
		Mat hsv;
		cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);
		for (int h=0; h<hsv.rows; h++)
		{
			for (int w=0; w<hsv.cols; w++)
			{
	   			Vec3b hsv_vec=hsv.at<Vec3b>(h,w);
		   		int H=hsv_vec.val[0];
				imgdata << H << ",";
			}
		}
		imgdata << 1 << "," << 0 << endl;
	}
	for ( int i=1 ; i < 276 ; i++ )
	{
		char file_name[255];
		sprintf(file_name,"image/basic_x (%d).png",i);
		Mat rgb = imread(file_name);
		resize(rgb,rgb,Size(28,28));

 		/*Mat src,mask;
		cvtColor(rgb, src, COLOR_BGR2GRAY);
		Canny(src, mask, 80, 160, 3);
		for (int h=0; h<mask.rows; h++) for (int w=0; w<mask.cols; w++) imgdata << (int)mask.data[h*mask.cols+w]<< ",";
		*/
		Mat hsv;
		cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);
		for (int h=0; h<hsv.rows; h++)
		{
			for (int w=0; w<hsv.cols; w++)
			{
	   			Vec3b hsv_vec=hsv.at<Vec3b>(h,w);
		   		int H=hsv_vec.val[0];
				imgdata << H << ",";
			}
		}
		imgdata << 0 << "," << 1 << endl;
	}
	imgdata.close();
	return 0;
}
