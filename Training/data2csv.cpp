#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

#define ATD at<double>
#define elif else if

int main (){

	vector<Mat> trainx, testx;
	Mat trainY = Mat::zeros(1, 645+343+328+176, CV_64FC1);
	Mat testY = Mat::zeros(1,200+100+100, CV_64FC1);

	for ( int i=1 ; i <= 645 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"image/img(%d).png",i);
		buf = imread(file_name,0);
		resize(buf,buf,Size(28,28));
		equalizeHist(buf,buf);
		//Canny(buf, buf, 80, 160, 3);
		trainx.push_back(buf);
	}

	for ( int i=1 ; i <= 343 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"image/neural_o (%d).png",i);
		buf = imread(file_name,0);
		resize(buf,buf,Size(28,28));
		equalizeHist(buf,buf);
		//Canny(buf, buf, 80, 160, 3);
		trainx.push_back(buf);
	}

	for ( int i=1 ; i <= 328 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"image/(%d).png",i);
		buf = imread(file_name,0);
		resize(buf,buf,Size(28,28));
		equalizeHist(buf,buf);
		//Canny(buf, buf, 80, 160, 3);
		trainx.push_back(buf);
	}

	for ( int i=1 ; i <= 176 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"image/basic_x (%d).png",i);
		buf = imread(file_name,0);
		resize(buf,buf,Size(28,28));
		equalizeHist(buf,buf);
		//Canny(buf, buf, 80, 160, 3);
		trainx.push_back(buf);
	}

	for(int j = 0; j < 645+343+328+176; j++){
		unsigned char temp = 1;
		if (j<645+343) trainY.ATD(0, j) = (double)temp;
		else trainY.ATD(0, j) = 0;
	}

	for ( int i=646 ; i <= 845 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"image/img(%d).png",i);
		buf = imread(file_name,0);
		resize(buf,buf,Size(28,28));
		equalizeHist(buf,buf);
		//Canny(buf, buf, 80, 160, 3);
		testx.push_back(buf);
	}

	for ( int i=344 ; i <= 443 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"image/neural_o (%d).png",i);
		buf = imread(file_name,0);
		resize(buf,buf,Size(28,28));
		equalizeHist(buf,buf);
		//Canny(buf, buf, 80, 160, 3);
		testx.push_back(buf);
	}

	for ( int i=177 ; i <= 276 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"image/basic_x (%d).png",i);
		buf = imread(file_name,0);
		resize(buf,buf,Size(28,28));
		equalizeHist(buf,buf);
		//Canny(buf, buf, 80, 160, 3);
		testx.push_back(buf);
	}

	for(int j = 0; j < 200+100+100; j++){
		unsigned char temp = 1;
		if (j < 200+100) testY.ATD(0, j) = (double)temp;
		else testY.ATD(0, j) = 0;
	}

	ofstream imgdata;
	imgdata.open("imgdata.csv",ios::out);
	if ( imgdata.is_open() )
	{
		for(int k = 0; k < trainx.size(); k++)
		{
			for(int j = 0; j < trainx[k].rows; j++)
			{
				for(int i = 0; i < trainx[k].cols; i++) imgdata << (int)trainx[k].data[j*trainx[k].cols+i] << ",";
			}
			if (trainY.ATD(0, k) == 1) imgdata << trainY.ATD(0, k) << "," << (double)0 << endl;
			else imgdata << trainY.ATD(0, k) << "," << (double)1 << endl;
		}
		for(int k = 0; k < testx.size(); k++)
		{
			for(int j = 0; j < testx[k].rows; j++)
			{
				for(int i = 0; i < testx[k].cols; i++) imgdata << (int)testx[k].data[j*testx[k].cols+i] << ",";
			}
			if (testY.ATD(0, k) == 1) imgdata << testY.ATD(0, k) << "," << (double)0 << endl;
			else imgdata << testY.ATD(0, k) << "," << (double)1 << endl;
		}		
	}
	imgdata.close();

	trainY.release();
	testY.release();

	return 0;
}
