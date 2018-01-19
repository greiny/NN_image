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

void fliplr(const Mat &, Mat &);
void flipud(const Mat &, Mat &);
void flipudlr(const Mat &, Mat &);
void rotateNScale(const Mat &, Mat &, double, double);
void addWhiteNoise(const Mat &, Mat &, double);
void dataEnlarge(vector<Mat>&, Mat&);

int main (){
	vector<Mat> trainx, testx;
	Mat trainY = Mat::zeros(1, 845+443+328+276, CV_64FC1);

	for ( int i=1 ; i <= 845 ; i++ )
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

	for ( int i=1 ; i <= 443 ; i++ )
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

	for ( int i=1 ; i <= 276 ; i++ )
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

	for(int j = 0; j < 845+443+328+276; j++){
		unsigned char temp = 1;
		if (j<845+443) trainY.ATD(0, j) = (double)temp;
		else trainY.ATD(0, j) = 0;
	}

	dataEnlarge(trainx, trainY);

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
	}

	imgdata.close();

	trainx.clear();
	trainY.release();

	return 0;
}

void
fliplr(const Mat &_from, Mat &_to){
    flip(_from, _to, 1);
}

void
flipud(const Mat &_from, Mat &_to){
    flip(_from, _to, 0);
}

void
flipudlr(const Mat &_from, Mat &_to){
    flip(_from, _to, -1);
}

void
rotateNScale(const Mat &_from, Mat &_to, double angle, double scale){
    Point center = Point(_from.cols / 2, _from.rows / 2);
   // Get the rotation matrix with the specifications above
    Mat rot_mat = getRotationMatrix2D(center, angle, scale);
   // Rotate the warped image
    warpAffine(_from, _to, rot_mat, _to.size());
}

void
addWhiteNoise(const Mat &_from, Mat &_to, double stdev){

    _to = Mat::ones(_from.rows, _from.cols, CV_64FC1);
    randu(_to, Scalar(-1.0), Scalar(1.0));
    _to *= stdev;
    _to += _from;
    // how to make this faster?
    for(int i = 0; i < _to.rows; i++){
        for(int j = 0; j < _to.cols; j++){
            if(_to.ATD(i, j) < 0.0) _to.ATD(i, j) = 0.0;
            if(_to.ATD(i, j) > 1.0) _to.ATD(i, j) = 1.0;
        }
    }
}

void 
dataEnlarge(vector<Mat>& data, Mat& label){
    int nSamples = data.size();
    Mat tmp;
    /*
    // flip left right
    for(int i = 0; i < nSamples; i++){
        fliplr(data[i], tmp);
        data.push_back(tmp);
    }
    // flip up down
    for(int i = 0; i < nSamples; i++){
        flipud(data[i], tmp);
        data.push_back(tmp);
    }
    // flip left right up down
    for(int i = 0; i < nSamples; i++){
        flipudlr(data[i], tmp);
        data.push_back(tmp);
    }

    // add white noise
    for(int i = 0; i < nSamples; i++){
        Mat tmp;
        addWhiteNoise(data[i], tmp, 0.05);
        data.push_back(tmp);
    }
    */

    // rotate
    for(int i = 0; i < nSamples; i++){
    	for(int j = -17 ; j < 19; j++){
			rotateNScale(data[i], tmp, j*10, 1.005);
			data.push_back(tmp);
    	}
    }

    // copy label matrix    ;
    repeat(label, 1, 37, tmp);
    label = tmp;
}

