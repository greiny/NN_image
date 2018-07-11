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
	vector<Mat> trainX;
	Mat trainY = Mat::zeros(845+338, 2, CV_64FC1);

	for ( int i=1 ; i <= 845 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"target/o/o(%d).png",i);
		buf = imread(file_name,0);
		//resize(buf,buf,Size(28,28));
		//equalizeHist(buf,buf);
		//Canny(buf, buf, 80, 160, 3);
		trainX.push_back(buf);
	}

	for ( int i=1 ; i <= 338 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"target/x/x(%d).png",i);
		buf = imread(file_name,0);
		//resize(buf,buf,Size(28,28));
		//equalizeHist(buf,buf);
		//Canny(buf, buf, 80, 160, 3);
		trainX.push_back(buf);
	}

	for(int j = 0; j < 845+338; j++){
		unsigned char temp = 1;
		if (j<845) { trainY.ATD(j, 0) = (double)temp; trainY.ATD(j, 1) = 0; }
		else { trainY.ATD(j, 0) = 0; trainY.ATD(j, 1) = (double)temp;}
	}

	dataEnlarge(trainX, trainY);

	ofstream imgdata;
	imgdata.open("imgdata.csv",ios::out);
	if ( imgdata.is_open() )
	{
		for(int k = 0; k < trainX.size(); k++)
		{
			resize(trainX[k],trainX[k],Size(28,28));
			for(int j = 0; j < trainX[k].rows; j++)
				for(int i = 0; i < trainX[k].cols; i++)
					imgdata << (int)trainX[k].data[j*trainX[k].cols+i] << ",";
			for(int j = 0; j < trainY.cols; j++) imgdata << trainY.ATD(k, j) << ",";
			imgdata << endl;	
		}

	}

	imgdata.close();

	trainX.clear();
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
  
    /*
    // flip left right
    for(int i = 0; i < nSamples; i++){
        fliplr(data[i], tmp);
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
    // rotate -10 degree
    for(int i = 0; i < nSamples; i++){
        Mat tmp;
        rotateNScale(data[i], tmp, -10, 1.2);
        data.push_back(tmp);
    }
    // rotate +10 degree
    for(int i = 0; i < nSamples; i++){
        Mat tmp;
        rotateNScale(data[i], tmp, 10, 1.2);
        data.push_back(tmp);
    }
    
    // copy label matrix    
    cv::Mat tmp;
    repeat(label, 3, 1, tmp); 
    label = tmp;
}
