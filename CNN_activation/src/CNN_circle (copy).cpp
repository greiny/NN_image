#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <pthread.h>
#include <cstdio>
#include <chrono>
#include <unistd.h>

#include <stdlib.h>
#include <ctime>
#include <vector>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//custom includes
#include "modules/neuralNetwork.h"
#include "modules/DisplayManyImages.h"

using namespace std;
using namespace cv;

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define _USE_MATH_DEFINES
#define pi       3.14159265358979323846

bool Feedforward(Mat mask);
void saveTemp(Mat frame, Rect roi);

int main(int argc, const char* argv[])
{

	// constants of Canny
	double canny_thes = 70;

	// constans of Gaussian blur
	Size ksize = Size(5, 5);
	double sigma1 = 2;
	double sigma2 = 2;

	dataReader dR;
	float neural_thres = 0.9;
	float prob = 0;
	int num = 0;

	VideoCapture capture("out2.avi");

	//Layer Construction
	int sImage = 28*28;
	int nOutput = 2;
	vector<ConvLayer> CLayer;
	vector<int> FCLayer{64}; // #neuron -> FCLayer{256,64}
	bool GAP = false;
	int nInput = 1;

	ConvLayer CLayer1, CLayer2;
	CLayer1.nKernel = 4;
	CLayer1.sKernel = 7;
	CLayer1.pdim = 2;
	CLayer.push_back(CLayer1);

	for(int i=0; i<CLayer.size(); i++) nInput = nInput*CLayer[i].nKernel;
	if (GAP==false){
		int sInput = 1;
		for(int i=0; i<CLayer.size(); i++) sInput = sInput*CLayer[i].pdim;
		sInput = (int)pow((int)(sqrt(sImage)/sInput),2);
		nInput = sInput*nInput;
	}

	neuralNetwork nn(nInput,FCLayer,nOutput);
	// Loads a csv file of weight matrix data
	char* kernel_file = "log/kernel4.csv";
	dR.loadKernels(kernel_file,CLayer);

	// Loads a csv file of weight matrix data
	char* weights_file = "log/weights37.txt";
	nn.loadWeights(weights_file);

	bool flag=1;
	Mat frame;

	// Time checking start
	int frames = 0;
	float time = 0, fps = 0;
	auto t0 = std::chrono::high_resolution_clock::now();

	while(1)
	{
		if (flag==1)
		{
			capture >> frame; 
			if (frame.empty()) break;
			flag = 0;
		}
		else
		{
			Mat src, mask;
			//Rect rec(0,0,frame.cols/2, frame.rows);
			//frame = frame(rec);
			//resize(frame,frame,Size(), 0.5, 0.5);

			cvtColor(frame, src, COLOR_BGR2GRAY);

			//cvtColor(src, mask, COLOR_GRAY2BGR);
			double *conv_pattern = dR.ConvNPooling(src,GAP);
			//check for FPS(Frame Per Second)
			auto t11 = std::chrono::high_resolution_clock::now();
			float count = std::chrono::duration<float>(t11-t0).count();
			// limit fps
			if (count < (1/30)) usleep(((1/30)-count)*1000000);

			auto t1 = std::chrono::high_resolution_clock::now();
			time += std::chrono::duration<float>(t1-t0).count();
			t0 = t1;
			++frames;
			if (time > 0.5f) 
			{
				fps = frames / time;
				frames = 0;
				time = 0;
			}

			Mat blank(Size(10,frame.rows),frame.type(),Scalar::all(0));
			Mat matDst(Size(frame.cols*3+blank.cols*2,frame.rows),frame.type(),Scalar::all(0));
			//hconcat(frame, blank, matDst);

			//imshow( "Targets", matDst );
			//video << matDst;
			frame.release();
			flag = 1;
			if(waitKey(10)==27)  break;
		} // end of else after capture
	} // end of while
    return 0;
}


bool Feedforward(Mat mask)
{
	Mat crop;
	resize(mask,crop,Size(28,28));
	double *res; 	
	Mat result(34,71,CV_64FC1);
	//double *conv_pattern = dR.ConvNPooling(mask,CLayer,GAP);

	/*for(int j = 0; j < 71; j++){
		for(int i = 0; i < 34; i++){
			res = nn.feedForwardPattern(conv_pattern[i*71+j]);
			result.data[i*71+j] = res[0];
		}
	}
	result = 0.8<result;
	result = result*255;
	//cvtColor(result, result, COLOR_GRAY2BGR);
	//video<<result;
	//imshow("result",result);
	//cout << res[0] << ", " << res[1] <<endl;
	prob = (float)res[0];
	//if (neural_thres < res[0] && res[0] > res[1]) return true;
	if (res[0] > res[1]) return true;
	else */
	return false;
}

void saveTemp(Mat frame, Rect roi)
{
	Mat crop = frame(roi);
	std::ostringstream name;
	//name << "data/img#" << num << ".png";
	imwrite(name.str(), crop);
	//num++;
}

