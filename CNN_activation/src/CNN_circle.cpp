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

int main(int argc, const char* argv[])
{
	dataReader dR;
	float neural_thres = 0.9;
	float prob = 0;
	int num = 0;

	//Layer Construction
	int sImage = 64*64;//480*320;
	int nOutput = 2;
	vector<ConvLayer> CLayer;
	vector<int> FCLayer{}; // #neuron -> FCLayer{256,64}
	bool GAP = true;
	int nInput = 1;

	ConvLayer CLayer1, CLayer2;
	CLayer1.nKernel = 3;
	CLayer1.sKernel = 15;
	CLayer1.pdim = 2;
	CLayer.push_back(CLayer1);
	CLayer2.nKernel = 5;
	CLayer2.sKernel = 9;
	CLayer2.pdim = 2;
	CLayer.push_back(CLayer2);

	for(int i=0; i<CLayer.size(); i++) nInput = nInput*CLayer[i].nKernel;
	if (GAP==false){
		int sInput = 1;
		for(int i=0; i<CLayer.size(); i++) sInput = sInput*CLayer[i].pdim;
		sInput = (int)pow((int)(sqrt(sImage)/sInput),2);
		nInput = sInput*nInput;
	}

	neuralNetwork nn(nInput,FCLayer,nOutput);
	// Loads a csv file of weight matrix data
	char* kernel_file = "log/kernel_crack.csv";
	dR.loadKernels(kernel_file,CLayer);
	char* weights_file = "log/weights2.txt";
	nn.loadWeights(weights_file);

	Mat frame;
	int k = 1;
	while(1)
	{
		stringstream open;
		open << "crack_image/crack(" << k << ").jpg";
		Mat src = imread(open.str(),0);
		if (src.empty() == true) break;

		// Create binary image from source image
		double *conv_pattern = dR.ConvNPooling(src,GAP);
		double *res = nn.feedForwardPattern(conv_pattern); // calculation results
		cout << res[0] << endl;
		k++;
	} // end of while
    return 0;
}

