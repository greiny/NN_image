#pragma once
#include "general_settings.h"
#include <fstream>

using namespace std;
using namespace cv;
void getNetworkCost(vector<Mat>&, Mat&, vector<Cvl>&, vector<Fcl>&, Smr&);
void getNetworkCost(vector<Mat>&, Mat&, vector<Cvl>&, vector<Fcl>&, Smr&, int, int);
void costLogging(const char* filename);
