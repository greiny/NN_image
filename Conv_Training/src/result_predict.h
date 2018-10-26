#pragma once
#include "general_settings.h"
#include <fstream>

using namespace std;
using namespace cv;

Mat resultPredict(const vector<Mat> &, const vector<Cvl> &, const vector<Fcl> &, const Smr &);
void testNetwork(const vector<Mat> &, const Mat&, const vector<Cvl> &, const vector<Fcl> &, const Smr &);
void predictLogging(const char* filename);
