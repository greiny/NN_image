#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;
///////////////////////////////////
// Network Layer Structures
///////////////////////////////////
typedef struct ConvKernel{
    Mat W;
    double b;
    Mat Wgrad;
    double bgrad;
    Mat Wd2;
    double bd2;
    double lr_b;
    double lr_w;
}ConvK;

typedef struct ConvLayer{
    vector<ConvK> layer;
}Cvl;

typedef struct FullConnectLayer{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
    Mat Wd2;
    Mat bd2;
    double lr_b;
    double lr_w;
}Fcl;

typedef struct SoftmaxRegession{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
    double cost;
    Mat Wd2;
    Mat bd2;
    double lr_b;
    double lr_w;
}Smr;

///////////////////////////////////
// Config Structures
///////////////////////////////////
struct ConvLayerConfig {
    int KernelSize;
    int KernelAmount;
    double WeightDecay;
    int PoolingDim;
    bool useLRN; //LocalResponseNormalization
    ConvLayerConfig(int a, int b, double c, int d, bool e) : KernelSize(a), KernelAmount(b), WeightDecay(c), PoolingDim(d) , useLRN(e){}
};

struct FullConnectLayerConfig {
    int NumHiddenNeurons;
    double WeightDecay;
    double DropoutRate;
    FullConnectLayerConfig(int a, double b, double c) : NumHiddenNeurons(a), WeightDecay(b), DropoutRate(c) {}
};

struct SoftmaxLayerConfig {
    int NumClasses;
    double WeightDecay;
};
