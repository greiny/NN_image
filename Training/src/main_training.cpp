/*******************************************************************
* Neural Network Training Example
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
*********************************************************************/
//standard libraries
#include <iostream>
#include <ctime>
#include <sstream>
#include <fstream>

//custom includes
#include "modules/neuralNetwork.h"
#include "modules/neuralNetworkTrainer.h"

//use standard namespace
using namespace std;

int main()
{		
	//seed random number generator
	srand( (unsigned int) time(0) );
	
	//Training condition
	int max_epoch = 10000000;
	double accuracy = 99.9;
	double max_time = 3000;
	float lr = 0.001;
	float momentum = 0.9;
	float tRatio = 0.7;
	float vRatio = 0.3;

	//Layer Construction
	int sImage = 28*28;
	int nOutput = 2;
	vector<ConvLayer> CLayer;
	ConvLayer CLayer1, CLayer2;
		CLayer1.nKernel = 3;
		CLayer1.sKernel = 7;
		CLayer1.pdim = 2;
		CLayer.push_back(CLayer1);
		CLayer2.nKernel = 5;
		CLayer2.sKernel = 3;
		CLayer2.pdim = 2;
		CLayer.push_back(CLayer2);
	vector<int> FCLayer{}; // #neuron -> FCLayer{256,64}
	bool GAP = true;

	int nInput = 1;
	for(int i=0; i<CLayer.size(); i++) nInput = nInput*CLayer[i].nKernel;
	if (GAP==false){
		int sInput = 1;
		for(int i=0; i<CLayer.size(); i++) sInput = sInput*CLayer[i].pdim;
		sInput = (int)pow((int)(sqrt(sImage)/sInput),2);
		nInput = sInput*nInput;
	}

	//create data set reader and load data file
	dataReader d;
	d.loadKernels("7(3)_3(5).csv",CLayer);
	d.loadImageFile4Train("image_data/imgdata2.csv",sImage,nOutput,tRatio,vRatio,GAP);
	d.setCreationApproach( STATIC, 10 );

	//create neural network
	neuralNetwork nn(nInput,FCLayer,nOutput);

	//save the Training Condition
	char file_name[255];
	int file_no = 0;
	sprintf(file_name,"log/condition%d.csv",file_no);
	for ( int i=0 ; i < 100 ; i++ )
	{
		ifstream test(file_name);
		if (!test) break;
		file_no++;
		sprintf(file_name,"log/condition%d.csv",file_no);
	}

	//create neural network trainer and save log
	neuralNetworkTrainer nT( &nn );
	nT.setTrainingParameters(lr, momentum, false);
	nT.setStoppingConditions(max_epoch, accuracy, max_time);

	ofstream logTrain;
	logTrain.open(file_name,ios::out);
	if ( logTrain.is_open() )
	{
		logTrain << "Input" << "," << nInput << endl;
		logTrain << "ConvLayer" << "," << CLayer.size() << endl;
		for (int i=0; i<CLayer.size(); i++)
			logTrain << "Kernel["<< i << "]," << CLayer[i].nKernel << "," << CLayer[i].sKernel << "x" << CLayer[i].sKernel << ","<< CLayer[i].pdim << endl;
		logTrain << "FCLayer" << "," << FCLayer.size() << endl;
		for (int i=0; i<FCLayer.size(); i++) logTrain << "Neuron["<< i << "]," << FCLayer[i] << endl;
		logTrain << "Output" << "," << nOutput << endl;
		logTrain << "TrainingSet" << "," << (int)d.trainingDataEndIndex << endl;
		logTrain << "ValidationSet" << "," << (int)((d.trainingDataEndIndex/tRatio)*vRatio) << endl;
		logTrain << "Accuracy(%)" << "," << accuracy << endl;
		logTrain << "Learning_rate" << "," << lr << endl;
		logTrain << "Momentum" << "," << momentum << endl;
		logTrain.close();
	}

	ostringstream log_name;
	log_name << "log/log" << file_no << ".csv";
	const char* log_name_char = new char[log_name.str().length()+1];
	log_name_char = log_name.str().c_str();
	nT.enableLogging(log_name_char, 10);

	char w_name[255];
	sprintf(w_name,"log/weights%d.txt",file_no);
	nn.enableLoggingWeight(w_name); // ((#input)*(#neuron)+(#neuron)*(#output)--Weight)+((#neuron+#output)--Bias)

	//train neural network on data sets
	for (int i=0; i < d.getNumTrainingSets(); i++ )
	{
		nT.trainNetwork( d.getTrainingDataSet() );
	}

	CLayer.clear();
	//print success
	cout << endl << "Neuron weights saved to '" << w_name << "'" << endl;
	cout << endl << endl << "-- END OF PROGRAM --" << endl;
}
