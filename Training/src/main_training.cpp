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

	int nKernel = 3;
	int sKernel = 7;
	int pdim = 2;
	
	//Training condition
	int sImage = 28*28;
	int nInput = (int)pow((int)sqrt(sImage)/pdim,2)*nKernel;
	vector<int> nLayer{64};
	int nOutput = 2;

	int max_epoch = 1000000;
	double accuracy = 99.9;
	double max_time = 300;

	float lr = 0.001;
	float momentum = 0.9;

	//create data set reader and load data file
	dataReader d;
	float tRatio = 0.7;
	float vRatio = 0.3;
	d.loadKernels("kernel3.csv",sKernel,nKernel);
	d.loadImageFile4Train("image_data/imgdata.csv",sImage,nOutput,tRatio,vRatio,sKernel,nKernel,pdim);
	d.setCreationApproach( STATIC, 10 );

	//create neural network
	neuralNetwork nn(nInput,nLayer,nOutput);

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
		logTrain << "#Input" << "," << nInput << endl;
		logTrain << "#Layer" << "," << nLayer.size() << endl;
		for (int i=0; i<nLayer.size(); i++) logTrain << "#Neuron["<< i << "]," << nLayer[i] << endl;
		logTrain << "#Output" << "," << nOutput << endl;
		logTrain << "#TrainingSet" << "," << (int)d.trainingDataEndIndex << endl;
		logTrain << "#ValidationSet" << "," << (int)((d.trainingDataEndIndex/tRatio)*vRatio) << endl;
		logTrain << "Accuracy(%)" << "," << accuracy << endl;
		logTrain << "Learning_rate" << "," << lr << endl;
		logTrain << "Momentum" << "," << momentum << endl;
		logTrain.close();
	}

	ostringstream log_name;
	log_name << "log/log" << file_no << ".csv";
	const char* log_name_char = new char[log_name.str().length()+1];
	log_name_char = log_name.str().c_str();
	nT.enableLogging(log_name_char, 1);

	//train neural network on data sets
	for (int i=0; i < d.getNumTrainingSets(); i++ )
	{
		nT.trainNetwork( d.getTrainingDataSet() );
	}

	ostringstream w_name;
	w_name << "log/weights" << file_no << ".txt";
	const string& buf = w_name.str();
	const char* w_name_char = buf.c_str();
	nn.saveWeights(w_name_char); // ((#input)*(#neuron)+(#neuron)*(#output)--Weight)+((#neuron+#output)--Bias)
	cout <<w_name_char << endl;
	cout << endl << endl << "-- END OF PROGRAM --" << endl;
}
