/*******************************************************************
* Basic Feed Forward Neural Network Class
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
********************************************************************/

#ifndef NNetwork
#define NNetwork

#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <string.h>
#include "dataReader.h"

using namespace std;

class neuralNetworkTrainer;

class neuralNetwork
{
	//class members
	//--------------------------------------------------------------------------------------------
private:

	//number of neurons
	int nInput;
	int nOutput;
	int nLayer;
	vector<int> nHidden;

	//neurons
	double* inputNeurons;
	vector<double*> hiddenNeurons;
	double* outputNeurons;

	//weights
	vector <double**> wHiddenHidden; // 3D-pointer [#layer][#prev_neuron][#next_neuron]
	double** wInputHidden;
	double** wHiddenOutput;

	const char* outputFilename;
	bool loggingEnabled;
	fstream outputFile;

	//Friends
	//--------------------------------------------------------------------------------------------
	friend neuralNetworkTrainer;
	
	//public methods
	//--------------------------------------------------------------------------------------------

public:

	//constructor & destructor
	neuralNetwork(int numInput, vector<int> numHidden, int numOutput);
	~neuralNetwork();

	//weight operations
	void enableLoggingWeight(const char* filename);
	bool loadWeights(const char* inputFilename);
	bool saveWeights(double hTAcc, double hGAcc, long hEpoch);
	double* feedForwardPattern( double* pattern );
	double getSetAccuracy( std::vector<dataEntry*>& set );
	void getRegression( std::vector<dataEntry*>& set, int i );
	double getSetMSE( std::vector<dataEntry*>& set );
	int clampOutput( double x );

	//private methods
	//--------------------------------------------------------------------------------------------

private: 

	void initializeWeights();
	inline double activationFunction( double x );
	void feedForward( double* pattern );
	
};

#endif
