//standard includes
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <string.h>

//include definition file
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"

#include "neuralNetwork.h"

using namespace std;
using namespace cv;

/*******************************************************************
* Constructor
********************************************************************/
neuralNetwork::neuralNetwork(int nInput, vector<int> nHidden, int nOutput) : nInput(nInput), nHidden(nHidden), nOutput(nOutput), nLayer(nHidden.size())
{
	int nLayer = nHidden.size();

	//create neuron lists
	inputNeurons = new( double[nInput + 1] );
	for ( int i=0; i < nInput; i++ ) inputNeurons[i] = 0;
	inputNeurons[nInput] = -1;

	outputNeurons = new( double[nOutput] );
	for ( int i=0; i < nOutput; i++ ) outputNeurons[i] = 0;

	for (int k=0; k<nLayer; k++)
	{
		hiddenNeurons.push_back(new(double[nHidden[k]+1])); // number of HiddenNeuron and bias
		for ( int i=0; i < nHidden[k]; i++ ) hiddenNeurons[k][i] = 0;
		hiddenNeurons[k][nHidden[k]] = -1; // order of bias
	}

	//create weight lists (include bias neuron weights)
	wInputHidden = new( double*[nInput + 1] );
	for ( int i=0; i <= nInput; i++ )
	{
		wInputHidden[i] = new (double[nHidden[0]]);
		for ( int j=0; j < nHidden[0]; j++ ) wInputHidden[i][j] = 0;
	}

	if (nLayer>1)
	{
		for (int k=0; k<nLayer-1; k++)
		{
			wHiddenHidden.push_back(new( double*[nHidden[k] + 1])); // hidden(k)-hidden(k+1)
			for ( int i=0; i <= nHidden[k]; i++ )
			{
				wHiddenHidden[k][i] = new (double[nHidden[k+1]]);
				for ( int j=0; j < nHidden[k+1]; j++ ) wHiddenHidden[k][i][j] = 0;
			}
		}
	}
	wHiddenOutput = new( double*[nHidden[nLayer-1] + 1] );
	for ( int i=0; i <= nHidden[nLayer-1]; i++ )
	{
		wHiddenOutput[i] = new (double[nOutput]);
		for ( int j=0; j < nOutput; j++ ) wHiddenOutput[i][j] = 0;
	}
	//initialize weights
	//--------------------------------------------------------------------------------------------------------
	initializeWeights();
}

/*******************************************************************
* Destructor
********************************************************************/
neuralNetwork::~neuralNetwork()
{
	//delete neurons
	delete[] inputNeurons;
	delete[] outputNeurons;
	for (int k=0; k<nLayer; k++) delete[] hiddenNeurons[k];

	//delete weight storage
	for (int i=0; i <= nInput; i++) delete[] wInputHidden[i];
	delete[] wInputHidden;
	for (int j=0; j <= nHidden[nLayer-1]; j++) delete[] wHiddenOutput[j];
	delete[] wHiddenOutput;
	if (nLayer>1) for (int k=0; k<nLayer-1; k++) for (int i=0; i<=nHidden[k]; i++) delete[] wHiddenHidden[k][i];
}

/*******************************************************************
* Load Neuron Weights
********************************************************************/
bool neuralNetwork::loadWeights(const char* filename)
{
	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);

	if ( inputFile.is_open() )
	{
		vector<double> weights;
		string line = "";
		
		//read data
		while ( !inputFile.eof() )
		{
			getline(inputFile, line);				
			
			//process line
			if (line.length() > 2 ) 
			{				
				//store inputs		
				char* cstr = new char[line.size()+1];
				char* t;
				strcpy(cstr, line.c_str());

				//tokenise
				int i = 0;
				t=strtok (cstr,",");
				
				while ( t!=NULL )
				{	
					weights.push_back( atof(t) );
				
					//move token onwards
					t = strtok(NULL,",");
					i++;			
				}

				//free memory
				delete[] cstr;
			}
		}

		//check if sufficient weights were loaded
		int num_weight = (nInput + 1) * nHidden[0] + (nHidden[nLayer-1] +  1) * nOutput;
		if (nLayer>1) for (int k=0; k<nLayer-1; k++) num_weight += (nHidden[k] +  1) * nHidden[k+1];
		if ( weights.size() != num_weight )
		{
			cout << endl << "Error - Incorrect number of weights in input file: " << filename << endl;
			//close file
			inputFile.close();

			return false;
		}
		else
		{
			//set weights
			int pos = 0;

			for ( int i=0; i <= nInput; i++ )
			{
				for ( int j=0; j < nHidden[0]; j++ ) wInputHidden[i][j] = weights[pos++];
			}
			if (nLayer>1)
			{
				for (int k=0; k<nLayer-1; k++)
				{
					for ( int i=0; i <= nHidden[k]; i++ )
					{
						for ( int j=0; j < nHidden[k+1]; j++ ) wHiddenHidden[k][i][j] = weights[pos++];
					}
				}
			}
			for ( int i=0; i <= nHidden[nLayer-1]; i++ )
			{
				for ( int j=0; j < nOutput; j++ ) wHiddenOutput[i][j] = weights[pos++];
			}

			//print success
			cout << endl << "Neuron weights loaded successfully from '" << filename << "'" << endl;

			//close file
			inputFile.close();
			
			return true;
		}		
	}
	else 
	{
		cout << endl << "Error - Weight input file '" << filename << "' could not be opened: " << endl;
		return false;
	}
}
/*******************************************************************
* Save Neuron Weights
********************************************************************/
bool neuralNetwork::saveWeights(const char* filename)
{
	//open file for reading
	fstream outputFile;
	outputFile.open(filename, ios::out);

	if ( outputFile.is_open() )
	{
		outputFile.precision(10); // # of floating point

		//output weights
		for ( int i=0; i <= nInput; i++ )  // '=' means loop including bias neuron
		{
			for ( int j=0; j < nHidden[0]; j++ ) outputFile << wInputHidden[i][j] << ","; // (nInput+1)*nHidden
		}
		outputFile << endl;
		if (nLayer>1)
		{
			for (int k=0; k<nLayer-1; k++)
			{
				for ( int i=0; i <= nHidden[k]; i++ )
				{
					for ( int j=0; j < nHidden[k+1]; j++ ) outputFile << wHiddenHidden[k][i][j] << ",";
				}
				outputFile << endl;
			}
		}
		for ( int i=0; i <= nHidden[nLayer-1]; i++ ) // '=' means loop including bias neuron
		{
			for ( int j=0; j < nOutput; j++ )
			{
				outputFile << wHiddenOutput[i][j];	// (nHidden+1)*nOutput
				if ( i * nOutput + j + 1 != (nHidden[nLayer-1] + 1) * nOutput ) outputFile << ",";
			}
		}

		//print success
		cout << endl << "Neuron weights saved to '" << filename << "'" << endl;

		//close file
		outputFile.close();
		
		return true;
	}
	else 
	{
		cout << endl << "Error - Weight output file '" << filename << "' could not be created: " << endl;
		return false;
	}
}
/*******************************************************************
* Feed pattern through network and return results
********************************************************************/
double* neuralNetwork::feedForwardPattern(double *pattern)
{
	feedForward(pattern);
	//create copy of output results
	double* results = new double[nOutput];
	//for (int i=0; i < nOutput; i++ ) results[i] = clampOutput(outputNeurons[i]);
	for (int i=0; i < nOutput; i++ ) results[i] = outputNeurons[i];
	return results;
}
/*******************************************************************
* Return the NN accuracy on the set
********************************************************************/
double neuralNetwork::getSetAccuracy( std::vector<dataEntry*>& set )
{
	double incorrectResults = 0;
		
	//for every training input array
	for ( int tp = 0; tp < (int) set.size(); tp++)
	{						
		//feed inputs through network and backpropagate errors
		feedForward( set[tp]->pattern );
		
		//correct pattern flag
		bool correctResult = true;

		//check all outputs against desired output values
		for ( int k = 0; k < nOutput; k++ )
		{					
			//set flag to false if desired and output differ
			if ( clampOutput(outputNeurons[k]) != set[tp]->target[k] ) correctResult = false;
		}
		
		//inc training error for a incorrect result
		if ( !correctResult ) incorrectResults++;	
		
	}//end for
	
	//calculate error and return as percentage
	return 100 - (incorrectResults/set.size() * 100);
}

/*******************************************************************
* Return the NN mean squared error on the set
********************************************************************/
double neuralNetwork::getSetMSE( std::vector<dataEntry*>& set )
{
	double mse = 0;
		
	//for every training input array
	for ( int tp = 0; tp < (int) set.size(); tp++)
	{						
		//feed inputs through network and backpropagate errors
		feedForward( set[tp]->pattern );
		
		//check all outputs against desired output values
		for ( int k = 0; k < nOutput; k++ )
		{					
			//sum all the MSEs together
			mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
		}		
		
	}//end for
	
	//calculate error and return as percentage
	return mse/(nOutput * set.size());
}


/*******************************************************************
* Initialize Neuron Weights
********************************************************************/
void neuralNetwork::initializeWeights()
{
	//set range
	double rH = 1/sqrt((double)nInput);
	vector<double> rO;
	for (int k=0; k<nLayer; k++) rO.push_back(1/sqrt((double)nHidden[k]));

	//set weights
	for(int i=0; i<=nInput; i++)
	{
		for(int j=0; j<nHidden[0]; j++) wInputHidden[i][j] = (((double)(rand()%100)+1)/100*2*rH)-rH;
	}

	if (nLayer>1)
	{
		for (int k=0; k<nLayer-1; k++)
		{
			for ( int i=0; i <= nHidden[k]; i++ )
			{
				for ( int j=0; j < nHidden[k+1]; j++ ) wHiddenHidden[k][i][j] = (((double)(rand()%100)+1)/100 * 2 * rO[k] ) - rO[k];
			}
		}
	}

	for(int i = 0; i <= nHidden[nLayer-1]; i++)
	{
		for(int j = 0; j < nOutput; j++) wHiddenOutput[i][j] = ( ( (double)(rand()%100)+1)/100 * 2 * rO[0] ) - rO[0];
	}

}
/*******************************************************************
* Activation Function
********************************************************************/
inline double neuralNetwork::activationFunction( double x )
{
	//sigmoid function
	//return tanh(x);
	return 1/(1+exp(-x));
}	
/*******************************************************************
* Output Clamping
********************************************************************/
int neuralNetwork::clampOutput( double x )
{
	if ( x > 0.9 ) return 1;
	else return 0;
}
/*******************************************************************
* Feed Forward Operation
********************************************************************/

void neuralNetwork::feedForward(double* pattern)
{
	//set input neurons to input values
	for(int i = 0; i < nInput; i++) inputNeurons[i] = pattern[i];

	for(int j=0; j < nHidden[0]; j++)
	{
		//clear value
		hiddenNeurons[0][j] = 0;
		//get weighted sum of pattern and bias neuron
		for( int i=0; i <= nInput; i++ ) hiddenNeurons[0][j] += inputNeurons[i] * wInputHidden[i][j];
		//set to result of sigmoid
		hiddenNeurons[0][j] = activationFunction( hiddenNeurons[0][j] );
	}

	if (nLayer>1)
	{
		for (int k=1; k<nLayer; k++)
		{
			//Calculate Hidden Layer values - include bias neuron
			for(int j=0; j < nHidden[k]; j++)
			{
				//clear value
				hiddenNeurons[k][j] = 0;
				//get weighted sum of pattern and bias neuron
				for( int i=0; i <= nHidden[k-1]; i++ )
				{
					hiddenNeurons[k][j] += hiddenNeurons[k][i] * wHiddenHidden[k-1][i][j];
				}
				//set to result of sigmoid
				hiddenNeurons[k][j] = activationFunction( hiddenNeurons[k][j] );
			}
		}
	}

	for(int j=0; j < nOutput; j++)
	{
		//clear value
		outputNeurons[j] = 0;
		//get weighted sum of pattern and bias neuron
		for( int i=0; i <= nHidden[nLayer-1]; i++ ) outputNeurons[j] += hiddenNeurons[nLayer-1][i] * wHiddenOutput[i][j];
		//set to result of sigmoid
		outputNeurons[j] = activationFunction( outputNeurons[j] );
	}
}


