//include definition file
#include "dataReader.h"

#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <algorithm>

using namespace std;
using namespace cv;

/*******************************************************************
* Destructor
********************************************************************/
dataReader::~dataReader()
{
	//clear data
	for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
	data.clear();		 
}
/*******************************************************************
* Loads a csv file of input data
********************************************************************/
bool dataReader::loadDataFile4Train( const char* filename, int nI, int nT ,float tratio, float gratio)
{
	//clear any previous data
	for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
	data.clear();
	tSet.clear();
	
	//set number of inputs and outputs
	nInputs = nI;
	nTargets = nT;

	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);	

	if ( inputFile.is_open() )
	{
		string line = "";

		//read data
		while ( !inputFile.eof() )
		{
			// get num of line
			getline(inputFile, line);				
			
			//process line
			if (line.length() > 2 ) processLine(line);
		}	

		//print success
		cout << "Input File: " << filename << "\nRead Complete: " << data.size() << " Patterns Loaded"  << endl;

		//normalize data
		double *buf[(int)nInputs];
		vector <double> minVal, maxVal;
		ofstream maxmin("log/maxmin.csv");
		for(int i=0; i < nInputs; i++)
		{
			buf[i] = new( double[(int)data.size()] );
			double min=0, max = 255;
			/*for(int j=0; j < (int)data.size(); j++)
			{
				double temp = (double)data[j]->pattern[i];
			    if (temp > max) max = temp;
			    if (temp < min) min = temp;
			    buf[i][j] = temp;
			}*/
			minVal.push_back(min);
			maxVal.push_back(max);
			maxmin << min << "," << max << endl;
			//cout << "#input = " << (int)i+1 << "  min = " << min << ", max = " << max << endl;
		}
		maxmin.close();

		for(int ii=0; ii < nInputs; ii++)
		{
			for(int jj=0; jj < data.size(); jj++)
			{
				double diff = maxVal[ii]-minVal[ii];
				if (diff==0)
				{
					data[jj]->pattern[ii]=ratio*1.0;
				}
				else
				{
					data[jj]->pattern[ii]=(double)((buf[ii][jj]-minVal[ii])/diff)*ratio;
				} // normalization to 0~1
			}

		}

		//shuffle data
		random_shuffle(data.begin(), data.end());

		//split data set
		trainingDataEndIndex = (int) ( tratio * data.size() );
		int gSize = (int) ( ceil(gratio * data.size()) );
		int vSize = (int) ( data.size() - trainingDataEndIndex - gSize );
							
		//generalization set
		for ( int i = trainingDataEndIndex; i < trainingDataEndIndex + gSize; i++ ) tSet.generalizationSet.push_back( data[i] );
				
		//validation set
		for ( int i = trainingDataEndIndex + gSize; i < (int) data.size(); i++ ) tSet.validationSet.push_back( data[i] );

		//close file
		inputFile.close();

		return true;
	}
	else 
	{
		cout << "Error Opening Input File: " << filename << endl;
		return false;
	}
}

void dataReader::loadMat4Test(const Mat frame, int nI, int nT)
{
	//clear any previous data
	for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
	data.clear();
	tSet.clear();
	
	//set number of inputs and outputs
	nInputs = nI;
	nTargets = nT;

	processMat(frame);
	for(int i=0; i < nInputs; i++)
	{
		for(int j=0; j < data.size(); j++)
		{
			double diff = ref[i]->p_max-ref[i]->p_min;
			data[j]->pattern[i]=(double)((data[j]->pattern[i]-ref[i]->p_min)/diff)*ratio;
		}
	}

	//validation set
	trainingDataEndIndex = (int)0;
	for ( int i = 0; i < (int) data.size(); i++ ) tSet.validationSet.push_back( data[i] );

}

bool dataReader::maxmin( const char* filename, int nI)
{
		//open file for maxmin value
		fstream maxminFile;
		maxminFile.open("log/maxmin.csv", ios::in);

		if ( maxminFile.is_open() )
		{
			string line2 = "";
			cout << "ManMin Reference File Loaded" << endl;

			//read data
			while ( !maxminFile.eof() )
			{
				// get num of line
				getline(maxminFile, line2);
				if (line2.length() > 2 ) processLine2(line2);  //ref[i]->p_min
			}

			//close file
			maxminFile.close();
			return true;
		}
		else
		{
			cout << "Error Opening MaxMin File " << endl;
			return false;
		}
}

/*******************************************************************
* Processes a single line from the data file
********************************************************************/

void dataReader::processMat( const Mat frame )
{
	//create new pattern and target
	double* pattern = new double[nInputs];
	double* target = new double[nTargets];
	
	int i = 0;
	while ( i < (nInputs + nTargets) )
	{	
		if ( i < nInputs ) pattern[i] = (unsigned char)(frame.data[i]);
		else target[i - nInputs] = (unsigned char)(frame.data[i]);
		i++;			
	}
	
	//add to records
	data.push_back( new dataEntry( pattern, target ) );		
}


void dataReader::processLine( string &line )
{
	//create new pattern and target
	double* pattern = new double[nInputs];
	double* target = new double[nTargets];
	
	//store inputs		
	char* cstr = new char[line.size()+1];
	char* t;
	strcpy(cstr, line.c_str());

	//tokenise
	int i = 0;
	t=strtok (cstr,",");
	
	while ( t!=NULL && i < (nInputs + nTargets) )
	{	
		if ( i < nInputs ) pattern[i] = atof(t);
		else target[i - nInputs] = atof(t);

		//move token onwards
		t = strtok(NULL,",");
		i++;			
	}
	
	//add to records
	data.push_back( new dataEntry( pattern, target ) );		
}

void dataReader::processLine2( string &line2 )
{
	//create new pattern and target
	double p_max = 0;
	double p_min = 0;

	//store inputs
	char* cstr = new char[line2.size()+1];
	char* t;
	strcpy(cstr, line2.c_str());

	//tokenise
	int i = 0;
	t=strtok (cstr,",");

	while ( t!=NULL && i < 2 )
	{
		if ( i < 1 ) p_min = atof(t);
		else p_max = atof(t);
		//move token onwards
		t = strtok(NULL,",");
		i++;
	}
	//add to records
	ref.push_back( new reference( p_max, p_min ) );
}

/*******************************************************************
* Selects the data set creation approach
********************************************************************/
void dataReader::setCreationApproach( int approach, double param1, double param2 )
{
	//static
	if ( approach == STATIC )
	{
		creationApproach = STATIC;
		
		//only 1 data set
		numTrainingSets = 1;
	}

	//growing
	else if ( approach == GROWING )
	{			
		if ( param1 <= 100.0 && param1 > 0)
		{
			creationApproach = GROWING;
		
			//step size
			growingStepSize = param1 / 100;
			growingLastDataIndex = 0;

			//number of sets
			numTrainingSets = (int) ceil( 1 / growingStepSize );				
		}
	}

	//windowing
	else if ( approach == WINDOWING )
	{
		//if initial size smaller than total entries and step size smaller than set size
		if ( param1 < data.size() && param2 <= param1)
		{
			creationApproach = WINDOWING;
			
			//params
			windowingSetSize = (int) param1;
			windowingStepSize = (int) param2;
			windowingStartIndex = 0;			

			//number of sets
			numTrainingSets = (int) ceil( (double) ( trainingDataEndIndex - windowingSetSize ) / windowingStepSize ) + 1;
		}			
	}

}

/*******************************************************************
* Returns number of data sets created by creation approach
********************************************************************/
int dataReader::getNumTrainingSets()
{
	return numTrainingSets;
}
/*******************************************************************
* Get data set created by creation approach
********************************************************************/
trainingDataSet* dataReader::getTrainingDataSet()
{		
	switch ( creationApproach )
	{	
		case STATIC : createStaticDataSet(); break;
		case GROWING : createGrowingDataSet(); break;
		case WINDOWING : createWindowingDataSet(); break;
	}
	
	return &tSet;
}
/*******************************************************************
* Get all data entries loaded
********************************************************************/
vector<dataEntry*>& dataReader::getAllDataEntries()
{
	return data;
}

/*******************************************************************
* Create a static data set (all the entries)
********************************************************************/
void dataReader::createStaticDataSet()
{
	//training set
	for ( int i = 0; i < trainingDataEndIndex; i++ ) tSet.trainingSet.push_back( data[i] );		
}
/*******************************************************************
* Create a growing data set (contains only a percentage of entries
* and slowly grows till it contains all entries)
********************************************************************/
void dataReader::createGrowingDataSet()
{
	//increase data set by step percentage
	growingLastDataIndex += (int) ceil( growingStepSize * trainingDataEndIndex );		
	if ( growingLastDataIndex > (int) trainingDataEndIndex ) growingLastDataIndex = trainingDataEndIndex;

	//clear sets
	tSet.trainingSet.clear();
	
	//training set
	for ( int i = 0; i < growingLastDataIndex; i++ ) tSet.trainingSet.push_back( data[i] );			
}
/*******************************************************************
* Create a windowed data set ( creates a window over a part of the data
* set and moves it along until it reaches the end of the date set )
********************************************************************/
void dataReader::createWindowingDataSet()
{
	//create end point
	int endIndex = windowingStartIndex + windowingSetSize;
	if ( endIndex > trainingDataEndIndex ) endIndex = trainingDataEndIndex;		

	//clear sets
	tSet.trainingSet.clear();
					
	//training set
	for ( int i = windowingStartIndex; i < endIndex; i++ ) tSet.trainingSet.push_back( data[i] );
			
	//increase start index
	windowingStartIndex += windowingStepSize;
}

