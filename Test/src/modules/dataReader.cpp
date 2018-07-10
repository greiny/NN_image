//include definition file
#include "dataReader.h"
#include "matrix_maths.h"
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
			double min=100000, max = 0;
			for(int j=0; j < (int)data.size(); j++)
			{
				double temp = (double)data[j]->pattern[i];
			    if (temp > max) max = temp;
			    if (temp < min) min = temp;
			    buf[i][j] = temp;
			}
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
					data[jj]->pattern[ii]=1.0;
				}
				else
				{
					data[jj]->pattern[ii]=(double)((buf[ii][jj]-minVal[ii])/diff);
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

bool dataReader::loadImageFile4Train( const char* filename, int nI, int nT ,float tratio, float gratio , int sK, int nK, int pdim)
{
	//clear any previous data
	for (int i=0; i < (int) data.size(); i++ ) delete data[i];
	data.clear();
	tSet.clear();

	//set number of inputs and outputs
	sImage = nI;
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
			if (line.length() > 2 ) processLine4Image(line);
		}

		//print success
		cout << "Input File: " << filename << "\nRead Complete: " << data.size() << " Patterns Loaded"  << endl;

		// normalize and convolve data
		for(int jj=0; jj < data.size(); jj++)
		{
			data[jj]-> pattern = ConvNPooling(data[jj]->pattern,sImage,sKernel,nKernel,pdim);
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

void dataReader::loadImage4Test(const Mat frame,int sI, int nT)
{
	//clear any previous data
	for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
	data.clear();
	tSet.clear();

	sImage = sI;
	nTargets = nT;
	processMat(frame);

	//validation set
	trainingDataEndIndex = (int)0;
	for ( int i = 0; i < (int) data.size(); i++ ) tSet.validationSet.push_back( data[i] );

}

bool dataReader::maxmin( const char* filename)
{
		//open file for maxmin value
		fstream maxminFile;
		maxminFile.open(filename, ios::in);

		if ( maxminFile.is_open() )
		{
			string line2 = "";
			cout << "ManMin Reference File Loaded" << endl;

			//read data
			while ( !maxminFile.eof() )
			{
				// get num of line
				getline(maxminFile, line2);
				if (line2.length() > 2 ) processLine4maxmin(line2);  //ref[i]->p_min
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
	double* pattern = new double[sImage];
	double* target = new double[nTargets];
	frame.convertTo(frame, CV_64FC1, 1.0/255, 0);
	pattern = ConvNPooling(frame,sKernel,nKernel,pdim);
	for (int i=0; i<nTargets; i++) if ( i < sImage ) target[i] = 0;

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

void dataReader::processLine4Image( string &line )
{
	//create new pattern and target
	double* pattern = new double[sImage];
	double* target = new double[nTargets];

	//store inputs
	char* cstr = new char[line.size()+1];
	char* t;
	strcpy(cstr, line.c_str());

	//tokenise
	int i = 0;
	t=strtok (cstr,",");

	while ( t!=NULL && i < (sImage + nTargets) )
	{
		if ( i < sImage ) pattern[i] = atof(t);
		else target[i - sImage] = atof(t);

		//move token onwards
		t = strtok(NULL,",");
		i++;
	}

	//add to records
	data.push_back( new dataEntry( pattern, target ) );
}

void dataReader::processLine4maxmin( string &line2 )
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

///////////////////////////////////////////////////////////////////////////////

Mat
dataReader::Pooling(const Mat &M, int pVert, int pHori, int poolingMethod){
    if(pVert == 1 && pHori == 1){
        Mat res;
        M.copyTo(res);
        return res;
    }
    int remX = M.cols % pHori;
    int remY = M.rows % pVert;
    Mat newM;
    if(remX == 0 && remY == 0) M.copyTo(newM);
    else{
        Rect roi = Rect(remX / 2, remY / 2, M.cols - remX, M.rows - remY);
        M(roi).copyTo(newM);
    }
    Mat res = Mat::zeros(newM.rows / pVert, newM.cols / pHori, CV_64FC1);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            Mat temp;
            Rect roi = Rect(j * pHori, i * pVert, pHori, pVert);
            newM(roi).copyTo(temp);
            double val = 0.0;
            // for Max Pooling
            if(POOL_MAX == poolingMethod){
                double minVal = 0.0;
                double maxVal = 0.0;
                Point minLoc;
                Point maxLoc;
                minMaxLoc( temp, &minVal, &maxVal, &minLoc, &maxLoc );
                val = maxVal;
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                val = sum(temp)[0] / (pVert * pHori);
            }elif(POOL_STOCHASTIC == poolingMethod){
                // Stochastic Pooling
                double sumval = sum(temp)[0];
                Mat prob = temp.mul(Reciprocal(sumval));
                val = sum(prob.mul(temp))[0];
                prob.release();
            }
            res.ATD(i, j) = val;
            temp.release();
        }
    }
    newM.release();
    return res;
}

double* dataReader::ConvNPooling(double *pattern, int sImage, int sKernel, int nKernel, int pdim)
{
	int rows = (int)sqrt(sImage);
	Mat img(rows,rows,CV_64FC1);
	for(int j = 0; j < rows; j++) for(int i = 0; i < rows; i++) img.ATD(j, i) = (double)pattern[j*rows+i];
	img.convertTo(img, CV_64FC1, 1.0/255, 0);

   	vector<Mat> Kernelset;
	vector<Mat> conv;
    for(int k = 0; k < Kernels.size(); k++)
	{
		Mat buf(sKernel,sKernel,CV_64FC1);
		for(int i = 0; i < sKernel; i++) for(int j = 0; j < sKernel; j++) buf.ATD(j,i) = (double) Kernels[k][j][i];
		Kernelset.push_back(buf);
		buf.release();
	}


	for(int k = 0; k < Kernelset.size(); k++)
	{
		Mat temp = rot90(Kernelset[k], 2);
		Mat tmpconv = convCalc(img, temp, CONV_SAME);
		tmpconv = nonLinearity(tmpconv,NL_RELU);
		conv.push_back(tmpconv);
		temp.release(); tmpconv.release();
	}

	for(int k = 0; k < Kernelset.size(); k++)
	{
		Mat temp = conv[k];
		conv[k] = Pooling(temp, pdim, pdim, POOL_MAX);
		temp.release();
	}

	double* conv_pattern = (new(double[Kernelset.size()*conv[0].cols*conv[0].rows]));
	for(int k = 0; k < Kernelset.size(); k++)
	{
		int nPixel = conv[k].cols*conv[k].rows;
		for(int j = 0; j < conv[k].rows; j++)
			for(int i = 0; i < conv[k].cols; i++) conv_pattern[k*nPixel+j*conv[k].rows+i] = conv[k].ATD(j,i);
	}
	conv.clear();
	return conv_pattern;
}

double* dataReader::ConvNPooling(Mat pattern, int sKernel, int nKernel,int pdim)
{
   	vector<Mat> Kernelset;
	vector<Mat> conv;

	//Normalization
	pattern.convertTo(pattern, CV_64FC1, 1.0/255, 0);

	//Kernel loaded
    for(int k = 0; k < Kernels.size(); k++)
	{
		Mat buf(sKernel,sKernel,CV_64FC1);
		for(int i = 0; i < sKernel; i++) for(int j = 0; j < sKernel; j++) buf.ATD(j,i) = (double) Kernels[k][j][i];
		Kernelset.push_back(buf);
		buf.release();
	}

	for(int k = 0; k < Kernelset.size(); k++)
	{
		Mat temp = rot90(Kernelset[k], 2);
		Mat tmpconv = convCalc(pattern, temp, CONV_SAME);
		tmpconv = nonLinearity(tmpconv,NL_RELU);
		conv.push_back(tmpconv);
		temp.release(); tmpconv.release();
	}

	for(int k = 0; k < Kernelset.size(); k++)
	{
		Mat temp = conv[k];
		conv[k] = Pooling(temp, pdim, pdim, POOL_MAX);
		temp.release();
	}

	double* conv_pattern = (new(double[Kernelset.size()*conv[0].cols*conv[0].rows]));
	for(int k = 0; k < Kernelset.size(); k++)
	{
		int nPixel = conv[k].cols*conv[k].rows;
		for(int j = 0; j < conv[k].rows; j++)
			for(int i = 0; i < conv[k].cols; i++) conv_pattern[k*nPixel+j*conv[k].rows+i] = conv[k].ATD(j,i);
	}
	conv.clear();
	return conv_pattern;
}

bool dataReader::loadKernels(const char* filename, int sk, int nk)
{
	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);

	sKernel = (int)sk;
	nKernel = (int)nk;

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
		int num_weight = sKernel * sKernel * nKernel;
		if ( weights.size() != num_weight )
		{
			cout << endl << "Error - Incorrect number of Kernels in input file: " << filename << endl;
			//close file
			inputFile.close();

			return false;
		}
		else
		{
			//set weights
			int pos = 0;
			for ( int k=0; k < nKernel; k++ )
			{
				double** buf = new( double*[sKernel] );
				for ( int i=0; i < sKernel; i++ )
				{
					buf[i] = new (double[sKernel]);
					for ( int j=0; j < sKernel; j++ ) buf[i][j] = weights[pos++];
				}
				Kernels.push_back(buf);
			}
			//print success
			cout << endl << "Convolution Kernels loaded successfully from '" << filename << "'" << endl;
			//close file
			inputFile.close();
			return true;
		}
	}
	else
	{
		cout << endl << "Error - Kernel input file '" << filename << "' could not be opened: " << endl;
		return false;
	}
}
