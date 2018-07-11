/*******************************************************************
* CSV Data File Reader and Training Set Creator
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
********************************************************************/

#ifndef _DATAREADER
#define _DATAREADER

//include standard header files
#include <vector>
#include <string>
#include <opencv2/core.hpp>

/*******************************************************************
* stores a data item
********************************************************************/
class dataEntry
{
public:	
	
	double* pattern;	//input patterns
	double* target;		//target result

public:	

	dataEntry(double* p, double* t): pattern(p), target(t) {}
		
	~dataEntry()
	{				
		delete[] pattern;
		delete[] target;
	}

};

/*******************************************************************
* stores a reference item
********************************************************************/
class reference
{
public:

	double p_max;	    //maximum pattern data
	double p_min;		//minimum pattern data

public:

	reference(double px, double pn): p_max(px), p_min(pn) {}

};

/*******************************************************************
* Training Sets Storage - stores shortcuts to data items
********************************************************************/
class trainingDataSet
{
public:

	std::vector<dataEntry*> trainingSet;
	std::vector<dataEntry*> generalizationSet;
	std::vector<dataEntry*> validationSet;

	trainingDataSet(){}
	
	void clear()
	{
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}
};

//dataset retrieval approach enum
enum { NONE, STATIC, GROWING, WINDOWING };

//data reader class
class dataReader
{
	
//private members
//----------------------------------------------------------------------------------------------------------------
private:

	//data storage
	std::vector<dataEntry*> data;
	std::vector<reference*> ref;
	int nInputs;
	int nTargets;
	int sImage;
	int sKernel;
	int nKernel;
	int pdim;

	std::vector <double**> Kernels;

	//current data set
	trainingDataSet tSet;

	//data set creation approach and total number of dataSets
	int creationApproach;
	int numTrainingSets;

	//creation approach variables
	double growingStepSize;			//step size - percentage of total set
	int growingLastDataIndex;		//last index added to current dataSet
	int windowingSetSize;			//initial size of set
	int windowingStepSize;			//how many entries to move window by
	int windowingStartIndex;		//window start index	
	
//public methods
//----------------------------------------------------------------------------------------------------------------
public:
	int trainingDataEndIndex;
	dataReader(): creationApproach(NONE), numTrainingSets(-1) {}
	~dataReader();
	bool loadDataFile4Train( const char* filename, int nI, int nT, float gratio, float tratio ); //maxmin->normalization
	bool loadImageFile4Train( const char* filename, int nI, int nT, float gratio, float tratio, int , int, int, bool); //normalization->convolution and pooling->ReLU
	void loadImage4Test( const cv::Mat frame,int,int, bool); //normalization->convolution and pooling->ReLU
	bool maxmin( const char* filename);
	void setCreationApproach( int approach, double param1 = -1, double param2 = -1 );
	int getNumTrainingSets();
	cv::Mat Pooling(const cv::Mat &, int , int , int );
	double* ConvNPooling(double *pattern,int , int , int, int, bool);
	double* ConvNPooling(cv::Mat pattern, int , int, int, bool);
	bool loadKernels(const char* inputFilename,int sKernel,int nKernel);
	
	trainingDataSet* getTrainingDataSet();
	std::vector<dataEntry*>& getAllDataEntries();


//private methods
//----------------------------------------------------------------------------------------------------------------
private:
	
	void createStaticDataSet();
	void createGrowingDataSet();
	void createWindowingDataSet();	
	
	void processLine( std::string &line );
	void processMat( cv::Mat frame, bool GAP );
	void processLine4maxmin( std::string &line );
	void processLine4Image( std::string &line );
};

#endif
