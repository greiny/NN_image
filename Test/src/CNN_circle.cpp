#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <pthread.h>
#include <cstdio>
#include <chrono>
#include <unistd.h>

#include <stdlib.h>
#include <ctime>
#include <vector>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//custom includes
#include "modules/neuralNetwork.h"
#include "modules/DisplayManyImages.h"

using namespace std;
using namespace cv;

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define _USE_MATH_DEFINES
#define pi       3.14159265358979323846

bool getError(vector<Point> ncontour, float thres);
bool Feedforward(Mat mask, Rect roi);
void saveTemp(Mat frame, Rect roi);
void main_ellipse ();

int nKernel = 4;
int sKernel = 7;
int pdim = 2;

//Training condition
int sImage= 28*28;
int nPattern = (int)pow((int)sqrt(sImage)/pdim,2)*nKernel;
vector<int> nLayer{64};
int nTarget = 2;

// constants of Canny
double canny_thes = 70;

// constans of Gaussian blur
Size ksize = Size(5, 5);
double sigma1 = 2;
double sigma2 = 2;

dataReader dR;
neuralNetwork nn(nPattern,nLayer,nTarget);
float neural_thres = 0.9;
float prob = 0;
int num = 0;

VideoCapture capture("out.avi");
VideoWriter video("CNN4.avi",CV_FOURCC('M','J','P','G'),15, Size(336*3+10*2,188+50),true);

int main(int argc, const char* argv[])
{
	// Loads a csv file of weight matrix data
	char* kernel_file = "log/kernel4.csv";
	dR.loadKernels(kernel_file,sKernel,nKernel);

	// Loads a csv file of weight matrix data
	char* weights_file = "log/weights37.txt";
	nn.loadWeights(weights_file);

	main_ellipse();
    return 0;
}

void main_ellipse ()
{
	bool flag=1;
	Mat frame;

	// Time checking start
	int frames = 0;
	float time = 0, fps = 0;
	auto t0 = std::chrono::high_resolution_clock::now();

	while(1)
	{
		if (flag==1)
		{
			capture >> frame; 
			if (frame.empty()) break;
			flag = 0;
		}
		else
		{
			Mat src, mask, dst_image1,dst_image2;
			//Rect rec(0,0,frame.cols/2, frame.rows);
			//frame = frame(rec);
			resize(frame,frame,Size(), 0.5, 0.5);

			cvtColor(frame, src, COLOR_BGR2GRAY);
			GaussianBlur(src, mask, ksize, sigma1, sigma2 );
			Canny(mask, mask, canny_thes, canny_thes*2, 3);

			cvtColor(mask, dst_image1, COLOR_GRAY2BGR);
			dst_image1.copyTo(dst_image2);

			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

			/// Get the moments and mass centers:
			vector<Moments> mu(contours.size());
			vector<Point2f> mc(contours.size());
			for( size_t i = 0; i < contours.size(); i++ )
			{
			  mu[i] = moments( contours[i], false );
			  mc[i] = Point2f( static_cast<float>(mu[i].m10/mu[i].m00) , static_cast<float>(mu[i].m01/mu[i].m00) );
			}

			////compute I1, I2, I3/////////
			float I1, I2, I3, II1;
			size_t j=0;
			for( size_t i = 0; i < contours.size(); i++ )
			{
				if ((size_t)contours[i].size() > 10)
				{
					if(mu[i].m00 != 0 )
					{
						I1=(mu[i].mu20*mu[i].mu02-mu[i].mu11*mu[i].mu11)/pow(mu[i].m00,4);
						I2=((mu[i].mu30*mu[i].mu30)*(mu[i].mu03*mu[i].mu03)-6*(mu[i].mu30)*(mu[i].mu21)*(mu[i].mu12)*(mu[i].mu03)+
							  4*(mu[i].mu30)*(pow(mu[i].mu12,3))+4*(mu[i].mu03)*(pow(mu[i].mu21,3))-
							  3*(mu[i].mu21*mu[i].mu21)*(mu[i].mu12*mu[i].mu12))/pow(mu[i].m00,10);
						I3=((mu[i].mu20)*(mu[i].mu21*mu[i].mu03-mu[i].mu12*mu[i].mu12)-
							  mu[i].mu11*(mu[i].mu30*mu[i].mu03-mu[i].mu21*mu[i].mu12)+
							  mu[i].mu02*(mu[i].mu30*mu[i].mu12-mu[i].mu21*mu[i].mu21))/pow(mu[i].m00,7);
						II1=fabs(I1-0.006332);

						if(fabs(I2)<0.0000001 && fabs(I3)<0.0000001 && II1<0.0003/*0.0003*/)
						{
							bool pass = getError(contours[i],0.1f);
							if(pass == true )
							{
								Rect roi = boundingRect(contours[i]);
								rectangle(dst_image1,roi,Scalar(0, 0, 255),1.5);
								bool pass_CNN = Feedforward(src,roi);
								if(pass_CNN == true )
								{
									saveTemp(frame,roi);
									stringstream stream;
									stream << fixed << setprecision(3) << prob;
									putText(dst_image2, stream.str(), Point(roi.x-2,roi.y-2), FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 255,255), 0.8);
									rectangle(dst_image2,roi,Scalar(0, 255, 255),1.5);
								}
							}
						}
						else
						{
							/// Find the convex hull object for each contour
							vector<Point> hull( contours[i].size() );
							convexHull( Mat(contours[i]), hull, false );

							if(hull.size() > 30)
							{
								bool pass = getError(hull,0.3f);
								if(pass == true )
								{
									Rect roi = boundingRect(contours[i]);
									rectangle(dst_image1,roi,Scalar(0, 0, 255),1.5);
									bool pass_CNN = Feedforward(src,roi);
									if(pass_CNN == true )
									{
										saveTemp(frame,roi);
										stringstream stream;
										stream << fixed << setprecision(3) << prob;
										putText(dst_image2, stream.str(), Point(roi.x-2,roi.y-2), FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 255,255), 0.8);
										rectangle(dst_image2,roi,Scalar(0, 255, 255),1.5);
									}
								}
							}
						}
					}
				}
			}
			//check for FPS(Frame Per Second)
			auto t11 = std::chrono::high_resolution_clock::now();
			float count = std::chrono::duration<float>(t11-t0).count();
			// limit fps
			if (count < 0.1) usleep((0.1-count)*1000000);

			auto t1 = std::chrono::high_resolution_clock::now();
			time += std::chrono::duration<float>(t1-t0).count();
			t0 = t1;
			++frames;
			if (time > 0.5f) 
			{
				fps = frames / time;
				frames = 0;
				time = 0;
			}

			Mat blank(Size(10,frame.rows),frame.type(),Scalar::all(0));
			Mat matDst(Size(frame.cols*3+blank.cols*2,frame.rows),frame.type(),Scalar::all(0));

			Mat info(Size(matDst.cols,50),frame.type(),Scalar::all(0));
			std::ostringstream ssframe;
			ssframe << "FPS : " << fps << " Resolution : " << frame.cols << " x " << frame.rows;
			putText(info, ssframe.str(), Point(30,30), FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 255,255), 2);
			putText(frame, "Original", Point(10,20), FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 255, 255), 2);			
			putText(dst_image1, "w/o Neural", Point(10,20), FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 255,255), 2);
			putText(dst_image2, "w/ Neural", Point(10,20), FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 255,255), 2);

			hconcat(frame, blank, matDst);
			hconcat(matDst, dst_image1, matDst);
			hconcat(matDst, blank, matDst);
			hconcat(matDst, dst_image2, matDst);
			vconcat(info, matDst, matDst);
			
			imshow( "Targets", matDst );
			//video << matDst;
			frame.release();
			flag = 1;
			if(waitKey(10)==27)  break;
		} // end of else after capture
	} // end of while
} // end of main ellipse


bool getError(vector<Point> ncontour, float thres)
{
	Mat pointsf;
	Mat(ncontour).convertTo(pointsf, CV_32F);
	RotatedRect box = fitEllipse(pointsf);
	Point2f bc = box.center;
	float box_angle = box.angle;
	float error = 0;
	float error1 = 0;
	box_angle = pi/2-box_angle*pi/180; //convert to radian

	for (size_t k = 0; k < ncontour.size(); k++)
		{
			  error1=((ncontour[k].x-bc.x)*cos(box_angle)+(ncontour[k].y-bc.y)*sin(box_angle))*
					  ((ncontour[k].x-bc.x)*cos(box_angle)+(ncontour[k].y-bc.y)*sin(box_angle))/(box.size.width*box.size.width*0.25f)+
					  ((ncontour[k].x-bc.x)*sin(box_angle)+(ncontour[k].y-bc.y)*cos(box_angle))*
					  ((ncontour[k].x-bc.x)*sin(box_angle)+(ncontour[k].y-bc.y)*cos(box_angle))/(box.size.height*box.size.height*0.25f)-1;
			  error = error+error1;
		}

	error = error/ncontour.size();
	error = fabs(error);
	if( MAX(box.size.width, box.size.height) < MIN(box.size.width, box.size.height)*20 && error < thres) return true;
	else return false;
}

bool Feedforward(Mat mask, Rect roi)
{
	Mat crop = mask(roi);
	resize(crop,crop,Size(28,28));
	
	double *res; 	
	double *conv_pattern = dR.ConvNPooling(crop,sKernel,nKernel,pdim);
	res = nn.feedForwardPattern(conv_pattern); // calculation results

	//cout << res[0] << ", " << res[1] <<endl;
	prob = (float)res[0];
	//if (neural_thres < res[0] && res[0] > res[1]) return true;
	if (res[0] > res[1]) return true;
	else return false;
}

void saveTemp(Mat frame, Rect roi)
{
	Mat crop = frame(roi);
	std::ostringstream name;
	name << "data/img#" << num << ".png";
	imwrite(name.str(), crop);
	num++;
}
