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
//#include <librealsense/rs.hpp>

//custom includes
#include "modules/neuralNetwork.h"
#include "modules/DisplayManyImages.h"

using namespace std;
using namespace cv;
//using namespace rs;

#define CLEAR(x) memset(&(x), 0, sizeof(x))
#define _USE_MATH_DEFINES
#define pi       3.14159265358979323846

void main_ellipse ();


int const INPUT_WIDTH      = 640;
int const INPUT_HEIGHT     = 480;
int const FRAMERATE        = 60;

int nKernel = 10;
int sKernel = 7;
int pdim = 2;

//Training condition
int sImage= 28*28;
int nPattern = (((int)sqrt(sImage))/pdim)^2;
vector<int> nLayer{256};
int nTarget = 1;

// constans of Gaussian blur
Size ksize = Size(5, 5);
double sigma1 = 2;
double sigma2 = 2;

dataReader dR;
neuralNetwork nn(nPattern,nLayer,nTarget);
float neural_thres = 0.95; // this is for threshold to determine if x is true or false
//VideoWriter video("out.avi",CV_FOURCC('M','J','P','G'),10, Size(INPUT_WIDTH,INPUT_HEIGHT),true);
//ofstream logfile("log.csv");

VideoCapture capture(0);

int main(int argc, const char* argv[])
{
	// Loads a csv file of weight matrix data
	char* kernel_file = "log/kernel.csv";
	dR.loadKernels(kernel_file,sKernel,nKernel);

	// Loads a csv file of weight matrix data
	char* weights_file = "log/weights.txt";
	nn.loadWeights(weights_file);

	//logfile << "#Contour" << "," << "Data" << endl;
	main_ellipse();

    return 0;
}

void main_ellipse ()
{
#if 0
	// Detect device 
	rs::log_to_console(rs::log_severity::warn);
	rs::context ctx;
	printf("There are %d connected RealSense devices.\n", ctx.get_device_count());
	if (ctx.get_device_count() == 0) throw std::runtime_error("No device detected. Is it plugged in?");

	// Get device parameters and Prepare streaming
	rs::device * dev = ctx.get_device(0);
	dev->enable_stream(rs::stream::color, INPUT_WIDTH, INPUT_HEIGHT, rs::format::bgr8, FRAMERATE);
	dev->start();
#endif

	bool flag=1;
	Mat frame;
	int num_x = 0;
	int num_o = 0;
	// Time checking start
	int frames = 0;
	float time = 0, fps = 0;
	auto t0 = std::chrono::high_resolution_clock::now();

	while (1)
	{
		if (flag==1)
		{
			//dev->wait_for_frames();
			//frame = Mat(Size(INPUT_WIDTH, INPUT_HEIGHT), CV_8UC3, (void*)dev->get_frame_data(rs::stream::color), Mat::AUTO_STEP);
			capture >> frame;
			flag = 0;
		}
		else
		{
			Mat src,mask;
			resize(frame,frame,Size(), 0.5, 0.5);
			cv::cvtColor(frame, src, COLOR_BGR2GRAY);
			//equalizeHist(src, src);
			//frame.release();
			GaussianBlur( src, src, ksize, sigma1, sigma2 );
			Canny(src, mask, 80, 160, 3);

			Mat mask_cp,mask_cp2;
			mask.copyTo(mask_cp);
			cv::cvtColor(mask_cp, mask_cp, COLOR_GRAY2BGR);
			mask_cp.copyTo(mask_cp2);

			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

			/// Get the moments and mass centers:
			vector<Moments> mu(contours.size());
			vector<Point2f> mc(contours.size());
			for( size_t i = 0; i < contours.size(); i++ ){
			  mu[i] = moments( contours[i], false );
			  mc[i] = Point2f( static_cast<float>(mu[i].m10/mu[i].m00) , static_cast<float>(mu[i].m01/mu[i].m00) );
			}

			/// Find the convex hull object for each contour
			vector<vector<Point> > hull( contours.size() );
			for( size_t i = 0; i < contours.size(); i++ ) convexHull( Mat(contours[i]), hull[i], false );

			////compute I1, I2, I3/////////
			float I1, I2, I3, II1;
			unsigned char data[100];
			size_t j=0;
			for( size_t i = 0; i < contours.size(); i++ )
			{
				size_t count = contours[i].size();
				if (count > 10)
				{
					Mat pointsf;
					Mat(contours[i]).convertTo(pointsf, CV_32F);
					RotatedRect box = fitEllipse(pointsf);
					Point2f bc = box.center;
					float box_angle = box.angle;
					float error = 0;
					float error1 = 0;
					box_angle = pi/2-box_angle*pi/180; //convert to radian

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

						if( count < 6 ) continue;
						for (size_t k = 0; k < contours[i].size(); k++)
						{
							  error1=((contours[i][k].x-bc.x)*cos(box_angle)+(contours[i][k].y-bc.y)*sin(box_angle))*
									  ((contours[i][k].x-bc.x)*cos(box_angle)+(contours[i][k].y-bc.y)*sin(box_angle))/(box.size.width*box.size.width*0.25f)+
									  ((contours[i][k].x-bc.x)*sin(box_angle)+(contours[i][k].y-bc.y)*cos(box_angle))*
									  ((contours[i][k].x-bc.x)*sin(box_angle)+(contours[i][k].y-bc.y)*cos(box_angle))/(box.size.height*box.size.height*0.25f)-1;
							  error = error+error1;
						}

						error = error/contours[i].size();
						error = fabs(error);

						if(fabs(I2)<0.0000001 && fabs(I3)<0.000001 && II1<0.0003/*0.0003*/)
						{
							data[j]=i; // i is contours.size()
							j++;
							if(error < 0.1f)
							{
								if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30 ) continue;

								//ellipse(dst_gpu, box, Scalar(0,0,255), 1, LINE_AA);
								//ellipse(dst_gpu, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(0,255,255), 1, LINE_AA);

								Rect roi = boundingRect(contours[i]);
								rectangle(mask_cp,roi,Scalar(0,0,255));

								Mat crop = src(roi); 
								resize(crop,crop,Size(sqrt(nPattern),sqrt(nPattern)));
								Canny(crop, crop, 80, 160, 3);
								
								//cvtColor(crop, crop, cv::COLOR_BGR2HSV);
								//Mat hsv[3];
								//split(crop,hsv);
								//hsv[0].copyTo(crop);
/*
								Mat crop3 = frame(roi);
								std::ostringstream name2;
								name2 << "data/basic#" << num_x << ".png";
								imwrite(name2.str(), crop3);
								num_x++;*/

								double *res;
								double *conv_pattern = dR.ConvNPooling(crop,sKernel,nKernel,pdim);
								res = nn.feedForwardPattern(conv_pattern); // calculation results
								cout << res[0] <<endl;
								if (res[0] > neural_thres )
								{
									//cout << "Found!" << endl;
									rectangle(mask_cp2,roi,Scalar(255,0,0));
									/*
									Mat crop2 = frame(roi);
									std::ostringstream name;
									name << "data/neural#" << num_o << ".png";
									imwrite(name.str(), crop2);
									num_o++;
									*/
								}

							}
						}
						else
						{
							if(hull[i].size() > 10)
							{
								Mat h_pointsf;
								Mat(hull[i]).convertTo(h_pointsf, CV_32F);
								RotatedRect h_box = fitEllipse(h_pointsf);
								Point2f h_bc = h_box.center;
								float h_box_angle = h_box.angle;
								float h_error = 0;
								float h_error1 = 0;
								h_box_angle = pi/2-h_box_angle*pi/180;

								for (size_t kk = 0; kk < contours[i].size(); kk++){
									h_error1=((contours[i][kk].x-h_bc.x)*cos(h_box_angle)+(contours[i][kk].y-h_bc.y)*sin(h_box_angle))*
										  ((contours[i][kk].x-h_bc.x)*cos(h_box_angle)+(contours[i][kk].y-h_bc.y)*sin(h_box_angle))/(h_box.size.width*h_box.size.width*0.25f)+
										  ((contours[i][kk].x-h_bc.x)*sin(h_box_angle)+(contours[i][kk].y-h_bc.y)*cos(h_box_angle))*
										  ((contours[i][kk].x-h_bc.x)*sin(h_box_angle)+(contours[i][kk].y-h_bc.y)*cos(h_box_angle))/(h_box.size.height*h_box.size.height*0.25f)-1;

									h_error = h_error+h_error1;
								}

								h_error = h_error/contours[i].size();
								h_error = fabs(h_error);

								if(h_error < 0.08f)
								{
									if( MAX(h_box.size.width, h_box.size.height) > MIN(h_box.size.width, h_box.size.height)*30 )
									  continue;
									//ellipse(dst_gpu, h_box, Scalar(0,0,255), 1, LINE_AA);
									//ellipse(dst_gpu, h_box.center, h_box.size*0.5f, h_box.angle, 0, 360, Scalar(0,255,0), 1, LINE_AA);

									Rect roi=  boundingRect(contours[i]);
									rectangle(mask_cp,roi,Scalar(0,0,255));
									
									Mat crop = src(roi); 
									resize(crop,crop,Size(sqrt(nPattern),sqrt(nPattern)));
									Canny(crop, crop, 80, 160, 3);
									//cvtColor(crop, crop, cv::COLOR_BGR2HSV);
									//Mat hsv[3];
									//split(crop,hsv);
									//hsv[0].copyTo(crop);
									
/*									
									Mat crop3 = frame(roi);
									std::ostringstream name2;
									name2 << "data/basic#" << num_x << ".png";
									imwrite(name2.str(), crop3);
									num_x++;
*/
									double *res;
									double *conv_pattern = dR.ConvNPooling(crop,sKernel,nKernel,pdim);
									res = nn.feedForwardPattern(conv_pattern); // calculation results
									cout << res[0] <<endl;
									if (res[0] > neural_thres )
									{
										//cout << "Found!" << endl;
										rectangle(mask_cp2,roi,Scalar(255,0,0));
										/*
										Mat crop2 = frame(roi);
										std::ostringstream name;
										name << "data/neural#" << num_o << ".png";
										imwrite(name.str(), crop2);
										num_o++;
										*/
									}
									
								}
							}
						}// end of else
					} // end of if(mu[i].m00 != 0 )
				} // end of count 
			}

			//check for FPS(Frame Per Second)
			auto t11 = std::chrono::high_resolution_clock::now();
			float count = std::chrono::duration<float>(t11-t0).count();
			// limit fps
			if (count < 0.1f) usleep((0.1-count)*1000000);

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

			Mat info(Size(frame.cols*3,50),frame.type(),Scalar::all(0));
			std::ostringstream ssgpu;
			ssgpu << "FPS : " << fps << " Resolution : " << frame.cols << " x " << frame.rows;
			putText(info, ssgpu.str(), Point(30,30), FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 255,255), 2);
			putText(frame, "Original", Point(10,20), FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 255, 255), 2);			
			putText(mask_cp, "w/o Neural", Point(10,20), FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 255,255), 2);
			putText(mask_cp2, "w/ Neural", Point(10,20), FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 255,255), 2);
			//video << buf;

			Mat matDst(Size(frame.cols*3,frame.rows),frame.type(),Scalar::all(0));
			hconcat(frame, mask_cp, matDst);
			hconcat(matDst, mask_cp2, matDst);
			vconcat(info, matDst, matDst);

			//ShowManyImages("Images", 3, frame, mask_cp, mask_cp2);

			// Show in a window
			//namedWindow( "Targets", 0 );
			imshow( "Targets", matDst );
			frame.release();
			flag = 1;
			if(waitKey(10)==27)  break;

		} // end of else after capture
	} // end of while
} // end of main ellipse
