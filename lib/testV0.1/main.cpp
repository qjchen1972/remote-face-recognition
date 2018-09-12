#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include<ctime>
#include <opencv2/opencv.hpp>
#include "src/dll_imp.h"
#include "src/mtcnn.h"

using namespace std;
using namespace cv;


void video()
{	
	init_face("../net/init_net.pb", "../net/predict_net.pb");

	std::vector<float> svec1,bvec1;
	std::vector<float> vec2;
	mtcnn *find = nullptr;
	char filename[128];

	cv::Mat img = cv::imread("../img/3.jpg");


	find = new mtcnn(img.rows, img.cols);
	Rect rect;
	vector<Point2f> five;
	if (!find->findFace(img, rect, five)) return ;

	cv::Mat  dst;
	set_distance_threshold(10000);
	if (!get_vec(img, rect, five, svec1, 1,4, 1,0,&dst)) return ;	

	set_distance_threshold(0);
	if (!get_vec(img, rect, five, bvec1, 1, 4, 1,0,&dst)) return;

	//sprintf_s(filename, "sval10/b%d.jpg", count);
	//imwrite(filename, dst);
	//set_distance_threshold(10000);
	
	delete find;
	five.clear();


	VideoCapture capture("../img/8.MP4");
	if (!capture.isOpened()) return;
	printf("video is %d \n", static_cast<long>(capture.get(CV_CAP_PROP_FRAME_COUNT)));

	Mat frame;
	int status = 1;
	float  sum0 = 0;
	float  sum01 = 0;
	float  sum1 = 0;
	float  sum2 = 0;
	float  sum3 = 0;
	float  sum4 = 0;

	int   t0 = 0;
	int   t01 = 0;
	int   t1 = 0;
	int   t2 = 0;
	int   t3 = 0;
	int   t4 = 0;

	
	//find = new mtcnn(540, 960);
	find = new mtcnn(960, 540);
	double elapsed_secs = 0.0;
	int p = 0;
	while (status)
	{
		if (!capture.read(frame))  break;

		Mat show;
		//resize(frame, show, Size(960,540));
		resize(frame, show, Size(540,960));

		imshow("video", show);
		moveWindow("video", 20, 20);

		five.clear();
		if (!find->findFace(show, rect, five)) continue ;
		vec2.clear();

		std::clock_t begin = clock();
		if (!get_vec(show, rect, five, vec2, 1,4, 1,0, &dst)) return ;		
		std::clock_t end = clock();
		elapsed_secs += double(end - begin);
		p++;
		
		float cos;
		cos = cal_cos(svec1, vec2);
		//if (cos < 0.25) continue;
		count++;
		//sprintf_s(filename, "modify/%d.jpg", count);
		//imwrite(filename, dst);

		if(count >= 1000) break;
		if (rect.width <= 22)
		{
			sum0 += cos;
			t0++;
		}
		else if (rect.width <= 26)
		{
			sum01 += cos;
			t01++;
		}
		else if (rect.width <= 32)
		{
			sum1 += cos;
			t1++;
		}
		else if (rect.width <= 48)
		{
			sum2 += cos;
			t2++;
		}
		else if (rect.width <= 64)
		{
			sum3 += cos;
			t3++;
		}
		else
		{
			sum4 += cos;
			t4++;
		}

		printf("%d: rect(%d,%d) ,cos is %f \n", count, rect.width,rect.height, cos);
				
		imshow("dst", dst);
		moveWindow("dst", 1200, 20);
		
		if (cv::waitKey(27) >= 0)
			status = false;
	}
	elapsed_secs = elapsed_secs / p;
	printf("time %d %f [0,22] = (%d,%f) [23,26] = (%d,%f) [27,32] = (%d,%f) [33,48] = (%d,%f) [49,64] = (%d,%f) [65,max] = (%d,%f)  %f\n",p,
		elapsed_secs / CLOCKS_PER_SEC,	t0,sum0,t01,sum01,t1,sum1,t2,sum2,t3,sum3,t4,sum4, sum0+sum01+sum1+sum2+sum3+sum4);

}

void test_modify_face()
{
	//先初始化
	init_modify_face("../net/modify_init_net", "../net/modify_pred_net");
	cv::Mat dst1, img1;
	char name[128];
	char dstname[128];
	int s = 5;
	for (int i = 26; i < 27; i++)
	{
		sprintf_s(name, "%d.jpg", i);
		sprintf_s(dstname, "test1/%d_%d.jpg", i, s);
		printf("%s\n", name);
		dst1 = cv::imread(name, cv::IMREAD_GRAYSCALE);
		resize(dst1, dst1, Size(64, 64));
		get_modify_face(dst1, dst1);
		imwrite(dstname, dst1);
	}
}


int  main()
{
	init_face_imp("hope.dll");
	init_face("../net/init_net.pb", "../net/predict_net.pb");
	//目前是4096维向量，腾讯他们是512维。以后在优化
	std::vector<float> vec1;
	std::vector<float> vec2;
	mtcnn *find = nullptr;
	
	//cv::Mat img = cv::imread("../img/6.jpg");// , cv::IMREAD_GRAYSCALE);	
	cv::Mat img = cv::imread("txt/0.jpg");
	find = new mtcnn(img.rows, img.cols);
	cv::Rect rect;
	vector<cv::Point2f> five;

	if (!find->findFace(img, rect, five)) return 0;
	cv::Mat  dst;
	if (!get_vec(img, rect, five, vec1, 2,2, 0, 0,&dst)) return 0;

	for (int i = 0; i < vec1.size(); i++)
	{
		if (i % 16 == 0) printf("\n");
		printf("%f,", vec1[i]);
	}
	printf("\n");	
	delete find;
	five.clear();

	img = cv::imread("../img/220.jpg");// , cv::IMREAD_GRAYSCALE);	
	//cv::Mat img = cv::imread("D:/BaiduNetdiskDownload/mtcnn/20180608/7.jpg");
	find = new mtcnn(img.rows, img.cols);
	
	if (!find->findFace(img, rect, five)) return 0;	
	if (!get_vec(img, rect, five, vec2, 2, 2, 0, 0, &dst)) return 0;

	printf("vec legth is %d ,cos is %f \n", vec1.size(), cal_cos(vec1, vec2));

	cv::imshow("ok", dst);
	cv::waitKey(0);

	delete find;
	five.clear();
	return 0;
}


