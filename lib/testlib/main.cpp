//#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include<ctime>
#include <opencv2/opencv.hpp>
#include "face/newface.h"
#include "mtcnn/mtcnn.h"
#include "ssd/ssd.h"



void test_ssd() {

	if (!initSsd("../net/ssd_init_net", "../net/ssd_pred_net")) {
		std::cout << "init ssd error" << std::endl;
		return;
	}

	std::clock_t start = clock();
	cv::Mat img = cv::imread("../img/16.jpg");
	//resize(img, img, Size(1096, 1096));
	std::vector<float> vec;
	if (!getSsdVec(img, vec)) {
		std::cout << "get ssd  ans error" << std::endl;
		return;
	}
	std::clock_t end = clock();
	
	printf("time is  %f, ans is %d \n", float(end - start) / CLOCKS_PER_SEC, vec.size());
}

void test_modify() {

	//先初始化
	init_modify_face("../net/modify_init_net", "../net/modify_pred_net");
	cv::Mat dst1,img1;
	char name[128];
	char dstname[128];
	int s = 5;
	for (int i = 3; i < 6; i++)
	{
		sprintf_s(name, "../img/%d.jpg",i);
		sprintf_s(dstname, "../img/%d_%d.jpg", i,s);
		printf("%s\n", name);
		dst1 = cv::imread(name, cv::IMREAD_GRAYSCALE);
		//resize(img1, dst1, Size(96, 96));
		//resize(dst1, dst1, Size(48, 48));
		//resize(dst1, dst1, Size(64, 64));
		get_modify_face(dst1, dst1);
		//resize(dst1, dst1, Size(96, 96));
		imwrite(dstname, dst1);
	}
	return;
}

int test_face() {
	init_face("../net/init_net.pb", "../net/predict_net.pb");
	
	//目前是4096维向量，腾讯他们是512维。以后在优化
	std::vector<float> vec0;
	std::vector<float> vec1;
	std::vector<float> vec2;

	mtcnn *find = nullptr;

	//cv::Mat img = cv::imread("../img/6.jpg");// , cv::IMREAD_GRAYSCALE);	
	cv::Mat img0 = cv::imread("../img/0.jpg");
	cv::Mat img1 = cv::imread("../img/1.jpg");
	cv::Mat img2 = cv::imread("../img/2.jpg");
	//resize(img0, img0, Size(64, 64));
	//resize(img1, img1, Size(64, 64));
	//resize(img2, img2, Size(64, 64));

	find = new mtcnn(img0.rows, img0.cols);
	cv::Rect rect;
	vector<cv::Point2f> five;

	if (!find->findFace(img0, rect, five)) return 0;

	//printf("(%f %f) (%f %f) (%f %f)  (%f  %f)  (%f %f)\n",five[0].x,five[0].y, five[1].x, five[1].y, five[2].x, five[2].y, 
		//five[3].x, five[3].y, five[4].x, five[4].y);
	//cv::imshow("ok", img0(rect));
	//moveWindow("ok", 80, 80);
	//cv::waitKey(0);	

	cv::Mat  dst;
	if (!get_vec(img0, rect, five, vec0, 1, 2, 1, 0, &dst)) return 0;
	five.clear();
	delete find;
	//return 0;
	cv::imshow("ok", dst);
	cv::waitKey(0);

	/*for (int i = 0; i < vec0.size(); i++)
	{
		if (i % 16 == 0) printf("\n");
		printf("%f,", vec0[i]);
	}
	printf("\n");*/

	find = new mtcnn(img1.rows, img1.cols);

	if (!find->findFace(img1, rect, five)) return 0;
	if (!get_vec(img1, rect, five, vec1, 1, 2, 1, 0, &dst)) return 0;
	five.clear();
	delete find;

	cv::imshow("ok", dst);
	cv::waitKey(0);


	/*for (int i = 0; i < vec1.size(); i++)
	{
		if (i % 16 == 0) printf("\n");
		printf("%f,", vec1[i]);
	}
	printf("\n");*/

	find = new mtcnn(img2.rows, img2.cols);

	if (!find->findFace(img2, rect, five)) return 0;
	if (!get_vec(img2, rect, five, vec2, 1, 2, 1, 0, &dst)) return 0;
	//five.clear();

	cv::imshow("ok", dst);
	cv::waitKey(0);

	//std::clock_t end = clock();

	//printf("time is  %f \n", double(end - start) / CLOCKS_PER_SEC);

	printf("vec0 and vec1: cos is %f \n", cal_cos(vec0, vec1));
	printf("vec0 and vec2: cos is %f \n", cal_cos(vec0, vec2));
	printf("vec1 and vec2: cos is %f \n", cal_cos(vec1, vec2));

	return 1;

}


void test_video()
{
	
	init_face("../net/init_net.pb", "../net/predict_net.pb");

	std::vector<float> svec1,bvec1;
	std::vector<float> vec2;
	mtcnn *find = nullptr;
	char filename[128];

	cv::Mat img = cv::imread("../img/0.jpg");


	find = new mtcnn(img.rows, img.cols);
	Rect rect;
	vector<Point2f> five;
	if (!find->findFace(img, rect, five)) return ;

	cv::Mat  dst;
	//set_distance_threshold(10000);
	printf("rect is ( %d  %d  %d  %d)\n", rect.x, rect.y, rect.height, rect.width);

	printf(" five is  (%f, %f)   (%f, %f )  (%f, %f) (%f, %f) (%f, %f)\n",
		five[0].x, five[0].y, five[1].x, five[1].y, five[2].x, five[2].y, five[3].x, five[3].y,
		five[4].x, five[4].y);
	if (!get_vec(img, rect, five, svec1, 1,2, 1,0,&dst)) return ;

	for (int i = 0; i < svec1.size(); i++)
	{
		if (i % 16 == 0) printf("\n");
		printf("%f,", svec1[i]);
	}
	printf("\n");
	
	

	int count = 0;
	sprintf_s(filename, "lmcp/s%d.jpg", count);
	//imwrite(filename, dst);
	set_distance_threshold(0);
	if (!get_vec(img, rect, five, bvec1, 1, 2, 1,0,&dst)) return;

	//float cos = cal_cos(svec1, bvec1);
	//printf("cos is %f \n", cos);
	sprintf_s(filename, "lmcp/b%d.jpg", count);
	//imwrite(filename, dst);
	set_distance_threshold(10000);
	
	delete find;
	five.clear();


	VideoCapture capture("../img/4.MP4");
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

		//printf("frame %d %d\n", frame.rows, frame.cols);
		five.clear();
		if (!find->findFace(show, rect, five)) continue ;
		vec2.clear();

		std::clock_t begin = clock();
		if (!get_vec(show, rect, five, vec2, 1,2, 1,0, &dst)) return ;		
		std::clock_t end = clock();
		elapsed_secs += double(end - begin);
		p++;
		
		float cos;
		//if(rect.width >= 64 ) cos =  cal_cos(bvec1, vec2);0.2128 0.206693
		//else cos = cal_cos(svec1, vec2);
		cos = cal_cos(svec1, vec2);
		//if (cos < 0.25) continue;
		count++;
		//sprintf_s(filename, "%d.jpg", count);
		//imwrite(filename, dst);

		//if(count >= 5) break;
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

int  main()
{
	test_face();
	//test_modify();
	//test_ssd();
	return 1;
}



