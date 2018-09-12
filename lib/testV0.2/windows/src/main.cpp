#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include<ctime>
#include <opencv2/opencv.hpp>
#include "dll_imp.h"

using namespace std;
using namespace cv;

int  main()
{
	init_imp("caffe2_net.dll");
	init_net(1, 0, "../netgpu/init_net.pb", "../netgpu/predict_net.pb");
	cv::Mat img = cv::imread("../img/0.jpg", cv::IMREAD_GRAYSCALE);
	resize(img, img, Size(64, 64));
	cv::Mat p[1];
	p[0] = img;
	cout << "ok0" << endl;

	set_input("input", p, 1);
	cout << "ok1" << endl;
	predict();
	cout << "ok2" << endl;

	std::vector<float> vec;
	get_output("output", vec);
	cout << "ok3" << endl;

	for (int i = 0; i < vec.size(); i++)
	{
		if (i % 16 == 0) printf("\n");
		printf("%f,", vec[i]);
	}
	printf("\n");
	return 0;
}

