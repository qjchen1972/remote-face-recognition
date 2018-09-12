#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include<ctime>
#include <opencv2/opencv.hpp>
#include "cnn_net.h"

using namespace std;
using namespace cv;

int  main()
{
	Cnn_Net  net;
	net.init_net(1, 1, "netgpu/init_net.pb", "netgpu/predict_net.pb");
	cv::Mat img = cv::imread("img/0.jpg", cv::IMREAD_GRAYSCALE);
	resize(img, img, Size(64, 64));

	cv::Mat p[1];
	p[0] = img;

	net.set_input("input", p, 1);
	net.predict();
	
	std::vector<float> vec;
	net.get_output("output", vec);
	
	for (int i = 0; i < vec.size(); i++)
	{
		if (i % 16 == 0) printf("\n");
		printf("%f,", vec[i]);
	}
	printf("\n");
	return 0;
}

