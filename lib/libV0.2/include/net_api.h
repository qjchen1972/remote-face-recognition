#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

//是否用GPU
#define  CNN_NET_CPU  0
#define  CNN_NET_GPU  1

//归一的方式
#define  CNN_NORM_ZERO  0  //归一在(0,1)
#define  CNN_NORM_NEGA  1  //归一在(-1,1)

#ifdef __cplusplus
extern "C" {
#endif

	_declspec(dllexport) int init_net(int type, int mode,const char* init_net, const char*  predict_net);
	_declspec(dllexport) bool predict();
	_declspec(dllexport) int set_input(std::string  str, cv::Mat *img, int size);
	_declspec(dllexport) int get_output(std::string  str, std::vector<float>  &vec);
	_declspec(dllexport) int get_dims(std::string str, std::vector<int>  &vec);
	_declspec(dllexport) void vec2mat(std::vector<float>  vec, int chn, int row, int col, cv::Mat &dst);


#ifdef __cplusplus
}
#endif 

