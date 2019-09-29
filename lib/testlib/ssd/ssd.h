#pragma once

#include <opencv2/opencv.hpp>
#include <vector>


int  initSsd(const char* init_net, const char*  predict_net);
int  getSsdVec(cv::Mat img, std::vector<float> &vec);







