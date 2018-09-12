#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

typedef int(*Init_Net)(int, int, const char*, const char*);
typedef bool(*Predict)();
typedef int(*Set_Input)(std::string, cv::Mat*, int);
typedef int(*Get_Output)(std::string, std::vector<float>&);
typedef int(*Get_Dims)(std::string, std::vector<int>&);
typedef void(*Vec2mat)(std::vector<float>, int, int, int, cv::Mat&);

extern Init_Net   init_net;
extern Predict   predict;
extern Set_Input  set_input;
extern Get_Output get_output;
extern Get_Dims   get_dims;
extern Vec2mat    vec2mat;

void init_imp(const char* dllfile);

