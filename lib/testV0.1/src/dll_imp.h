#pragma once
//#include "newface.h"

#include <opencv2/opencv.hpp>
#include <vector>

//typedef int(*Ini_Modify_Face)(const char*, const char*);
//typedef void(*Get_Modify_Face)(cv::Mat , cv::Mat&);
typedef int(*Ini_Face)(const char*, const char*);
typedef int(*Get_Vec)(cv::Mat, cv::Rect, std::vector<cv::Point2f>, std::vector<float>&, int, int, int,int,cv::Mat*);
typedef float(*Cal_Cos)(std::vector<float>, std::vector<float>);
typedef void(*Set_Face_Ext)(float, float);
typedef void(*Set_Dis_Threshold)(int);

//extern Ini_Modify_Face init_modify_face;
//extern Get_Modify_Face get_modify_face;
extern Ini_Face init_face;
extern Get_Vec   get_vec;
extern Cal_Cos  cal_cos;
extern Set_Face_Ext set_face_ext;
extern Set_Dis_Threshold set_distance_threshold;
void init_face_imp(const char* dllfile);

