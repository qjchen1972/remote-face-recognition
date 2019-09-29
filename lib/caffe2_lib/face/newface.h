#pragma once
#ifndef NEWFACE_H
#define NEWFACE_H

#include <opencv2/opencv.hpp>
#include <vector>

//输入图像做前处理
#define  FACE_IMG_DEFAULT  0 //图像不做前处理
#define  FACE_IMG_GRAY  1   //灰度处理，用于单通道的输入的网络
#define  FACE_IMG_LMCP  2   // LMcp处理，用于单通道的输入的网络

//区域切割方式
#define  NO_CROP    0  //不切割
#define  RECT_CROP  1  //用传入的rect切割
#define  SEETA_CROP 2  //用于seeta方式切割
#define  FACE_CROP  3  //我们的割脸切割
#define  CCL_CROP   4  //基于ccl论文的切割

//归一的方式
#define NORM_DEFAULT  0  //归一在(0,1)
#define NORM_NEGA    1  //归一在(-1,1)

//脸修正
#define NO_MODIFY  0  //不修正
#define IR_MODIFY  1  //分辨率修正

//predict方式
#define FACE_SINGLE  0  //单图预测
#define FACE_TWO     1  //用一张图的正图和水平翻转图合并来预测



	int  init_face(const char* init_net, const char*  predict_net);
	int  init_modify_face(const char* init_net, const char*  predict_net);

//3通道
	int  get_vec(cv::Mat img,
	         cv::Rect rect, 
	         std::vector<cv::Point2f> five, 
	         std::vector<float> &vec, 
	         int img_type = FACE_IMG_LMCP, 
	         int crop_mode = SEETA_CROP,
		     int norm_mode = NORM_DEFAULT,
			// int modify_mode = NO_MODIFY,
	         int predict_type = FACE_SINGLE,
	         cv::Mat *dst = NULL);

	//用于测试，不需要导出
    void get_modify_face(cv::Mat src, cv::Mat &dst);
//计算cos距离
	float  cal_cos(std::vector<float> a, std::vector<float> b);

//设置区域内收比率,不设置就是缺省区域
	void set_face_ext(float top_x, float top_y);
//设置多远距离开始进行内收,缺省是rect.width=64时内收
	void set_distance_threshold(int width);




#endif