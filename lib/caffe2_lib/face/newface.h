#pragma once
#ifndef NEWFACE_H
#define NEWFACE_H

#include <opencv2/opencv.hpp>
#include <vector>

//����ͼ����ǰ����
#define  FACE_IMG_DEFAULT  0 //ͼ����ǰ����
#define  FACE_IMG_GRAY  1   //�Ҷȴ������ڵ�ͨ�������������
#define  FACE_IMG_LMCP  2   // LMcp�������ڵ�ͨ�������������

//�����иʽ
#define  NO_CROP    0  //���и�
#define  RECT_CROP  1  //�ô����rect�и�
#define  SEETA_CROP 2  //����seeta��ʽ�и�
#define  FACE_CROP  3  //���ǵĸ����и�
#define  CCL_CROP   4  //����ccl���ĵ��и�

//��һ�ķ�ʽ
#define NORM_DEFAULT  0  //��һ��(0,1)
#define NORM_NEGA    1  //��һ��(-1,1)

//������
#define NO_MODIFY  0  //������
#define IR_MODIFY  1  //�ֱ�������

//predict��ʽ
#define FACE_SINGLE  0  //��ͼԤ��
#define FACE_TWO     1  //��һ��ͼ����ͼ��ˮƽ��תͼ�ϲ���Ԥ��



	int  init_face(const char* init_net, const char*  predict_net);
	int  init_modify_face(const char* init_net, const char*  predict_net);

//3ͨ��
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

	//���ڲ��ԣ�����Ҫ����
    void get_modify_face(cv::Mat src, cv::Mat &dst);
//����cos����
	float  cal_cos(std::vector<float> a, std::vector<float> b);

//�����������ձ���,�����þ���ȱʡ����
	void set_face_ext(float top_x, float top_y);
//���ö�Զ���뿪ʼ��������,ȱʡ��rect.width=64ʱ����
	void set_distance_threshold(int width);




#endif