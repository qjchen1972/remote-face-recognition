#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2/opencv.hpp>


class CPreProcess
{
public:
	CPreProcess(){}
	~CPreProcess() {}

	void set_face_ext(float top_x = 0.3, float top_y = 0.3)
	{
		m_offset_pctx = top_x;
		m_offset_pcty = top_y;
	}
	
	void set_param(float exp_gem, float gass_sigma0, float gass_sigma1, float alpha, float tau)
	{
		m_exp_gemma = exp_gem;
		m_gass_sigma0 = gass_sigma0;
		m_gass_sigma1 = gass_sigma1;
		m_alpha = alpha;
		m_tau = tau;
	}

	void set_img_size(int w, int h)
	{
		img_size = cv::Size(w, h);
	}

	void crop_img(cv::Mat  &img)
	{
		cv::resize(img, img, img_size);
	}

	//图像的中心点为中心旋转，进行脸部对齐
	cv::Mat  get_face_with_center(cv::Rect rect, cv::Mat  img, std::vector<cv::Point2f> five)
	{
		cv::Point eye_direction(five[1].x - five[0].x, five[1].y - five[0].y);
		//calc rotation angle in radians
		float angle = atan2(eye_direction.y, eye_direction.x);
	
		cv::Mat roil;

		if (abs(angle) > 0.01)
		{
			int center_x = rect.x + rect.width / 2;
			int center_y = rect.y + rect.height / 2;
			int line = round(sqrt(rect.width * rect.width + rect.height*rect.height) / 2.0);

			cv::Rect temp_rect;
			temp_rect.x = center_x - line;
			temp_rect.y = center_y - line;
			temp_rect.width = line * 2;
			temp_rect.height = line * 2;
			modify_rect(temp_rect, img.cols, img.rows);

			img(temp_rect).copyTo(roil);

			std::vector<cv::Point2f> newfive;
			for (int i = 0; i < five.size(); i++)
				newfive.push_back(cv::Point2f(five[i].x - temp_rect.x, five[i].y - temp_rect.y));

			align_face_with_center(roil, angle, newfive);

			center_x = temp_rect.width / 2;
			center_y = temp_rect.height / 2;

			cv::Rect r;
			r.x = center_x - rect.width / 2;
			r.y = center_y - rect.height / 2;
			r.width = rect.width;
			r.height = rect.height;
			modify_rect(r, roil.cols, roil.rows);
			roil = roil(r);
		}
		else
		{
			img(rect).copyTo(roil);
		}
		return  roil;
	}

	//使用左眼为中心进行旋转，对齐
	cv::Mat  get_face(/*cv::Rect rect, */cv::Mat  img, std::vector<cv::Point2f> five)
	{
	
		//get the direction of eye
		cv::Point eye_direction(five[1].x - five[0].x, five[1].y - five[0].y);
		//calc rotation angle in radians
		float angle = atan2(eye_direction.y, eye_direction.x);
		// calculate offsets in original image
		int offset_h = floor(m_offset_pctx * img_size.width);
		int offset_v = floor(m_offset_pcty*img_size.height);
		
		//distance betwee eyes
		float eye_dis = sqrt(eye_direction.x*eye_direction.x + eye_direction.y * eye_direction.y);
		
		//calculate the reference eye-width 
		float reference = img_size.width - 2.0*offset_h;
		
		//scale factor
		float scale = eye_dis / reference;
		align_face(img, angle, five);

		//crop the rotated image
		cv::Rect temp_rect;
		temp_rect.x = round(five[0].x - scale * offset_h);
		temp_rect.y = round(five[0].y - scale * offset_v);
		temp_rect.width = round(img_size.width *scale);
		temp_rect.height = round(img_size.height*scale);

		modify_rect(temp_rect, img.cols, img.rows);	
		img = img(temp_rect);
		return img;
	}
	
	//参考 http://www.researchgate.net/publication/322568078_Face_Recognition_via_Centralized_Coordinate_Learning
	//做对齐
	cv::Mat  get_face_with_ccl(/*cv::Rect rect,*/ cv::Mat  img, std::vector<cv::Point2f> five)
	{		
		//get the direction of eye
		cv::Point eye_direction(five[1].x - five[0].x, five[1].y - five[0].y);
		//calc rotation angle in radians
		float angle = atan2(eye_direction.y, eye_direction.x);
		align_face(img, angle, five);

		float center_x = (five[2].x + five[3].x + five[4].x) / 3.0;
		float center_y = (five[2].y + five[3].y + five[4].y) / 3.0;

		float lc = abs(center_x - five[0].x);
		float rc = abs(center_x - five[1].x);

		float max = lc > rc ? lc : rc;

		// calculate offsets in original image
		int offset_h = floor(m_offset_pctx * img_size.width);
		int offset_v = floor(m_offset_pcty*img_size.height);
		//calculate the reference eye-width 
		float reference = img_size.width - 2.0*offset_h;
		//scale factor
		float scale = 2.0*max / reference;

		
		//crop the rotated image
		cv::Rect temp_rect;
		temp_rect.x = round(center_x - img_size.width *scale/2);
		temp_rect.y = round(five[0].y - scale * offset_v);
		temp_rect.width = round(img_size.width *scale);
		temp_rect.height = round(img_size.height*scale);

		modify_rect(temp_rect, img.cols, img.rows);
		img = img(temp_rect);
		return img;
	}
	
	//参考论文 Enhanced-local-texture-feature-sets-for-face-recognition-under-difficult-lighting-condition
	//对图像进行预处理
	cv::Mat preprocess(cv::Mat srcImg )
	{
		cv::Mat resultImage,resultImage1, resultImage2;
		cv::Mat dst;
		/* gamma correction */
		srcImg.convertTo(resultImage, CV_32FC(srcImg.channels()), 1.0 / 255);
		cv::pow(resultImage, m_exp_gemma, resultImage);

		cv::normalize(resultImage, dst, 0, 1, CV_MINMAX);
		cv::convertScaleAbs(dst, dst, 255);
		cv::imwrite("s1.jpg", dst);

		/* DoG filter */
		int dia = 9;
		cv::GaussianBlur(resultImage, resultImage1, cv::Size(dia, dia), m_gass_sigma0, m_gass_sigma0);
		cv::normalize(resultImage1, dst, 0, 1, CV_MINMAX);
		cv::convertScaleAbs(dst, dst, 255);
		cv::imwrite("s2.jpg", dst);

		cv::GaussianBlur(resultImage, resultImage2, cv::Size(dia, dia), m_gass_sigma1, m_gass_sigma1);
		cv::normalize(resultImage2, dst, 0, 1, CV_MINMAX);
		cv::convertScaleAbs(dst, dst, 255);
		cv::imwrite("s3.jpg", dst);

		resultImage = resultImage1 - resultImage2;

		cv::normalize(resultImage, dst, 0, 1, CV_MINMAX);
		cv::convertScaleAbs(dst, dst, 255);
		cv::imwrite("s4.jpg", dst);

		
		/* Contrast Equalization */
		// img = img/(mean2(abs(img).^alpha)^(1/alpha));  
		resultImage1 = cv::abs(resultImage);
		cv::pow(resultImage1, m_alpha, resultImage1);
		resultImage /= cv::pow(cv::mean(resultImage1).val[0], 1.0/ m_alpha);

		cv::normalize(resultImage, dst, 0, 1, CV_MINMAX);
		cv::convertScaleAbs(dst, dst, 255);
		cv::imwrite("s5.jpg", dst);

		// img = img/(mean2(min(tau,abs(img)).^alpha)^(1/alpha)); 
		resultImage1 = cv::abs(resultImage);
		resultImage1 = cv::min(m_tau, resultImage1);
		cv::pow(resultImage1, m_alpha, resultImage1);
		resultImage = resultImage / cv::pow(cv::mean(resultImage1).val[0], 1.0/ m_alpha) / m_tau;

		cv::normalize(resultImage, dst, 0, 1, CV_MINMAX);
		cv::convertScaleAbs(dst, dst, 255);
		cv::imwrite("s6.jpg", dst);

		// tanh :tanh x = (e^(x)-e^(-x)) /(e^x+e^(-x)) 
		cv::exp(resultImage, resultImage1);
		cv::exp((0 - resultImage), resultImage2);
		resultImage = resultImage1 - resultImage2;
		resultImage1 = resultImage1 + resultImage2;
		resultImage = resultImage / resultImage1;

		//normalize
		cv::normalize(resultImage, resultImage, 0, 1, CV_MINMAX);
		cv::convertScaleAbs(resultImage, resultImage, 255);
		return resultImage;
	}

private:
	
	float m_offset_pctx = 0.30;
	float m_offset_pcty = 0.30;

	
	float m_exp_gemma = 0.2;
	float m_gass_sigma0 = 1.0;
	float m_gass_sigma1 = 2.0;
	float m_alpha = 0.1;
	float m_tau = 10.0;

	cv::Size img_size = cv::Size(64, 64);
	
	//纠正 rect
	void modify_rect(cv::Rect &r, int width, int height)
	{
		if (r.x < 0) r.x = 0;
		if (r.y < 0) r.y = 0;
		if (r.width + r.x > width) r.width = width - r.x;
		if (r.height + r.y > height) r.height = height - r.y;
	}	

	void align_face(cv::Mat &img, float angle, std::vector<cv::Point2f> &five)
	{		
		std::vector<cv::Point2f> src, dest;
		float center_x = five[0].x;
		float center_y = five[0].y;

		dest.push_back(cv::Point2f(five[0].x, five[0].y));
		src.push_back(cv::Point2f(five[0].x, five[0].y));

		for (int i = 1; i < five.size(); i++)
		{
			float src_x = (five[i].x - center_x)  * cos(angle) + (five[i].y - center_y)  * sin(angle) + center_x;
			float src_y = (five[i].y - center_y) * cos(angle) - (five[i].x - center_x)* sin(angle) + center_y;
			dest.push_back(cv::Point2f(src_x, src_y));
			src.push_back(five[i]);
			five[i].x = src_x;
			five[i].y = src_y;
		}
		cv::Mat warp_mat = cv::estimateRigidTransform(src, dest, false);
		cv::warpAffine(img, img, warp_mat, img.size());
	}

	void align_face_with_center(cv::Mat &img, float angle, std::vector<cv::Point2f> five)
	{
		std::vector<cv::Point2f> src, dest;
		float center_x = 1.0*img.cols / 2;
		float center_y = 1.0*img.rows / 2;

		for (int i = 0; i < five.size(); i++)
		{
			float src_x = (five[i].x - center_x)  * cos(angle) + (five[i].y - center_y)  * sin(angle) + center_x;
			float src_y = (five[i].y - center_y) * cos(angle) - (five[i].x - center_x)* sin(angle) + center_y;
			dest.push_back(cv::Point2f(src_x, src_y));
			src.push_back(five[i]);
		}
		cv::Mat warp_mat = cv::estimateRigidTransform(src, dest, false);
		cv::warpAffine(img, img, warp_mat, img.size());
	}

};
 