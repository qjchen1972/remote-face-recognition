#include<math.h>
#include "../include/preprocess.h"
#include "../include/predictor.h"
#include "../include/modify_face.h"
#include "../seeta_include/aligner.h"
#include "newface.h"

using namespace std;
using namespace cv;
using namespace seeta;

static CModeify_Face modify;
static CPredictor predictor;
static CPreProcess pre;
static std::shared_ptr<Aligner> small_aligner_;
static std::shared_ptr<Aligner> big_aligner_;

static int rect_limit = 64;
static float seeta_ext_x = 0.05;
static float seeta_ext_y = 0.05;
static int start_chn = 1;
static int start_width = 0;
static int start_height = 0;
static int ext_width = 0;
static int ext_height = 0;


void set_distance_threshold(int width)
{
	rect_limit = width;
}

int init_face(const char* init_net, const char*  predict_net)
{
	predictor.init_net(init_net, predict_net);
	predictor.get_img_dims(start_chn, start_height, start_width);
	pre.set_img_size(start_width,start_height);
	ext_width = start_width + 2*round(seeta_ext_x * start_width);
	ext_height = start_height + 2*round(seeta_ext_y * start_height);
	big_aligner_.reset(new seeta::Aligner(ext_height, ext_width,"linear"));
	small_aligner_.reset(new seeta::Aligner(start_height, start_width , "linear"));
	//big_aligner_.reset(new seeta::Aligner(ext_height, ext_width));
	//small_aligner_.reset(new seeta::Aligner(start_height, start_width));

	return 1;
}

int init_modify_face(const char* init_net, const char*  predict_net)
{
	modify.init_net(init_net, predict_net);
	return 1;
}

void get_modify_face(Mat src,Mat &dst)
{
	modify.predict(src, dst);
}

static Mat  crop(cv::Mat img,
	             cv::Rect rect,
	             std::vector<cv::Point2f> five,
	             int crop_mode)
{
	Mat src;
	if (crop_mode == NO_CROP)
	{
		img.copyTo(src);
		pre.crop_img(src);
		return src;
	}
	else if (crop_mode == RECT_CROP)
	{
		src = pre.get_face_with_center(rect, img, five);
		pre.crop_img(src);
		return src;
	}
	else if (crop_mode == SEETA_CROP)
	{
		float point_data[10];
		for (int i = 0; i < 5; ++i)
		{
			point_data[i * 2] = five[i].x;
			point_data[i * 2 + 1] = five[i].y;
		}
		ImageData src_img_data(img.cols, img.rows, img.channels());
		src_img_data.data = img.data;
		if (rect.width >= rect_limit)
		{
			cv::Mat dst_img(ext_height, ext_width, CV_8UC(img.channels()));
			ImageData dst_img_data(ext_width, ext_height, img.channels());
			dst_img_data.data = dst_img.data;
			big_aligner_->Alignment(src_img_data, point_data, dst_img_data);
			cv::Rect real;
			real.x = (ext_width - start_width) / 2;
			real.y = (ext_height - start_height ) / 2;
			real.width = start_width;
			real.height = start_height;
			dst_img = dst_img(real);
			return dst_img;
		}
		else
		{
			cv::Mat dst_img(start_height, start_width, CV_8UC(img.channels()));
			ImageData dst_img_data(start_width, start_height, img.channels());
			dst_img_data.data = dst_img.data;
			small_aligner_->Alignment(src_img_data, point_data, dst_img_data);
			return dst_img;
		}
	}
	else if (crop_mode == FACE_CROP)
	{
		src = pre.get_face(rect, img, five);
		pre.crop_img(src);
		return src;
	}
	else if (crop_mode == CCL_CROP)
	{
		src = pre.get_face_with_ccl(rect, img, five);
		pre.crop_img(src);
		return src;
	}
	else;
	return src;
}

//3Í¨µÀ
int  get_vec(cv::Mat img,
	cv::Rect rect,
	std::vector<cv::Point2f> five,
	std::vector<float> &vec,
	int img_type, //= FACE_IMG_LMCP
	int crop_mode, //= SEETA_CROP
	int norm_mode,
	//int modify_mode,// = NO_MODIFY,
	int predict_type, //= FACE_SINGLE
	cv::Mat *dst ) //= NULL
{
	
	Mat src;

	if (img_type == FACE_IMG_GRAY)
	{
		/*if (start_chn != 1)
		{
			printf("ERROR:net input need channel 3\n");
			return 0;
		}*/
		if (img.channels() != 1)	cvtColor(img, img, CV_BGR2GRAY);		
		src = crop(img, rect, five, crop_mode);
		if (start_chn != 1)
		{
			Mat rgb_channel[3];
			rgb_channel[0] = src;
			rgb_channel[1] = src;
			rgb_channel[2] = src;
			merge(rgb_channel, 3, src);
		}
	}
	else if (img_type == FACE_IMG_LMCP)
	{
		/*if (start_chn != 1)
		{
			printf("ERROR:net input need channel 3\n");
			return 0;
		}*/
		//if (start_chn == 1 && img.channels() != 1)	cvtColor(img, img, CV_BGR2GRAY);
		if (img.channels() != 1)	cvtColor(img, img, CV_BGR2GRAY);
		src = crop(img, rect, five, crop_mode);
		src = pre.preprocess(src);
		if (start_chn != 1)
		{
			Mat rgb_channel[3];
			rgb_channel[0] = src;
			rgb_channel[1] = src;
			rgb_channel[2] = src;
			merge(rgb_channel, 3, src);
		}
	}
	else 
	{
		if(start_chn == 1 && img.channels() != 1) cvtColor(img, img, CV_BGR2GRAY);
		src = crop(img, rect, five, crop_mode);
	}

	/*if (modify_mode != NO_MODIFY)
	{
		modify.predict(src, src);
	}*/

	if(dst != NULL ) *dst= src;
	return predictor.predict(src, vec, predict_type, norm_mode);
}

void set_face_ext(float top_x, float top_y)
{
	seeta_ext_x = top_x;
	seeta_ext_y = top_y;
	//pre.set_face_ext(top_x, top_y);
}

float  cal_cos(std::vector<float> a, std::vector<float> b)
{
	float a_dis = 0;
	float b_dis = 0;
	float sum = 0;
	for (int i = 0; i < a.size(); i++)
	{
		sum += a[i] * b[i];
		a_dis += a[i] * a[i];
		b_dis += b[i] * b[i];
	}
	return sum / (sqrt(a_dis) * sqrt(b_dis));
}

