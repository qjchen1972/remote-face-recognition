#pragma once

#define USE_OPENCV
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif


namespace caffe2net {
	class Utils {
	public:
		Utils() {}
		~Utils() {}

#ifdef USE_OPENCV

		void Mat2array(cv::Mat src, unsigned char *data) {
			int p = 0;

#pragma omp parallel for
			for (int h = 0; h < src.rows; h++)
			{
				unsigned char* s = src.ptr<unsigned char>(h);
				int img_index = 0;
				for (int w = 0; w < src.cols; w++)
				{
					for (int c = 0; c < src.channels(); c++)
					{
						int data_index;
						if(src.channels()>1)
							data_index = (2-c) * src.rows*src.cols + h * src.cols + w;
						else
							data_index = c * src.rows*src.cols + h * src.cols + w;
						unsigned char g = s[img_index++];
						data[data_index] = g;
					}
				}
		}
	}

		void array2Mat(std::vector<float> data, int chn, int width, int height, cv::Mat &img) {

			cv::Mat dst_img(width, height, CV_32FC(chn));

#pragma omp parallel for
			for (int h = 0; h < dst_img.rows; h++) {
				float* s = dst_img.ptr<float>(h);
				int img_index = 0;
				for (int w = 0; w < dst_img.cols; w++) {
					for (int c = 0; c < dst_img.channels(); c++) {
						int data_index = c * dst_img.rows * dst_img.cols + h * dst_img.cols + w;
						s[img_index++] = data[data_index];
					}
				}
			}
			cv::convertScaleAbs(dst_img, img, 255);
		}
#endif
	};
}
