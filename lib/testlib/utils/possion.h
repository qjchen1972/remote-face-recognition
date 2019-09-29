#pragma once
#ifndef _POSSION__H
#define _POSSION__H

#include "opencv2/photo.hpp"

#include <vector>

namespace cv
{
	class Possion
	{
	public:
		Possion()
		{
			chan = 1;
		}

		void set_outchan(int value)
		{
			chan = value;
		}
		void evaluate(const cv::Mat &I, cv::Mat &gx, cv::Mat &gy, cv::Mat &cloned)
		{
			initVariables(I);

			if (chan < I.channels()) chan = I.channels();
			if (chan == 1)
				cloned = Mat(I.size(), CV_8UC1);
			else
				cloned = Mat(I.size(), CV_8UC3);
			poisson(I, gx, gy);
			if (chan == 1)
				cloned = res;
			else
				merge(output, 3, cloned);
		}

		

		void computeGradientX(const cv::Mat &img, cv::Mat &gx)
		{
			Mat kernel = Mat::zeros(1, 3, CV_8S);
			kernel.at<char>(0, 2) = 1;
			kernel.at<char>(0, 1) = -1;
			if (chan == 1)
			{
				gx = Mat(img.size(), CV_32FC1);
				filter2D(img, gx, CV_32F, kernel);	
			}
			else
			{
				gx = Mat(img.size(), CV_32FC3);
				if (img.channels() == 3)
				{
					filter2D(img, gx, CV_32F, kernel);
				}
				else if (img.channels() == 1)
				{
					Mat tmp[3];
					for (int chan = 0; chan < 3; ++chan)
					{
						filter2D(img, tmp[chan], CV_32F, kernel);
					}
					merge(tmp, 3, gx);
				}
			}
		}


		void computeGradientY(const cv::Mat &img, cv::Mat &gy)
		{
			Mat kernel = Mat::zeros(3, 1, CV_8S);
			kernel.at<char>(2, 0) = 1;
			kernel.at<char>(1, 0) = -1;
			if (chan == 1)
			{
				gy = Mat(img.size(), CV_32FC1);
				filter2D(img, gy, CV_32F, kernel);
			}
			else
			{
				gy = Mat(img.size(), CV_32FC3);
				if (img.channels() == 3)
				{
					filter2D(img, gy, CV_32F, kernel);
				}
				else if (img.channels() == 1)
				{
					Mat tmp[3];
					for (int chan = 0; chan < 3; ++chan)
					{
						filter2D(img, tmp[chan], CV_32F, kernel);
					}
					merge(tmp, 3, gy);
				}
			}
		}

		void computeLaplacianX(const cv::Mat &img, cv::Mat &gxx)
		{
			Mat kernel = Mat::zeros(1, 3, CV_8S);
			kernel.at<char>(0, 0) = -1;
			kernel.at<char>(0, 1) = 1;

			if (chan == 1)
				gxx = Mat(img.size(), CV_32FC1);
			else
				gxx = Mat(img.size(), CV_32FC3);

			filter2D(img, gxx, CV_32F, kernel);
		}



		void computeLaplacianY(const cv::Mat &img, cv::Mat &gyy)
		{
			Mat kernel = Mat::zeros(3, 1, CV_8S);
			kernel.at<char>(0, 0) = -1;
			kernel.at<char>(1, 0) = 1;
			if (chan == 1)
				gyy = Mat(img.size(), CV_32FC1);
			else
				gyy = Mat(img.size(), CV_32FC3);
			filter2D(img, gyy, CV_32F, kernel);
		}


	private:

		void initVariables(const cv::Mat &destination)
		{
			if (chan == 1)
				res = Mat(destination.size(), CV_8UC1);
			//init of the filters used in the dst
			const int w = destination.cols;
			filter_X.resize(w - 2);
			for (int i = 0; i < w - 2; ++i)
				filter_X[i] = 2.0f * std::cos(static_cast<float>(CV_PI) * (i + 1) / (w - 1));

			const int h = destination.rows;
			filter_Y.resize(h - 2);
			for (int j = 0; j < h - 2; ++j)
				filter_Y[j] = 2.0f * std::cos(static_cast<float>(CV_PI) * (j + 1) / (h - 1));
		}



		void poisson(const cv::Mat &destination, cv::Mat &gx, cv::Mat &gy)
		{
			Mat laplacianX;
			Mat laplacianY;
			computeLaplacianX(gx, laplacianX);
			computeLaplacianY(gy, laplacianY);

			if (chan == 1)
				poissonSolver(destination, laplacianX, laplacianY, res);
			else
			{
				split(laplacianX, rgbx_channel);
				split(laplacianY, rgby_channel);
				split(destination, output);
				if (destination.channels() == 1)
				{
					poissonSolver(output[0], rgbx_channel[0], rgby_channel[0], output[0]);
					output[1] = output[0];
					output[2] = output[0];
				}				
				else
				{
					for (int chan = 0; chan < 3; ++chan)
					{
						poissonSolver(output[chan], rgbx_channel[chan], rgby_channel[chan], output[chan]);
					}
				}
			}
		}	


		void dst(const Mat& src, Mat& dest, bool invert = false)
		{
			Mat temp = Mat::zeros(src.rows, 2 * src.cols + 2, CV_32F);

			int flag = invert ? DFT_ROWS + DFT_SCALE + DFT_INVERSE : DFT_ROWS;

			src.copyTo(temp(Rect(1, 0, src.cols, src.rows)));

			for (int j = 0; j < src.rows; ++j)
			{
				float * tempLinePtr = temp.ptr<float>(j);
				const float * srcLinePtr = src.ptr<float>(j);
				for (int i = 0; i < src.cols; ++i)
				{
					tempLinePtr[src.cols + 2 + i] = -srcLinePtr[src.cols - 1 - i];
				}
			}

			Mat planes[] = { temp, Mat::zeros(temp.size(), CV_32F) };
			Mat complex;

			merge(planes, 2, complex);
			dft(complex, complex, flag);
			split(complex, planes);
			temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);

			for (int j = 0; j < src.cols; ++j)
			{
				float * tempLinePtr = temp.ptr<float>(j);
				for (int i = 0; i < src.rows; ++i)
				{
					float val = planes[1].ptr<float>(i)[j + 1];
					tempLinePtr[i + 1] = val;
					tempLinePtr[temp.cols - 1 - i] = -val;
				}
			}

			Mat planes2[] = { temp, Mat::zeros(temp.size(), CV_32F) };

			merge(planes2, 2, complex);
			dft(complex, complex, flag);
			split(complex, planes2);

			temp = planes2[1].t();
			dest = Mat::zeros(src.size(), CV_32F);
			temp(Rect(0, 1, src.cols, src.rows)).copyTo(dest);
		}



		void idst(const Mat& src, Mat& dest)
		{
			dst(src, dest, true);
		}



		void solve(const Mat &img, Mat& mod_diff, Mat &result)
		{
			const int w = img.cols;
			const int h = img.rows;

			Mat res;
			dst(mod_diff, res);

			for (int j = 0; j < h - 2; j++)
			{
				float * resLinePtr = res.ptr<float>(j);
				for (int i = 0; i < w - 2; i++)
				{
					resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);
				}
			}

			idst(res, mod_diff);

			unsigned char *  resLinePtr = result.ptr<unsigned char>(0);
			const unsigned char * imgLinePtr = img.ptr<unsigned char>(0);
			const float * interpLinePtr = NULL;

			//first col
			for (int i = 0; i < w; ++i)
				result.ptr<unsigned char>(0)[i] = img.ptr<unsigned char>(0)[i];

			for (int j = 1; j < h - 1; ++j)
			{
				resLinePtr = result.ptr<unsigned char>(j);
				imgLinePtr = img.ptr<unsigned char>(j);
				interpLinePtr = mod_diff.ptr<float>(j - 1);

				//first row
				resLinePtr[0] = imgLinePtr[0];

				for (int i = 1; i < w - 1; ++i)
				{
					//saturate cast is not used here, because it behaves differently from the previous implementation
					//most notable, saturate_cast rounds before truncating, here it's the opposite.
					float value = interpLinePtr[i - 1];
					if (value < 0.)
						resLinePtr[i] = 0;
					else if (value > 255.0)
						resLinePtr[i] = 255;
					else
						resLinePtr[i] = static_cast<unsigned char>(value);
				}
				//last row
				resLinePtr[w - 1] = imgLinePtr[w - 1];
			}

			//last col
			resLinePtr = result.ptr<unsigned char>(h - 1);
			imgLinePtr = img.ptr<unsigned char>(h - 1);
			for (int i = 0; i < w; ++i)
				resLinePtr[i] = imgLinePtr[i];
		}


		void poissonSolver(const cv::Mat &img, cv::Mat &gxx, cv::Mat &gyy, cv::Mat &result)
		{
			const int w = img.cols;
			const int h = img.rows;

			Mat lap = Mat(img.size(), CV_32FC1);

			lap = gxx + gyy;

			Mat bound = img.clone();

			rectangle(bound, Point(1, 1), Point(img.cols - 2, img.rows - 2), Scalar::all(0), -1);
			Mat boundary_points;
			Laplacian(bound, boundary_points, CV_32F);

			boundary_points = lap - boundary_points;

			Mat mod_diff = boundary_points(Rect(1, 1, w - 2, h - 2));

			solve(img, mod_diff, result);
		}

		//std::vector <cv::Mat> rgbx_channel, rgby_channel, output;
		Mat rgbx_channel[3];
		Mat rgby_channel[3];
		Mat output[3];
		std::vector<float> filter_X, filter_Y;
		char chan;
		Mat res;
	};
}
#endif
