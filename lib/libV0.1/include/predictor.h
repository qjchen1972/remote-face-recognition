#pragma once
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>
#include<math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace caffe2;


class CPredictor
{
public:
	CPredictor(){}
	~CPredictor(){}

	void init_net(const char* init_net, const char* predict_net)
	{		
		NetDef init_net_def, predict_net_def;
		CAFFE_ENFORCE(ReadProtoFromFile(init_net, &init_net_def));
		CAFFE_ENFORCE(ReadProtoFromFile(predict_net, &predict_net_def));
		face_workspace.RunNetOnce(init_net_def);

		init_net_def.mutable_device_option()->set_device_type(CPU);
		predict_net_def.mutable_device_option()->set_device_type(CPU);
		face_predict = CreateNet(predict_net_def, &face_workspace);

		auto tensor = face_workspace.CreateBlob(inputstr)->GetMutable<TensorCPU>();
		input_dims = tensor->dims();
		return;
	}

	void get_img_dims(int &chn, int &height, int &width)
	{
		chn = input_dims[1];
		height = input_dims[2];
		width = input_dims[3];
	}

	int predict(Mat img, std::vector<float>  &vec, int  type, int norm_mode)
	{
		TensorCPU input;
		if (!preProcess(img, type, input,norm_mode)) return 0;

		auto tensor = face_workspace.CreateBlob(inputstr)->GetMutable<TensorCPU>();
		tensor->ResizeLike(input);
		tensor->ShareData(input);

		face_predict->Run();
		TensorCPU output = TensorCPU(face_workspace.GetBlob(outputstr)->Get<TensorCPU>());
		const float * probs = output.data<float>();
		vector<TIndex> dims = output.dims();
		vec.resize(dims[0]*dims[1]);
		copy(probs, probs + dims[0] * dims[1], vec.begin());
		return 1;
	}
		

private:
	const string inputstr = "input";
	const string outputstr = "output";

	Workspace  face_workspace;
	unique_ptr<NetBase> face_predict;
	vector<TIndex> input_dims;
	
	int preProcess(Mat img, int type, TensorCPU &tensor, int norm_mode)
	{
		vector<TIndex> dims;
		vector<float> data;

		if (img.channels() != input_dims[1] || img.rows != input_dims[2] || img.cols != input_dims[3])
		{
			printf("ERROR:input need (%d, %d,%d) \n", input_dims[1], input_dims[2], input_dims[3]);
			return 0;
		}
	
		int batch_size = img.channels() * img.rows * img.cols;
		
		if (type == 0)
		{
			vector<TIndex> tdims({ 1, img.channels(), img.rows, img.cols });
			vector<float> tdata(1 * batch_size);
			dims = tdims;
			data = tdata;

			if (norm_mode == 0)
			{
				img.convertTo(img, CV_32FC(img.channels()), 1.0 / 255, 0);
			}
			else
			{
				img.convertTo(img, CV_32FC(img.channels()), 1.0 / 255, 0);
				img = img - 0.5;
				img = img / 0.5;
				//img = img - 127.5;
				//img = img / 127.5;
				//img = 2.0*(img - 0.5);
			}
		
			if (img.channels() == 1)
			{
				copy((float *)img.datastart, (float *)img.dataend, data.begin());
			}
			else
			{
				int size = img.rows * img.cols;
				Mat rgb_channel[3];
				split(img, rgb_channel);
				for (int h = 0; h < img.rows; h++)
					for (int w = 0; w < img.cols; w++)
					{
						data[h * img.cols + w] = rgb_channel[2].at<float>(h, w);
						data[size + h * img.cols + w] = rgb_channel[1].at<float>(h, w);
						data[2 * size + h * img.cols + w] = rgb_channel[0].at<float>(h, w);
					}
			}
		}
		else
		{

			vector<TIndex> tdims({ 2, img.channels(), img.rows, img.cols });
			vector<float> tdata(2 * batch_size);

			dims = tdims;
			data = tdata;

			Mat himg;
			flip(img, himg, 1);		

			if (norm_mode == 0)
			{
				img.convertTo(img, CV_32FC(img.channels()), 1.0 / 255, 0);
				himg.convertTo(img, CV_32FC(img.channels()), 1.0 / 255, 0);
			}
			else
			{
				img.convertTo(img, CV_32FC(img.channels()), 1.0, 0);
				img = img - 127.5;
				img = img / 127.5;
				himg.convertTo(img, CV_32FC(img.channels()), 1.0, 0);
				img = img - 127.5;
				img = img / 127.5;

			}

			if (img.channels() == 1)
			{
				copy((float *)img.datastart, (float *)img.dataend, data.begin());
				copy((float *)himg.datastart, (float *)himg.dataend, data.begin()+ batch_size);
			}
			else
			{
				int size = img.rows * img.cols;
				Mat rgb_channel[3], hrgb_channel[3];
				split(img, rgb_channel);
				split(himg, hrgb_channel);
				for (int h = 0; h < img.rows; h++)
					for (int w = 0; w < img.cols; w++)
					{
						data[h * img.cols + w] = rgb_channel[2].at<float>(h, w);
						data[size + h * img.cols + w] = rgb_channel[1].at<float>(h, w);
						data[2 * size + h * img.cols + w] = rgb_channel[0].at<float>(h, w);
						data[batch_size + h * img.cols + w] = hrgb_channel[2].at<float>(h, w);
						data[batch_size + size + h * img.cols + w] = hrgb_channel[1].at<float>(h, w);
						data[batch_size + 2 * size + h * img.cols + w] = hrgb_channel[0].at<float>(h, w);
					}
			}
		}

		tensor.CopyFrom(TensorCPU(dims, data, NULL));
		return 1;
	}	
};
