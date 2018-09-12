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


class CModeify_Face
{
public:
	CModeify_Face() {}
	~CModeify_Face() {}

	void init_net(const char* init_net, const char* predict_net)
	{
		NetDef init_net_def, predict_net_def;
		CAFFE_ENFORCE(ReadProtoFromFile(init_net, &init_net_def));
		CAFFE_ENFORCE(ReadProtoFromFile(predict_net, &predict_net_def));
		face_workspace.RunNetOnce(init_net_def);

		init_net_def.mutable_device_option()->set_device_type(CPU);
		predict_net_def.mutable_device_option()->set_device_type(CPU);
		face_predict = CreateNet(predict_net_def, &face_workspace);

		auto in_tensor = face_workspace.CreateBlob(inputstr)->GetMutable<TensorCPU>();
		input_dims = in_tensor->dims();
	
		return;
	}

	void get_img_dims(int &chn, int &height, int &width)
	{
		chn = input_dims[1];
		height = input_dims[2];
		width = input_dims[3];
	}

	int predict(Mat img, Mat &dst)
	{
		TensorCPU input,up;

		if(img.cols != input_dims[3] || img.rows != input_dims[2])
			resize(img, img, Size(input_dims[3], input_dims[2]));
		if (!preProcess(img, input, up)) return 0;

		auto in_tensor = face_workspace.CreateBlob(inputstr)->GetMutable<TensorCPU>();
		in_tensor->ResizeLike(input);
		in_tensor->ShareData(input);

		auto up_tensor = face_workspace.CreateBlob(upstr)->GetMutable<TensorCPU>();
		up_tensor->ResizeLike(up);
		up_tensor->ShareData(up);

		face_predict->Run();
		TensorCPU output = TensorCPU(face_workspace.GetBlob(outputstr)->Get<TensorCPU>());
		const float * probs = output.data<float>();

		Mat dst_img(input_dims[2], input_dims[3], CV_32FC(img.channels()));
		if (input_dims[1] == 1)
		{
			for (int h = 0; h < dst_img.rows; h++)
			{
				for (int w = 0; w < dst_img.cols; w++)
				{
					dst_img.at<float>(h, w) = probs[h * dst_img.cols + w];
				}
			}
		}
		else
		{
			int size = dst_img.rows * dst_img.cols;
			Mat rgb_channel[3];
			split(dst_img, rgb_channel);
			for (int h = 0; h < dst_img.rows; h++)
				for (int w = 0; w < dst_img.cols; w++)
				{
					rgb_channel[2].at<float>(h, w) = probs[h * img.cols + w];
					rgb_channel[1].at<float>(h, w) = probs[size+h * img.cols + w];
					rgb_channel[0].at<float>(h, w) = probs[2 * size+h * img.cols + w];
				}
			merge(rgb_channel, 3, dst_img);
		}
		convertScaleAbs(dst_img, dst, 255);
		return 1;
	}	

private:
	const string inputstr = "input";
	const string outputstr = "output";
	const string upstr = "up";
	Workspace  face_workspace;
	unique_ptr<NetBase> face_predict;
	vector<TIndex> input_dims;
	
	int preProcess(Mat img, TensorCPU &in_tensor, TensorCPU &up_tensor)
	{
		Mat dst;
		vector<TIndex> in_dims, up_dims;
		vector<float> in_data, up_data;	

		int batch_size = img.channels() * img.rows * img.cols;

		vector<TIndex> tdims({ 1, img.channels(), img.rows, img.cols });
		vector<float> tdata(1 * batch_size);
		in_dims = tdims;
		in_data = tdata;
		img.convertTo(img, CV_32FC(img.channels()), 1.0 / 255, 0);
		if (img.channels() == 1)
		{
			std::copy((float *)img.datastart, (float *)img.dataend, in_data.begin());
		}
		else
		{
			int size = img.rows * img.cols;
			Mat rgb_channel[3];
			split(img, rgb_channel);
			for (int h = 0; h < img.rows; h++)
				for (int w = 0; w < img.cols; w++)
				{
					in_data[h * img.cols + w] = rgb_channel[2].at<float>(h, w);
					in_data[size + h * img.cols + w] = rgb_channel[1].at<float>(h, w);
					in_data[2 * size + h * img.cols + w] = rgb_channel[0].at<float>(h, w);
				}
		}
		in_tensor.CopyFrom(TensorCPU(in_dims, in_data, NULL));
		up_tensor.CopyFrom(TensorCPU(in_dims, in_data, NULL));
		return 1;
	}
	
};
