#pragma once
#include <stdio.h>
#include <iostream>
#include <fstream>
//#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
//#include "caffe2/predictor/predictor.h"
//#include <caffe2/core/workspace.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/context_gpu.h>
#include<math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace caffe2;

#define  CNN_NET_CPU  0
#define  CNN_NET_GPU  1

#define  CNN_NORM_ZERO  0
#define  CNN_NORM_NEGA  1

class Cnn_Net
{
public:
	Cnn_Net(){}
	~Cnn_Net() {}

	int set_input(std::string  str, Mat *img, int size)
	{
		Blob* blob = net_workspace.CreateBlob(str);
		if (!blob)
		{
			cout << str << "  no exist!" << endl;
			return 0;
		}

		TensorCPU cpuinput;
		preProcess(img, size, cpuinput);
	
		if (m_proc_type == CNN_NET_GPU)
		{
			TensorCUDA input = TensorCUDA(cpuinput);			
			TensorCUDA *in_tensor = blob->GetMutable<TensorCUDA>();
			
			//cout << in_tensor << endl;
			if (!in_tensor)
			{
				cout << str << "  get  cuda tensor error!" << endl;
				return 0;
			}
			in_tensor->ResizeLike(input);
			in_tensor->ShareData(input);
			
		}
		else
		{			
			auto in_tensor = blob->GetMutable<TensorCPU>();

			if (!in_tensor)
			{
				cout << str << "  get tensor error!" << endl;
				return 0;
			}
			in_tensor->ResizeLike(cpuinput);
			in_tensor->ShareData(cpuinput);
		}
		return 1;
	}

	int get_output(std::string  str, std::vector<float>  &vec )
	{
		vector<TIndex> dims;
		const float * probs;

		Blob* blob = net_workspace.GetBlob(str);
		if (!blob)
		{
			cout << str << "  no exist!" << endl;
			return 0;
		}

		if (m_proc_type == CNN_NET_GPU)
		{
			TensorCPU output = TensorCPU(blob->Get<TensorCUDA>());
			//TensorCUDA output =blob->Get<TensorCUDA>();
			dims = output.dims();
			probs = output.data<float>();
			if(!probs) printf("it is null\n");
			else printf("it is not null\n");
		}
		else
		{
			TensorCPU output = TensorCPU(blob->Get<TensorCPU>());
			dims = output.dims();

			probs = output.data<float>();
		}
		int size = 1;
		for (int i = 0; i < dims.size(); i++)
		{
			cout << dims.at(i) << endl;
			size *= dims.at(i);
		}
		printf(" k3  %d\n",size);
		vec.resize(size);
		printf("k1\n");
		for(int i =0; i<size; i++)
		{
	           if( i % 16 == 0 ) cout << endl;
                   cout<<probs[i] <<"   ";
		   vec[i] = probs[i];
		   
                }
		copy(probs, probs + size, vec.begin());
		printf("ok\n");
		return 1;
	}	

	int init_net(int type, int mode,const char* init_net, const char* predict_net)
	{
		m_proc_type = type;
		m_norm_mode = mode;

		//if (m_proc_type == CNN_NET_GPU)
		//{
			DeviceOption option;
			option.set_device_type(CUDA);
			new CUDAContext(option);
		//}	

		NetDef init_net_def, predict_net_def;
		CAFFE_ENFORCE(ReadProtoFromFile(init_net, &init_net_def));
		CAFFE_ENFORCE(ReadProtoFromFile(predict_net, &predict_net_def));
		if (m_proc_type == CNN_NET_GPU)
		{		
			init_net_def.mutable_device_option()->set_device_type(CUDA);
			predict_net_def.mutable_device_option()->set_device_type(CUDA);		
		}
		else
		{
			init_net_def.mutable_device_option()->set_device_type(CPU);
			predict_net_def.mutable_device_option()->set_device_type(CPU);
		}
		if (!net_workspace.RunNetOnce(init_net_def))
		{
			cout <<" net  RunNetOnce error!" << endl;
			return 0;
		}	
		net_predict = CreateNet(predict_net_def, &net_workspace);
		if (!net_predict)
		{
			cout << " create net error!" << endl;
			return 0;
		}
		return 1;
	}

	int get_dims(std::string str, std::vector<int>  &vec)
	{
		vector<TIndex> dims;
		Blob* blob = net_workspace.CreateBlob(str);
		
		if (!blob)
		{
			cout << str << "  no exist!" << endl;
			return 0;
		}

		if (m_proc_type == CNN_NET_GPU)
		{
			auto tensor = blob->GetMutable<TensorCUDA>();
			dims = tensor->dims();
		}
		else
		{
			auto tensor = blob->GetMutable<TensorCPU>();
			dims = tensor->dims();
		}
		for (int i = 0; i < dims.size(); i++) vec.at(i) = dims.at(i);
		return 1;
	}

	

	bool predict()
	{		
		CAFFE_ENFORCE(net_predict->Run());
		return true;
	}

	void vec2mat(std::vector<float>  vec, int chn, int row, int col, cv::Mat &dst )
	{		
		Mat dst_img(row, col, CV_32FC(chn));
		if (chn == 1)
		{
			for (int h = 0; h < dst_img.rows; h++)
			{
				for (int w = 0; w < dst_img.cols; w++)
				{
					dst_img.at<float>(h, w) = vec[h * dst_img.cols + w];
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
					rgb_channel[2].at<float>(h, w) = vec[h * dst_img.cols + w];
					rgb_channel[1].at<float>(h, w) = vec[size + h * dst_img.cols + w];
					rgb_channel[0].at<float>(h, w) = vec[2 * size + h * dst_img.cols + w];
				}
			merge(rgb_channel, 3, dst_img);
		}
		convertScaleAbs(dst_img, dst, 255);
	}


private:
	
	Workspace  net_workspace;
	unique_ptr<NetBase> net_predict;
	int m_proc_type = CNN_NET_CPU;
	int m_norm_mode = CNN_NORM_ZERO;

	void preProcess(Mat *img, int size, TensorCPU &in_tensor)
	{
		Mat dst;
		int img_size = img[0].rows * img[0].cols;
		int batch_size = img[0].channels() * img_size;

		vector<TIndex> in_dims({ size, img[0].channels(), img[0].rows, img[0].cols });
		vector<float> in_data(size * batch_size);

		for (int i = 0; i < size; i++)
		{
			img[i].copyTo(dst);
			if (m_norm_mode == CNN_NORM_ZERO)
			{
				dst.convertTo(dst, CV_32FC(dst.channels()), 1.0 / 255, 0);
			}
			else
			{
				dst.convertTo(dst, CV_32FC(dst.channels()), 1.0 / 255, 0);
				dst = dst - 0.5;
				dst = dst / 0.5;
			}
			if (dst.channels() == 1)
			{
				std::copy((float *)dst.datastart, (float *)dst.dataend, in_data.begin() + i * batch_size);
			}
			else
			{
				Mat rgb_channel[3];
				split(dst, rgb_channel);
				for (int h = 0; h < dst.rows; h++)
					for (int w = 0; w < dst.cols; w++)
					{
						in_data[i*batch_size + h * dst.cols + w] = rgb_channel[2].at<float>(h, w);
						in_data[i*batch_size + img_size + h * dst.cols + w] = rgb_channel[1].at<float>(h, w);
						in_data[i*batch_size + 2 * img_size + h * dst.cols + w] = rgb_channel[0].at<float>(h, w);
					}
			}
		}
		if (m_proc_type == CNN_NET_GPU)
		{		
			CUDAContext* context;
			context = new CUDAContext();
			//in_tensor.CopyFrom(TensorCPU(in_dims, in_data, context));
			in_tensor.CopyFrom(TensorCPU(in_dims, in_data, NULL));
		}
		else
		{
			CPUContext* context;
			context = new CPUContext();
			//in_tensor.CopyFrom(TensorCPU(in_dims, in_data, context));
			in_tensor.CopyFrom(TensorCPU(in_dims, in_data, NULL));
		}
	}
};
