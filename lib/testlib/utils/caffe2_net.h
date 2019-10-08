#pragma once
#include <stdio.h>
//#include <tchar.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <caffe2/core/workspace.h>
#include <caffe2/utils/proto_utils.h>



#ifdef GPU
#include <caffe2/core/context_gpu.h>
#endif

namespace caffe2net{

	class Mat {
	public:
		Mat()  {}
		~Mat() {}

		void setMat(unsigned char *src, int chn, int width,int height) {
			m_chn = chn;
			m_width = width;
			m_height = height;

			int size = chn * width*height;
			img.resize(size);
			data.resize(size);			
			std::copy((unsigned char*)src, (unsigned char*)src + size, img.begin());			
		}

		void normalize(const float* mean, const float* norm) {

			int size =  m_chn * m_width * m_height;
			int batch = m_width * m_height;

#pragma omp parallel for
			for (int i = 0; i < size; i++) {
				int chn = i / batch;
				data[i] = ((float)img[i] - mean[chn]) * norm[chn];
			}
		}

		std::vector<float> data;
	private:
		int m_chn;
		int m_width;
		int m_height;
		std::vector<unsigned char> img;
		
	};


	class Net {
	public:
		Net() {}

		~Net() {
			/* inputHost的value使用ShareExternalPointer，不需要做删除操作

			auto hiter = inputHost.begin();
			while (hiter != inputHost.end()) {
				printf("delete %s \n", hiter->first.c_str());
				delete hiter->second;				
				++hiter;				
			}

#ifdef GPU
			auto diter = inputDevice.begin();
			while (diter != inputDevice.end()) {
				delete diter->second;
				++diter;
			}
#endif
          */

		}

		int initNet(const char* model,const char* param) {
			if (!caffe2::ReadProtoFromFile(model, &init_net_def)) {
				printf("load model %s error!\n", model);
				return 0;
			}
			if (!caffe2::ReadProtoFromFile(param, &predict_net_def)) {
				printf("load param %s error!\n",param);
				return 0;
			}
#ifdef GPU
			const auto device = caffe2::TypeToProto(caffe2::CUDA);
			init_net_def.mutable_device_option()->set_device_type(device);
			predict_net_def.mutable_device_option()->set_device_type(device);

#else
			const auto device = caffe2::TypeToProto(caffe2::CPU);
			init_net_def.mutable_device_option()->set_device_type(device);
			predict_net_def.mutable_device_option()->set_device_type(device);

#endif

			face_workspace = caffe2::make_unique<caffe2::Workspace>();
			if (!face_workspace->RunNetOnce(init_net_def)) {
				printf("create net error \n");
				return 0;
			}
			face_predict = caffe2::CreateNet(predict_net_def, face_workspace.get());
			return 1;
		}

		int initInput(const char* input, std::vector<int64_t> *dim = nullptr) {

			caffe2::Blob* blob = face_workspace->CreateBlob(input);
			if (!blob) {
				printf("create blob error\n");
				return 0;
			}

			auto tensor = blob->GetMutable<caffe2::TensorCPU>();
			if (dim != nullptr)		*dim = tensor->sizes().vec();
			
			auto input_host_tensor = new caffe2::TensorCPU(tensor->sizes(), caffe2::CPU);
			inputHost[input] = input_host_tensor;
			//printf("it is %d \n",input_host_tensor->numel());
	
#ifdef GPU
			auto input_device_tensor = new caffe2::TensorCUDA(tensor->sizes(), caffe2::CUDA);
			inputDevice[input] = input_device_tensor;
			blob->Reset(input_device_tensor);
#else
			blob->Reset(input_host_tensor);
#endif
			return 1;
		}

		void setInput(const char* input, Mat &in) {

			inputHost[input]->ShareExternalPointer(in.data.data());
#ifdef GPU
			inputDevice[input]->CopyFrom(*inputHost[input]);
#endif
		}

		int run() {

			if (!face_predict->Run()) {
				printf("predict error\n");
				return 0;
			}
			return 1;
		}

		int setOutput(const char* output, std::vector<float> &data) {

			caffe2::Blob* blob = face_workspace->GetBlob(output);
			if (!blob) {
				printf("get %s blob error\n",output);
				return 0;
			}			
			

#ifdef GPU
			caffe2::TensorCPU tensor = blob->Get<caffe2::TensorCUDA>().UnsafeSharedInstance();
#else
			caffe2::TensorCPU tensor = blob->Get<caffe2::TensorCPU>().UnsafeSharedInstance();

#endif
			const float * probs = tensor.data<float>();
			data.resize(tensor.numel());
			std::copy(probs, probs + tensor.numel(), data.begin());
			return 1;
		}

	private:
		caffe2::NetDef init_net_def;
		caffe2::NetDef predict_net_def;

		std::unique_ptr<caffe2::Workspace>  face_workspace;
		std::unique_ptr<caffe2::NetBase> face_predict;


		std::unordered_map<std::string, caffe2::TensorCPU*> inputHost;
	
#ifdef GPU
		std::unordered_map<std::string, caffe2::TensorCUDA*> inputDevice;
#endif

	};

}