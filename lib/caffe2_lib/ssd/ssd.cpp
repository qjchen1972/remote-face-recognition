#define USE_OPENCV

#include "../utils/caffe2_net.h"
#include "../utils/utils.h"
#include "detect.h"
#include "ssd.h"

static caffe2net::Net predictor;
static std::vector<float>  prior_data;


//static const float mean[3] = { 104,117,123 };
static const float mean[3] = { 0,0,0 };
static const float norms[3] = { 1.0f,1.0f, 1.0f };

//static const float mean[3] = { 127.5,127.5,127.5 };
//static const float norms[3] = { 1.0f / 127.5,1.0f / 127.5, 1.0f / 127.5 };

//static const float mean[3] = { 0,0,0 };
//static const float norms[3] = { 1.0f / 255,1.0f / 255, 1.0f / 255 };

int  initSsd(const char* init_net, const char*  predict_net) {
	if (!predictor.initNet(init_net, predict_net)) {
		std::cout << "init net error" << std::endl;
		return 0;
	}
	std::vector<int64_t> dim;
	predictor.initInput("input", &dim);
	//std::cout << "init net error" << std::endl;
	//printf("init net error\n");
	printf("input dim is %lld  %lld %lld \n", dim[1],dim[2],dim[3]);
	//std::cout << "init net error"<<std::endl;
	PriorBox box;
	box.createPriorBox(prior_data);
	printf("init  prior box %d \n", prior_data.size());
	return 1;
}


int  getSsdVec(cv::Mat img, std::vector<float> &vec) {
	caffe2net::Mat cm;
	caffe2net::Utils util;

	cv::Mat src;
	cv::resize(img, src,cv::Size(320, 320));

	//cv::imshow("ok", src);
	//cv::waitKey(0);

	unsigned char *data = new unsigned char[src.channels()*src.cols*src.rows];
	util.Mat2array(src, data);
	cm.setMat(data, src.channels(), src.cols, src.rows);
	cm.normalize(mean, norms);
	predictor.setInput("input", cm);
	if (!predictor.run()) {
		std::cout << "predict error" << std::endl;
		return 0;
	}

	std::vector<float>  loc;
	std::vector<float>  conf;

	predictor.setOutput("loc_output", loc);
	predictor.setOutput("conf_output", conf);

	printf("loc size %d\n", loc.size());
	printf("conf size %d\n", conf.size());

	/*int flag = 0;
	for (int i = 0; i < conf.size(); i++)
	{
		
		if (i % 21 == 0) {
			if(flag == 1) printf("\n");
			if (conf[i] > 0.5) flag =0;
			else flag = 1;
			
		}
		if(flag == 1)		printf("%f,", conf[i]);
	}
	printf("\n");
	*/


	Detect dect;
	dect.getAnswer(loc.data(),conf.data(), prior_data, vec);

	int len = vec.size() / 6;
	for (int i = 0; i < len; i++) {
		if (vec[i * 6 + 5] * 1.0 < 0.5) continue;
		//if (vec[i * 6 + 0] < 0 || vec[i * 6 + 1] < 0 || vec[i * 6 + 2] < 0 || vec[i * 6 + 3] < 0) continue;
		//if (vec[i * 6 + 0] > 1.0 || vec[i * 6 + 1] > 1.0 || vec[i * 6 + 2] > 1.0 || vec[i * 6 + 3] > 1.0) continue;

		cv::Point point1(vec[i * 6 + 0] * img.cols, vec[i * 6 + 1] * img.rows);
		cv::Point point2(vec[i * 6 + 2] * img.cols, vec[i * 6 + 3] * img.rows);
		printf("(%f  %f  %d  %d )   (%f  %f %d  %d)\n", vec[i * 6 + 0], vec[i * 6 + 1],point1.x, point1.y, vec[i * 6 + 2], vec[i * 6 + 3],point2.x, point2.y);
		cv::rectangle(img, cv::Rect(point1, point2), cv::Scalar(0, 225, 255), 1);
		char ch[100];
		sprintf(ch, "%s %.2f", dect.getClass(int(vec[i * 6 + 4])).c_str(), vec[i * 6 + 5] * 1.0);
		std::string temp(ch);
		cv::putText(img, temp, point1, CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 255));
		//if(i >=0)	break;
	}
	cv::imshow("ok", img);
	cv::waitKey(0);
	delete[] data;
	return 1;
}