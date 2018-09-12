#include<math.h>
#include "../include/cnn_net.h"
#include "../include/net_api.h"

using namespace std;
using namespace cv;

static Cnn_Net  net;

int init_net(int type, int mode, const char* init_net, const char*  predict_net)
{
	return net.init_net(type, mode, init_net, predict_net);
}

bool predict()
{
	return net.predict();
}

int set_input(std::string  str, cv::Mat *img, int size)
{
	return net.set_input(str, img, size);
}

int get_output(std::string  str, std::vector<float>  &vec)
{
	return net.get_output(str, vec);
}

int get_dims(std::string str, std::vector<int>  &vec)
{
	return net.get_dims(str, vec);
}

void vec2mat(std::vector<float>  vec, int chn, int row, int col, cv::Mat &dst)
{
	net.vec2mat(vec, chn, row, col, dst);
}


