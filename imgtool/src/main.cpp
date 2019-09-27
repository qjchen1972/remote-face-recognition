#include "mtcnn/network.h"
#include "mtcnn/mtcnn.h"
#include <time.h>
#include <io.h>
#include <string.h>
#include "util/preprocess.h"
#include "seeta/aligner.h" 
#include "util/possion.h"

//图像放缩到处理尺寸
const int  srcimg_width = 256;
const int  srcimg_height = 256;
//输出尺寸
const int  dstimg_width = 64;
const int  dstimg_height = 64;


//中科院的头像切割
std::shared_ptr<seeta::Aligner> aligner_;
//头像切割和预处理
CPreProcess pre;



//type: 0 --- 中科院切割  1 -- ccl 切割    2---我们自己的切割  3--- 图像中心切割 
cv::Mat  getFaceMat(char* file, mtcnn *find, int type=0) {
	cv::Mat dst;
	cv::Mat img = cv::imread(file);

	if (img.empty()){
		printf("%s  no exst\n", file);
		return dst;
	}

	cv::Rect rect;
	vector<Point2f> five;
	cv::resize(img, img, cv::Size(srcimg_width, srcimg_height));
	if (!find->findFace(img, rect, five)){
		printf("%s have no face\n", file);
		return dst;
	}

	switch (type) {
	case 0: 
	{
		if (!aligner_) aligner_.reset(new seeta::Aligner(dstimg_height, dstimg_width, "linear"));
		float point_data[10];
		for (int i = 0; i < 5; ++i) {
			point_data[i * 2] = five[i].x;
			point_data[i * 2 + 1] = five[i].y;
		}
		seeta::ImageData src_img_data(img.cols, img.rows, img.channels());
		src_img_data.data = img.data;

		cv::Mat dst_img(dstimg_height, dstimg_width, CV_8UC(img.channels()));
		seeta::ImageData dst_img_data(dstimg_width, dstimg_height, img.channels());
		dst_img_data.data = dst_img.data;
		aligner_->Alignment(src_img_data, point_data, dst_img_data);
		return dst_img;
	}
	case 1:		
		dst = pre.get_face_with_ccl(img, five);
		pre.crop_img(dst);
		return dst;

	case 2:
		dst = pre.get_face(img, five);
		pre.crop_img(dst);
		return dst;
	case 3:
		dst = pre.get_face_with_center(rect, img, five);
		pre.crop_img(dst);
		return dst;
	default:
		;

	}

	return dst;
}


//得到图像类型
// type : 0--- gray  1 --- rgb 2 --- tt  3--- gradient 
cv::Mat  getMatType(cv::Mat img,int type) {

	cv::Mat dst;
	switch (type) {
	case 0:
		cvtColor(img, dst, cv::COLOR_BGR2GRAY);
		return dst;
	case 1:
	{
		if (img.channels() == 3) return img;
		cv::Mat rgb_channel[3];
		rgb_channel[0] = img;
		rgb_channel[1] = img;
		rgb_channel[2] = img;
		cv::merge(rgb_channel, 3, dst);
		return dst;
	}
	case 2:
		if (img.channels() != 1)
			cvtColor(img, dst, cv::COLOR_BGR2GRAY);
		dst = pre.preprocess(dst);
		return dst;

	case 3:
	{
		cv::Possion p;
		cv::Mat X, Y;
		img.convertTo(img, CV_32FC(img.channels()), 1.0, 0);
		p.computeGradientX(img, X);
		p.computeGradientY(img, Y);
		convertScaleAbs((abs(X) + abs(Y)), dst);
		return dst;
	}
	default:
		;
	}
	return dst;
}





void create_train_data(int total_type) {

	const char* srcdir = "d:/BaiduNetdiskDownload/result00/result00";
	const char* label_file = "d:/BaiduNetdiskDownload/result00/MS-Celeb-1M_clean_list.txt";
	const char* dstdir = "d:/BaiduNetdiskDownload/g1";
	const char* tr_label_file = "d:/BaiduNetdiskDownload/g1_label.txt";

	std::ofstream trf;
	trf.open(tr_label_file, ios::out | ios::trunc);
	if (!trf.is_open()) {
		printf("open file %s error!\n", tr_label_file);
		return;
	}

	std::ifstream f;
	f.open(label_file, ios::in);
	if (!f.is_open()) {
		printf("open file %s error!\n", label_file);
		return;
	}

	srand(time(NULL));
	pre.set_img_size(dstimg_width, dstimg_height);
	mtcnn *find = new mtcnn(srcimg_height, srcimg_width);

	int  current_id = -1;
	int  our_id = -1;
	int id_num = 0;

	char facefile[128];
	char lastfacefile[128];
	char realfile[128];
	char lastfile[128];

	char file[128];
	char imgfile[256];
	cv::Mat img;
	cv::Mat roil;
	cv::Mat lastmat;
	int count;
	int totalnum = 0;
	int flag = -1;
	int real = 0;
	int s0 = 0;
	int s1 = 0;
	int s2 = 0;
	int s3 = 0;
	int s4 = 0;
	int s5 = 0;

	while (1) {

		f >> file >> count;
		if (f.eof()) break;
		totalnum++;

		sprintf_s(imgfile, "%s/%s", srcdir, file);
		img = getFaceMat(imgfile, find, 3);
		if (img.empty()) continue;		
		img = getMatType(img, 3);
		if (img.empty()) continue;

		if (current_id != count) {
			flag = 0;
			current_id = count;
			id_num = 0;
		}

		int seed = rand() % 8;
		if (seed <= 1) {
			//24,24
			cv::resize(img, roil, Size(24, 24));
			cv::resize(roil, img, Size(dstimg_width, dstimg_height));
			s0++;
		}
		else if (seed <= 3) {
			//32,32
			cv::resize(img, roil, Size(32, 32));
			cv::resize(roil, img, Size(dstimg_width, dstimg_height));
			s1++;
		}
		else if (seed <= 4) {
			// 40
			cv::resize(img, roil, Size(40, 40));
			cv::resize(roil, img, Size(dstimg_width, dstimg_height));
			s2++;
		}
		else if (seed <= 5) {
			//48,48
			cv::resize(img, roil, Size(48, 48));
			cv::resize(roil, img, Size(dstimg_width, dstimg_height));
			s3++;
		}
		else if (seed <= 6) {
			//56,56
			cv::resize(img, roil, Size(56, 56));
			cv::resize(roil, img, Size(dstimg_width, dstimg_height));
			s4++;
		}
		else s5++;

		flag++;
		sprintf_s(facefile, "%s/%d_%d.jpg", dstdir, current_id, id_num);
		sprintf_s(realfile, "%d_%d.jpg", current_id, id_num);
		if (flag == 1)
		{
			strcpy_s(lastfile, realfile);
			strcpy_s(lastfacefile, facefile);
			img.copyTo(lastmat);
			id_num++;
		}
		else if (flag == 2)
		{
			our_id++;
			if (our_id > total_type) break;
			trf << lastfile << "  " << our_id << endl;
			trf << realfile << "  " << our_id << endl;
			id_num++;
			real = real + 2;
			cv::imwrite(lastfacefile, lastmat);
			cv::imwrite(facefile, img);
		}
		else
		{
			trf << realfile << "  " << our_id << endl;
			cv::imwrite(facefile, img);
			id_num++;
			real++;
		}
	}
	printf("num = %d, total = %d  real = %d ( %d,%d,%d,%d,%d,%d)\n", our_id, totalnum, real, s0, s1, s2, s3, s4, s5);
	f.close();
	trf.close();
}


void test() {

	cv::Mat image = imread("../img/17.jpg");	
	mtcnn find(image.rows, image.cols);

	clock_t start;
	start = clock();
	cv::Rect rect;
	vector<Point2f> five;

	find.findFace(image, rect, five);
	start = clock() - start;
	cout << "time is  " << start / 10e3 << endl;

	imshow("result", image(rect));
	waitKey(0);
	printf("%d %d %d %d \n", rect.x, rect.y, rect.width, rect.height);
	for (int i = 0; i < 5; i++)
		printf("(%d,%d)\n", int(five[i].x), int(five[i].y));

	CPreProcess pre;
	Mat roil = pre.get_face_with_ccl(image, five);
	imshow("result", roil);
	waitKey(0);
}


int main()
{
	test();
	//create_train_data(4);
	return 0;
}