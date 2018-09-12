#include "network.h"
#include "mtcnn.h"
#include <time.h>
#include <io.h>
#include <string.h>
#include "preprocess.h"
#include "aligner.h"" 
#include "possion.h"
using namespace seeta;

const int  srcimg_width = 256;
const int  srcimg_height = 256;
const int  dstimg_width = 96;
const int  dstimg_height = 96;

Mat get_seeta_mat(char* file, mtcnn *find, Size s, std::shared_ptr<Aligner> aligner_, CPreProcess *pre)
{
	Mat img;
	Mat dst;
	img = imread(file);
	if (true == img.empty())
	{
		printf("%s  no exst\n", file);
		return dst;
	}

	cv::Rect rect;
	vector<Point2f> five;
	cv::resize(img, img, s);
	if (!find->findFace(img, rect, five))
	{
		printf("%s have no face\n", file);
		return dst;
	}

	//使用中科院的脸部对齐
	/*float point_data[10];
	for (int i = 0; i < 5; ++i)
	{
		point_data[i * 2] = five[i].x;
		point_data[i * 2 + 1] = five[i].y;
	}
	ImageData src_img_data(img.cols, img.rows, img.channels());
	src_img_data.data = img.data;

	cv::Mat dst_img(dstimg_height, dstimg_width, CV_8UC(img.channels()));
	ImageData dst_img_data(dstimg_width, dstimg_height, img.channels());
	dst_img_data.data = dst_img.data;
	aligner_->Alignment(src_img_data, point_data, dst_img_data);
	//cvtColor(dst_img, img, cv::COLOR_BGR2GRAY);
	return dst_img;
	*/

	//使用ccl脸部对齐
	dst = pre->get_face_with_ccl(rect, img, five);

	//使用自己的脸部对齐
	//dst = pre->get_face(rect, img, five);

	cvtColor(dst, img, cv::COLOR_BGR2GRAY);
	pre->crop_img(img);
	//转为lmcp
	//img = pre->preprocess(img);
	return img;
}

void get_gray(Mat img, Mat &dst)
{	
	cvtColor(img, dst, cv::COLOR_BGR2GRAY);
}

void get_tt(Mat img, Mat &dst, CPreProcess *pre)
{
	//cvtColor(img, dst, cv::COLOR_BGR2GRAY);
	dst = pre->preprocess(img);
}

void get_grad(Mat img, Mat &dst)
{
	Possion p;
	//cvtColor(img, img, cv::COLOR_BGR2GRAY);
	Mat X,Y;
	img.convertTo(img, CV_32FC(img.channels()), 1.0, 0);
	p.computeGradientX(img, X);
	p.computeGradientY(img, Y);
	convertScaleAbs((abs(X) + abs(Y)), dst);
}

void get_rgb(Mat img, Mat &dst, CPreProcess *pre)
{
	Mat rgb_channel[3];
	//get_gray(img, rgb_channel[0]);
	rgb_channel[0] = img;
	get_tt(img, rgb_channel[1], pre);
	get_grad(rgb_channel[1], rgb_channel[2]);
	merge(rgb_channel, 3, dst);
}

void update_face()
{
	const char* srcdir = "d:/result01/result01";
	const char* dstdir = "d:/BaiduNetdiskDownload/tt_dst";
	const char* label_file = "d:/BaiduNetdiskDownload/tt_label.txt";

	char file[128];
	Mat img;
	int count = 0;
	ofstream f;
	f.open(label_file, ios::out | ios::app);
	if (!f.is_open())
	{
		printf("open file %s error!\n", label_file);
		return;
	}

	long long hFile = 0;
	struct _finddata_t fileinfo;

	char facefile[128];
	char realfile[128];
	int num = 5000;
	int  totalnum = 0;
	int  real = 0;
	Mat roil;


	CPreProcess pre;
	Size size(srcimg_width, srcimg_height);	
	mtcnn *find = new mtcnn(srcimg_height, srcimg_width);
	std::shared_ptr<Aligner> aligner_;
	aligner_.reset(new seeta::Aligner(dstimg_height, dstimg_width, "linear"));
	srand(time(NULL));

	
	
	for (int i = 12282; i < 12783; i++)
	{
		sprintf_s(file, "%s/test%d/*", srcdir, i+1);
		int count = 0;
		num++;
		if ((hFile = _findfirst(file, &fileinfo)) != -1)
		{
			do
			{
				if (strlen(fileinfo.name) < 3) continue;
				sprintf(facefile, "%s/test%d/%s", srcdir, i+1, fileinfo.name);
				img = get_seeta_mat(facefile, find, size, aligner_, &pre);
			
				totalnum++;
				if (img.empty()) continue;
				real++;

				int seed = 7;// rand() % 8;

				if (seed <= 0)
				{
					//24,24
					resize(img, roil, Size(24, 24));
					resize(roil, img, Size(dstimg_width, dstimg_height));
				}
				else if (seed <= 1)
				{
					//32,32
					resize(img, roil, Size(32, 32));
					resize(roil, img, Size(dstimg_width, dstimg_height));
				}
				else if (seed <= 2)
				{
					// 40
					resize(img, roil, Size(40, 40));
					resize(roil, img, Size(dstimg_width, dstimg_height));
				}
				else if (seed <= 3)
				{
					//48,48
					resize(img, roil, Size(48, 48));
					resize(roil, img, Size(dstimg_width, dstimg_height));
				}
				else if (seed <= 6)
				{
					//56,56
					resize(img, roil, Size(56, 56));
					resize(roil, img, Size(dstimg_width, dstimg_height));
				}
				else;
				
				/*std::vector<int> com;
				com.push_back(IMWRITE_JPEG_QUALITY);
				int p = 5 + (real-1) * 10;
				if (p > 100) break;
				com.push_back(p);
				*/
				//int type = 0;
				sprintf(facefile, "%s/ch%d_%d.jpg", dstdir, i + 1, count);
				sprintf(realfile, "ch%d_%d.jpg", i + 1, count);
				f << realfile << "  " << num << endl;
				Mat dst;
				get_tt(img, dst, &pre);
				//get_rgb(img, dst, &pre);
				cv::imwrite(facefile, dst);

				/*type++;
				sprintf(facefile, "%s/ch%d_%d_%d.jpg", dstdir, i + 1, count, type);
				get_tt(img, dst, &pre);
				cv::imwrite(facefile, dst);

				type++;
				sprintf(facefile, "%s/ch%d_%d_%d.jpg", dstdir, i + 1, count, type);
				get_grad(dst, dst);
				cv::imwrite(facefile, dst);

				type++;
				sprintf(facefile, "%s/ch%d_%d_%d.jpg", dstdir, i + 1, count, type);
				get_rgb(img, dst, &pre);
				cv::imwrite(facefile, dst);
				*/
				count++;

			} while(_findnext(hFile, &fileinfo) == 0);
		}			
	}	
	printf("num = %d, total = %d  real = %d\n", num, totalnum, real);
	f.close();
}

void  get_cebla(int  total_type)
{
	const char* srcdir = "d:/BaiduNetdiskDownload/result00/result00";
	const char* dstdir = "d:/BaiduNetdiskDownload/tt_dst";
	const char* label_file = "d:/BaiduNetdiskDownload/result00/MS-Celeb-1M_clean_list.txt";
	const char* tr_label_file = "d:/BaiduNetdiskDownload/tt_label.txt";


	char file[128];
	char imgfile[256];

	Mat img;
	int count;

	ofstream trf;
	trf.open(tr_label_file, ios::out | ios::trunc);
	if (!trf.is_open())
	{
		printf("open file %s error!\n", tr_label_file);
		return;
	}

	ifstream f;
	f.open(label_file, ios::in);
	if (!f.is_open())
	{
		printf("open file %s error!\n", label_file);
		return;
	}

	int  current_id = -1;
	int  our_id = -1;

	int id_num = 0;

	char facefile[128];
	char lastfacefile[128];
	char realfile[128];
	char lastfile[128];

	int  totalnum = 0;
	int  real = 0;
	Mat roil;
	Mat lastmat;

	CPreProcess pre;
	Size size(srcimg_width, srcimg_height);
	mtcnn *find = new mtcnn(srcimg_height, srcimg_width);
	std::shared_ptr<Aligner> aligner_;
	aligner_.reset(new seeta::Aligner(dstimg_height, dstimg_width, "linear"));
	srand(time(NULL));

	int flag = 0;
	int s0 = 0;
	int s1 = 0;
	int s2 = 0;
	int s3 = 0;
	int s4 = 0;
	int s5 = 0;
	Mat dst;
	while (1)
	{
		f >> file >> count;
		if (f.eof()) break;
		sprintf_s(imgfile, "%s/%s", srcdir, file);
		dst = get_seeta_mat(imgfile, find, size, aligner_, &pre);
		totalnum++;
		if (dst.empty()) continue;		
		get_tt(dst, img, &pre);

		if (current_id != count)
		{
			flag = 0;
			current_id = count;
			id_num = 0;
		}

		int seed = 7;// rand() % 8;

		if (seed <= 1)
		{
			//24,24
			resize(img, roil, Size(24, 24));
			resize(roil, img, Size(dstimg_width, dstimg_height));
			s0++;
		}
		else if (seed <= 3)
		{
			//32,32
			resize(img, roil, Size(32, 32));
			resize(roil, img, Size(dstimg_width, dstimg_height));
			s1++;
		}
		else if (seed <= 4)
		{
			// 40
			resize(img, roil, Size(40, 40));
			resize(roil, img, Size(dstimg_width, dstimg_height));
			s2++;
		}
		else if (seed <= 5)
		{
			//48,48
			resize(img, roil, Size(48, 48));
			resize(roil, img, Size(dstimg_width, dstimg_height));
			s3++;
		}
		else if (seed <= 6)
		{
			//56,56
			resize(img, roil, Size(56, 56));
			resize(roil, img, Size(dstimg_width, dstimg_height));
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
			imwrite(lastfacefile, lastmat);
			imwrite(facefile, img);
		}
		else
		{
			trf << realfile << "  " << our_id << endl;
			imwrite(facefile, img);
			id_num++;
			real++;
		}
	}
	printf("num = %d, total = %d  real = %d ( %d,%d,%d,%d,%d,%d)\n", our_id, totalnum, real,s0,s1,s2,s3,s4,s5);
	f.close();
	trf.close();
}


void  get_img(int  total_type)
{
	const char* srcdir = "d:/BaiduNetdiskDownload/result00/result00";
	const char* dstdir1 = "d:/BaiduNetdiskDownload/img_src";
	const char* dstdir2 = "d:/BaiduNetdiskDownload/img_dst";
	const char* label_file = "d:/BaiduNetdiskDownload/result00/MS-Celeb-1M_clean_list.txt";
	const char* tr_label_file = "d:/BaiduNetdiskDownload/img_label.txt";


	char file[128];
	char imgfile[256];

	Mat img;
	int count;

	ofstream trf;
	trf.open(tr_label_file, ios::out | ios::trunc);
	if (!trf.is_open())
	{
		printf("open file %s error!\n", tr_label_file);
		return;
	}

	ifstream f;
	f.open(label_file, ios::in);
	if (!f.is_open())
	{
		printf("open file %s error!\n", label_file);
		return;
	}

	char facefile[128];
	char lastfacefile[128];
	char realfile[128];
	char lastfile[128];

	
	int  totalnum = 0;
	int  real = 0;
	Mat roil;

	CPreProcess pre;
	Size size(srcimg_width, srcimg_height);
	mtcnn *find = new mtcnn(srcimg_height, srcimg_width);
	std::shared_ptr<Aligner> aligner_;
	aligner_.reset(new seeta::Aligner(dstimg_height, dstimg_width, "linear"));
	srand(time(NULL));

	int flag = 0;
	int s0 = 0;
	int s1 = 0;
	int s2 = 0;
	int s3 = 0;
	int s4 = 0;
	int s5 = 0;
	while (1)
	{
		f >> file >> count;
		if (f.eof()) break;
		sprintf_s(imgfile, "%s/%s", srcdir, file);
		img = get_seeta_mat(imgfile, find, size, aligner_, &pre);
		totalnum++;
		if (img.empty()) continue;		

		int seed = rand() % 7;

		if (seed <= 1)
		{
			//24,24
			resize(img, roil, Size(16, 16));
			resize(roil, roil, Size(dstimg_width, dstimg_height));
			s0++;
		}
		else if (seed <= 3)
		{
			//32,32
			resize(img, roil, Size(24, 24));
			resize(roil, roil, Size(dstimg_width, dstimg_height));
			s1++;
		}
		else if (seed <= 4)
		{
			// 40
			resize(img, roil, Size(32, 32));
			resize(roil, roil, Size(dstimg_width, dstimg_height));
			s2++;
		}
		else if (seed <= 5)
		{
			//48,48
			resize(img, roil, Size(40, 40));
			resize(roil, roil, Size(dstimg_width, dstimg_height));
			s3++;
		}
		else if (seed <= 6)
		{
			//56,56
			resize(img, roil, Size(48, 48));
			resize(roil, roil, Size(dstimg_width, dstimg_height));
			s4++;
		}	
		
		sprintf_s(facefile, "%s/%d.jpg", dstdir1, real);
		sprintf_s(realfile, "%d.jpg", real);
		sprintf_s(lastfacefile, "%s/%d.jpg", dstdir2, real);
		sprintf_s(lastfile, "%d.jpg", real);

		imwrite(facefile, img);

		std::vector<int> com;
		com.push_back(IMWRITE_JPEG_QUALITY);
		int quality = 30+ rand() % 65;
		com.push_back(quality);
		imwrite(lastfacefile, roil, com);
		trf << realfile << "  " << lastfile << endl;
		real++;
		if (real > total_type) break;		
	}
	printf("total = %d  real = %d ( %d,%d,%d,%d,%d,%d)\n", totalnum, real, s0, s1, s2, s3, s4, s5);
	f.close();
	trf.close();
}



/*void  get_cebla(int  total_type)
{
	const char* srcdir = "d:/BaiduNetdiskDownload/result00/result00";
	const char* dstdir = "d:/BaiduNetdiskDownload/result00/tt_dst";
	const char* label_file = "d:/BaiduNetdiskDownload/result00/MS-Celeb-1M_clean_list.txt";
	const char* tr_label_file = "d:/BaiduNetdiskDownload/result00/tt_label.txt";


	char file[128];
	char imgfile[256];

	Mat img;
	CPreProcess pre;
	int count = 0;

	ofstream trf;
	trf.open(tr_label_file, ios::out | ios::trunc);
	if (!trf.is_open())
	{
		printf("open file %s error!\n", tr_label_file);
		return;
	}

	ifstream f;
	f.open(label_file, ios::in);
	if (!f.is_open())
	{
		printf("open file %s error!\n", label_file);
		return;
	}

	int  current_id = -1;
	int  our_id = -1;

	int id_num = 0;

	char facefile[128];
	char lastfacefile[128];
	char realfile[128];
	char lastfile[128];

	int  totalnum = 0;
	int  real = 0;
	Mat lastmat;
	Mat roil;

	int flag = 0;
	Size model_size(64, 64);

	mtcnn *find1 =  new mtcnn(96, 112, 40);
	Size s1(96, 112);

	mtcnn *find2 = new mtcnn(24, 24, 20);
	Size s2(24, 24);

	mtcnn *find3 = new mtcnn(32, 32,20);
	Size s3(12, 12);

	mtcnn *find4 = new mtcnn(48, 48,20);
	Size s4(16, 16);

	clock_t start;

	srand(time(NULL));

	std::shared_ptr<Aligner> aligner_;
	aligner_.reset(new seeta::Aligner(64, 64, "linear"));
	float point_data[10];

	while (1)
	{
		f >> file >> count;
		if (f.eof()) break;
		sprintf_s(imgfile, "%s/%s", srcdir, file);
		img = imread(imgfile);
		if (true == img.empty())
		{
			//printf("%s  %d no exst\n", imgfile, count);
			continue;
		}
		cv::Rect rect;
		vector<Point2f> five;

		totalnum++;
		int seed = 0;// rand() % 10;
		if (seed  <= 3 )
		{
			resize(img, img, s1);
			if (!find1->findFace(img, rect, five))
			{
				printf("%s have no face\n", imgfile);
				continue;
			}
		}
		else if (seed <= 6)
		{
			//resize(img, img, s2);
			if (!find2->findFace(img, rect, five))
			{
				printf("%s have no face\n", imgfile);
				continue;
			}
		}
		else if (seed <= 8 )
		{
			//resize(img, img, s3);
			if (!find3->findFace(img, rect, five))
			{
				printf("%s have no face\n", imgfile);
				continue;
			}
		}
		else
		{
			resize(img, img, s4);
			resize(img, img, model_size);
		
			if (!find4->findFace(img, rect, five))
			{
				printf("%s have no face\n", imgfile);
				continue;
			}
		}
		
		
		for (int i = 0; i < 5; ++i)
		{
			point_data[i * 2] = five[i].x;
			point_data[i * 2 + 1] = five[i].y;
		}
		ImageData src_img_data(img.cols, img.rows, img.channels());
		src_img_data.data = img.data;
		
			cv::Mat dst_img(64, 64, CV_8UC(img.channels()));
			ImageData dst_img_data(64, 64, img.channels());
			dst_img_data.data = dst_img.data;
			aligner_->Alignment(src_img_data, point_data, dst_img_data);

		//roil = pre.get_face(rect, img, five);
		cvtColor(dst_img, img, cv::COLOR_BGR2GRAY);
		//resize(img, img, s4);
		img = pre.preprocess(img);
		resize(img, img, s3);
		resize(img, img, model_size);
		if (current_id != count)
		{
			flag = 0;
			current_id = count;
			id_num = 0;
		}
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
			imwrite(lastfacefile, lastmat);
			imwrite(facefile, img);
		}
		else
		{
			trf << realfile << "  " << our_id << endl;
			imwrite(facefile, img);
			id_num++;
			real++;
		}

	}
	printf("num = %d, total = %d  real = %d\n", our_id, totalnum, real);

	f.close();
	trf.close();
}
*/

int main()
{
	//get_img(20);
	get_cebla(5000);
	//update_face();
	return 0;
}