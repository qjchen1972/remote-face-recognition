#pragma once

#include<vector>
/*
# SSD300 CONFIGS
voc = {
	'num_classes': 21,
	'lr_steps': (80000, 100000, 120000),
	'max_iter': 120000,
	'feature_maps': [40, 20, 10, 5, 3, 1],
	'min_dim': 320,
	'steps': [8, 16, 32, 64, 100, 300],
	'min_sizes': [30, 60, 111, 162, 213, 264],
	'max_sizes': [60, 111, 162, 213, 264, 315],
	'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
	'variance': [0.1, 0.2],
	'clip': True,
	'name': 'VOC',
}
*/



class PriorBox {
public:
	PriorBox(){}
	~PriorBox() {}

	int imageSize = 320;
	int numPriors = 6;
	float variance[2] = {0.1,0.2};
	int featureMaps[6] = { 40, 20, 10, 5, 3, 1 };
	int minSizes[6] = { 30, 60, 111, 162, 213, 264 };
	int maxSizes[6] = { 60, 111, 162, 213, 264, 315 };
	int steps[6] = { 8, 16, 32, 64, 100, 320 };
	int aspectRatios[6][2] = { {2,-1},{2,3}, {2, 3},{2, 3},{2,-1},{2,-1} };
	bool clip = true;
	
	void createPriorBox(std::vector<float> &mean) {
		int size = sizeof(featureMaps)/sizeof(int);

		//printf("start size  %d\n", size);

//#pragma omp parallel for
		for(int k = 0 ;k < size;k++)
			for(int i = 0; i< featureMaps[k];i++)
				for (int j = 0; j < featureMaps[k]; j++){
					float f_k = 1.0*imageSize / steps[k];
					float c_x = (j + 0.5) / f_k;
					float c_y = (i + 0.5) / f_k;
					float s_k = 1.0*minSizes[k] / imageSize;

					insertMean(mean,c_x,c_y,s_k,s_k);

					float s_k_prime = sqrt(s_k * maxSizes[k] / imageSize);
					insertMean(mean, c_x,c_y,s_k_prime,s_k_prime );

					for (int m = 0; m < 2; m++) {
						if (aspectRatios[k][m] > 0) {
							insertMean(mean, c_x,c_y,s_k*float(sqrt(aspectRatios[k][m])), s_k / float(sqrt(aspectRatios[k][m])) );
							insertMean(mean, c_x,c_y,s_k / float(sqrt(aspectRatios[k][m])),s_k*float(sqrt(aspectRatios[k][m])) );
						}
					}
				}

		//printf("end size  %d\n", mean.size());
		if (!clip) return;
#pragma omp parallel for
		for (int i = 0; i < mean.size(); i++) {
			mean[i] = std::max(mean[i], 0.0f);
			mean[i] = std::min(mean[i], 1.0f);
		}

/*#pragma omp parallel for
		for (int i = 0; i < mean.size(); i++) {
			if (mean[i] < 0) {
				mean[i] = 0;
				printf(" %d  %f \n", i, mean[i]);
			}
			else if (mean[i] > 1.0) {
				mean[i] = 1.0;
				printf(" %d  %f \n", i, mean[i]);
			}
		}*/

	}

private:
	void insertMean(std::vector<float> &mean, float x1, float y1, float x2, float y2) {
		mean.push_back(x1);
		mean.push_back(y1);
		mean.push_back(x2);
		mean.push_back(y2);
	}
	
};

/*
Detect(num_classes, 0, 200, 0.01, 0.45)

struct dection_t {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	int numclass;
	float confidence;
};
*/

#define  VOCCLASSNUM  21

class Detect {
public:
	int numClass = 21;
	int bkgLabel = 0;
	float nmsThresh = 0.45;
	float variance[2] = { 0.1,0.2 };
	
	void getAnswer(const float *loc_data, const float *conf_data, std::vector<float>  prior_data, std::vector<float> &ans) {
		int num_priors = prior_data.size() / 4;

		std::vector<std::pair<std::pair<int, int>, float> > idx_class_conf;
		proConf(conf_data, num_priors, 0.5, idx_class_conf);

		printf("idx_class_conf size  %d\n", idx_class_conf.size());

		std::vector<std::vector<float> > bboxes;
		decode(loc_data, prior_data, bboxes);

		std::vector<std::pair<std::pair<int, int>, float>> result;

		for (int i = 0; i < idx_class_conf.size(); ++i) {
			std::vector<float> bbox = bboxes[idx_class_conf[i].first.first];
			printf("%d  ( %f,%f,%f,%f,%s,%f )\n", idx_class_conf[i].first.first, bbox[0], bbox[1], bbox[2], bbox[3],
				VocClass[idx_class_conf[i].first.second].c_str(), idx_class_conf[i].second);
		}


		nms(bboxes, idx_class_conf, 0.45, result);
		printf("now size is %d\n", result.size());

		for (int i = 0; i < result.size(); ++i) {
			std::vector<float> bbox = bboxes[result[i].first.first];

			ans.push_back(bbox[0]);
			ans.push_back(bbox[1]);
			ans.push_back(bbox[2]);
			ans.push_back(bbox[3]);
			ans.push_back(result[i].first.second);
			ans.push_back(result[i].second);

			printf("%d  ( %f,%f,%f,%f,%s,%f )\n", result[i].first.first, bbox[0], bbox[1], bbox[2], bbox[3],
				VocClass[result[i].first.second].c_str(), result[i].second);
		}
	}		
	
	
	std::string getClass(int index) {
		return VocClass[index];
	}

private:


	const std::string VocClass[VOCCLASSNUM] = { "background","aeroplane", "bicycle", "bird", "boat",
									   "bottle", "bus", "car", "cat", "chair",
									   "cow","diningtable", "dog", "horse", "motorbike",
									   "person","pottedplant","sheep", "sofa", "train", "tvmonitor" };
	
	/*
	def decode(loc, priors, variances):
	"""Decode locations from predictions using priors to undo
	the encoding we did for offset regression at train time.
	Args:
		loc (tensor): location predictions for loc layers,
			Shape: [num_priors,4]
		priors (tensor): Prior boxes in center-offset form.
			Shape: [num_priors,4].
		variances: (list[float]) Variances of priorboxes
	Return:
		decoded bounding box predictions
	"""

	boxes = torch.cat((
		priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
		priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
	boxes[:, :2] -= boxes[:, 2:] / 2
	boxes[:, 2:] += boxes[:, :2]
	return boxes
	*/
	void decode(const float *loc_data, std::vector<float> prior_data, std::vector<std::vector<float>> &bboxes) {

		int num_priors = prior_data.size() / 4;

//#pragma omp parallel for
		for (int i = 0; i < num_priors; ++i) {
			std::vector<float> temp;
			temp.resize(4);
			temp[0] = prior_data[i * 4 + 0] + variance[0] * loc_data[i * 4 + 0] * prior_data[i * 4 + 2];
			temp[1] = prior_data[i * 4 + 1] + variance[0] * loc_data[i * 4 + 1] * prior_data[i * 4 + 3];
			temp[2] = prior_data[i * 4 + 2] * exp(variance[1] * loc_data[i * 4 + 2]);
			temp[3] = prior_data[i * 4 + 3] * exp(variance[1] * loc_data[i * 4 + 3]);

			temp[0] -= temp[2] / 2.0;
			temp[1] -= temp[3] / 2.0;
			temp[2] += temp[0];
			temp[3] += temp[1];
			/*if (temp[0] < 0 || temp[1] < 0 || temp[2] < 0 || temp[3] < 0) {

				printf("%d  < 0 ( %f, %f,%f,%f) (%f,%f,%f,%f)\n", i, temp[0], temp[1], temp[2], temp[3], 
					prior_data[i * 4 + 0], prior_data[i * 4 + 1], prior_data[i * 4 + 2], prior_data[i * 4 + 3]);

				temp[0] = temp[0] < 0 ? 0 : temp[0];
				temp[1] = temp[1] < 0 ? 0 : temp[1];
				temp[2] = temp[2] < 0 ? 0 : temp[2];
				temp[3] = temp[3] < 0 ? 0 : temp[3];
				
			}
			if (temp[0] > 1 || temp[1] > 1 || temp[2] > 1 || temp[3] > 1) {
				printf("%d  > 1 ( %f, %f,%f,%f) (%f,%f,%f,%f)\n", i, temp[0], temp[1], temp[2], temp[3],
					prior_data[i * 4 + 0], prior_data[i * 4 + 1], prior_data[i * 4 + 2], prior_data[i * 4 + 3]);
				temp[0] = temp[0] > 1 ? 1 : temp[0];
				temp[1] = temp[1] > 1 ? 1 : temp[1];
				temp[2] = temp[2] > 1 ? 1 : temp[2];
				temp[3] = temp[3] > 1 ? 1 : temp[3];
			}*/

			bboxes.push_back(temp);
		}
	}

	/*float Iou(std::vector<float> a, std::vector<float> b)
	{
		float x1 = std::max(a[0], b[0]);
		float y1 = std::max(a[1], b[1]);
		float x2 = std::min(a[2], b[2]);
		float y2 = std::min(a[3], b[3]);
		float over_area = (x2 - x1) * (y2 - y1);
		float iou = over_area / ((a[2] - a[0])*(a[3] - a[1]) + (b[2] - b[0])*(b[3] - b[1]) - over_area);
		return iou;
	}*/

	void proConf(const float *conf_data, int num_priors, float thresh, std::vector<std::pair<std::pair<int, int>, float> > &idx_class_conf) {


		for (int prior_idx = 0; prior_idx < num_priors; ++prior_idx) {
			int idx = prior_idx * VOCCLASSNUM;
			float max = 0;
			int max_idx = -1;
			for (int class_idx = 1; class_idx < VOCCLASSNUM; ++class_idx) { //class_idx = 0 is background
				if (conf_data[idx + class_idx] > max) {
					max = conf_data[idx + class_idx];
					max_idx = class_idx;
				}
			}
			if (max > thresh) {
				idx_class_conf.push_back(std::make_pair(std::make_pair(prior_idx, max_idx), conf_data[idx + max_idx]));
			}
		}
	}

	void nms(std::vector<std::vector<float> > bboxes, std::vector<std::pair<std::pair<int, int>, float>> idx_class_conf,
		float nmsthresh, std::vector<std::pair<std::pair<int, int>, float>> &result) {

		for (int i = 0; i < idx_class_conf.size(); ++i) {
			std::vector<float> bbox = bboxes[idx_class_conf[i].first.first];
			printf("%d  ( %f,%f,%f,%f,%s,%f )\n", idx_class_conf[i].first.first, bbox[0], bbox[1], bbox[2], bbox[3],
				VocClass[idx_class_conf[i].first.second].c_str(), idx_class_conf[i].second);
		}
		std::sort(idx_class_conf.begin(), idx_class_conf.end(),
			[&](std::pair<std::pair<int, int>, float> a, std::pair<std::pair<int, int>, float> b) {
			return a.second > b.second;
		});
		//printf("sort ... \n");

		//int p = 0;
		std::vector<int> del(idx_class_conf.size(), 0);

		for (int i = 0; i < idx_class_conf.size(); i++){
		//while (idx_class_conf.size() > 0 ) {

			/*std::sort(idx_class_conf.begin(), idx_class_conf.end(),
				[&](std::pair<std::pair<int, int>, float> a, std::pair<std::pair<int, int>, float> b) {
				return a.second < b.second;
			});*/
			//if (p == 2) exit(0);
			/*for (int i = 0; i < idx_class_conf.size(); ++i) {
				std::vector<float> bbox = bboxes[idx_class_conf[i].first.first];
				printf("%d  ( %f,%f,%f,%f,%s,%f )\n", idx_class_conf[i].first.first, bbox[0], bbox[1], bbox[2], bbox[3],
					VocClass[idx_class_conf[i].first.second].c_str(), idx_class_conf[i].second);
			}
			p++;*/
			if (del[i]) continue;
			std::vector<float> bbox = bboxes[idx_class_conf[i].first.first];
			if (bbox[0] < 0 || bbox[1] < 0 || bbox[2] < 0 || bbox[3] < 0) {
				del[i] = 1;
				continue;
			}
			if (bbox[0] > 1.0 || bbox[1] > 1.0 || bbox[2] > 1.0 || bbox[3] > 1.0) {
				del[i] = 1;
				continue;
			}

			result.push_back(idx_class_conf[i]);
			printf("push %d \n", idx_class_conf[i].first.first);
			del[i] = 1;

			for (int j = i+1; j < idx_class_conf.size(); j++)
			{
				if (del[j]) continue;
				float iou_value = Iou(bboxes[idx_class_conf[i].first.first], bboxes[idx_class_conf[j].first.first]);
				//printf("(%d,%d)   %f \n", idx_class_conf[i].first.first, idx_class_conf[j].first.first, iou_value);
				if (iou_value > nmsthresh)
				{
					del[j] = 1;
					//idx_class_conf.erase(idx_class_conf.begin() + i + 1);
				}
			}
			//idx_class_conf.erase(idx_class_conf.begin());
		}
	}


	/*
	# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
	"""Apply non-maximum suppression at test time to avoid detecting too many
	overlapping bounding boxes for a given object.
	Args:
		boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
		scores: (tensor) The class predscores for the img, Shape:[num_priors].
		overlap: (float) The overlap thresh for suppressing unnecessary boxes.
		top_k: (int) The Maximum number of box preds to consider.
	Return:
		The indices of the kept boxes with respect to num_priors.
	"""

	keep = scores.new(scores.size(0)).zero_().long()
	if boxes.numel() == 0:
		return keep
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	area = torch.mul(x2 - x1, y2 - y1)
	v, idx = scores.sort(0)  # sort in ascending order
	# I = I[v >= 0.01]
	idx = idx[-top_k:]  # indices of the top-k largest vals
	xx1 = boxes.new()
	yy1 = boxes.new()
	xx2 = boxes.new()
	yy2 = boxes.new()
	w = boxes.new()
	h = boxes.new()

	# keep = torch.Tensor()
	count = 0
	while idx.numel() > 0:
		i = idx[-1]  # index of current largest val
		# keep.append(i)
		keep[count] = i
		count += 1
		if idx.size(0) == 1:
			break
		idx = idx[:-1]  # remove kept element from view
		# load bboxes of next highest vals
		torch.index_select(x1, 0, idx, out=xx1)
		torch.index_select(y1, 0, idx, out=yy1)
		torch.index_select(x2, 0, idx, out=xx2)
		torch.index_select(y2, 0, idx, out=yy2)
		# store element-wise max with next highest score
		xx1 = torch.clamp(xx1, min=x1[i])
		yy1 = torch.clamp(yy1, min=y1[i])
		xx2 = torch.clamp(xx2, max=x2[i])
		yy2 = torch.clamp(yy2, max=y2[i])
		w.resize_as_(xx2)
		h.resize_as_(yy2)
		w = xx2 - xx1
		h = yy2 - yy1
		# check sizes of xx1 and xx2.. after each iteration
		w = torch.clamp(w, min=0.0)
		h = torch.clamp(h, min=0.0)
		inter = w*h
		# IoU = i / (area(a) + area(b) - i)
		rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
		union = (rem_areas - inter) + area[i]
		IoU = inter/union  # store result in iou
		# keep only elements with an IoU <= overlap
		idx = idx[IoU.le(overlap)]
	return keep, count
	*/
	float Iou(std::vector<float> a, std::vector<float> b)
	{
		float x1 = std::max(a[0], b[0]);
		float y1 = std::max(a[1], b[1]);
		float x2 = std::min(a[2], b[2]);
		float y2 = std::min(a[3], b[3]);
		float w = std::max(x2 - x1, 0.0f);
		float h = std::max(y2 - y1, 0.0f);

		float over_area = w * h;// (x2 - x1) * (y2 - y1);
		float iou = over_area / ((a[2] - a[0])*(a[3] - a[1]) + (b[2] - b[0])*(b[3] - b[1]) - over_area);
		return iou;
	}
};
	