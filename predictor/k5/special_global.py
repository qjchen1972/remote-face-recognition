# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division


class Global:
	class_out = 20002
	loss_log = 'k5/answer.txt'

	net = 'FACE_NET'
	model_path='./models/model-46.pth.tar'

	onnx_out='k5/facenet.proto'
	initnet='k5/init_net.pb'
	prednet='k5/predict_net.pb'

	train_img_path = '/home/pp_dst'
	train_label_path = '/home/pp_label.txt'
	train_batch_size = 64
	checkpoint = True
	lr = 0.00001
	max_epoch = 5000

	test_img_path=''
	test_label_path=''
	test_batch_size=1


	predict_img_path='/home/test'
	predict_label_path='/home/test/test_label.txt'
	predict_batch_size=1



	vecnum = 512
	mcp_size = 30
	mcp_dis = 0.35

	val_img_path = '/home/val'
	val_label_path = '/home/val_label.txt'

	img_w = 64
	img_h = 64
	img_chn = 1


	init_features =32
	growth_rate = 16
	block_config = (6,12,24,16)
	bn_size = 2

	k_size = 3
	stride = 1
	padding = 1

