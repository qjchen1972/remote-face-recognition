# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import k5.special_global as sglobal

class Global:
	class_out = sglobal.Global.class_out
	loss_log = sglobal.Global.loss_log

	net = sglobal.Global.net
	model_path = sglobal.Global.model_path
	onnx_out=sglobal.Global.onnx_out
	initnet=sglobal.Global.initnet
	prednet=sglobal.Global.prednet

	train_img_path = sglobal.Global.train_img_path
	train_label_path = sglobal.Global.train_label_path
	train_batch_size = sglobal.Global.train_batch_size
	checkpoint = sglobal.Global.checkpoint
	lr = sglobal.Global.lr
	max_epoch = sglobal.Global.max_epoch

	test_img_path = sglobal.Global.test_img_path 
	test_label_path = sglobal.Global.test_label_path 
	test_batch_size = sglobal.Global.test_batch_size


	predict_img_path = sglobal.Global.predict_img_path
	predict_label_path = sglobal.Global.predict_label_path
	predict_batch_size = sglobal.Global.predict_batch_size



	vecnum = sglobal.Global.vecnum
	mcp_size = sglobal.Global.mcp_size
	mcp_dis = sglobal.Global.mcp_dis

	val_img_path = sglobal.Global.val_img_path
	val_label_path = sglobal.Global.val_label_path

	img_w = sglobal.Global.img_w
	img_h = sglobal.Global.img_h
	img_chn = sglobal.Global.img_chn


	init_features = sglobal.Global.init_features
	growth_rate = sglobal.Global.growth_rate
	block_config = sglobal.Global.block_config
	bn_size = sglobal.Global.bn_size 

	k_size = sglobal.Global.k_size 
	stride = sglobal.Global.stride 
	padding = sglobal.Global.padding

