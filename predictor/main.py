#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import os
import numpy as np
import time
import sys
import logging
from torch.autograd.gradcheck import *
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn
import math
import argparse

from train_proc import Train_Proc
from global_pam import *

logging.basicConfig(level=logging.INFO, format='%(message)s')

parser = argparse.ArgumentParser(description='chenqianjiang  model')
parser.add_argument('-m', type=int, default=0, metavar='mode', help='input run mode (default: 0)')
parser.add_argument('-c', action='store_true', default=False, help='disables CUDA training')

def main():
	args = parser.parse_args()
	print(args.m,args.c)

	if args.m == 0:run_train(args.c)
	elif args.m == 1:run_test(args.c)
	elif args.m == 2:run_predict(args.c)
	elif args.m == 3:run_onnx(args.c)
	else: pass

def run_train( iscpu ):    
	net = Global.net
	img_path = Global.train_img_path
	label_path = Global.train_label_path
	model_path = Global.model_path
	batch_size = Global.train_batch_size
	checkpoint = Global.checkpoint
	lr = Global.lr
	max_epoch = Global.max_epoch
	Train_Proc.train(img_path, label_path, model_path, net, batch_size, max_epoch, checkpoint,lr,iscpu)

def run_test( iscpu ):    
	net = Global.net
	img_path = Global.test_img_path
	label_path = Global.test_label_path
	model_path = Global.model_path
	batch_size = Global.test_batch_size
	Train_Proc.test(img_path, label_path, model_path, net, batch_size, iscpu)

def run_predict( iscpu ):    
	net = Global.net
	img_path = Global.predict_img_path
	label_path = Global.predict_label_path
	model_path = Global.model_path
	batch_size = Global.predict_batch_size
	Train_Proc.predict(img_path, label_path, model_path, net, batch_size, iscpu)

def run_onnx(iscpu):
	net = Global.net
	model_path = Global.model_path
	Train_Proc.create_onnx(net,model_path,iscpu)


if __name__ == '__main__':
	main()
