# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from torch.autograd import Variable
from math import sqrt
from net import DenseNet
from global_pam import *

def face_net(classes,**kwargs):

	model = DenseNet(num_init_features=Global.init_features, growth_rate=Global.growth_rate, 
					block_config=Global.block_config, bn_size=Global.bn_size, num_classes=classes,**kwargs)
	return model

class FaceNet(nn.Module):

	def __init__(self, classCount):
	
		super(FaceNet, self).__init__()		
		self.face_net = face_net(classCount)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
				m.weight.data.normal_(0, sqrt(2. / n))
				if m.bias is not None:m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.in_features
				m.weight.data.normal_(0, sqrt(2. / n))
				if m.bias is not None:m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, x):
		out = self.face_net(x)
		return out

class DenseNet121(nn.Module):

    def __init__(self, classCount):	
        super(DenseNet121, self).__init__()		
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        kernelCount = self.densenet121.classifier.in_features		
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


class DenseNet169(nn.Module):
    
    def __init__(self, classCount):
        
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=True)
        
        kernelCount = self.densenet169.classifier.in_features
        
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet169(x)
        return x
    

class DenseNet201(nn.Module):
    
    def __init__ (self, classCount):
        
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=True)        
        kernelCount = self.densenet201.classifier.in_features        
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet201(x)
        return x
