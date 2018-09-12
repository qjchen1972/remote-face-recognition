# -*- coding: utf-8 -*-
from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from math import floor
from collections import OrderedDict
from math import sqrt
from global_pam import *

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features,affine=True)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate,affine=True)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
	def __init__(self, growth_rate=32, block_config=(6,12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
		super(DenseNet, self).__init__()

        # First convolution
		self.features = nn.Sequential(OrderedDict([
            		('conv0', nn.Conv2d(Global.img_chn, num_init_features, 
					kernel_size=Global.k_size, stride=Global.stride,
					padding=Global.padding, bias=False))
			,
            #('norm0', nn.BatchNorm2d(num_init_features)),
            #('relu0', nn.ReLU(inplace=True)),
            #('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
			]))

        # Each denseblock
		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			if num_layers == 0:continue
			block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
			self.features.add_module('denseblock%d' % (i + 1), block)
			num_features = num_features + num_layers * growth_rate
			if i != len(block_config) - 1:
				trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
				self.features.add_module('transition%d' % (i + 1), trans)
				num_features = num_features // 2

        # Final batch norm
		self.features.add_module('norm5', nn.BatchNorm2d(num_features))
		self.features.add_module('relu5', nn.ReLU(inplace=True))

		self.fc1 = nn.Linear(num_features, num_classes)
        
	def forward(self, x):
		features = self.features(x)
		p = tuple(int(i) for i in features.size()[2:])
		out = F.avg_pool2d(features, kernel_size=p, stride=1).view(features.size(0), -1)
		out = self.fc1(out)
		return out


class _SubBlock(nn.Module):

	def __init__(self, in_channels, out_channels):
		
		super(_SubBlock, self).__init__()
		self.bn = nn.BatchNorm2d(in_channels)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

	def forward(self, x):
		out = self.conv(F.relu(self.bn(x)))
		return torch.cat((x, out), 1)


class _DenseBlock40(nn.Module):
	
	def __init__(self, num_layers, in_channels, growth_rate):
		
		super(_DenseBlock40, self).__init__()

		layers = []
		for i in range(num_layers):
			cumul_channels = in_channels + i * growth_rate
			layers.append(_SubBlock(cumul_channels, growth_rate))

		self.block = nn.Sequential(*layers)
		self.out_channels = cumul_channels + growth_rate

	def forward(self, x):		
		out = self.block(x)
		return out


class _TransitionLayer40(nn.Module):
	
	def __init__(self, in_channels, theta):
		super(_TransitionLayer40, self).__init__()
		self.out_channels = int(floor(theta*in_channels))

		self.bn = nn.BatchNorm2d(in_channels)
		self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
		self.pool = nn.AvgPool2d(2)

	def forward(self, x):
		out = self.pool(self.conv(F.relu(self.bn(x))))
		return out


class DenseNet40(nn.Module):
    
	def __init__(self, num_blocks,num_layers_total, growth_rate,theta, num_classes):
    
		super(DenseNet40, self).__init__()

		error_msg = "[!] Total number of layers must be 3*n + 4..."
		assert (num_layers_total - 4) % num_blocks == 0, error_msg

		num_layers_dense = int((num_layers_total - 4) / num_blocks)
		out_channels = 64
		self.conv = nn.Conv2d(1, out_channels, kernel_size=3, padding=1)

		blocks = []
		for i in range(num_blocks - 1):
			dblock = _DenseBlock40(num_layers_dense, out_channels, growth_rate)
			blocks.append(dblock)

			out_channels = dblock.out_channels
			trans = _TransitionLayer40(out_channels, theta)
			blocks.append(trans)
			out_channels = trans.out_channels

		dblock = _DenseBlock40(num_layers_dense, out_channels, growth_rate)
		blocks.append(dblock)
		self.block = nn.Sequential(*blocks)
        	self.out_channels = dblock.out_channels
        	self.fc = nn.Linear(self.out_channels, num_classes)

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
		out = self.conv(x)
		out = self.block(out)
		out = F.avg_pool2d(out, kernel_size=8, stride=1).view(out.size(0), -1)
        	out = self.fc(out)
		return out     


	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
