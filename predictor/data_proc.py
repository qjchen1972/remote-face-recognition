# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import os
import numpy as np
from PIL import Image,ImageEnhance
import torch
import  torchvision
from torch.utils.data import Dataset
import struct
try:
    import accimage
except ImportError:
    accimage = None
import random
from io import BytesIO
from global_pam import *

__all__ = ["Binary_Data","Img_Data"]

class Binary_Data(Dataset):

	data_type = 2
	label_type = 2

	input_chn = 2
	input_row = 64
	input_col = 64

	lable_len = 7413

	def __init__(self, data_filepath, label_filepath,transform=None):
    
		self.datafile = data_filepath
		self.labelfile = label_filepath
		self.transform = transform
		self.label_size = label_type*label_len
		self.data_len = input_chn*input_row*input_col
		self.data_size = data_type * input_chn*input_row*input_col

	def __getitem__(self, index):        
		with open(self.datafile , 'rb') as df, open(self.labelfile , 'rb') as lf:
			lf.seek(index*self.label_size,0)
			read_buf = lf.read(self.label_size)			
			label_tuple = struct.unpack(str(lable_len)+'h', read_buf)	
			df.seek(index*self.data_size,0)
			read_buf = df.read(self.data_size)
			data_tuple = struct.unpack(str(self.data_len)+'h', read_buf)
			imagedata = torch.Tensor(np.reshape(data_tuple,(input_chn,input_row,input_col)))
		if self.transform != None:
			img = transforms.ToPILImage(imagedata)
			imagedata = self.transform(img)
		return imagedata, torch.Tensor(label_tuple)

	def __len__(self):	
		return int(os.path.getsize(self.labelfile) / self.label_size)
    

class Img_Data(Dataset):
	def __init__(self, img_filepath, label_filepath, transform=None):
    
		self.list_image = []
		self.list_label = []
		self.transform = transform
		self.img_augmenter = ImageAugment(3)
		#self.count = 0

		with open(label_filepath, "r") as lf:			
			while True:                
				line = lf.readline()
				if not line:break
				
				items = line.split()
				if not items: break            
				img = os.path.join(img_filepath, items[0])
				if os.path.exists(img):
					self.list_image.append(img)
					self.list_label.append(int(items[1]))
					
	def __getitem__(self, index):        
		image_path = self.list_image[index]
		image_data = Image.open(image_path).convert('RGB')
		#image_data.save('jpg/%d.jpg' %self.count)
		if Global.img_chn == 1:
			image_data = self.img_augmenter.process(image_data).convert('L')
			
		else:
			image_data = self.img_augmenter.process(image_data)
		image_label=self.list_label[index]
		#self.count = self.count + 1
		if self.transform != None: 
			image_data = self.transform(image_data)
		#tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
                #pil = tensor_to_pil(image_data)
                #pil.save('jpg/%d.jpg' %self.count)
		#self.count = self.count + 1
		#if self.count >= 100: exit(0)
		return image_data, image_label

	def __len__(self):        
		return len(self.list_label)

class ImageAugment:
    def __init__(self,noise_level=2 ):
        # noise_level (int): 0: no noise; 1: 5-95% quality; 2:25-95%
        if noise_level == 0:
            self.noise_level = [0, 0]
        elif noise_level == 1:
            self.noise_level = [0, 25]
        elif noise_level == 2:
            self.noise_level = [0, 50]
	elif noise_level == 3:
            self.noise_level = [0, 65]
        else:
            raise KeyError("Noise level should be either 0, 1, 2, 3")
	self.resize_level = [Global.img_w//3, Global.img_w]

     
  
    def add_jpeg_noise(self, hr_img):
        quality = 100 - round(random.uniform(*self.noise_level))
        lr_img = BytesIO()
        hr_img.save(lr_img, format='JPEG', quality=int(quality))
        lr_img.seek(0)
        lr_img = Image.open(lr_img)
        return lr_img

    def process(self, hr_img):
	
	type = int(round(random.uniform(0,10)))
	w = int(round(random.uniform(*self.resize_level)))
	h = int(round(random.uniform(*self.resize_level)))
	if type >= 7:
		lr_img = hr_img.resize((w, h),Image.BILINEAR)
	else:
		lr_img = hr_img
	
	
	type = int(round(random.uniform(0,10)))
	if type >= 8:
		lr_img = ImageEnhance.Brightness(lr_img).enhance(random.uniform(4,13) / 10.0)
	else:
		lr_img = lr_img

	type = int(round(random.uniform(0,10)))
	if type >= 9:
		lr_img = ImageEnhance.Color(lr_img).enhance(random.uniform(6,15) / 10.0)
        else:    
                lr_img = lr_img

	
        type = int(round(random.uniform(0,10)))
        if type >= 9:
		lr_img = ImageEnhance.Contrast(lr_img).enhance(random.uniform(6,15) / 10.0)
        else:
                lr_img = lr_img
  
	type = int(round(random.uniform(0,10)))
        if type >= 9:
		lr_img = ImageEnhance.Sharpness(lr_img).enhance(random.uniform(6,20) / 10.0)
        else:
                lr_img = lr_img
	#if self.noise_level[1] > 0:
            #lr_img = self.add_jpeg_noise(lr_img)
        return lr_img


    
