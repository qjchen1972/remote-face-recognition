# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import os
import numpy as np
from PIL import Image,ImageEnhance
import torch
from torch.utils.data import Dataset
import struct
try:
    import accimage
except ImportError:
    accimage = None
import random
from io import BytesIO

__all__ = ["Binary_Data","Img_Data","Img_img","Img_test"]

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
		with open(label_filepath, "r") as lf:			
			while True:                
				line = lf.readline()
				if not line:break
				
				items = line.split()
				if not items: break            
				img = os.path.join(img_filepath, items[0])
				if os.path.exists(img):
				    #one hot 
					#label = [0 for x in range(0, self.label_len)]
					#label[int(items[1])] = 1
					#create tensor
					#label = [int(items[1])]
					self.list_image.append(img)
					self.list_label.append(int(items[1]))
					
	def __getitem__(self, index):        
		image_path = self.list_image[index]
		image_data = Image.open(image_path).convert('L')
		#image_label= torch.FloatTensor(self.list_label[index])
		image_label=self.list_label[index]
		if self.transform != None: image_data = self.transform(image_data)
		return image_data, image_label

	def __len__(self):        
		return len(self.list_label)


class Img_test(Dataset):
        def __init__(self, img_filepath, label_filepath,transform=None):

                self.list_image = []
                self.transform = transform
                with open(label_filepath, "r") as lf:
                        while True:
                                line = lf.readline()
                                if not line:break
                                items = line.split()
                                if not items: break
                                img = os.path.join(img_filepath, items[0])
                                if os.path.exists(img):
                                    self.list_image.append(img)


        def __getitem__(self, index):
                image_path = self.list_image[index]
                image_data = Image.open(image_path).convert('RGB')
                #image_data = image_data.resize((64,64))
                #image_data = image_data.convert('L')
                width, height = image_data.size
		dstimg_data = image_data
                #dstimg_data = image_data.resize((48, 48), Image.BILINEAR)
                upimg_data = dstimg_data.resize((192, 192), Image.BICUBIC)
		if index == 16:upimg_data.save('2.jpg')
		#upimg_data = image_data
                if self.transform != None:
                        image_data = self.transform(image_data)
                        dstimg_data = self.transform(dstimg_data)
                        upimg_data = self.transform(upimg_data)
                return (dstimg_data,upimg_data),image_data

        def __len__(self):
                return len(self.list_image)

class Img_img(Dataset):
	def __init__(self, img_filepath, label_filepath,transform=None):
    
		self.list_image = []
		self.transform = transform
		self.img_augmenter = ImageAugment()
		with open(label_filepath, "r") as lf:			
			while True:                
				line = lf.readline()
				if not line:break				
				items = line.split()
				if not items: break            
				img = os.path.join(img_filepath, items[0])
				if os.path.exists(img):
				    self.list_image.append(img)
					

	def __getitem__(self, index):        
		image_path = self.list_image[index]
		image_data = Image.open(image_path).convert('L')
		image_data = image_data.resize((64,64),Image.BILINEAR)
		#dstimg_data = self.img_augmenter.process(image_data).convert('L')
		#image_data = image_data.convert('L')
		#image_data,_,_=image_data.convert('YCbCr').split()
		dstimg_data = self.img_augmenter.process(image_data)
		#width, height = image_data.size
		#dstimg_data = image_data.resize((width//2, height//2))
		#dstimg_data =  dstimg_data.resize((width, height))
		if self.transform != None: 
			image_data = self.transform(image_data)
			dstimg_data = self.transform(dstimg_data)
		return (dstimg_data,dstimg_data),image_data

	def __len__(self):        
		return len(self.list_image)

class ImageAugment:
    def __init__(self,noise_level=3 ):
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
	self.resize_level = [18, 32]
  
    def add_jpeg_noise(self, hr_img):
        quality = 100 - round(random.uniform(*self.noise_level))
        lr_img = BytesIO()
        hr_img.save(lr_img, format='JPEG', quality=int(quality))
        lr_img.seek(0)
        lr_img = Image.open(lr_img)
        return lr_img
	
    def process(self, hr_img):
	size = int(random.uniform(*self.resize_level))
	lr_img = hr_img.resize((size, size),Image.BILINEAR)
	width, height = hr_img.size
	lr_img = lr_img.resize((width, height),Image.BILINEAR)
	#lr_img = ImageEnhance.Brightness(lr_img).enhance(random.uniform(2,13) / 10.0)
	#if self.noise_level[1] > 0:
            #lr_img = self.add_jpeg_noise(lr_img)
        return lr_img
