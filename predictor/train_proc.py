# -*- coding:utf-8 -*-
from __future__ import print_function
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import numpy as np
import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
#import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix,roc_curve, auc
import time
import math
import torch.onnx

from data_proc import * 
from models import *
from global_pam import *
import layer
from torch.autograd import Variable
import transforms

logging.basicConfig(level=logging.INFO, format='%(message)s')


class Train_Proc():
	
	@staticmethod
	def validate(model, data_loader, isgpu):   		
		
		model.eval()

		alllist=[]
		for i, (input, target) in enumerate(data_loader):
			if isgpu: input = input.cuda(async=True)
			input_var = torch.autograd.Variable(input)
			vec = model(input_var)
			vec = torch.squeeze(vec, 0 )
			alllist.append(vec.data)
		sum1 = 0
		for n in range(len(alllist)):
			if n < 369  : sum1 +=  float(torch.dot(alllist[0],alllist[n])/(torch.norm(alllist[0]) * torch.norm(alllist[n])))
			elif n < 700: sum1 +=  float(torch.dot(alllist[369],alllist[n])/(torch.norm(alllist[369]) * torch.norm(alllist[n])))
			else: 
				sum1 +=  float(torch.dot(alllist[700],alllist[n])/(torch.norm(alllist[700]) * torch.norm(alllist[n])))

		sum2 = 0
        	for n in range(len(alllist)):
            		if n < 369 :
				sum2 +=  float(torch.dot(alllist[369],alllist[n])/(torch.norm(alllist[369]) * torch.norm(alllist[n])))
				sum2 +=  float(torch.dot(alllist[700],alllist[n])/(torch.norm(alllist[700]) * torch.norm(alllist[n])))
			elif n < 700:
				sum2 +=  float(torch.dot(alllist[0],alllist[n])/(torch.norm(alllist[0]) * torch.norm(alllist[n])))
				sum2 +=  float(torch.dot(alllist[700],alllist[n])/(torch.norm(alllist[700]) * torch.norm(alllist[n])))
			else:
				sum2 +=  float(torch.dot(alllist[369],alllist[n])/(torch.norm(alllist[369]) * torch.norm(alllist[n])))
				sum2 +=  float(torch.dot(alllist[0],alllist[n])/(torch.norm(alllist[0]) * torch.norm(alllist[n])))
		return sum1,sum2


	@staticmethod
	def epoch_train(model, data_loader, optimizer, loss,MCP,isgpu):

		model.train()
		loss_train = 0
	    
		for batch_id, (input, target) in enumerate(data_loader):
			if isgpu: input = input.cuda(async=True)
			if isgpu: target = target.cuda(async=True)
			var_input = torch.autograd.Variable(input)
			var_target = torch.autograd.Variable(target)
			var_output =  model(var_input)
			var_output = MCP(var_output, var_target)
			lossvalue = loss(var_output, var_target)
			loss_train += lossvalue.item()
			optimizer.zero_grad()
			lossvalue.backward()
			optimizer.step()
			temploss = loss_train/(batch_id + 1)
			if batch_id % 500 == 0 :
				logging.info(' %d : training_loss = %.6f  whole loss = %.6f' % ( batch_id, lossvalue.item(),temploss ))
		if isgpu:torch.cuda.empty_cache()
		return temploss

	@staticmethod     
	def train(img_path, label_path, model_path, net, batch_size, max_epoch, checkpoint,lr,iscpu):
		
		isgpu = not iscpu and torch.cuda.is_available()
		if isgpu: cudnn.benchmark = True
		rtime = time.time()

		if net == 'FACE_NET':
			model = FaceNet(Global.vecnum)
			MCP = layer.MarginCosineProduct(Global.vecnum, Global.class_out,s=Global.mcp_size,m=Global.mcp_dis)
		if not isgpu:
			model = model.cpu()
			MCP = MCP.cpu()
			#model = torch.nn.DataParallel(model)
			torch.manual_seed(int(rtime))
		else: 
			model = model.cuda()
			#model = torch.nn.DataParallel(model).cuda()
			torch.cuda.manual_seed(int(rtime))
			MCP = MCP.cuda()

		print(model)

		kwargs = {'num_workers': 16, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.RandomHorizontalFlip())
		#transformList.append(transforms.RandomResizedCrop((Global.img_w,Global.img_h),scale=(0.07,1.0),ratio=(0.4,2.5)))
		#transformList.append(transforms.ColorJitter(0.7,0.3,0.3,0.3))
		transformList.append(transforms.Resize((Global.img_w,Global.img_h)))
		transformList.append(transforms.RandomCrop((Global.img_w,Global.img_h),12))
		#transformList.append(transforms.Resize((Global.img_w,Global.img_h)))
		transformList.append(transforms.ToTensor())
		transformList.append(transforms.Randomblack(mean = [0.0]))
		transformList.append(transforms.RandomErasing(mean = [0.0]))
		transformList.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))) 
		transformSequence=transforms.Compose(transformList)

		data = Img_Data(img_filepath = img_path, label_filepath = label_path,transform=transformSequence)
		data_loader = DataLoader(dataset = data,batch_size=batch_size,shuffle=True,**kwargs)

		val_kwargs = {'num_workers': 1, 'pin_memory': True} if isgpu else {}
		val_transformList = []
		val_transformList.append(transforms.Resize((Global.img_w,Global.img_h)))
		val_transformList.append(transforms.ToTensor())
		val_transformList.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
		val_transformSequence=transforms.Compose(val_transformList)
		valdata = Img_Data(img_filepath =Global.val_img_path,label_filepath = Global.val_label_path,transform=val_transformSequence)
		val_data_loader = DataLoader(dataset = valdata,batch_size=1,shuffle=False,**val_kwargs)

		loss = torch.nn.CrossEntropyLoss()
	
		#optimizer = optim.SGD([{'params': model.parameters()}, {'params': MCP.parameters()}], lr=lr,momentum=0.9)
		optimizer = optim.Adam([{'params': model.parameters()}, {'params': MCP.parameters()}], lr=lr,weight_decay=0,amsgrad=True)
			
		if checkpoint == True:
			modelCheckpoint = torch.load(model_path)
			model.load_state_dict(modelCheckpoint['state_dict'])
			#optimizer.load_state_dict(modelCheckpoint['optimizer'])
			MCP.load_state_dict(modelCheckpoint['MCP'])
			start = modelCheckpoint['epoch'] + 1
		else:
			start = 0
		
		for epoch_id in range(start, max_epoch):
			oldtime = time.time()                        
			loss_train = Train_Proc.epoch_train(model, data_loader, optimizer,loss,MCP,isgpu)
			newtime = time.time()            
			torch.save({'epoch':epoch_id, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(),'MCP':MCP.state_dict()}, 
				'./models/'+'model-'+str(epoch_id)+'.pth.tar')			
			totaltime = newtime - oldtime
			val_sum1,val_sum2 = Train_Proc.validate(model,val_data_loader,isgpu)
			logging.info('Epoch %d : training_loss =  %.6f  time = %d   val1 = %.6f val2 = %.6f  %.6f' % ( epoch_id, loss_train , totaltime,val_sum1,val_sum2,val_sum1-val_sum2))
			with open(Global.loss_log,'a+') as ftxt:		
				ftxt.write('\nEpoch %d : training_loss =  %.6f  time = %d  val1 = %.6f val2 = %.6f  %.6f\n' % ( epoch_id, loss_train , totaltime,val_sum1,val_sum2,val_sum1-val_sum2))

	@staticmethod		
	def test(img_path, label_path, model_path, net, batch_size, iscpu):                 
		
		oldtime = time.time()	
		isgpu = not iscpu and torch.cuda.is_available()
		if isgpu : cudnn.benchmark = True

		if net == 'FACE_NET':
			model = FaceNet(Global.vecnum)

		if not isgpu :
			model = model.cpu()
			model = torch.nn.DataParallel(model)
			modelCheckpoint = torch.load(model_path,map_location='cpu')
		else: 
			model = model.cuda()
			model = torch.nn.DataParallel(model).cuda()
			modelCheckpoint = torch.load(model_path)
		
		model.load_state_dict(modelCheckpoint['state_dict'])

		kwargs = {'num_workers': 1, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.Resize((Global.img_w,Global.img_h)))
		transformList.append(transforms.ToTensor())
		val_transformList.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))		
		transformSequence=transforms.Compose(transformList)

		data = Img_Data(img_filepath = img_path, label_filepath = label_path, transform=transformSequence)
		data_loader = DataLoader(dataset = data,batch_size=batch_size,shuffle=False,**kwargs)

		model.eval()	
		
		out_label = torch.FloatTensor().cuda()
		out_prect = torch.FloatTensor().cuda()
		
		for i, (input, target) in enumerate(data_loader):
			if isgpu : input = input.cuda(async=True)
			input_var = torch.autograd.Variable(input,volatile=True)
			output,vec = model(input_var)
			out_label = torch.cat((out_label, target), 0)
			out_prect = torch.cat((out_prect, output.data), 0)
		
		 # save the output
        	predicted = torch.max(out_prect, 1)[1]
       		Y = np.reshape(predicted, -1)
        	np.savetxt('Y.csv', Y, delimiter=",", fmt='%5s')

        	# compute loss & accuracy 
        	predicted = torch.max(out_prect, 1)[1]
        	label = torch.max(out_label, 1)[1]
        	total = len(label)
        	print('Total number of test data is', total)
        	correct = (predicted == label).sum()
        	acc = 100 * (correct / total)
        	print('The accuracy is', acc)

	@staticmethod
	def predict(img_path, label_path, model_path, net, batch_size, iscpu):   

		isgpu = not iscpu and torch.cuda.is_available()
		if isgpu: cudnn.benchmark = True

		if net == 'FACE_NET':
			model = FaceNet(Global.vecnum)
		
		if not isgpu :
			model = model.cpu()
			model = torch.nn.DataParallel(model)
			modelCheckpoint = torch.load(model_path,map_location='cpu')
		else: 
			model = model.cuda()
			model = torch.nn.DataParallel(model).cuda()
			modelCheckpoint = torch.load(model_path)
		
		model.load_state_dict(modelCheckpoint['state_dict'])

		kwargs = {'num_workers': 1, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.Resize((Global.img_w,Global.img_h)))
		transformList.append(transforms.ToTensor())
		transformList.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
		transformSequence=transforms.Compose(transformList)

		data = Img_Data(img_filepath = img_path, label_filepath = label_path, transform=transformSequence)
		data_loader = DataLoader(dataset = data,batch_size=batch_size,shuffle=False,**kwargs)

		model.eval()

		alllist=[]
		loss = torch.nn.CrossEntropyLoss()
		for i, (input, target) in enumerate(data_loader):

			if isgpu: input = input.cuda(async=True)
			if isgpu: target = target.cuda(async=True)
			input_var = torch.autograd.Variable(input)
			var_target = torch.autograd.Variable(target)
			vec = model(input_var)
			vec = torch.squeeze(vec, 0 )
			alllist.append(vec.data)
						
		oldtime = time.time()
		for n in range(len(alllist)):
			for j in range(len(alllist)):
				if n == j:continue
				print(n,'---',j,' == ', float(torch.dot(alllist[n],alllist[j])/(torch.norm(alllist[n]) * torch.norm(alllist[j]))),'\n')
		newtime = time.time()
		print('cal cos is %d, one time is %f' %(newtime-oldtime, (newtime-oldtime)*1.0/((n+1)*n)))

	@staticmethod
	def create_onnx(net,model_path,iscpu):

		isgpu = not iscpu and torch.cuda.is_available()
		
		if net == 'FACE_NET':
			model = FaceNet(Global.vecnum)
	
		if not isgpu:
			#model = torch.nn.DataParallel(model)
			modelCheckpoint = torch.load(model_path,map_location='cpu')
			dummy_input = Variable(torch.randn(1, Global.img_chn, Global.img_h, Global.img_w))
		else:
			model = model.cuda()
            		#model = torch.nn.DataParallel(model).cuda()		
			dummy_input = Variable(torch.randn(1, Global.img_chn, Global.img_h, Global.img_w)).cuda()
			modelCheckpoint = torch.load(model_path)
			print('it is gpu')

		model.load_state_dict(modelCheckpoint['state_dict'])
		input_names = [ "input" ] 
		output_names = [ "output" ]
		torch.onnx.export(model, dummy_input, Global.onnx_out, verbose=True, input_names=input_names, output_names=output_names)
		

		
