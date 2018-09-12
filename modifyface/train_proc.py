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
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix,roc_curve, auc
import time
import math
import torch.onnx
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_proc import * 
from models import *
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO, format='%(message)s')

loss_log = './answer.txt'

def tensor_to_jpg(tensor, filename,isgpu):
	tensor = tensor.view(tensor.shape[1:])
	if isgup:
        	tensor = tensor.cpu()
	tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
	pil = tensor_to_pil(tensor)
	pil.save(filename)


class Train_Proc():
	
	@staticmethod
	def epoch_train(model, data_loader, optimizer, loss,isgpu):

		model.train()
		loss_train = 0
	    
		for batch_id, (input,target) in enumerate(data_loader):
			if isgpu: 
				lr,lr_up = input
				lr = lr.cuda(async=True)
				lr_up = lr_up.cuda(async=True)
				input = (lr,lr_up)
			if isgpu: target = target.cuda(async=True)
			output =  model(input)
			lossvalue = loss(output, target)
			loss_train += lossvalue.item()
			optimizer.zero_grad()
			lossvalue.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 5)
			optimizer.step()
			temploss = loss_train/(batch_id + 1)
			if batch_id % 1 == 0 :
				logging.info(' %d : training_loss = %.6f  whole loss = %.6f' % ( batch_id, lossvalue.item(),temploss ))
		
		if isgpu:torch.cuda.empty_cache()
		return temploss

	@staticmethod     
	def train(img_path, label_path, model_path, net, batch_size, max_epoch, checkpoint,lr,iscpu):
		
		isgpu = not iscpu and torch.cuda.is_available()
		if isgpu: cudnn.benchmark = True
		rtime = time.time()

		if net == 'DCSCN_NET':
			model = DCSCN(color_channel=1,
					up_scale=1,
					feature_layers=6,
					first_feature_filters=96,
					last_feature_filters=16,
					reconstruction_filters=32,
					up_sampler_filters=32)

		if not isgpu:
			model = model.cpu()
			torch.manual_seed(int(rtime))
		else: 
			model = model.cuda()
			torch.cuda.manual_seed(int(rtime))
		
		print(model)

		kwargs = {'num_workers': 16, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.ToTensor())
		transformSequence=transforms.Compose(transformList)

		data = Img_img(img_filepath = img_path, label_filepath = label_path,transform=transformSequence)
		data_loader = DataLoader(dataset = data,batch_size=batch_size,shuffle=True,**kwargs)

		loss = nn.L1Loss()
		
		#pre_train = torch.load("./model_results/DCSCN_model_135epos.pt", map_location='cpu')
		#new = model.state_dict()
		#a = {}
		#for i, j in zip(pre_train, new):
			#a[j] = pre_train[i]

		#model.load_state_dict(a, strict=False)
		#torch.save(model.state_dict(), "./model_results/DCSCN_model_135epos_2.pt")
		optimizer = optim.Adam(model.parameters(), lr=lr,  amsgrad=True)
		#scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
		start = 0
		if checkpoint == True:
			modelCheckpoint = torch.load(model_path)
			model.load_state_dict(modelCheckpoint['state_dict'])
			#optimizer.load_state_dict(modelCheckpoint['optimizer'])
			start = modelCheckpoint['epoch'] + 1

		
		for epoch_id in range(start, max_epoch):
			oldtime = time.time()                        
			loss_train = Train_Proc.epoch_train(model, data_loader, optimizer,loss,isgpu)
			newtime = time.time()            
			torch.save({'epoch':epoch_id, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}, 
				'./models/'+'model-'+str(epoch_id)+'.pth.tar')			
			totaltime = newtime - oldtime
			for i, param_group in enumerate(optimizer.param_groups):
				oldlr = float(param_group['lr'])
				break
			logging.info('Epoch %d : training_loss =  %.6f  time = %d lr = %.6f' % ( epoch_id, loss_train , totaltime,oldlr))
			with open(loss_log,'a+') as ftxt:		
				ftxt.write('\nEpoch %d : training_loss =  %.6f  time = %d lr = %.6f \n' % ( epoch_id, loss_train , totaltime,oldlr))
			#scheduler.step(loss_train)

	@staticmethod		
	def test(img_path, label_path, model_path, net, te_batch_size, iscpu):                 
		
		oldtime = time.time()	
		isgpu = not iscpu and torch.cuda.is_available()
		if isgpu : cudnn.benchmark = True

		if net == 'FACE_NET':
			model = FaceNet(class_out)

		if not isgpu :
			model = model.cpu()
			model = torch.nn.DataParallel(model)
		else: 
			model = model.cuda()
			model = torch.nn.DataParallel(model).cuda()
		
		modelCheckpoint = torch.load(model_path)
		model.load_state_dict(modelCheckpoint['state_dict'])

		kwargs = {'num_workers': 1, 'pin_memory': True} if isgpu else {}
		transformList = []
		transformList.append(transforms.Resize((64,64)))
		transformList.append(transforms.ToTensor())
		transformSequence=transforms.Compose(transformList)

		#data = Binary_Data(data_filepath = img_path, label_filepath = label_path )
		data = Img_Data(img_filepath = img_path, label_filepath = label_path, transform=transformSequence)
		data_loader = DataLoader(dataset = data,batch_size=te_batch_size,shuffle=False,**kwargs)

		model.eval()	

		if isgpu :
			out_label = torch.FloatTensor().cuda()
			out_prect = torch.FloatTensor().cuda()
		else:
			out_label = torch.FloatTensor()
			out_prect = torch.FloatTensor()

		for i, (input, target) in enumerate(data_loader):
			if isgpu : input = input.cuda(async=True)
			input_var = torch.autograd.Variable(input,volatile=True)
			output,vec = model(input_var)

			if isgpu :
				out_label = torch.cat((out_label, target), 0)
				out_prect = torch.cat((out_prect, output.data), 0)
			else:
				out_label = torch.cat((out_label, target.cpu()), 0)
				out_prect = torch.cat((out_prect, output.data.cpu()), 0)
		newtime = time.time()
		
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
		print('total time is %d, one time is %f' %(newtime-oldtime, (newtime-oldtime)*1.0/total))

        	# Confussion matrix.
        	y_gt = np.reshape(label, [-1])
        	y_pd = np.reshape(predicted, [-1])
        	c_matrix = confusion_matrix(y_gt, y_pd)
        	print('Confussion Matrix:\n', c_matrix)
        	tn, fp, fn, tp = confusion_matrix(y_gt, y_pd).ravel()
        	print('TN FP FN TP:\n',tn, fp, fn, tp)

	@staticmethod
	def predict(img_path, label_path, model_path, net, te_batch_size, iscpu):   
		isgpu = not iscpu and torch.cuda.is_available()
		if isgpu: cudnn.benchmark = True
		
		if net == 'DCSCN_NET':
			model = DCSCN(color_channel=1,
                                        up_scale=1,
                                        feature_layers=12,
                                        first_feature_filters=196,
                                        last_feature_filters=48,
                                        reconstruction_filters=64,
                                        up_sampler_filters=32)


		if not isgpu :
			model = model.cpu()
			#model = torch.nn.DataParallel(model)
		else: 
			model = model.cuda()
			#model = torch.nn.DataParallel(model).cuda()
		
		modelCheckpoint = torch.load(model_path)
		#model.load_state_dict(modelCheckpoint['state_dict'])
		
		model.load_state_dict(modelCheckpoint)

		kwargs = {'num_workers': 1, 'pin_memory': True} if isgpu else {}
		transformList = []
		#transformList.append(transforms.Resize((64,64)))
		transformList.append(transforms.ToTensor())
		transformSequence=transforms.Compose(transformList)

		#data = Binary_Data(data_filepath = img_path, label_filepath = label_path )
		data = Img_test(img_filepath = img_path, label_filepath = label_path, transform=transformSequence)
		data_loader = DataLoader(dataset = data,batch_size=te_batch_size,shuffle=False,**kwargs)

		model.eval()

		for i, (input, target) in enumerate(data_loader):
			if(i != 16): continue

			if isgpu:
				lr,lr_up = input
                                lr = lr.cuda(async=True)
                                lr_up = lr_up.cuda(async=True)
                                input = (lr,lr_up) 
			
			vec = model(input)
			print(type(vec),vec.size())
			vec = vec.view(vec.shape[1:])
			vec= vec.cpu()
			tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
			pil = tensor_to_pil(vec)
			pil.save('1.jpg')
			print(vec.size())			
			#for k in range(vec.size(2)):
				#print("\n")
				#for m in range(vec.size(3)):
			    		#print(vec[0][0][k][m].item(),end='  ')
			#print("\n")
			exit(0)		
	
	@staticmethod
	def create_onnx(net,model_path,iscpu):

		chn = 1
		height = 64
		width = 64
		protofile="./modify_proto"

		isgpu = not iscpu and torch.cuda.is_available()
		if net == 'DCSCN_NET':
			model = DCSCN(color_channel=1,
                                        up_scale=1,
                                        feature_layers=6,
                                        first_feature_filters=96,
                                        last_feature_filters=16,
                                        reconstruction_filters=32,
                                        up_sampler_filters=32)
		
		if not isgpu:
			model = model.cpu()
			#model = torch.nn.DataParallel(model)
			input = torch.randn(1, chn, height, width)
			up_input = torch.randn(1, chn, height, width)
			dummy_input = [torch.randn(1, chn, height, width),torch.randn(1, chn, height, width)]
		else:
			model = model.cuda()
            		#model = torch.nn.DataParallel(model).cuda()		
			dummy_input = Variable(torch.randn(1,  chn, height, width)).cuda()
	
		modelCheckpoint = torch.load(model_path, map_location='cpu')
		model.load_state_dict(modelCheckpoint['state_dict'])
		input_names = [ "input" ]+["up"]+["learned_%d" % i for i in range(21) ] 
		output_names = ["output" ]
		torch.onnx.export(model, dummy_input, protofile, verbose=True, input_names=input_names, output_names=output_names)
		

		
