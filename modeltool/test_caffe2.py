# -*- coding:utf-8 -*-
#%matplotlib inline
from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
from PIL import Image
from matplotlib import pyplot
import os
from caffe2.python import core, net_drawer, net_printer, visualize, workspace,utils
import urllib2
print("Required modules imported.")

IMAGE_LOCATION =  "/home/0.jpg"
INIT_NET = "k5/init_net.pb"
PREDICT_NET = "k5/predict_net.pb"
output="output"
width=64
height=64

srcimg = skimage.io.imread(IMAGE_LOCATION,as_grey=True)
#print srcimg
#[0,1]
#srcimg = skimage.transform.resize(srcimg, (width, height))
#print srcimg
#img = skimage.img_as_float(srcimg).astype(np.float32)

#[-1,1]
img = srcimg - 127.5
img = img / 127.5
img = skimage.transform.resize(img, (width, height))

img = img[np.newaxis, :, :].astype(np.float32)
img = img[np.newaxis, :, :, :].astype(np.float32)


with open(INIT_NET) as f:
    init_net = f.read()
with open(PREDICT_NET) as f:
    predict_net = f.read()


workspace.RunNetOnce(init_net)
workspace.CreateNet(predict_net)
p = workspace.Predictor(init_net, predict_net)
results = p.run([img])
img_out = workspace.FetchBlob(output)
print(type(img_out),img_out.size, img_out.shape)
for i in range(img_out.shape[1]):
    if i%16 == 0  : print("\n")
    print img_out[0][i],
print("\n")

