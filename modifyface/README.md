In order to improve low resolution images, we trained this deep learning network. Our model is modified on the model provided in this paper.[Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network ](https://github.com/qjchen1972/remote-face-recognition/blob/master/modifyface/dcscnn.pdf)


Dependencies
===
* python2.7 or later
* pytorchV0.3 or later

Training image
===
*  the label is 64X64 images of 500 thousand of MS-Celeb-1M
*  In training, we use label images to randomly generate  images size between 18 and 32 with BILINEAR. then resize to 64X64 image  for inputing


run
====
* train :  ./main.py
* predictor: ./main.py -m 2
* create onnx:  ./main.py -m 3 -c
