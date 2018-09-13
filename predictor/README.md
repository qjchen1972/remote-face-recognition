Reference paper [CosFace: LargeMarginCosineLossforDeepFaceRecognition
](https://github.com/qjchen1972/remote-face-recognition/blob/master/predictor/cosface.pdf), we used cosface to replace SoftMax.


Dependencies
====
* python2.7 or later
* pytorchV0.3 or later

data 
====

*  the train is 64X64 images of 1078299 of MS-Celeb-1M
*  the label is 20001
*  The validation set is the 3 class of 500 faces with different resolutions. The result of the validation set is that the sum of COS distance is calculated by using a normal face and all 3 types of faces. 

run
===
* train : ./main.py
* predictor: ./main.py -m 2
* test : ./main.py -m 1
* create onnx: ./main.py -m 3 -c
