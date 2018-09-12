The face recognition system embedded in the police enforcement device can identify criminals and alert them in real time. The project provides some features:

* The AI framework is based on pytorch + onnx + caffe2

* Aiming at the remote and weak light and low resolution, the training image is specially processed, and the data augment is also specially designed.

   *  Transform the image and remove the shadow. detail in [lib](https://github.com/qjchen1972/remote-face-recognition/blob/master/lib/README.md)
   
      ![](https://github.com/qjchen1972/remote-face-recognition/blob/master/img/lmcp.png)
      
   *  A convolutional neural network with modified resolution specially added.detail in [modifyface](https://github.com/qjchen1972/remote-face-recognition/blob/master/modifyface/README.md)
   
      ![](https://github.com/qjchen1972/remote-face-recognition/blob/master/img/modify.png)
