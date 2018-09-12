The face recognition system embedded in the police enforcement device can identify criminals and alert in real time. The project provides some features:

* I try to share everything , but it's a commercial project, so the open source code is not complete. 

* The AI framework is based on pytorch + onnx + caffe2

* Aiming at the remote and weak light and low resolution, the training image is specially processed, and the data augment is also specially designed.

   *  Transform the image and remove the shadow. detail in [lib](https://github.com/qjchen1972/remote-face-recognition/blob/master/lib/README.md)
   
      ![](https://github.com/qjchen1972/remote-face-recognition/blob/master/img/lmcp.png)
      
   *  A convolutional neural network with modified resolution specially added.detail in [modifyface](https://github.com/qjchen1972/remote-face-recognition/blob/master/modifyface/README.md)
   
      ![](https://github.com/qjchen1972/remote-face-recognition/blob/master/img/modify.png)

*  Fast and small network, the network of face recognitio is only 15228kb and the network of modifying face resolution is 757kb


Here are some test results. The test video is in [img](https://github.com/qjchen1972/remote_recognition/tree/master/img).

* Identifying twins

  ![](https://github.com/qjchen1972/remote_recognition/blob/master/img/twns.png)

* Long distance  identification, face area is only 20X20

 ![](https://github.com/qjchen1972/remote_recognition/blob/master/img/remote.png)



