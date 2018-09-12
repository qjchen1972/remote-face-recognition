#!/usr/bin/env python
# -*- coding:utf-8 -*-
import onnx
import io
from global_pam import *
# Load the ONNX model
model = onnx.load(Global.onnx_out)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

import caffe2.python.onnx.backend as backend
import numpy as np


rep = backend.prepare(model, device="CUDA:0") # or "CPU"
#rep = backend.prepare(model, device="CPU")
# For the Caffe2 backend:
#     rep.predict_net is the Caffe2 protobuf for the network
#     rep.workspace is the Caffe2 workspace for the network
#       (see the class caffe2.python.onnx.backend.Workspace)
outputs = rep.run(np.random.randn(1, Global.img_chn, Global.img_h, Global.img_w).astype(np.float32))
# To run networks with more than one input, pass a tuple
# rather than a single numpy ndarray.
print(outputs[0])

c2_workspace = rep.workspace
c2_graph = rep.predict_net

from caffe2.python.predictor import mobile_exporter

init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_graph, c2_graph.external_input)

with io.open(Global.initnet, 'wb') as f:
    f.write(init_net.SerializeToString())
with open(Global.prednet, 'wb') as f:
    f.write(predict_net.SerializeToString())
