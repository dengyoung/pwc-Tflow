#!/usr/bin/env python

import argparse
import os
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
# from model_flow_middle_area import Model
from PWCNet import pwc_net

weight = sys.argv[1]

state_dict = torch.load(weight, map_location='cpu')


class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = pwc_net.PWCNet().eval()
        self.model.load_state_dict(state_dict)

    def forward(self, image1, image2):

        flow_estimation = self.model(image1, image2)
        
        return flow_estimation

model = _M()


flow = model(
    torch.randn(1, 3 ,480 ,640),
    torch.randn(1, 3 ,480 ,640)
    )

print(flow.shape)

torch.onx.export(model,                     # model being run
                (torch.randn(1, 3 ,480 ,640),
                torch.randn(1, 3 ,480 ,640)),
                f"model_pwc.onnx",             # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=11,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['img1', 'img2'],   # the model's input names
                output_names = ['flow'])


print("export to", os.path.abspath("model_pwc.onnx"))
