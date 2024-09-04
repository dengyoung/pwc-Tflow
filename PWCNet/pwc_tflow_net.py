import numpy as np
import torch

from PWCNet import encoder
# from PWCNet import decoder
from PWCNet import decoder_tflow
from PWCNet import refiner

class PWCNet_tflow(torch.nn.Module):
    def __init__(self):
        super(PWCNet_tflow, self).__init__()

        self.encoder = encoder.Encoder()
        self.netTwo = decoder_tflow.Decoder(2)
        self.netThr = decoder_tflow.Decoder(3)
        self.netFou = decoder_tflow.Decoder(4)
        self.netFiv = decoder_tflow.Decoder(5)
        self.netSix = decoder_tflow.Decoder(6)
        self.refiner = refiner.Refiner()
        
    def init_cam_intri(self, cam_intri, cam_intri_inv, height, width):

        self.netTwo.init_cam_intri(cam_intri, cam_intri_inv, height, width)
        self.netThr.init_cam_intri(cam_intri, cam_intri_inv, height, width)
        self.netFou.init_cam_intri(cam_intri, cam_intri_inv, height, width)
        self.netFiv.init_cam_intri(cam_intri, cam_intri_inv, height, width)
        self.netSix.init_cam_intri(cam_intri, cam_intri_inv, height, width)


    def forward(self, image1, image2, rotation_quat):
        feature1 = self.encoder(image1)
        feature2 = self.encoder(image2)

        flow_estimation = self.netSix(feature1[-1], feature2[-1], None, rotation_quat)
        flow_estimation = self.netFiv(feature1[-2], feature2[-2], flow_estimation, rotation_quat)
        flow_estimation = self.netFou(feature1[-3], feature2[-3], flow_estimation, rotation_quat)
        flow_estimation = self.netThr(feature1[-4], feature2[-4], flow_estimation, rotation_quat)
        flow_estimation = self.netTwo(feature1[-5], feature2[-5], flow_estimation, rotation_quat)

        return (flow_estimation['tenflow'] + self.refiner(flow_estimation['tenFeat'])) * 20.0