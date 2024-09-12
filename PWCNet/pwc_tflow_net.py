import numpy as np
import torch

import sys
sys.path.append('/home/zhangyuang/huyu/pwc-Tflow/')
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

        flow_estimation_6 = self.netSix(feature1[-1], feature2[-1], None, rotation_quat)
        flow_estimation_5 = self.netFiv(feature1[-2], feature2[-2], flow_estimation_6, rotation_quat)
        flow_estimation_4 = self.netFou(feature1[-3], feature2[-3], flow_estimation_5, rotation_quat)
        flow_estimation_3 = self.netThr(feature1[-4], feature2[-4], flow_estimation_4, rotation_quat)
        flow_estimation_2 = self.netTwo(feature1[-5], feature2[-5], flow_estimation_3, rotation_quat)

        # return (flow_estimation['tenFlow'] + self.refiner(flow_estimation['tenFeat'])) * 20.0
        # return (flow_estimation_2['tenFlow'] + self.refiner(flow_estimation_2['tenFeat'])), flow_estimation_3['tenFlow'], \
        #         flow_estimation_4['tenFlow'], flow_estimation_5['tenFlow'], flow_estimation_6['tenFlow']
        return flow_estimation_6['tenFlow'], flow_estimation_5['tenFlow'], flow_estimation_4['tenFlow'], \
                flow_estimation_3['tenFlow'], (flow_estimation_2['tenFlow'] + self.refiner(flow_estimation_2['tenFeat']))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model = PWCNet_tflow().to(device)
    model.init_cam_intri(torch.randn(3,3), torch.randn(3,3), 480, 640)
    image1 = torch.randn(3, 3, 480, 640).to(device)
    image2 = torch.randn(3, 3, 480, 640).to(device)
    rotation_quat = torch.randn(3,2,4).to(device)
    import time
    for i in range(10):
        t0 = time.time()
        output = model(image1, image2, rotation_quat)
        print(output[0].shape)
        print(output[0].shape)
        print(output[1].shape)
        print(output[2].shape)
        print(output[3].shape)
        print(output[4].shape)
        print("time:", time.time()-t0)
   