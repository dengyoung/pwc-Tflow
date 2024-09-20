import numpy as np
import torch

import sys
sys.path.append('/home/zhangyuang/huyu/pwc-Tflow/')
from PWCNet import encoder
# from PWCNet import decoder
from PWCNet import decoder_tflow
from PWCNet import refiner
from PWCNet import utils
import torch.nn.functional as F



class PWCNet_tflow(torch.nn.Module):
    def __init__(self):
        super(PWCNet_tflow, self).__init__()

        self.encoder = encoder.Encoder()
        self.netOne = decoder_tflow.Decoder(1)
        self.netTwo = decoder_tflow.Decoder(2)
        self.netThr = decoder_tflow.Decoder(3)
        self.netFou = decoder_tflow.Decoder(4)
        self.refiner = refiner.Refiner()

        self.upsampler = torch.nn.Sequential(torch.nn.Conv2d(2 + 439, 256, 3, 1, 1),
                                       torch.nn.ReLU(inplace=True),
                                       torch.nn.Conv2d(256, 2 ** 2 * 9, 1, 1, 0))
        
    def init_cam_intri(self, cam_intri, cam_intri_inv, height, width):

        self.netOne.init_cam_intri(cam_intri, cam_intri_inv, height, width)
        self.netTwo.init_cam_intri(cam_intri, cam_intri_inv, height, width)
        self.netThr.init_cam_intri(cam_intri, cam_intri_inv, height, width)
        self.netFou.init_cam_intri(cam_intri, cam_intri_inv, height, width)

    def upsample_flow(self, flow, feature, upsample_factor=2,
                      ):
        # convex upsampling
        concat = torch.cat((flow, feature), dim=1)

        mask = self.upsampler(concat)
        b, flow_channel, h, w = flow.shape
        mask = mask.view(b, 1, 9, upsample_factor, upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(upsample_factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

        up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
        up_flow = up_flow.reshape(b, flow_channel, upsample_factor * h,
                                    upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    def forward(self, image1, image2, rotation_quat):

        image1, image2 = utils.normalize_img(image1, image2)
        
        feature1 = self.encoder(image1)
        feature2 = self.encoder(image2)

        flow_estimation_4 = self.netFou(feature1[-1], feature2[-1], None, rotation_quat)
        flow_estimation_3 = self.netThr(feature1[-2], feature2[-2], flow_estimation_4, rotation_quat)
        flow_estimation_2 = self.netTwo(feature1[-3], feature2[-3], flow_estimation_3, rotation_quat)
        flow_estimation_1 = self.netOne(feature1[-4], feature2[-4], flow_estimation_2, rotation_quat)

        flow_estimation_1_refine = (flow_estimation_1['tenFlow'] + self.refiner(flow_estimation_1['tenFeat']))

        
        return flow_estimation_4['tenFlow'], flow_estimation_3['tenFlow'], \
            flow_estimation_2['tenFlow'] , flow_estimation_1_refine,\
            self.upsample_flow(flow_estimation_1_refine, flow_estimation_1['tenFeat'], upsample_factor=2) 

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    device = torch.device('cuda')
    torch.cuda.set_device(3)
    model = PWCNet_tflow().to(device)
    model.init_cam_intri(torch.randn(3,3), torch.randn(3,3), 480, 640)
    image1 = torch.randn(1, 3, 480, 640).to(device)
    image2 = torch.randn(1, 3, 480, 640).to(device)
    rotation_quat = torch.randn(1,2,4).to(device)
    import time
    for i in range(10):
        t0 = time.time()
        output = model(image1, image2, rotation_quat)
        print(output[0].shape)
        print(output[1].shape)
        print(output[2].shape)
        print(output[3].shape)
        print(output[4].shape)
        print("time:", time.time()-t0)
   