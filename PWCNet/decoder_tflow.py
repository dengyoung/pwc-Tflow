import torch
import torch.nn.functional as F
import sys
sys.path.append('/home/zhangyuang/huyu/pwc-Tflow/')
from PWCNet import correlation
from data_utils import rotation

from PWCNet import warping

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        return x1

class Decoder(torch.nn.Module):
    def __init__(self, intLevel):
        super().__init__()
        #intchannel: correlation + pyramid-feature + tenFlow + tenFeat + R_flow
        #intchannel-1: correlation + R_flow
        intPrevious = [ None, None, 81 + 32 + 2 + 2 + 2, 81 + 64 + 2 + 2 + 2, 81 + 96 + 2 + 2 + 2, 81 + 128 + 2 + 2 + 2, 81 + 2, None ][intLevel + 1]
        intCurrent = [ None, None, 81 + 32 + 2 + 2 + 2, 81 + 64 + 2 + 2 + 2, 81 + 96 + 2 + 2 + 2, 81 + 128 + 2 + 2 + 2, 81 + 2, None ][intLevel + 0]
        
        intCurrent = [81+2+64+2+2, 81+2+64+2+2, 81+2+96+2+2, 81+2+128+2+2, 81+2+128][intLevel]
        intPrevious = [81+2+64+2+2, 81+2+64+2+2, 81+2+96+2+2, 81+2+128+2+2, 81+2+128, None][intLevel + 1]
        self.intResize = [2, 2, 4, 8, 8][intLevel]


        if intLevel < 4: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 4: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 96 + 64, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 4: self.fltBackwarp = [1, 2, 2, 1][intLevel]


        self.netOne = ConvBlock(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.netTwo = ConvBlock(in_channels=intCurrent + 128, out_channels=96, kernel_size=3, stride=1, padding=1)

        self.netThr = ConvBlock(in_channels=intCurrent + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.netFou = ConvBlock(in_channels=intCurrent + 128 +  96 + 64, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.cor =  correlation.Correlation(4)


    def init_cam_intri(self, cam_intri, cam_intri_inv, height, width):

        self.cam_intri = cam_intri
        self.cam_intri_inv = cam_intri_inv
        self.height = height
        self.width = width
    

    def forward(self, tenOne, tenTwo, flowEst_Prev, rotation_quat):
        tenFlow = None
        tenFeat = None

        if flowEst_Prev is None:
            tenFlow = None
            tenFeat = None

            # tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)

            R_Batch = rotation.quaternion_to_matrix(rotation_quat)
            delt_R = warping.get_delt_R(R_Batch)

            R_flow = warping.get_Rflow(delt_R, self.cam_intri, self.cam_intri_inv, self.height, self.width)
            resize_H = (self.height + self.intResize -1) // self.intResize
            resize_W = (self.width + self.intResize -1)  // self.intResize
            R_flow_resized = F.interpolate(R_flow, size=(resize_H, resize_W), mode='bilinear', align_corners=False) / self.intResize

            tenTwo = warping.warpping(input=tenTwo, flow=R_flow_resized)

            tenVolume = torch.nn.functional.leaky_relu(input = self.cor(tenOne, tenTwo), negative_slope=0.1, inplace=False)

            tenFeat = torch.cat([ tenVolume, R_flow_resized, tenOne], 1)

        elif flowEst_Prev is not None:
            tenFlow = self.netUpflow(flowEst_Prev['tenFlow'])
            tenFeat = self.netUpfeat(flowEst_Prev['tenFeat'])

            tenFlow = F.interpolate(tenFlow, size=(tenTwo.shape[-2], tenTwo.shape[-1]), mode='bilinear', align_corners=False)
            tenFeat = F.interpolate(tenFeat, size=(tenTwo.shape[-2], tenTwo.shape[-1]), mode='bilinear', align_corners=False)

            R_Batch = rotation.quaternion_to_matrix(rotation_quat)
            delt_R = warping.get_delt_R(R_Batch)

            # R_flow = self.R_flow(delt_R, tenTwo.device)
            R_flow = warping.get_Rflow(delt_R, self.cam_intri, self.cam_intri_inv, self.height, self.width)
            resize_H = (self.height + self.intResize -1) // self.intResize
            resize_W = (self.width + self.intResize -1)  // self.intResize
            R_flow_resized = F.interpolate(R_flow, size=(resize_H, resize_W), mode='bilinear', align_corners=False) / self.intResize

            tenTwo = warping.warpping(tenTwo, R_flow_resized)
            tenVolume = torch.nn.functional.leaky_relu(input=self.cor(feature1=tenOne,feature2=warping.warpping(input=tenTwo,flow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

            # tenFeat = torch.cat([ tenVolume, R_flow_pool, tenOne, tenFlow, tenFeat ], 1)
            tenFeat = torch.cat([ tenVolume, R_flow_resized, tenOne, tenFlow, tenFeat ], 1)

        # end
    


        tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)

        tenFlow = self.netFou(tenFeat)

        return {
            'tenFlow': tenFlow,
            'tenFeat': tenFeat
        }
    # end
# end

from PWCNet import encoder
import time
if __name__ == '__main__':
    device = torch.device('cuda')
    torch.cuda.set_device(3)
    encoder = encoder.Encoder().to(device)
    image1 = torch.randn(1, 3, 480, 640).to(device)
    image2 = torch.randn(1, 3, 480, 640).to(device)

    feature1 = encoder(image1)
    feature2 = encoder(image2)

    decoder_1 = Decoder(1).to(device)
    decoder_2 = Decoder(2).to(device)
    decoder_3 = Decoder(3).to(device)
    decoder_4 = Decoder(4).to(device)

    decoder_1.init_cam_intri(torch.randn(3, 3), torch.randn(3, 3), 480, 640)
    decoder_2.init_cam_intri(torch.randn(3, 3), torch.randn(3, 3), 480, 640)
    decoder_3.init_cam_intri(torch.randn(3, 3), torch.randn(3, 3), 480, 640)
    decoder_4.init_cam_intri(torch.randn(3, 3), torch.randn(3, 3), 480, 640)
    for i in range(10):
        time0 = time.time()
        output_4 = decoder_4(feature1[-1], feature2[-1], None, torch.randn(1, 2, 4).to(device))
        output_3 = decoder_3(feature1[-2], feature2[-2], output_4, torch.randn(1, 2, 4).to(device))
        output_2 = decoder_2(feature1[-3], feature2[-3], output_3, torch.randn(1, 2, 4).to(device))
        output_1 = decoder_1(feature1[-4], feature2[-4], output_2, torch.randn(1, 2, 4).to(device))
        print(time.time() - time0)
        

    print(output_1['tenFlow'].shape)
    print(output_2['tenFlow'].shape)
    print(output_3['tenFlow'].shape)
    print(output_4['tenFlow'].shape)

    print(output_1['tenFeat'].shape)
    print(output_2['tenFeat'].shape)
    print(output_3['tenFeat'].shape)
    print(output_4['tenFeat'].shape)
