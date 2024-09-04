import torch

from PWCNet import encoder
from PWCNet import decoder
from PWCNet import refiner

class PWCNet(torch.nn.Module):
    def __init__(self):
        super(PWCNet, self).__init__()

        self.encoder = encoder.Encoder()
        self.netTwo = decoder.Decoder(2)
        self.netThr = decoder.Decoder(3)
        self.netFou = decoder.Decoder(4)
        self.netFiv = decoder.Decoder(5)
        self.netSix = decoder.Decoder(6)
        self.refiner = refiner.Refiner()


    def forward(self, image1, image2):
        feature1 = self.encoder(image1)
        feature2 = self.encoder(image2)

        flow_estimation = self.netSix(feature1[-1], feature2[-1],None)
        flow_estimation = self.netFiv(feature1[-2], feature2[-2],flow_estimation)
        flow_estimation = self.netFou(feature1[-3], feature2[-3],flow_estimation)
        flow_estimation = self.netThr(feature1[-4], feature2[-4],flow_estimation)
        flow_estimation = self.netTwo(feature1[-5], feature2[-5],flow_estimation)

        return (flow_estimation['tenflow'] + self.refiner(flow_estimation['tenFeat'])) * 20.0

