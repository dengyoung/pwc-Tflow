import torch
import torch.nn.functional as F

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):

        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        
        return x3
    

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = ConvBlock(3, 16, 3, 2, 1)

        self.conv2 = ConvBlock(16, 32, 3, 2, 1)

        self.conv3 = ConvBlock(32, 64, 3, 2, 1)

        self.conv4 = ConvBlock(64, 96, 3, 2, 1)

        self.conv5 = ConvBlock(96, 128, 3, 2, 1)

        self.conv6 = ConvBlock(128, 196, 3, 2, 1)
    
    def forward(self, image):
        x1 = self.conv1(image)

        x2 = self.conv2(x1)

        x3 = self.conv3(x2)

        x4 = self.conv4(x3)

        x5 = self.conv5(x4)
        
        x6 = self.conv6(x5)
        
        return [ x1, x2, x3, x4, x5, x6 ]