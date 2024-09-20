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
    
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=torch.nn.InstanceNorm2d, stride = 1, dilation = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                     dilation=dilation, stride=stride, padding=dilation, bias=False)
        

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                     dilation=dilation, stride=1, padding=dilation, bias=False)

        self.relu = torch.nn.ReLU(inplace=True)

        self.norm1 = norm_layer(out_channels)
        self.norm2 = norm_layer(out_channels)

        if not stride == 1 or in_channels != out_channels:
            self.norm3 = norm_layer(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.downsample = None
        else:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x

        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)
        
        return self.relu(x + y) 

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1/2
        self.norm1 = torch.nn.InstanceNorm2d(64)
        self.relu1 = torch.nn.ReLU(inplace=True)

        self.res1_1 = ResidualBlock(in_channels=64, out_channels=64, stride=1, norm_layer=torch.nn.InstanceNorm2d, dilation=1) # 1/2
        self.res1_2 = ResidualBlock(in_channels=64, out_channels=64, stride=1, norm_layer=torch.nn.InstanceNorm2d, dilation=1) # 1/2

        self.res2_1 = ResidualBlock(in_channels=64, out_channels=96, stride=2, norm_layer=torch.nn.InstanceNorm2d, dilation=1) # 1/4
        self.res2_2 = ResidualBlock(in_channels=96, out_channels=96, stride=1, norm_layer=torch.nn.InstanceNorm2d, dilation=1) # 1/4

        self.res3_1 = ResidualBlock(in_channels=96, out_channels=128, stride=2, norm_layer=torch.nn.InstanceNorm2d, dilation=1) # 1/8
        self.res3_2 = ResidualBlock(in_channels=128, out_channels=128, stride=1, norm_layer=torch.nn.InstanceNorm2d, dilation=1) # 1/8


        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0) # 1/8



    
    def forward(self, image):
        
        x1 = self.conv1(image)
        x1 = self.norm1(x1)
        x1 = self.relu1(x1)

        x2 = self.res1_1(x1)
        x2 = self.res1_2(x2)

        x3 = self.res2_1(x2)
        x3 = self.res2_2(x3)

        x4 = self.res3_1(x3)
        x4 = self.res3_2(x4)

        x5 = self.conv2(x4)

        
        return [ x2, x3, x4, x5]
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder().to(device)
    image1 = torch.randn(3, 3, 480, 640).to(device)
    import time
    for i in range(10):
        t0 = time.time()
        output = model(image1)
        print(time.time() - t0)
        print(output[0].shape)
        print(output[4].shape)