import torch
import torch.nn.functional as F

from PWCNet import correlation

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

        intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
        intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

        if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]


        self.netOne = ConvBlock(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.netTwo = ConvBlock(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.netThr = ConvBlock(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1)

        self.netFou = ConvBlock(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.netFiv = ConvBlock(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.netSix = ConvBlock(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.cor =  correlation.Correlation(4)

    # end

    def flow_warp(self, feature, flow: torch.Tensor):
        B, C, H, W = feature.size()
        ys, xs = torch.meshgrid(torch.arange(H,device=flow.device), torch.arange(W,device=flow.device), indexing='ij')  # [H, W]
        
        stacks = [xs, ys]
        grid = torch.stack(stacks, dim=0).float()  # [2, H, W]
        grid = grid[None].repeat(B, 1, 1, 1)  # [B, 2, H, W]

        grid = grid + flow
        
        x_grid = 2 * grid[:, 0] / (W - 1) - 1
        y_grid = 2 * grid[:, 1] / (H - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

        feature = F.grid_sample(feature, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return feature


    def forward(self, tenOne, tenTwo, flowEst_Prev):
        tenFlow = None
        tenFeat = None

        if flowEst_Prev is None:
            tenFlow = None
            tenFeat = None

            # tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)

            tenVolume = torch.nn.functional.leaky_relu(input = self.cor(tenOne, tenTwo), negative_slope=0.1, inplace=False)

            tenFeat = torch.cat([ tenVolume ], 1)

        elif flowEst_Prev is not None:
            tenFlow = self.netUpflow(flowEst_Prev['tenFlow'])
            tenFeat = self.netUpfeat(flowEst_Prev['tenFeat'])

            tenVolume = torch.nn.functional.leaky_relu(input=self.cor(feature1=tenOne,feature=self.flow_warp(feature2=tenTwo,flow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

            tenFeat = torch.cat([ tenVolume, tenOne, tenFlow, tenFeat ], 1)

        # end

        tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

        tenFlow = self.netSix(tenFeat)

        return {
            'tenFlow': tenFlow,
            'tenFeat': tenFeat
        }
    # end
# end