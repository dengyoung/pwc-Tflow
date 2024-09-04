import torch
import torch.nn.functional as F

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

        if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]


        self.netOne = ConvBlock(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.netTwo = ConvBlock(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.netThr = ConvBlock(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1)

        self.netFou = ConvBlock(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.netFiv = ConvBlock(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.netSix = ConvBlock(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.cor =  correlation(4)

        self.intLevel = intLevel

    def init_cam_intri(self, cam_intri, cam_intri_inv, height, width):

        self.cam_intri = cam_intri
        self.cam_intri_inv = cam_intri_inv
        self.height = height
        self.width = width
    
    # def R_flow(self, _delt_R, device):

    #     delt_R = _delt_R

    #     B = delt_R.size(0)
    #     self.cam_intri = self.cam_intri.repeat(B, 1, 1)
    #     self.cam_intri_inv = self.cam_intri_inv.repeat(B, 1, 1)

    #     H = self.cam_intri @ delt_R @ self.cam_intri_inv

    #     pix_v, pix_u = torch.meshgrid(torch.arange(self.height), torch.arange(self.width), indexing='ij')
    #     pix_v = pix_v.repeat(B, 1, 1).to(device)
    #     pix_u = pix_u.repeat(B, 1, 1).to(device)

    #     ones_h = torch.ones((self.height, self.width)).repeat(B, 1, 1).to(device)

    #     # pix_z = H[2][0] * pix_u + H[2][1] * pix_v + H[2][2] * ones_h + 1e-5
    #     pix_z = H[:,2,0].unsqueeze(1).unsqueeze(2) * pix_u + H[:,2,1].unsqueeze(1).unsqueeze(2) * pix_v + H[:,2,2].unsqueeze(1).unsqueeze(2) * ones_h + 1e-5

    #     # pix_u_cur = (H[0][0] * pix_u + H[0][1] * pix_v + H[0][2]) / pix_z * 1.0
    #     # pix_v_cur = (H[1][0] * pix_u + H[1][1] * pix_v + H[1][2]) / pix_z * 1.0
    #     pix_u_cur = (H[:,0,0].unsqueeze(1).unsqueeze(2) * pix_u + H[:,0,1].unsqueeze(1).unsqueeze(2) * pix_v + H[:,0,2].unsqueeze(1).unsqueeze(2)) / pix_z * 1.0
    #     pix_v_cur = (H[:,1,0].unsqueeze(1).unsqueeze(2) * pix_u + H[:,1,1].unsqueeze(1).unsqueeze(2) * pix_v + H[:,1,2].unsqueeze(1).unsqueeze(2)) / pix_z * 1.0


    #     r_flow = torch.zeros((2, self.height, self.width), dtype=torch.float32).repeat(B, 1, 1, 1).to(device)

    #     r_flow[:,0] = torch.where(abs(pix_u_cur - pix_u) < 400, pix_u_cur - pix_u, 1e-5)
    #     r_flow[:,1] = torch.where(abs(pix_v_cur - pix_v) < 400, pix_v_cur - pix_v, 1e-5)

    #     return r_flow


    # def flow_warp(self, feature, flow: torch.Tensor):
    #     B, C, H, W = feature.size()
    #     ys, xs = torch.meshgrid(torch.arange(H,device=flow.device), torch.arange(W,device=flow.device), indexing='ij')  # [H, W]
        
    #     stacks = [xs, ys]
    #     grid = torch.stack(stacks, dim=0).float()  # [2, H, W]
    #     grid = grid[None].repeat(B, 1, 1, 1)  # [B, 2, H, W]

    #     grid = grid + flow
        
    #     x_grid = 2 * grid[:, 0] / (W - 1) - 1
    #     y_grid = 2 * grid[:, 1] / (H - 1) - 1

    #     grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    #     feature = F.grid_sample(feature, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    #     return feature

    def forward(self, tenOne, tenTwo, flowEst_Prev, rotation_quat):
        tenFlow = None
        tenFeat = None

        if flowEst_Prev is None:
            tenFlow = None
            tenFeat = None

            # tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)

            delt_R = rotation.quaternion_to_matrix(rotation_quat)

            # R_flow = self.R_flow(delt_R, tenTwo.device)
            R_flow = warping.get_Rflow(delt_R, self.cam_intri, self.cam_intri_inv, self.height, self.width, tenTwo.device)
            resize_H = self.height // 2**self.intLevel
            resize_W = self.width // 2**self.intLevel
            R_flow_resized = F.interpolate(R_flow, size=(resize_H, resize_W), mode='bilinear', align_corners=False) / 2**self.intLevel
            # R_flow_pool = F.avg_pool2d(R_flow, 2**self.intLevel, 2**self.intLevel)

            # tenTwo = self.flow_warp(tenTwo, R_flow_pool)
            # tenTwo = self.flow_warp(tenTwo, R_flow_resized)
            tenTwo = warping.warpping(tenTwo, R_flow_resized)

            tenVolume = torch.index_select(self.cor(tenOne, tenTwo), dim=1, index=self.index.to(tenOne).long())

            tenFeat = torch.cat([ tenVolume, R_flow_resized ], 1)

        elif flowEst_Prev is not None:
            tenFlow = self.netUpflow(flowEst_Prev['tenFlow'])
            tenFeat = self.netUpfeat(flowEst_Prev['tenFeat'])

            # R_flow = self.R_flow(delt_R, tenTwo.device)
            R_flow = warping.get_Rflow(delt_R, self.cam_intri, self.cam_intri_inv, self.height, self.width, tenTwo.device)
            resize_H = self.height // 2**self.intLevel
            resize_W = self.width // 2**self.intLevel
            R_flow_resized = F.interpolate(R_flow, size=(resize_H, resize_W), mode='bilinear', align_corners=False) / 2**self.intLevel
            # R_flow_pool = F.avg_pool2d(R_flow, 2**self.intLevel, 2**self.intLevel)

            # tenTwo = self.flow_warp(tenTwo, R_flow_pool)
            tenTwo = warping.warpping(tenTwo, R_flow_resized)
            tenVolume = torch.nn.functional.leaky_relu(input=self.cor(feature=tenOne,feature=self.flow_warp(feature=tenTwo,flow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

            # tenFeat = torch.cat([ tenVolume, R_flow_pool, tenOne, tenFlow, tenFeat ], 1)
            tenFeat = torch.cat([ tenVolume, R_flow_resized, tenOne, tenFlow, tenFeat ], 1)

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