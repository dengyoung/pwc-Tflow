import torch
import torch.nn.functional as F

class Correlation(torch.nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()

        self.max_displacement = max_displacement 
        self.kernel_size = 2 * max_displacement + 1 #卷积核大小

    def forward(self, feature1, feature2):
        # b, c, h, w = x.shape
        # return self.corr(x, y).view(b, -1, h, w) / c

        b, c, h, w = feature1.shape
        # 1, 6, 12, 16
        _feature2 = F.unfold(feature2, kernel_size=self.kernel_size, dilation=1, padding=self.kernel_size//2).reshape(b, c, self.kernel_size*self.kernel_size, h, w)
        correlation = torch.sum(feature1[:, :, None, :, :] *_feature2, dim=1)
        return correlation.view(b, -1, h, w) / c


    # def feature_expand(self,input_feature):

    #     B, C, H, W = input_feature.size()
    #     d = self.max_displacement / 2

    #     padded_feature = F.pad(input_feature, (d, d, d, d))
    #     expanded_feature = padded_feature.expand(B, C, H + 2 * d, W + 2 * d)

    #     return expanded_feature
    
    # def correlation_forward(self, feature_1, feature_2):
    #     B, C, H, W = feature_1.size()
    #     max_displacement = self.max_displacement
    #     stride_1 = self.stride_1
    #     stride_2 = self.stride_2
    #     pad = self.pad
    #     kernel_size = self.kernel_size

    #     correlation = torch.zeros(B, max_displacement * max_displacement, H, W).cuda()

    #     feature_2 = self.feature_expand(feature_2)
    #     for y in range(H):
    #         for x in range(W):
    #             feature2_at_xy = feature_2[:, :, y:y + max_displacement, x:x + max_displacement]
    #             feature1_at_xy = feature_1[:, :, y, x].unsqueeze(-1).unsqueeze(-1).expand_as(feature2_at_xy)

    #             correlation_at_xy = torch.sum(feature1_at_xy * feature2_at_xy, dim=1)
    #             correlation[:, :, y, x] = correlation_at_xy

    #     return correlation