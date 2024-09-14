import torch
import torch.nn.functional as F

def get_delt_R(R_Batch):
    """
    计算旋转矩阵的相对变换矩阵。

    参数:
    R_Batch (torch.Tensor): 形状为 (B, 2, 3, 3) 的张量，其中 B 是批次大小，2 表示前后两个旋转矩阵对。

    返回:
    torch.Tensor: 形状为 (B, 3, 3) 的张量，表示每一对旋转矩阵的相对旋转矩阵。
    """
    # batch_size = R_Batch.size(0)
    # delt_R = torch.zeros((batch_size, 3, 3), dtype=torch.float32, device=R_Batch.device)

    # for i in range(batch_size):
    #     R1 = R_Batch[i, 0]
    #     R2 = R_Batch[i, 1]

    #     # delt_R[i] = R2 @ R1.inverse()
    #     delt_R[i] = R2 @ torch.transpose(R1, 0, 1)
    # delt_R = delt_R.to(R_Batch.device)
    B = R_Batch.size(0)
    R_cam2body = torch.tensor([[0, 0, 1], 
                               [1, 0, 0], 
                               [0, 1, 0]], dtype=torch.float32).to(R_Batch.device)
    R_cam2body = R_cam2body.repeat(B, 1, 1)

    R1 = R_Batch[:, 0] @ R_cam2body  # 形状 (B, 3, 3)
    R2 = R_Batch[:, 1] @ R_cam2body # 形状 (B, 3, 3)

    delt_R = torch.transpose(R2,1,2) @ R1
    delt_R = delt_R.to(R_Batch.device)
    
    return delt_R


def get_Rflow(_delt_R, cam_intri, cam_intri_inv, height, width):

    delt_R = _delt_R

    B = delt_R.size(0)
    cam_intri = cam_intri.repeat(B, 1, 1).to(_delt_R.device)
    cam_intri_inv = cam_intri_inv.repeat(B, 1, 1).to(_delt_R.device)

    H = cam_intri @ delt_R @ cam_intri_inv

    pix_v, pix_u = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    pix_v = pix_v.repeat(B, 1, 1).to(_delt_R.device)
    pix_u = pix_u.repeat(B, 1, 1).to(_delt_R.device)

    ones_h = torch.ones((height, width)).repeat(B, 1, 1).to(_delt_R.device) # [B, H, W]

    pix_homog = torch.stack([pix_u, pix_v, ones_h], dim=1).float() # [B, 3, H, W]

    pix_homog = pix_homog.view(B, 3, -1) # [B, 3, H*W]

    pix_homog_cur = H @ pix_homog # [B, 3, H*W]

    pix_homog_cur = pix_homog_cur.view(B, 3, height, width) # [B, 3, H, W]

    pix_u_cur = pix_homog_cur[:, 0] / pix_homog_cur[:, 2] # [B, H, W]
    pix_v_cur = pix_homog_cur[:, 1] / pix_homog_cur[:, 2] # [B, H, W]

    r_flow = torch.zeros((2, height, width), dtype=torch.float32).repeat(B, 1, 1, 1).to(_delt_R.device)

    r_flow[:,0] = torch.where(abs(pix_u_cur - pix_u) < 640, pix_u_cur - pix_u, 1e-5)
    r_flow[:,1] = torch.where(abs(pix_v_cur - pix_v) < 640, pix_v_cur - pix_v, 1e-5)

    # # pix_z = H[2][0] * pix_u + H[2][1] * pix_v + H[2][2] * ones_h + 1e-5
    # pix_z = H[:,2,0].unsqueeze(1).unsqueeze(2) * pix_u + H[:,2,1].unsqueeze(1).unsqueeze(2) * pix_v + H[:,2,2].unsqueeze(1).unsqueeze(2) * ones_h + 1e-5

    # # pix_u_cur = (H[0][0] * pix_u + H[0][1] * pix_v + H[0][2]) / pix_z * 1.0
    # # pix_v_cur = (H[1][0] * pix_u + H[1][1] * pix_v + H[1][2]) / pix_z * 1.0
    # pix_u_cur = (H[:,0,0].unsqueeze(1).unsqueeze(2) * pix_u + H[:,0,1].unsqueeze(1).unsqueeze(2) * pix_v + \
    #              H[:,0,2].unsqueeze(1).unsqueeze(2) * ones_h) / pix_z * 1.0
    # pix_v_cur = (H[:,1,0].unsqueeze(1).unsqueeze(2) * pix_u + H[:,1,1].unsqueeze(1).unsqueeze(2) * pix_v + \
    #              H[:,1,2].unsqueeze(1).unsqueeze(2) * ones_h) / pix_z * 1.0


    # r_flow = torch.zeros((2, height, width), dtype=torch.float32).repeat(B, 1, 1, 1).to(_delt_R.device)

    # r_flow[:,0] = torch.where(abs(pix_u_cur - pix_u) < 640, pix_u_cur - pix_u, 1e-5)
    # r_flow[:,1] = torch.where(abs(pix_v_cur - pix_v) < 640, pix_v_cur - pix_v, 1e-5)

    return r_flow


def warpping(input, flow: torch.Tensor):
    B, C, H, W = input.size()
    ys, xs = torch.meshgrid(torch.arange(H,device=flow.device), torch.arange(W,device=flow.device), indexing='ij')  # [H, W]
    
    stacks = [xs, ys]
    grid = torch.stack(stacks, dim=0).float()  # [2, H, W]
    grid = grid[None].repeat(B, 1, 1, 1)  # [B, 2, H, W]

    grid = grid + flow
    
    x_grid = 2 * grid[:, 0] / (W - 1) - 1
    y_grid = 2 * grid[:, 1] / (H - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    input = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return input