import torch
import torch.nn.functional as F


def get_Rflow(_delt_R, cam_intri, cam_intri_inv, height, width, device):

    delt_R = _delt_R

    B = delt_R.size(0)
    cam_intri = cam_intri.repeat(B, 1, 1)
    cam_intri_inv = cam_intri_inv.repeat(B, 1, 1)

    H = cam_intri @ delt_R @ cam_intri_inv

    pix_v, pix_u = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    pix_v = pix_v.repeat(B, 1, 1).to(device)
    pix_u = pix_u.repeat(B, 1, 1).to(device)

    ones_h = torch.ones((height, width)).repeat(B, 1, 1).to(device)

    # pix_z = H[2][0] * pix_u + H[2][1] * pix_v + H[2][2] * ones_h + 1e-5
    pix_z = H[:,2,0].unsqueeze(1).unsqueeze(2) * pix_u + H[:,2,1].unsqueeze(1).unsqueeze(2) * pix_v + H[:,2,2].unsqueeze(1).unsqueeze(2) * ones_h + 1e-5

    # pix_u_cur = (H[0][0] * pix_u + H[0][1] * pix_v + H[0][2]) / pix_z * 1.0
    # pix_v_cur = (H[1][0] * pix_u + H[1][1] * pix_v + H[1][2]) / pix_z * 1.0
    pix_u_cur = (H[:,0,0].unsqueeze(1).unsqueeze(2) * pix_u + H[:,0,1].unsqueeze(1).unsqueeze(2) * pix_v + H[:,0,2].unsqueeze(1).unsqueeze(2)) / pix_z * 1.0
    pix_v_cur = (H[:,1,0].unsqueeze(1).unsqueeze(2) * pix_u + H[:,1,1].unsqueeze(1).unsqueeze(2) * pix_v + H[:,1,2].unsqueeze(1).unsqueeze(2)) / pix_z * 1.0


    r_flow = torch.zeros((2, height, width), dtype=torch.float32).repeat(B, 1, 1, 1).to(device)

    r_flow[:,0] = torch.where(abs(pix_u_cur - pix_u) < 400, pix_u_cur - pix_u, 1e-5)
    r_flow[:,1] = torch.where(abs(pix_v_cur - pix_v) < 400, pix_v_cur - pix_v, 1e-5)

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