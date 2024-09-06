import torch
from data_utils import rotation
import torch.nn.functional as F
from PWCNet import warping

def flow_loss_func(flow_preds, flow_gt, valid, max_flow=400, rotation_quat=None, cam_intri=None, cam_intri_inv=None):

    flow_loss = 0.0
    height, width = flow_gt.shape[-2], flow_gt.shape[-1]

    if rotation_quat is not None:
        # flow_gt = rotation(flow_gt, rotation_quat)
        # rotation_R = rotation.quaternion_to_matrix(rotation_quat)
        R_Batch = rotation.quaternion_to_matrix(rotation_quat)
        delt_R = warping.get_delt_R(R_Batch)
        R_flow = warping.get_Rflow(delt_R, cam_intri, cam_intri_inv, height, width)
        flow_gt = warping.warpping(flow_gt, -R_flow)
    
    flow_gt = F.interpolate(flow_gt, size=(flow_preds.shape[-2], flow_preds.shape[-1]), mode='bilinear', align_corners=False) \
                * flow_preds.shape[-2] / flow_gt.shape[-2]
    
    valid = F.interpolate(valid[:,None,:], size=(flow_preds.shape[-2], flow_preds.shape[-1]), mode='bilinear', align_corners=False).squeeze(1)

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)

    weight =  1.0

    i_loss = (flow_preds - flow_gt).abs()
    flow_loss += weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        'mag': mag.mean().item()
    }

    return flow_loss, metrics, flow_gt