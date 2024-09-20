import torch
from data_utils import rotation
import torch.nn.functional as F
from PWCNet import warping

def flow_loss_func(flow_preds, flow_gt, valid, max_flow=400, rotation_quat=None, cam_intri=None, cam_intri_inv=None):

    height, width = flow_gt.shape[-2], flow_gt.shape[-1]

    if rotation_quat is not None:
        # flow_gt = rotation(flow_gt, rotation_quat)
        # rotation_R = rotation.quaternion_to_matrix(rotation_quat)
        R_Batch = rotation.quaternion_to_matrix(rotation_quat)
        delt_R = warping.get_delt_R(R_Batch)
        R_flow = warping.get_Rflow(delt_R, cam_intri, cam_intri_inv, height, width)
        # flow_gt = warping.warpping(flow_gt, -R_flow)
        flow_gt = flow_gt - R_flow
    
    # flow5, flow4, flow3, flow2, flow1 = flow_preds
    weights = [0.30, 0.15, 0.08, 0.08, 0.08]

    # weights = [0.0025, 0.005, 0.01, 0.02, 0.08, 0.32]
    epe_all = 0.0
    flow_loss = 0.0
    for flow_pred in flow_preds:

        flow_gt_resize = F.interpolate(flow_gt, size=(flow_pred.shape[-2], flow_pred.shape[-1]), mode='bilinear', align_corners=False) \
                * flow_pred.shape[-2] / flow_gt.shape[-2]
        
        valid_resize = F.interpolate(valid[:,None], size=(flow_pred.shape[-2], flow_pred.shape[-1]), mode='bilinear', align_corners=False).squeeze(1)

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt_resize ** 2, dim=1).sqrt()

        valid_mask = (valid_resize >= 0.5) & (mag < max_flow)

        weight =  weights.pop(0)

        i_loss = (flow_pred - flow_gt_resize).abs()
        flow_loss += weight * (valid_mask[:, None] * i_loss).mean()

        epe = torch.sum(i_loss ** 2, dim=1).sqrt()

        epe_all += epe.view(-1)[valid_mask.view(-1)].mean().item()
    
    if valid.max() < 0.5:
        pass

    metrics = {
        'epe': epe_all if epe_all > 0 else float('nan'),
        'mag': mag.mean().item()
    }

    return flow_loss, metrics, flow_gt_resize