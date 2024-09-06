from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from data_utils import datasets

from data_utils import frame_utils
from data_utils import flow_viz
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from data_utils import rotation
from PWCNet import warping


@torch.no_grad()
def validate_tartanair(model, _base_root=None):
    """ Perform evaluation on the TartanAir (test) split """
    model.eval()
    epe_list = []
    results = {}

    val_dataset = datasets.Tartanair(split='validation', root_base=_base_root, test_set=True, read_rotations = False)

    print('Number of validation image pairs: %d' % len(val_dataset))

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        # model.init_bhw(image1.shape[0], image1.shape[-2], image1.shape[-1])

        results_dict = model(image1, image2)

        flow_pr = results_dict[-1]  # [B, 2, H, W]

        assert flow_pr.size()[-2:] == flow_gt.size()[-2:]

        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)

    print("Validation tartanair EPE: %.3f" % (epe))
    results['tartanair_epe'] = epe

    return results

def validate_tartanair_rot(model, _base_root=None, cam_intri=None, cam_intri_inv=None):
    """ Perform evaluation on the TartanAir (test) split """
    model.eval()
    epe_list = []
    results = {}


    val_dataset = datasets.Tartanair(split='validation', root_base=_base_root, test_set=True, read_rotations = True)


    
    print('Number of validation image pairs: %d' % len(val_dataset))

    
    for val_id in range(len(val_dataset)):

        image1, image2, flow_gt, _, rotation_quat = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        flow_gt = flow_gt[None].cuda()
        rotation_quat = rotation_quat[None].cuda()

        height, width = flow_gt.shape[-2], flow_gt.shape[-1]

        # Debug log
        # print(f"Processing batch with images {image1.shape}, {image2.shape}")

        R_Batch = rotation.quaternion_to_matrix(rotation_quat)
        delt_R = warping.get_delt_R(R_Batch)
        R_flow = warping.get_Rflow(delt_R, cam_intri, cam_intri_inv, height, width)
        flow_gt = warping.warpping(flow_gt, -R_flow)


        # model.init_bhw(image1.shape[0], image1.shape[-2], image1.shape[-1])

        results_dict = model(image1, image2, rotation_quat)

        flow_pr = results_dict[-1]

        flow_gt = F.interpolate(flow_gt, size=(flow_pr.shape[-2], flow_pr.shape[-1]), mode='bilinear', align_corners=False) \
                * flow_pr.shape[-2] / flow_gt.shape[-2]
        
        assert flow_pr.size()[-2:] == flow_gt.size()[-2:]

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        
        epe_list.append(epe.cpu().detach().numpy())


    epe_list = np.mean(np.concatenate(epe_list))
    epe = np.mean(epe_list)
    print("Validation tartanair EPE: %.3f" % (epe))
    results['tartanair_rot_epe'] = epe

    return results

def validate_chairs(model, _base_root=None):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    results = {}

    val_dataset = datasets.FlyingChairs(split='validation', root_base=_base_root)

    print('Number of validation image pairs: %d' % len(val_dataset))

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        # model.init_bhw(image1.shape[0], image1.shape[-2], image1.shape[-1])

        results_dict = model(image1, image2)

        flow_pr = results_dict[-1]  # [B, 2, H, W]

        assert flow_pr.size()[-2:] == flow_gt.size()[-2:]

        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)

    print("Validation Chairs EPE: %.3f" % (epe))
    results['chairs_epe'] = epe

    return results


@torch.no_grad()
def validate_things(model,
                    dstype,
                    validate_subset,
                    padding_factor=16,
                    max_val_flow=400,
                    _base_root=None
                    ):
    """ Peform validation using the Things (test) split """
    model.eval()
    results = {}

    val_dataset = datasets.FlyingThings3D(dstype=dstype, test_set=True, validate_subset=validate_subset, 
                                          root_base=_base_root)
    print('Number of validation image pairs: %d' % len(val_dataset))
    epe_list = []

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = frame_utils.InputPadder(image1.shape, padding_factor=padding_factor)
        image1, image2 = padder.pad(image1, image2)

        # model.init_bhw(image1.shape[0], image1.shape[-2], image1.shape[-1])

        results_dict = model(image1, image2)
        flow_pr = results_dict[-1]

        flow = padder.unpad(flow_pr[0]).cpu()

        # Evaluation on flow <= max_val_flow
        flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
        valid_gt = valid_gt * (flow_gt_speed < max_val_flow)
        valid_gt = valid_gt.contiguous()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        val = valid_gt >= 0.5
        epe_list.append(epe[val].cpu().numpy())

    epe_list = np.mean(np.concatenate(epe_list))

    epe = np.mean(epe_list)

    print("Validation Things test set (%s) EPE: %.3f" % (dstype, epe))
    results[dstype + '_epe'] = epe

    return results


@torch.no_grad()
def validate_sintel(model,
                    dstype,
                    padding_factor=16,
                    _base_root=None
                    ):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    val_dataset = datasets.MpiSintel(split='training', dstype=dstype, root_base=_base_root)

    print('Number of validation image pairs: %d' % len(val_dataset))
    epe_list = []

    for val_id in range(len(val_dataset)):
        
        image1, image2, flow_gt, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = frame_utils.InputPadder(image1.shape, padding_factor=padding_factor)
        image1, image2 = padder.pad(image1, image2)

        # model.init_bhw(image1.shape[0], image1.shape[-2], image1.shape[-1])

        results_dict = model(image1, image2)

        # useful when using parallel branches
        flow_pr = results_dict[-1]

        flow = padder.unpad(flow_pr[0]).cpu()
        # flow = flow_pr[0].cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)

    print("Validation Sintel (%s) EPE: %.3f" % (dstype, epe))

    dstype = 'sintel_' + dstype

    results[dstype + '_epe'] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model,
                   padding_factor=16,
                     _base_root=None
                   ):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()

    val_dataset = datasets.KITTI(split='training', root_base=_base_root)
    print('Number of validation image pairs: %d' % len(val_dataset))

    out_list, epe_list = [], []
    results = {}

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = frame_utils.InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
        image1, image2 = padder.pad(image1, image2)

        # model.init_bhw(image1.shape[0], image1.shape[-2], image1.shape[-1])

        results_dict = model(image1, image2)

        # useful when using parallel branches
        flow_pr = results_dict[-1]

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()

        epe_list.append(epe[val].cpu().numpy())

        out_list.append(out[val].cpu().numpy())

    epe_list = np.concatenate(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI EPE: %.3f" % (epe))
    print("Validation KITTI F1: %.3f" % (f1))
    results['kitti_epe'] = epe
    results['kitti_f1'] = f1

    return results

@torch.no_grad()
def create_kitti_submission(model, output_path='datasets/kitti_submission/flow', padding_factor=16, save_vis_flow=False):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = frame_utils.InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
        image1_pad, image2_pad = padder.pad(image1[None].cuda(), image2[None].cuda())

        # model.init_bhw(image1_pad.shape[0], image1_pad.shape[-2], image1_pad.shape[-1])

        results_dict = model(image1_pad, image2_pad)

        flow_pr = results_dict[-1]

        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)

        if save_vis_flow:
            vis_flow_file = output_filename
            flow_viz.save_vis_flow_tofile(flow, vis_flow_file)
        else:
            frame_utils.writeFlowKITTI(output_filename, flow)
