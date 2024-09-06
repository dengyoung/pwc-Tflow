import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data.dataloader as dataloader
import torch.distributed as dist

from tqdm import tqdm
from tensorboardX import SummaryWriter
from PIL import Image

import torch
from torch.optim import AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2

from PWCNet.pwc_net import PWCNet
from PWCNet.pwc_tflow_net import PWCNet_tflow

from data_utils import rotation

from data_utils.datasets import build_train_dataset
from data_utils.evaluate import validate_things, validate_sintel, validate_kitti, validate_tartanair_rot
from dist_utils import get_dist_info, init_dist, setup_for_distributed
from loss import flow_loss_func

import os 

parser = argparse.ArgumentParser()

#dataset parameters
parser.add_argument('--dataset_dir', type=str, default=None, help='datset dir')
parser.add_argument('--val_dataset', type=str, default='things', help='validation dataset')
parser.add_argument('--stage', type=str, default='tartanair', help='training stage')

#traning parameters
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', default=8, type=int, help='number of GPU')
parser.add_argument('--num_iters', type=int, default=1000000, help='number of epochs for training')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--val_freq', default=1000, type=int, help='validation frequency')
parser.add_argument('--cam_intri', type=float, nargs=9, required=True, 
                    help="Camera intrinsic matrix as 9 space-separated values")
parser.add_argument('--cam_intri_inv', type=float, nargs=9, required=True, 
                    help="Camera intrinsic matrix as 9 space-separated values")

parser.add_argument('--max_flow', type=int, default=400, help='max flow value')

parser.add_argument('--resume', default=None, help='state dict to load')
parser.add_argument('--output_dir', default='.', help='output dir')

# distributed training
# parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--distributed', action='store_true')


args = parser.parse_args()
print(args)

torch.backends.cudnn.benchmark = True

writer = SummaryWriter(args.output_dir, flush_secs=1)

cam_intri = torch.tensor(args.cam_intri,dtype=torch.float32).view(3, 3)
cam_intri_inv = torch.tensor(args.cam_intri_inv,dtype=torch.float32).view(3, 3)

args.distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1 & args.distributed

# if args.distributed:
#     # adjust batch size for each gpu
#     assert args.batch_size % torch.cuda.device_count() == 0
#     args.batch_size = args.batch_size // torch.cuda.device_count()

#     dist_params = dict(backend='nccl')
#     init_dist('pytorch', **dist_params)
#     # re-set gpu_ids with distributed training mode
#     _, world_size = get_dist_info()
#     args.gpu_ids = range(world_size)
#     device = torch.device('cuda:{}'.format(args.local_rank))

#     setup_for_distributed(args.local_rank == 0)

# else:
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.distributed:

    dis_params = dict(backend='nccl')
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    assert args.batch_size % torch.cuda.device_count() == 0
    args.batch_size = args.batch_size // torch.cuda.device_count()
    device = torch.device(f'cuda:{local_rank}')

    setup_for_distributed(local_rank == 0)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('load the dataset')
train_dataset = build_train_dataset(args.stage, args.dataset_dir)
print('Number of training images:', len(train_dataset))

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=local_rank)
else:
    train_sampler = None

shuffle = not args.distributed
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=shuffle, num_workers=args.num_workers,
                                            pin_memory=True, drop_last=True,
                                            sampler=train_sampler)

# model = PWCNet().to(device)
model = PWCNet_tflow().to(device)

if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank)
    model = model.module

num_params = sum(p.numel() for p in model.parameters())
print('Number of model params:', num_params)

if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing_key, unexpected_key = model.load_state_dict(state_dict, strict=False)
    if missing_key:
        print('missing_key:', missing_key)
    if unexpected_key:
        print('unexpected_key:', unexpected_key)

optimizer = AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
sheduler = CosineAnnealingLR(optimizer, args.num_iters, args.lr * 0.01)

# helper function for smooth loss logging (average values before writing to tensorboard)
scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

def flow_to_color(flow):
    h, w = flow.shape[1:]
    mag, ang = cv2.cartToPolar(flow[0], flow[1])
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = np.clip(51 * mag, 0, 255)
    hsv[..., 2] = 255
    return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)


#############training loop#############
total_steps = 0
epoch = 0
counter = 0
video_imag_out = []
video_flow_out = []
video_flow_gt_out = []

# pbar = tqdm(range(args.num_iters),ncols=80)
pbar = tqdm(range(args.num_iters), ncols=80, dynamic_ncols=False)
# for epoch in pbar:
# while total_steps < args.num_iters:
for epoch in pbar:
    
    model.train()

    if args.distributed:
        train_sampler.set_epoch(epoch)
    
    for j, datas in enumerate(train_loader):
        optimizer.zero_grad()
        image1, image2, flow_gt, valid, rotation_quat = [data.to(device) for data in datas]

        model.init_cam_intri(cam_intri=cam_intri, cam_intri_inv=cam_intri_inv, height=image1.shape[-2], width=image1.shape[-1])
        
        flow_pred = model(image1, image2, rotation_quat)

        loss, metrics, flow_gt_tflow = flow_loss_func(flow_pred, flow_gt, valid, max_flow=args.max_flow, 
                                        rotation_quat=rotation_quat, cam_intri=cam_intri, cam_intri_inv=cam_intri_inv)

        loss.backward()
        optimizer.step()
        sheduler.step()
        pbar.set_description('loss:{}'.format(loss.item()))
        total_steps += 1
        
        smooth_dict({
            'loss':loss,
            'traning_epe':metrics['epe'],
        })


        with torch.no_grad():
            if total_steps % 10 == 0 and local_rank == 0:
                for k, v in scaler_q.items():
                    writer.add_scalar(k, np.mean(v), total_steps)
                scaler_q.clear()
                
                flow_gt_image = flow_to_color(flow_gt_tflow[6].cpu().numpy()).transpose(2, 0, 1)
                flow_pred_image = flow_to_color(flow_pred[6].detach().cpu().numpy()).transpose(2, 0, 1)
                image1 = image1[6].cpu().numpy()


                writer.add_image('flow_gt', flow_gt_image, dataformats='CHW', global_step=total_steps)
                writer.add_image('flow_pred', flow_pred_image, dataformats='CHW', global_step=total_steps)
                writer.add_image('image1', image1, dataformats='CHW', global_step=total_steps)

            
            if total_steps % args.val_freq == 0:
                # Ensure all processes complete their training step before validation
                dist.barrier() if args.distributed else None
                if local_rank == 0:

                    torch.save(model.state_dict(), f'{args.output_dir}/model_{total_steps}.pth')
                    print('model saved')
                    print('metrics:', metrics)
                
                #validation 
                val_results = {}
                
                if 'tartanair' in args.val_dataset:
                    test_results_dict = validate_tartanair_rot(model, _base_root=args.dataset_dir, cam_intri=cam_intri, cam_intri_inv=cam_intri_inv)
                    if local_rank == 0:
                        val_results.update(test_results_dict)
                        writer.add_scalar('tartanair_validation_epe', test_results_dict['tartanair_rot_epe'], total_steps)

                if local_rank == 0:

                    counter += 1

                    if counter >= 20:

                        for group in optimizer.param_groups:
                            group['lr'] *= 0.7

                        counter = 0

                    # Save validation results
                    val_file = os.path.join(args.output_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('step: %06d lr: %.6f\n' % (total_steps, optimizer.param_groups[-1]['lr']))

                        for k, v in val_results.items():
                            f.write("| %s: %.3f " % (k, v))

                        f.write('\n\n')

                model.train()

    epoch += 1
