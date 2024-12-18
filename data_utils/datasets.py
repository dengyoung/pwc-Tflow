# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp

from data_utils import frame_utils
from data_utils.transforms import FlowAugmentor, SparseFlowAugmentor
from itertools import islice

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, virtual=False,
                 load_occlusion=False,
                 ):
        self.augmentor = None
        self.sparse = sparse
        self.virtual = virtual

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        #flow_list、image_list都是存储的图像以及光流的路径
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        self.load_occlusion = load_occlusion
        self.occ_list = []

        self.read_rotations = False

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        if self.sparse:
            if self.virtual:
                flow, valid = frame_utils.read_vkitti_png_flow(self.flow_list[index])  # [H, W, 2], [H, W]
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])  # [H, W, 2], [H, W]
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        if self.load_occlusion:
            occlusion = frame_utils.read_gen(self.occ_list[index])  # [H, W], 0 or 255 (occluded)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if self.read_rotations:
            rotation_dir_base_1 = osp.split(osp.split(self.image_list[index][0])[0])[0]
            rotation_dir_1 = osp.join(rotation_dir_base_1, 'pose_left.txt')
            # print(pose_dir)
            image_name_1 = osp.split(self.image_list[index][0])[1]
            rotation_id_1 = int(image_name_1.split('_')[0])

            rotation_dir_base_2 = osp.split(osp.split(self.image_list[index][1])[0])[0]
            rotation_dir_2 = osp.join(rotation_dir_base_2, 'pose_left.txt')
            # print(pose_dir)
            image_name_2 = osp.split(self.image_list[index][1])[1]
            rotation_id_2 = int(image_name_2.split('_')[0])

            with open(rotation_dir_1, 'r', encoding='utf-8') as file:
                # 使用 islice 跳转到指定行
                line = next(islice(file, rotation_id_1, rotation_id_1 + 1), None)
                if line is not None:
                    # line.strip()
                    # print(line)
                    # 获取旋转四元数并转换为浮点数列表
                    rotation_quat_1 = [float(x) for x in line.split()[3:]]
                    
                    # 转换为 PyTorch Tensor
                    rotation_quat_1 = torch.tensor(rotation_quat_1, dtype=torch.float32)
                    rotation_quat_1 = torch.cat((rotation_quat_1[3].view(1), rotation_quat_1[:3]))

            with open(rotation_dir_2, 'r', encoding='utf-8') as file:    
                line_next = next(islice(file, rotation_id_2, rotation_id_2 + 1), None)
                if line_next is not None:

                    rotation_quat_2 = [float(x) for x in line_next.split()[3:]]
                    rotation_quat_2 = torch.tensor(rotation_quat_2, dtype=torch.float32)
                    rotation_quat_2 = torch.cat((rotation_quat_2[3].view(1), rotation_quat_2[:3]))
                    # print(rotation_dir_1)
                    # print(rotation_dir_2)
                    # print(rotation_quat_1)
                    # print(rotation_quat_2)
            
                    rotation_quat = torch.stack((rotation_quat_1, rotation_quat_2), dim=0)
                

        if self.load_occlusion:
            occlusion = np.array(occlusion).astype(np.float32)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                if self.load_occlusion:
                    img1, img2, flow, occlusion = self.augmentor(img1, img2, flow, occlusion=occlusion)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)

        if self.load_occlusion:

            if np.count_nonzero(occlusion) / (occlusion.shape[0]*occlusion.shape[1]) < 0.3:
                valid = np.zeros(flow.shape[:-1])

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        # if self.load_occlusion:
        #     occlusion = torch.from_numpy(occlusion)  # [H, W]

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        # # mask out occluded pixels
        # if self.load_occlusion:
        #     # non-occlusion: 0, occlusion: 255
        #     noc_valid = 1 - occlusion / 255.  # 0 or 1

        #     return img1, img2, flow, valid.float(), noc_valid.float()

        if self.read_rotations:
            return img1, img2, flow, valid.float(), rotation_quat
        else:
            return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.occ_list = v * self.occ_list

        return self

    def __len__(self):
        return len(self.image_list)

class Tartanair(FlowDataset):
    def __init__(self, aug_params=None, split='training',
                 root_base=None,
                 root='TartanAir',
                 test_set=False,
                 read_rotations=False,):
        super(Tartanair, self).__init__(aug_params)

        self.read_rotations = read_rotations
        image_list_all = []
        flow_list_all = []
        # Define the base root path
        root_path = osp.join(root_base, root, 'test' if test_set else 'train')

        # Iterate over all scenes and sub-scenes
        for scene in os.listdir(root_path):
            scene_path = osp.join(root_path, scene)
            for sub_scene in os.listdir(scene_path):
                sub_scene_path = osp.join(scene_path, sub_scene)
                for image_folder in os.listdir(sub_scene_path):
                    image_folder_path = osp.join(sub_scene_path, image_folder)

                    # Collect image pairs and corresponding flow files
                    images = sorted(glob(osp.join(image_folder_path, 'image_left', '*.png')))
                    flows = sorted(glob(osp.join(image_folder_path, 'flow', '*flow.npy')))

                    # Add image pairs
                    self.image_list.extend([[images[i], images[i + 1]] for i in range(len(images) - 1)])
                    self.flow_list.extend(flows)

        # Subsample data if in test mode
        if test_set:
            step = max(1, len(self.image_list) // 500)
            self.image_list = self.image_list[::step]
            self.flow_list = self.flow_list[::step]
            self.image_list = self.image_list[:500]
            self.flow_list = self.flow_list[:500]
        # root = osp.join(root_base, root)
        # if test_set:
        #     train_root = osp.join(root, 'test')
        #     for scene in os.listdir(train_root):
        #         image_floder_1 = osp.join(train_root, scene)
        #         for sub_scene in os.listdir(image_floder_1):
        #             image_floder_2 = osp.join(image_floder_1, sub_scene)
        #             for image_floder_3 in os.listdir(image_floder_2):
        #                 image_list = sorted(glob(osp.join(image_floder_2, image_floder_3, 'image_left', '*.png')))
        #                 for i in range(len(image_list) - 1):
        #                     image_list_all += [[image_list[i], image_list[i + 1]]]
        #                 flow_list_all += sorted(glob(osp.join(image_floder_2, image_floder_3, 'flow', '*flow.npy')))
        

        #     step = max(1, len(image_list_all) // 100)
        #     for i in range(0, len(image_list_all), step):
        #         self.image_list += [image_list_all[i]]
        #         self.flow_list += [flow_list_all[i]]

        # else:
        #     train_root = osp.join(root, 'train')
        #     for scene in os.listdir(train_root):
        #         image_floder_1 = osp.join(train_root, scene)
        #         for sub_scene in os.listdir(image_floder_1):
        #             image_floder_2 = osp.join(image_floder_1, sub_scene)
        #             for image_floder_3 in os.listdir(image_floder_2):
        #                 image_list = sorted(glob(osp.join(image_floder_2, image_floder_3, 'image_left', '*.png')))
        #                 for i in range(len(image_list) - 1):
        #                     self.image_list += [[image_list[i], image_list[i + 1]]]
        #                 self.flow_list += sorted(glob(osp.join(image_floder_2, image_floder_3, 'flow', '*flow.npy')))

            # image_list = sorted(glob(osp.join(train_root, '*/*/*/image_left', '*.png')))
            # for i in range(len(image_list) - 1):
            #     self.image_list += [[image_list[i], image_list[i + 1]]]
            
            # self.flow_list = sorted(glob(osp.join(train_root, '*/*/*/flow', '*flow.npy')))
class small_Tartanair(FlowDataset):
    def __init__(self, aug_params=None, data_dir=None,
                 test_set=False,
                 read_rotations=False,):
        super(small_Tartanair, self).__init__(aug_params)

        self.read_rotations = read_rotations

        images = sorted(glob(osp.join(data_dir, 'image_left', '*.png')))
        flows = sorted(glob(osp.join(data_dir, 'flow', '*flow.npy')))
        self.image_list.extend([[images[i], images[i + 1]] for i in range(len(images) - 1)])
        self.flow_list.extend(flows)

        # Subsample data if in test mode
        if test_set:
            step = max(1, len(self.image_list) // 50)
            self.image_list = self.image_list[::step]
            self.flow_list = self.flow_list[::step]
            self.image_list = self.image_list[:50]
            self.flow_list = self.flow_list[:50]


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training',
                 root_base=None,
                 root='Sintel',
                 dstype='clean',
                 load_occlusion=False,
                 ):
        super(MpiSintel, self).__init__(aug_params,
                                        load_occlusion=load_occlusion,
                                        )
        
        root = osp.join(root_base, root)

        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if load_occlusion:
            occlusion_root = osp.join(root, split, 'occlusions')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

                if load_occlusion:
                    self.occ_list += sorted(glob(osp.join(occlusion_root, scene, '*.png')))

class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train',
                 root_base=None,
                 root='FlyingChairs_release/data',
                 ):
        super(FlyingChairs, self).__init__(aug_params)

        root = osp.join(root_base, root)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chairs_split.txt')
        split_list = np.loadtxt(split_file, dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None,
                 root_base=None,
                 root='FlyingThings3D',
                 dstype='frames_cleanpass',
                 test_set=False,
                 validate_subset=True,
                 load_occlusion=False,
                 only_left=True,
                 ):
        super(FlyingThings3D, self).__init__(aug_params, load_occlusion=load_occlusion)

        root = osp.join(root_base, root)
        img_dir = root
        flow_dir = root

        if only_left:
            cam_list = ['left']
        else: 
            cam_list = ['left', 'right']

        for cam in cam_list:
            for direction in ['into_future', 'into_past']:
                if test_set:
                    image_dirs = sorted(glob(osp.join(img_dir, dstype, 'TEST/*/*')))
                else:
                    image_dirs = sorted(glob(osp.join(img_dir, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                if test_set:
                    flow_dirs = sorted(glob(osp.join(flow_dir, 'optical_flow/TEST/*/*')))
                else:
                    flow_dirs = sorted(glob(osp.join(flow_dir, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    occs = sorted(glob(osp.join(fdir, '*.png')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                            if load_occlusion:
                                self.occ_list += [occs[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]
                            if load_occlusion:
                                self.occ_list += [occs[i]]

        if test_set and validate_subset:
            num_val_samples = 1024
            all_test_samples = len(self.image_list)  # 7866

            stride = all_test_samples // num_val_samples
            remove = all_test_samples % num_val_samples

            self.image_list = self.image_list[:-remove][::stride]
            self.flow_list = self.flow_list[:-remove][::stride]

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training',
                 root_base=None,
                 root='KITTI',
                 ):
        super(KITTI, self).__init__(aug_params, sparse=True,
                                    )
        root = osp.join(root_base, root)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root_base = None, root='HD1K'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        root = osp.join(root_base, root)
        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1

class NeuSim(FlowDataset):
    def __init__(self, aug_params=None,
                 root_base=None,
                 root='NeuSim'
                 ):
        super(NeuSim, self).__init__(aug_params)

        root = osp.join(root_base, root)

        image_dirs = sorted(glob(osp.join(root, '*/image')))

        fw_flow_dirs = sorted(glob(osp.join(root, '*/forward_flow')))
        bw_flow_dirs = sorted(glob(osp.join(root, '*/backward_flow')))

        for image_dir, fw_flow_dir, bw_flow_dir in zip(image_dirs, fw_flow_dirs, bw_flow_dirs):
            images = sorted(glob(osp.join(image_dir, '*.png')))
            fw_flows = sorted(glob(osp.join(fw_flow_dir, '*.npy')))
            bw_flows = sorted(glob(osp.join(bw_flow_dir, '*.npy')))
            for i in range(len(fw_flows) - 1):
                self.image_list += [[images[i], images[i + 1]]]
                self.flow_list += [fw_flows[i]]
                self.image_list += [[images[i + 1], images[i]]]
                self.flow_list += [bw_flows[i]]

def build_train_dataset(stage, _root_base= None):
    if stage == 'chairs':
        aug_params = {'crop_size': (384, 512), 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}

        train_dataset = FlyingChairs(aug_params, split='training', root_base=_root_base)
    
    elif stage == 'tartanair':
        aug_params = {'crop_size': (480, 640), 'min_scale': -0.25, 'max_scale': 0.9, 'do_flip': True}

        train_dataset = Tartanair(aug_params, split='training', root_base=_root_base, read_rotations=False)
    
    elif stage == 'small_tartanair_r':
        aug_params = {'crop_size': (480, 640), 'min_scale': -0.25, 'max_scale': 0.9, 'do_flip': False}

        train_dataset = small_Tartanair(aug_params, data_dir=_root_base, read_rotations=True)
    elif stage == 'tartanair_r':
        aug_params = {'crop_size': (480, 640), 'min_scale': -0.25, 'max_scale': 0.9, 'do_flip': False}

        train_dataset = Tartanair(aug_params, split='training', root_base=_root_base, read_rotations=True)
    
    elif stage == 'things':
        aug_params = {'crop_size': (384, 768), 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}

        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass', load_occlusion=True, root_base=_root_base)
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=True, root_base=_root_base)
        train_dataset = clean_dataset + final_dataset

    elif stage == 'sintel':
        crop_size = (320, 896)
        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass', load_occlusion=True, root_base=_root_base)
        things_final = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=True, root_base=_root_base)

        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', root_base=_root_base)
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', root_base=_root_base)

        aug_params = {'crop_size': crop_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}

        kitti = KITTI(aug_params=aug_params, val=False, root_base=_root_base)

        aug_params = {'crop_size': crop_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}

        hd1k = HD1K(aug_params=aug_params, root_base=_root_base)

        train_dataset = 40 * sintel_clean + 40 * sintel_final + 200 * kitti + 10 * hd1k + things_clean + things_final

    elif stage == 'kitti':
        aug_params = {'crop_size': (320, 1152), 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}

        train_dataset = KITTI(aug_params, split='training', val=False, root_base=_root_base)

    elif stage == 'neusim':
        crop_size = (320, 896)
        aug_params = {'crop_size': crop_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        things_clean = FlyingThings3D(aug_params, dstype='frames_cleanpass', load_occlusion=True, root_base=_root_base)
        things_final = FlyingThings3D(aug_params, dstype='frames_finalpass', load_occlusion=True, root_base=_root_base)

        aug_params = {'crop_size': crop_size, 'min_scale': -1, 'max_scale': 0, 'do_flip': False}
        neu_dataset = NeuSim(aug_params, root_base=_root_base)

        train_dataset = things_clean + things_final + 2 * neu_dataset

    return train_dataset
