import torch
import os 
import torch.distributed as dist
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
from glob import glob
num_replicas=torch.cuda.device_count()

print(num_replicas)

# local_rank = int(os.environ['LOCAL_RANK'])

# print(local_rank)

print(dist.get_world_size)

for i in range(10):
    print(i)
root = '/mnt/sdc/TartanAir'
train_root = osp.join(root, 'train')
image_list_all = []
flow_list_all = []
for scene in os.listdir(train_root):
    image_floder_1 = osp.join(train_root, scene)
    for sub_scene in os.listdir(image_floder_1):
        image_floder_2 = osp.join(image_floder_1, sub_scene)
        for image_floder_3 in os.listdir(image_floder_2):
            image_list = sorted(glob(osp.join(image_floder_2, image_floder_3, 'image_left', '*.png')))
            for i in range(len(image_list) - 1):
                image_list_all += [[image_list[i], image_list[i + 1]]]
                print(image_list[i])
            flow_list_all += sorted(glob(osp.join(image_floder_2, image_floder_3, 'flow', '*flow.npy')))
            print(flow_list_all)