import torch
import sys
sys.path.append('/home/zhangyuang/huyu/pwc-Tflow/')
from data_utils import rotation
import torch.nn.functional as F
from PWCNet import warping
from glob import glob
import os.path as osp
import numpy as np
import cv2
from itertools import islice

# def test_warping(input, flow):
#     B, C, H, W = input.size()
#     output = torch.zeros_like(input)

#     for i in range(H):
#         for j in range(W):
#             u = j + flow[:,0, i, j]
#             v = i + flow[:,1, i, j]
#             if u < W and u >= 0 and v < H and v >= 0:
#                 output[:,0,i,j] = input[:,0, v.long(), u.long()]
#                 output[:,1,i,j] = input[:,1, v.long(), u.long()]
#     return output
            


def get_Tflow_gt(flow_gt, rotation_quat=None, cam_intri=None, cam_intri_inv=None):

    height, width = flow_gt.shape[-2], flow_gt.shape[-1]
    if rotation_quat is not None:
        # flow_gt = rotation(flow_gt, rotation_quat)
        # rotation_R = rotation.quaternion_to_matrix(rotation_quat)
        R_Batch = rotation.quaternion_to_matrix(rotation_quat)
        delt_R = warping.get_delt_R(R_Batch[None, ...])
        R_flow = warping.get_Rflow(delt_R, cam_intri, cam_intri_inv, height, width)
        flow_gt_1 = flow_gt - R_flow.squeeze(0)
        flow_gt_2 = warping.warpping(flow_gt[None, ...], -R_flow).squeeze(0)

    
    return flow_gt_1, flow_gt_2, R_flow.squeeze(0)

def flow_to_color(flow):
    h, w = flow.shape[1:]
    mag, ang = cv2.cartToPolar(flow[0], flow[1])
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = np.clip(51 * mag, 0, 255)
    hsv[..., 2] = 255
    return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)

def get_p_and_quaternion(filename, rotation_id):

    with open(filename, 'r', encoding='utf-8') as file:
        # 使用 islice 跳转到指定行
        line = next(islice(file, rotation_id, rotation_id + 1), None)
        if line is not None:
            # 获取旋转四元数并转换为浮点数列表
            rotation_quat_1 = [float(x) for x in line.split()[3:]]
            
            # 转换为 PyTorch Tensor
            rotation_quat_1 = torch.tensor(rotation_quat_1, dtype=torch.float32)
            rotation_quat_1 = torch.cat((rotation_quat_1[3].view(1), rotation_quat_1[:3]))

            # 获取平移向量并转换为浮点数列表
            translation_1 = [float(x) for x in line.split()[:3]]
            # 转换为 PyTorch Tensor
            translation_1 = torch.tensor(translation_1, dtype=torch.float32)

    with open(filename, 'r', encoding='utf-8') as file:    
        line_next = next(islice(file, rotation_id + 1, rotation_id + 2), None)
        if line_next is not None:

            rotation_quat_2 = [float(x) for x in line_next.split()[3:]]
            rotation_quat_2 = torch.tensor(rotation_quat_2, dtype=torch.float32)
            rotation_quat_2 = torch.cat((rotation_quat_2[3].view(1), rotation_quat_2[:3]))

            translation_2 = [float(x) for x in line_next.split()[:3]]
            translation_2 = torch.tensor(translation_2, dtype=torch.float32)

            translation = torch.stack((translation_1, translation_2), dim=0)
            rotation_quat = torch.stack((rotation_quat_1, rotation_quat_2), dim=0)
    
    return rotation_quat, translation

def get_flow_from_depth(depth, cam_intri, P, R, P_old, R_old):
    height, width = depth.shape

    flow_gt = torch.zeros(2, height, width)

    pix_v, pix_u = torch.meshgrid(torch.arange(height), torch.arange(width),indexing='ij')


    cam_pix_u = (pix_u - cam_intri[0,2]) * depth / cam_intri[0,0]
    cam_pix_v = (pix_v - cam_intri[1,2]) * depth / cam_intri[1,1]
    
    # 将相机坐标转换到世界坐标
    cam_pix = torch.stack([cam_pix_u, cam_pix_v, depth], dim=0)  # 3 x H x W
    world_pix = torch.einsum('ij,jhw->ihw', R_old, cam_pix)  # 使用einsum进行批量矩阵乘法

    world_pix[0] = world_pix[0] + P_old[0]
    world_pix[1] = world_pix[1] + P_old[1]
    world_pix[2] = world_pix[2] + P_old[2]

    # 将世界坐标转换到新的相机坐标
    new_world_pix = torch.stack([world_pix[0] - P[0], world_pix[1] - P[1], world_pix[2] - P[2]], dim=0)  # 3 x H x W
    new_cam_pix = torch.einsum('ij,jhw->ihw', R.T, new_world_pix)  # 使用 R 的转置进行批量矩阵乘法

    # 将新的相机坐标投影回像素平面
    new_cam_pix_u = new_cam_pix[0] / (new_cam_pix[2] + 1e-5)
    new_cam_pix_v = new_cam_pix[1] / (new_cam_pix[2] + 1e-5)

    # world_pix_u = R_old[0,0] * cam_pix_u + R_old[0,1] * cam_pix_v + R_old[0,2]
    # world_pix_v = R_old[1,0] * cam_pix_u + R_old[1,1] * cam_pix_v + R_old[1,2]
    # world_pix_z = R_old[2,0] * cam_pix_u + R_old[2,1] * cam_pix_v + R_old[2,2]

    # world_pix_u = world_pix_u + P_old[0]
    # world_pix_v = world_pix_v + P_old[1]
    # world_pix_z = world_pix_z + P_old[2]

    # new_world_pix_u = world_pix_u - P[0]
    # new_world_pix_v = world_pix_v - P[1]
    # new_world_pix_z = world_pix_z - P[2]

    # new_world_pix_u = R[0,0] * new_world_pix_u + R[1,0] * new_world_pix_v + R[2,0] * new_world_pix_z
    # new_world_pix_v = R[0,1] * new_world_pix_u + R[1,1] * new_world_pix_v + R[2,1] * new_world_pix_z
    # new_world_pix_z = R[0,2] * new_world_pix_u + R[1,2] * new_world_pix_v + R[2,2] * new_world_pix_z

    # new_cam_pix_u = new_world_pix_u / new_world_pix_z
    # new_cam_pix_v = new_world_pix_v / new_world_pix_z

    new_pix_u = cam_intri[0,0] * new_cam_pix_u + cam_intri[0,2]
    new_pix_v = cam_intri[1,1] * new_cam_pix_v + cam_intri[1,2]

    flow_gt[0] = new_pix_u - pix_u
    flow_gt[1] = new_pix_v - pix_v

    return flow_gt

def get_transformed_depth(depth, cam_intri, R_old, P_old, R, P):
    height, width = depth.shape

    # 创建像素网格
    pix_v, pix_u = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

    # 将像素坐标转换到相机坐标系
    cam_pix_u = (pix_u + 0.5 - cam_intri[0, 2]) * depth / cam_intri[0, 0]
    cam_pix_v = (pix_v + 0.5  - cam_intri[1, 2]) * depth / cam_intri[1, 1]

    # 构建相机坐标的3D点（注意深度图提供的是z坐标）
    cam_pix = torch.stack([cam_pix_u, cam_pix_v, depth], dim=0)  # 3 x H x W
    
    # 将相机坐标转换到世界坐标系
    world_pix = torch.einsum('ij,jhw->ihw', R_old, cam_pix) + P_old[:, None, None]

    # 将世界坐标转换回新的相机坐标系
    new_cam_pix = torch.einsum('ij,jhw->ihw', R.T, world_pix - P[:, None, None])

    # 检查 new_cam_pix[2] 是否为有效值（> 1e-6 并且不是 NaN）
    valid_mask = (new_cam_pix[2] > 1e-6) & torch.isfinite(new_cam_pix[2])

    # 计算新的像素坐标
    new_u = (new_cam_pix[0] * cam_intri[0, 0] / new_cam_pix[2]) + cam_intri[0, 2] - 0.5
    new_v = (new_cam_pix[1] * cam_intri[1, 1] / new_cam_pix[2]) + cam_intri[1, 2] - 0.5

    # # 对超出边界的像素进行限制（确保它们在图像尺寸范围内）
    # new_u = torch.clamp(new_u, 0, width - 1).long()
    # new_v = torch.clamp(new_v, 0, height - 1).long()

    # # 生成新的深度图
    # new_depth = torch.full_like(depth, 0)  # 初始化为无穷大的深度
    # mask_valid_indices = valid_mask & (new_v >= 0) & (new_v < height) & (new_u >= 0) & (new_u < width)
    # new_depth[new_v[mask_valid_indices], new_u[mask_valid_indices]] = new_cam_pix[2][mask_valid_indices]

        # 限制坐标在图像范围内
    new_u = torch.clamp(new_u, 0, width - 1)
    new_v = torch.clamp(new_v, 0, height - 1)

    # 获取最近的四个像素坐标
    u0 = new_u.floor().long()
    v0 = new_v.floor().long()
    u1 = torch.clamp(u0 + 1, 0, width - 1)
    v1 = torch.clamp(v0 + 1, 0, height - 1)

    # 计算插值权重
    w_u = new_u - u0.float()
    w_v = new_v - v0.float()

    # 使用广播进行向量化的双线性插值
    new_depth = (1 - w_u) * (1 - w_v) * depth[v0, u0] + \
                w_u * (1 - w_v) * depth[v0, u1] + \
                (1 - w_u) * w_v * depth[v1, u0] + \
                w_u * w_v * depth[v1, u1]

    # 保留有效区域
    new_depth[~valid_mask] = 100000

    return new_depth

def get_depth_image(depth):
    depth = depth.numpy().astype(np.float32)
    # 归一化深度图以便显示：将深度图缩放到 0-255 范围
    depth_map_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

    # 将归一化后的深度图转换为 8 位无符号整数
    depth_map_normalized = depth_map_normalized.astype(np.uint8)

    # 使用颜色映射将深度图转换为彩色图像（伪彩色）
    depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    return depth_colormap

# depth = torch.ones(480, 640)
# #绕着y旋转90度的旋转矩阵
# R = torch.tensor([[0.707, 0, -0.707],
#                   [0, 1, 0],
#                   [0.707, 0, 0.707]], dtype=torch.float32)
# R0 = torch.tensor([[1, 0, 0],
#                    [0, 1, 0],
#                    [0, 0, 1]], dtype=torch.float32)
# #绕着x轴旋转45度
# R = torch.tensor([[1, 0, 0],
#                     [0, 0.707, 0.707],
#                     [0, -0.707, 0.707]], dtype=torch.float32)
# p_old = torch.tensor([0, 0, 0], dtype=torch.float32)
# p = torch.tensor([0, -1, 0], dtype=torch.float32)

# R_cam2body = torch.tensor([[0, 0, 1], 
#                           [1, 0, 0], 
#                           [0, 1, 0]], dtype=torch.float32)

# flow_from_depth = get_flow_from_depth(depth, cam_intri, p, R_cam2body, p_old, R_cam2body)
# print(flow_from_depth)

# print(flow_from_depth[0,240,320])
# print(flow_from_depth[1,240,320])

cam_intri = torch.tensor([[320, 0, 320],
                         [0, 320, 240],
                         [0, 0, 1]], dtype=torch.float32).view(3, 3)
cam_intri_inv = torch.tensor([[1/320, 0, -1],
                             [0, 1/320, -0.75],
                             [0, 0, 1]], dtype=torch.float32).view(3, 3)

# # 初始化一个 480x640 的张量，所有值为 100000
# depth = torch.full((480, 640), 0, dtype=torch.float32)

# # 定义中间区域的索引范围
# center_start_x = (480 - 50) // 2  # 中间区域起始位置（高度方向）
# center_start_y = (640 - 50) // 2  # 中间区域起始位置（宽度方向）

# # 将中间 50x50 区域的深度值设置为 5
# depth[center_start_x:center_start_x+50, center_start_y:center_start_y+50] = 5

# #绕着y旋转90度的旋转矩阵
# R = torch.tensor([[0.707, 0, -0.707],
#                   [0, 1, 0],
#                   [0.707, 0, 0.707]], dtype=torch.float32)
# R_old = torch.tensor([[1, 0, 0],
#                         [0, 1, 0],
#                         [0, 0, 1]], dtype=torch.float32)

# # R = torch.tensor([[0.707, -0.707, 0],
# #                    [0.707, 0.707, 0],
# #                    [0, 0, 1]], dtype=torch.float32)
# # #绕着x轴旋转45度
# # R = torch.tensor([[1, 0, 0],
# #                     [0, 0.707, 0.707],
# #                     [0, -0.707, 0.707]], dtype=torch.float32)
# # p_old = torch.tensor([0, 0, 0], dtype=torch.float32)
# p_old = torch.tensor([0, 0, 0], dtype=torch.float32)
# new_depth = get_transformed_depth(depth=depth, cam_intri = cam_intri, R_old=R_old, P_old=p_old, R=R)


# depth_image = get_depth_image(depth)
# new_depth_image = get_depth_image(new_depth)

# print(torch.norm(depth - new_depth).mean())
# print(depth[240, 320])
# print(new_depth[240, 320])

# combined_image = cv2.hconcat([depth_image, new_depth_image])
# cv2.imwrite('/home/zhangyuang/huyu/pwc-Tflowdepth.png', combined_image)

if __name__ == '__main__':
    dataset_path = '/mnt/sdc/TartanAir/test/seasonsforest_winter/Easy/P009'

    flows_paths = sorted(glob(osp.join(dataset_path, 'flow', '*flow.npy')))
    depth_paths = sorted(glob(osp.join(dataset_path, 'depth_left', '*depth.npy')))

    for i in range(len(flows_paths)):
        flow_path = flows_paths[i]
        depth_path = depth_paths[i]

        depth_path_2 = depth_paths[i+1]

        flow_gt = np.load(flow_path)
        flow_gt = np.array(flow_gt).astype(np.float32)
        flow_gt = torch.from_numpy(flow_gt).permute(2, 0, 1).float()

        flow_name = osp.split(flow_path)[1]
        # rotation_id = int(flow_name.split('_')[0])
        # print(rotation_id)

        quaternion, position = get_p_and_quaternion(osp.join(dataset_path, 'pose_left.txt'), i)

        R_batch = rotation.quaternion_to_matrix(quaternion)
        R_cam2body = torch.tensor([[0, 0, 1], 
                                   [1, 0, 0], 
                                   [0, 1, 0]], dtype=torch.float32)
        
        R_batch = R_batch @ R_cam2body[None,...]  # 形状 (B, 3, 3)

        depth = np.load(depth_path)
        depth = torch.tensor(depth, dtype=torch.float32)

        depth_2 = np.load(depth_path_2)
        depth_2 = torch.tensor(depth_2, dtype=torch.float32)

        print(i)

        # new_depth = get_transformed_depth(depth=depth, cam_intri = cam_intri, R_old=R_batch[0], P_old=position[0], R=R_batch[1])
        new_depth_2 = get_transformed_depth(depth=depth_2, cam_intri = cam_intri, R_old=R_batch[1], P_old=position[1], R=R_batch[1], P=position[0])

        # flow_from_depth = get_flow_from_depth(new_depth, cam_intri, position[1], R_batch[1], position[0], R_batch[1])
        # flow_from_depth = get_flow_from_depth(depth, cam_intri, position[1], R_batch[0], position[0], R_batch[0])
        
        flow_from_depth = get_flow_from_depth(depth=new_depth_2, cam_intri=cam_intri, P_old=position[0], R_old=R_batch[1], P=position[1], R=R_batch[1])

        flow_gt_image = flow_to_color(flow_gt.numpy())
        flow_from_depth_image = flow_to_color(flow_from_depth.numpy())

        
        # print(depth)
        depth_image = get_depth_image(depth_2)
        new_depth_image = get_depth_image(new_depth_2)

        combined_image_1 = cv2.hconcat([flow_gt_image, flow_from_depth_image])
        combined_image_2 = cv2.hconcat([depth_image, new_depth_image])
        combined_image = cv2.vconcat([combined_image_1, combined_image_2])


        cv2.imwrite(osp.join('/home/zhangyuang/huyu/pwc-Tflow/exps/gtflow_test', f'{i}flowfromdepth.png'), combined_image)

        
        
        flow_gt_1, flow_gt_2, R_flow = get_Tflow_gt(flow_gt, rotation_quat=quaternion, cam_intri=cam_intri, cam_intri_inv=cam_intri_inv)

        flow_gt_1_image = flow_to_color(flow_gt_1.numpy())
        flow_gt_2_image = flow_to_color(flow_gt_2.numpy())
        flow_gt_image = flow_to_color(flow_gt.numpy())
        R_flow_image = flow_to_color(R_flow.numpy())

        delt_flow = flow_gt_1 - flow_from_depth
        epe = torch.norm(delt_flow, dim=0)
        print(epe.mean())

        # cv2.imshow('flow_gt_1', flow_gt_1_image)
        # cv2.imshow('flow_gt_2', flow_gt_2_image)
        # cv2.imshow('flow_gt', flow_gt_image)
        combined_image_1 = cv2.hconcat([flow_gt_image, R_flow_image])
        combined_image_2 = cv2.hconcat([flow_gt_1_image, flow_gt_2_image])
        combined_image = cv2.vconcat([combined_image_1, combined_image_2])


        cv2.imwrite(osp.join('/home/zhangyuang/huyu/pwc-Tflow/exps/gtflow_test', f'{i}.png'), combined_image)


#     # for flow_path in flows_paths:
        
#     #     flow_gt = np.load(flow_path)
#     #     flow_gt = np.array(flow_gt).astype(np.float32)
#     #     flow_gt = torch.from_numpy(flow_gt).permute(2, 0, 1).float()

#     #     flow_name = osp.split(flow_path)[1]
#     #     rotation_id = int(flow_name.split('_')[0])
#     #     print(rotation_id)

#     #     quaternion = get_quaternion(osp.join(dataset_path, 'pose_left.txt'), rotation_id)
#     #     depth_path = osp.join(dataset_path, 'depth_left', f'{rotation_id}_left_depth.npy')
        
#     #     flow_gt_1, flow_gt_2, R_flow = get_Tflow_gt(flow_gt, rotation_quat=quaternion, cam_intri=cam_intri, cam_intri_inv=cam_intri_inv)

#     #     flow_gt_1_image = flow_to_color(flow_gt_1.numpy())
#     #     flow_gt_2_image = flow_to_color(flow_gt_2.numpy())
#     #     flow_gt_image = flow_to_color(flow_gt.numpy())
#     #     R_flow_image = flow_to_color(R_flow.numpy())

#     #     # cv2.imshow('flow_gt_1', flow_gt_1_image)
#     #     # cv2.imshow('flow_gt_2', flow_gt_2_image)
#     #     # cv2.imshow('flow_gt', flow_gt_image)
#     #     combined_image_1 = cv2.hconcat([flow_gt_image, R_flow_image])
#     #     combined_image_2 = cv2.hconcat([flow_gt_1_image, flow_gt_2_image])
#     #     combined_image = cv2.vconcat([combined_image_1, combined_image_2])


#     #     cv2.imwrite(osp.join('/home/zhangyuang/huyu/pwc-Tflow/exps/gtflow_test', f'{rotation_id}.png'), combined_image)
        