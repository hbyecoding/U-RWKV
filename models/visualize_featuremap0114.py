import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

from src.dataloader.dataset import MedicalDataSets
from albumentations.core.composition import Compose
from albumentations import Resize, Normalize
from src.network.conv_based.compnet1225_enc_rwkv import LoRA__5

import torch.nn as nn
from src.network.conv_based.compnet1225_enc_rwkv_ERF import LoRA__5

class LoRA__5_FMap(LoRA__5):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(LoRA__5_FMap, self).__init__(input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5)
        
    def forward(self,x):
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        # x4 = self.Brwkv_4(x4)
        # p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        
        
        
        x5_ = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        # x5 = self.Brwkv(x5)
        # p5 = self.Brwkv(p5)
        """ Decoder 1 """
        # print(self.s1)
        # s1 = self.s1(p5, x5)
        
        # x5_ = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # # x5 = self.Brwkv(x5)
        # # p5 = self.Brwkv(p5)
        # """ Decoder 1 """
        # # print(self.s1)
        s1 = self.s1(p5, x5_)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        # return out1, out2        
        return out1, out2 
        # return x1, x2, x3, x4, x5
# def visualize_feature_maps(feature_maps, save_path=None):
#     """
#     使用 OpenCV 可视化 Feature Map
#     :param feature_maps: 输入的 Feature Map（Tensor）
#     :param save_path: 保存路径（可选）
#     """
#     # 将 Feature Map 转换为 NumPy 数组
#     feature_maps = feature_maps.detach().cpu().numpy()
    
#     # 遍历每个通道
#     for i in range(feature_maps.shape[1]):  # 遍历通道维度
#         # 获取单个通道的 Feature Map
#         single_channel = feature_maps[0, i, :, :]  # 取 batch 中的第一个样本
        
#         # 归一化到 [0, 255]
#         single_channel = (single_channel - single_channel.min()) / (single_channel.max() - single_channel.min()) * 255
#         single_channel = single_channel.astype(np.uint8)
        
#         # 使用 OpenCV 显示 Feature Map
#         cv2.imshow(f'Feature Map Channel {i}', single_channel)
#         cv2.waitKey(0)  # 等待按键
        
#         # 保存 Feature Map（可选）
#         if save_path:
#             cv2.imwrite(f'{save_path}_channel_{i}.png', single_channel)
    
#     cv2.destroyAllWindows()
import random
# 2025年1月15日
# import random
# import numpy as np
# import cv2

# def visualize_feature_maps(feature_maps, save_path):
#     """
#     保存所有 Feature Map 通道为图像文件，并进行二值化处理
#     :param feature_maps: 输入的 Feature Map（Tensor）
#     :param save_path: 保存路径
#     """
#     # 将 Feature Map 转换为 NumPy 数组
#     feature_maps = feature_maps.detach().cpu().numpy()
    
#     # 获取通道数
#     num_channels = feature_maps.shape[1]
    
#     # 遍历所有通道
    
#     # for channel_idx in range(num_channels):
#     channel_idx = 0
#         # 获取当前通道的 Feature Map
#     single_channel = feature_maps[0, channel_idx, :, :]  # 取 batch 中的第一个样本
#     print("single_channel.shape",single_channel.shape)
#     # 归一化到 [0, 1]
#     single_channel = (single_channel - single_channel.min()) / (single_channel.max() - single_channel.min())
#     print("single channel",single_channel)
#     # 二值化处理
#     binary_channel = np.where(single_channel >= 0.5, 255, 0).astype(np.uint8)
#     print("binary_channel",np.unique(binary_channel))
#     # 保存 Feature Map
#     cv2.imwrite(f'{save_path}_channel_{channel_idx}.png', binary_channel)
#     print("cv2 imwrite finished")
#     return "cv2 imwrite finished"
# 示例调用
# feature_maps = ...  # 你的 Feature Map 数据
# visualize_feature_maps(feature_maps, 'path/to/save')

# # 这是不带resize的 因为 是 正常的 out size
def visualize_feature_maps(feature_maps, save_path):
    """
    保存随机选择的 Feature Map 通道为图像文件
    :param feature_maps: 输入的 Feature Map（Tensor）
    :param save_path: 保存路径
    """
    # 将 Feature Map 转换为 NumPy 数组
    feature_maps = feature_maps.detach().cpu().numpy()
    
    # 随机选择一个通道
    num_channels = feature_maps.shape[1]  # 获取通道数
    random_channel = random.randint(0, num_channels - 1)  # 随机选择一个通道索引
    
    # 获取随机通道的 Feature Map
    single_channel = feature_maps[0, random_channel, :, :]  # 取 batch 中的第一个样本
    
    # 归一化到 [0, 255]
    single_channel = (single_channel - single_channel.min()) / (single_channel.max() - single_channel.min()) * 255
    single_channel = single_channel.astype(np.uint8)
    
    # 保存 Feature Map
    cv2.imwrite(f'{save_path}_random_channel2_{random_channel}.png', single_channel)


# def visualize_feature_maps(feature_maps, save_path, resize_to=(256, 256)):
#     """
#     保存随机选择的 Feature Map 通道为图像文件，并调整大小为 256x256
#     :param feature_maps: 输入的 Feature Map（Tensor）
#     :param save_path: 保存路径
#     :param resize_to: 调整大小后的尺寸，默认为 (256, 256)
#     """
#     # 将 Feature Map 转换为 NumPy 数组
#     feature_maps = feature_maps.detach().cpu().numpy()
    
#     # 随机选择一个通道
#     num_channels = feature_maps.shape[1]  # 获取通道数
#     random_channel = random.randint(0, num_channels - 1)  # 随机选择一个通道索引
    
#     # 获取随机通道的 Feature Map
#     single_channel = feature_maps[0, random_channel, :, :]  # 取 batch 中的第一个样本
    
#     # 归一化到 [0, 255]
#     single_channel = (single_channel - single_channel.min()) / (single_channel.max() - single_channel.min()) * 255
#     single_channel = single_channel.astype(np.uint8)
    
#     # 调整大小为 256x256
#     single_channel_resized = cv2.resize(single_channel, resize_to, interpolation=cv2.INTER_LINEAR)
    
#     # 保存调整大小后的 Feature Map
#     cv2.imwrite(f'{save_path}_random_channel_{random_channel}_resized.png', single_channel_resized)

def get_dataloader(base_dir, dataset_num=1, img_size=256):
    """
    加载数据集
    :param base_dir: 数据集根目录
    :param dataset_num: 数据集编号（例如 1）
    :param img_size: 图像尺寸（模型期望的输入尺寸）
    :return: DataLoader
    """
    val_transform = Compose([Resize(img_size, img_size), Normalize()])
    
    # 加载 poly 数据集的验证集
    db_val = MedicalDataSets(
        base_dir=base_dir,
        split="val",
        dataset_pre="poly",
        _ext=".jpg",
        transform=val_transform,
        train_file_dir=f"poly_train1.txt",
        val_file_dir=f"poly_val1.txt"
    )
    
    print(f"Val num: {len(db_val)}")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    return valloader

def main():
    # 加载模型
    import os
    model_path = '/data/hongboye/projects/checkpoint/LoRA__5/2024-12-28/225/poly_1_BS_4_0.8083_LoRA__5_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LoRA__5_FMap(dims=[24, 48, 96, 192,384]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载数据集
    base_dir = '/data/hongboye/projects/Tan9/data'  # 替换为你的数据集路径
    valloader = get_dataloader(base_dir, dataset_num=1, img_size=256)
    
    # 获取第 1 张图片
    for idx, data in enumerate(valloader):
        if idx == 0:  
            image = data['image']
            image = image.to(device)
            break
    
    # 前向传播，获取 Feature Map
    # with torch.no_grad():
    #     x1, x2, x3, x4, x5 = model(image)
    
    # # 可视化每一层的 Feature Map
    # os.makedirs("visual_fmap", exist_ok=True)
    # visualize_feature_maps(x1, save_path='./visual_fmap/x1_feature_map')
    # visualize_feature_maps(x2, save_path='./visual_fmap/x2_feature_map')
    # visualize_feature_maps(x3, save_path='./visual_fmap/x3_feature_map')
    # visualize_feature_maps(x4, save_path='./visual_fmap/x4_feature_map')
    # visualize_feature_maps(x5, save_path='./visual_fmap/x5_feature_map')
    
    with torch.no_grad():
        out1,out2 = model(image)
    print("out1.shape",out1.shape)
    # 可视化某一层的 Feature Map
    os.makedirs("visual_fmap", exist_ok=True)
    visualize_feature_maps(out1, save_path='visual_fmap')
    # visualize_feature_maps(out1, save_path='./visual_fmap/woBrwkv5_out1_feature_map')
    # visualize_feature_maps(out2, save_path='./visual_fmap/woBrwkv5_out2_feature_map')
    # visualize_feature_maps(x3, save_path='./visual_fmap/x3_feature_map')
    # visualize_feature_maps(x4, save_path='./visual_fmap/x4_feature_map')
    # visualize_feature_maps(x5, save_path='./visual_fmap/x5_feature_map')

if __name__ == '__main__':
    main()