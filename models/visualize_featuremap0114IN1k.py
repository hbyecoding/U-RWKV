import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
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
        # p5 = self.Brwkv_5(p5)
        # x5 = self.Brwkv(x5)
        # p5 = self.Brwkv(p5)
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        
        # x5_ = self.Brwkv_5(x5)
        # p5 = self.Brwkv_5(p5)
        # # x5 = self.Brwkv(x5)
        # # p5 = self.Brwkv(p5)
        # """ Decoder 1 """
        # # print(self.s1)
        # s1 = self.s1(p5, x5_)
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
# 这是不带resize的 因为
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
    cv2.imwrite(f'{save_path}_random_channel_{random_channel}.png', single_channel)


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
def get_input_grad(model, samples):
    outputs = model(samples)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map

"""
'/data/hongboye/dataset/IN1k' 
    root = os.path.join(data_path, 'val')
    sampler_val = torch.utils.data.SequentialSampler(dataset)
    dataset = datasets.ImageFolder(root, transform=transform)
    data_loader_val = torch.utils.data.DataLoader(dataset, sampler=sampler_val,
        batch_size=1, num_workers=1, pin_memory=True, drop_last=False)


"""

def get_dataloader(data_path = '/data/hongboye/dataset/IN1k' ):
    #  dataset_num=1, img_size=256
    import os
    from torch.utils.data import DataLoader
    from src.dataloader.dataset import MedicalDataSets
    from albumentations.core.composition import Compose
    from albumentations import Resize, Normalize
    # from torchvision import transforms as transforms
    from torchvision import datasets, transforms
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from PIL import Image
    
    root = os.path.join(data_path, 'val')
    t = [
        transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    transform = transforms.Compose(t)
    dataset = datasets.ImageFolder(root, transform=transform)
    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader_val = torch.utils.data.DataLoader(dataset, sampler=sampler_val,
        batch_size=1, num_workers=1, pin_memory=True, drop_last=False)
    # dataset = datasets.ImageFolder(root, transform=transform)   /
    return data_loader_val     


# def get_dataloader(base_dir='/data/hongboye/projects/Tan9/data', dataset_num=1, img_size=256):
#     """
#     加载数据集
#     :param base_dir: 数据集根目录
#     :param dataset_num: 数据集编号（例如 1）
#     :param img_size: 图像尺寸（模型期望的输入尺寸）
#     :return: DataLoader
#     """
#     from torch.utils.data import DataLoader
#     from src.dataloader.dataset import MedicalDataSets
#     from albumentations.core.composition import Compose
#     from albumentations import Resize, Normalize
#     val_transform = Compose([Resize(img_size, img_size), Normalize()])
    
#     # 加载 poly 数据集的验证集
#     db_val = MedicalDataSets(
#         base_dir=base_dir,
#         split="val",
#         dataset_pre="poly",
#         _ext=".jpg",
#         transform=val_transform,
#         train_file_dir=f"poly_train1.txt",
#         val_file_dir=f"poly_val1.txt"
#     )
    
#     print(f"Val num: {len(db_val)}")
#     valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
#     return valloader

def main():
    # 加载模型
    import os
    from torch import optim as optim
    from timm.utils import AverageMeter
    
    model_path = '/data/hongboye/projects/checkpoint/LoRA__5/2024-12-28/225/poly_1_BS_4_0.8083_LoRA__5_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LoRA__5_FMap(dims=[24, 48, 96, 192,384]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    save_root = "visual_save_npy"
    save_path = "./visual_save_npy/temp.npy"
    os.makedirs(save_root,exist_ok=True)
    # # 加载数据集
    # base_dir = '/data/hongboye/projects/Tan9/data'  # 替换为你的数据集路径
    # valloader = get_dataloader(base_dir, dataset_num=1, img_size=256)
    
    
    # # 获取第 1 张图片
    # for idx, data in enumerate(valloader):
    #     if idx == 0:  
    #         image = data['image']
    #         image = image.to(device)
    #         break
    data_loader_val = get_dataloader()
    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()

    for _, (samples, _) in enumerate(data_loader_val):

        if meter.count == 50: #50args.num_images
            np.save(save_path, meter.avg)
            exit()

        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            print('accumulate')
            meter.update(contribution_scores)    
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
    
    # 可视化某一层的 Feature Map
    os.makedirs("visual_fmap", exist_ok=True)
    visualize_feature_maps(out1, save_path='./visual_fmap/woBrwkv5_out1_feature_map')
    visualize_feature_maps(out2, save_path='./visual_fmap/woBrwkv5_out2_feature_map')
    # visualize_feature_maps(x3, save_path='./visual_fmap/x3_feature_map')
    # visualize_feature_maps(x4, save_path='./visual_fmap/x4_feature_map')
    # visualize_feature_maps(x5, save_path='./visual_fmap/x5_feature_map')

if __name__ == '__main__':
    main()