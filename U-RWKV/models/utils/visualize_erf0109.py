import os
import argparse
import numpy as np
import torch
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image

# from src.network.conv_based.compnet1225_enc_rwkv import LoRA_4_5
from src.network.conv_based.compnet1225_enc_rwkv_ERF import LoRA__5
# class LoRA_4_5ForERF(LoRA_4_5):
#     def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
#         super(LoRA_4_5ForERF, self).__init__(input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5)
        
#     def forward(self,x):
#         x1, p1 = self.e1(x)
#         x2, p2 = self.e2(p1)
#         x3, p3 = self.e3(p2)
#         x4, p4 = self.e4(p3)
#         x4 = self.Brwkv_4(x4)
#         p4 = self.Brwkv_4(p4)
#         x5, p5 = self.e5(p4)
        
#         x5 = self.Brwkv_5(x5)
#         p5 = self.Brwkv_5(p5)        
        
#         return x5
    
class LoRA__5ForERF(LoRA__5):
    def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(LoRA__5ForERF, self).__init__(input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5)
        
    def forward(self,x):
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        # x4 = self.Brwkv_4(x4)
        # p4 = self.Brwkv_4(p4)
        x5, p5 = self.e5(p4)
        
        x5 = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)        
        
        return x5

 
# def load_model(model_path, args, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
#     # if 
#     if args.model.startswith("LoRA_4_5"):
#         model = LoRA_4_5ForERF()
        
#     # 加载模型权重
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)

#     # 将模型移动到指定设备
#     model.to(device)
#     model.eval()  # 设置为评估模式
#     return model

def load_model(model_path, device='cuda'):
    model = LoRA__5(dims=[24, 48, 96, 192, 384])  # 初始化模型
    state_dict = torch.load(model_path, map_location=device)  # 加载权重
    model.load_state_dict(state_dict)  # 加载权重到模型
    model.to(device)  # 将模型移动到指定设备
    model.eval()  # 设置为评估模式
    return model

# 加载数据集  IN1k 也是可以， 故：

def get_valloader(args):
    t = [
        transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    transform = transforms.Compose(t)

    print("reading from datapath", args.data_path)
    root = os.path.join(args.data_path, 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    # nori_root = os.path.join('/home/dingxiaohan/ndp/', 'imagenet.val.nori.list')
    # from nori_dataset import ImageNetNoriDataset      # Data source on our machines. You will never need it.
    # dataset = ImageNetNoriDataset(nori_root, transform=transform)

    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader_val = torch.utils.data.DataLoader(dataset, sampler=sampler_val,
        batch_size=1, num_workers=1, pin_memory=True, drop_last=False)
    
    return data_loader_val


def compute_erf(model, input_size=(1024, 1024), device='cuda'):
    model.eval()
    model.to(device)
    
    # 创建一个全零的输入图像
    input_img = torch.zeros((1, 3, input_size[0], input_size[1]), device=device)
    input_img.requires_grad = True
    
    # 前向传播，获取输出
    output = model(input_img)
    
    # 选择一个输出点（通常是中心点）
    output_point = output[0, :, output.shape[2] // 2, output.shape[3] // 2].sum()
    
    # 反向传播计算梯度
    output_point.backward()
    
    # 获取输入图像的梯度
    grad = input_img.grad.data.abs().sum(dim=1).squeeze().cpu().numpy()
    
    return grad

def plot_erf(erf, save_path, title='Effective Receptive Field'):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(erf, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Gradient Magnitude')
    plt.title(title)
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 保存图像
    plt.close()  # 关闭图像，避免内存泄漏

# def main about opti  batch loss(criterion)

# def main(args):
#     import torch.nn as nn
#     from torch import optim as optim
    
#     model = load_model()
#     meter = AverageMeter()
#     optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)
    
#     criterion = nn.BCEWithLogitsLoss()
#     valloader = get_valloader()
#     for samples in valloader:
        
        
def main():
    # 模型权重路径
    model_path = '/data/hongboye/projects/checkpoint/LoRA__5/2024-12-28/225/poly_1_BS_4_0.8083_LoRA__5_model.pth'
    
    # 加载模型（带 Brwkv 模块）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_with_brwkv = load_model(model_path, device=device) #, use_brwkv=True
    
    # 计算带 Brwkv 模块的 ERF
    erf_with_brwkv = compute_erf(model_with_brwkv, input_size=(1024, 1024), device=device)
    
    # 保存带 Brwkv 模块的 ERF 图像
    plot_erf(erf_with_brwkv, save_path='LoRA5_Brwkv_erf.png', title='Effective Receptive Field of LoRA5 with Brwkv')
    
    # # 加载模型（不带 Brwkv 模块）
    # model_without_brwkv = load_model(model_path, device=device)
    
    # # 计算不带 Brwkv 模块的 ERF
    # erf_without_brwkv = compute_erf(model_without_brwkv, input_size=(1024, 1024), device=device)
    
    # # 保存不带 Brwkv 模块的 ERF 图像
    # plot_erf(erf_without_brwkv, save_path='LoRA5_woBrwkv_erf.png', title='Effective Receptive Field of LoRA5 without Brwkv')

if __name__ == '__main__':
    main()        
