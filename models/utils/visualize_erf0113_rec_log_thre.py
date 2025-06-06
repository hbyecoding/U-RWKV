import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到 sys.path
sys.path.append('/data/hongboye/projects/Tan9')

# 导入模型
from src.network.conv_based.compnet1225_enc_rwkv_ERF import LoRA__5
from src.network.conv_based.cval_LoRA5_enc_rwkv import v_enc_256_fffse_dec_fusion_rwkv_with2x4

#/data/hongboye/projects/checkpoint2025/v_enc_768_fffse_dec_fusion_rwkv_with2x4/2025-02-19/262/busi_3_busi_train.txt_BS_4_0.7072_v_enc_768_fffse_dec_fusion_rwkv_with2x4_model.pth
# class LoRA__5ForERF(LoRA__5):
#     def __init__(self, input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
#         super(LoRA__5ForERF, self).__init__(input_channel=3, num_classes=1, dims=[24,48,96,192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5)
        
#     def forward(self,x):
#         x1, p1 = self.e1(x)
#         x2, p2 = self.e2(p1)
#         x3, p3 = self.e3(p2)
#         x4, p4 = self.e4(p3)
#         # x4 = self.Brwkv_4(x4)
#         # p4 = self.Brwkv_4(p4)
#         x5, p5 = self.e5(p4)
        
        
        
#         x5_ = self.Brwkv_5(x5)
#         p5_ = self.Brwkv_5(p5)
#         # x5 = self.Brwkv(x5)
#         # p5 = self.Brwkv(p5)
#         """ Decoder 1 """
#         # print(self.s1)
#         s1 = self.s1(p5, x5)
#         a1 = self.a1(p5, x5)
#         m1 = self.m1(a1)
#         x6 = s1 * m1

#         """ Decoder 2 """
#         s2 = self.s2(x6, x4)
#         a2 = self.a2(a1, x4)
#         m2 = self.m2(a2)
#         x7 = s2 * m2

#         """ Decoder 3 """
#         s3 = self.s3(x7, x3)
#         a3 = self.a3(a2, x3)
#         m3 = self.m3(a3)
#         x8 = s3 * m3

#         """ Decoder 4 """
#         s4 = self.s4(x8, x2)
#         a4 = self.a4(a3, x2)
#         m4 = self.m4(a4)
#         x9 = s4 * m4

#         """ Decoder 5 """
#         s5 = self.s5(x9, x1)
#         a5 = self.a5(a4, x1)
#         m5 = self.m5(a5)
#         x10 = s5 * m5

#         """ Output """
#         out1 = self.output1(x10)
#         out2 = self.output2(a5)
#         # print("pause")
#         # return out1, out2        
#         return x5,out1, out2 
    
# 设置图像参数
large = 24; med = 24; small = 24
params = {
    'axes.titlesize': large,
    'legend.fontsize': med,
    'figure.figsize': (16, 10),
    'axes.labelsize': med,
    'xtick.labelsize': med,
    'ytick.labelsize': med,
    'figure.titlesize': large
}
plt.rcParams.update(params)
# plt.style.use('seaborn-v0_8')  # 使用 seaborn 样式
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_style("white")
plt.rcParams['axes.unicode_minus'] = False

def load_model(model_path, device='cuda'):
    model = v_enc_256_fffse_dec_fusion_rwkv_with2x4()  # 初始化模型dims = [48,96,192,384,768]
    state_dict = torch.load(model_path, map_location=device)  # 加载权重
    model.load_state_dict(state_dict)  # 加载权重到模型
    model.to(device)  # 将模型移动到指定设备
    model.eval()  # 设置为评估模式
    return model

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


def get_dataloader(base_dir='/data/hongboye/projects/Tan9/data', dataset_num=3, img_size=256):
    """
    加载数据集
    :param base_dir: 数据集根目录
    :param dataset_num: 数据集编号（例如 1）
    :param img_size: 图像尺寸（模型期望的输入尺寸）
    :return: DataLoader
    """
    from torch.utils.data import DataLoader
    from src.dataloader.dataset import MedicalDataSets
    from albumentations.core.composition import Compose
    from albumentations import Resize, Normalize
    val_transform = Compose([Resize(img_size, img_size), Normalize()])
    
    # 加载 isic 数据集的验证集
    db_val = MedicalDataSets(
        base_dir=base_dir,
        split="val",
        dataset_pre="isic18",
        _ext=".png",
        transform=val_transform,
        train_file_dir="isic18_train"+dataset_num+".txt",
        val_file_dir="isic18_val"+dataset_num+".txt"
    )
    
    print(f"Val num: {len(db_val)}")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    return valloader


def compute_erf(model, input_size=(256, 256), device='cuda'):
    model.eval()
    model.to(device)
    
    # # 创建一个全零的输入图像
    # input_img = torch.zeros((1, 3, input_size[0], input_size[1]), device=device)
    # input_img.requires_grad = True
    
    
    valloader = get_dataloader(base_dir='/data/hongboye/projects/Tan9/data', dataset_num="3", img_size=256)
    
    # 获取第 1 张图片
    for idx, data in enumerate(valloader):
        if idx == 4:  
            image = data['image']
            input_img = image.to(device)
            input_img.requires_grad = True
            break    
    # 前向传播，获取输出
    output = model(input_img)
    
    # 选择一个输出点（通常是中心点）
    output_point = output[0, :, output.shape[2] // 2, output.shape[3] // 2].sum()
    
    # 反向传播计算梯度
    output_point.backward()
    
    # 获取输入图像的梯度
    grad = input_img.grad.data.abs().sum(dim=1).squeeze().cpu().numpy()
    
    return grad

def get_rectangle(data, thresh):
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, h//2):
        selected_area = data[h//2 - i:h//2 +1 +i, w//2 - i:w//2 +1 +i]
        area_sum=np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
    return None

 

def heatmap(data, save_path, camp='RdYlGn', figsize=(10, 10.75)):
    plt.figure(figsize=figsize, dpi=40)
    ax = sns.heatmap(data,
                     xticklabels=False,
                     yticklabels=False, 
                     cmap=camp,  # 使用绿色系颜色映射
                     center=0, 
                     annot=False, 
                     cbar=False, 
                     annot_kws={"size": 24}, 
                     fmt='.2f')
    
    # 添加颜色条
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from matplotlib.colorbar import Colorbar 
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('top', size='5%', pad='2%')
    cbar = plt.colorbar(ax.get_children()[0], cax=cax, orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # 模型权重路径
    # model_path = 'checkpoint/compnet_/2024-12-22/249/poly_1_BS_8_0.7960_compnet__model.pth'
    # model_path = '/data/hongboye/projects/checkpoint/LoRA__5/2024-12-28/225/poly_1_BS_4_0.8083_LoRA__5_model.pth'
    # model_path = '/data/hongboye/projects/checkpoint2025/v_enc_768_fffse_dec_fusion_rwkv_with2x4/2025-02-19/262/busi_3_busi_train.txt_BS_4_0.7072_v_enc_768_fffse_dec_fusion_rwkv_with2x4_model.pth'
    # model_path = '/data/hongboye/projects/checkpoint2025/v_enc_768_fffse_dec_fusion_rwkv_with2x4/2025-02-19/275/clinicDB_4_BS_4_0.8543_v_enc_768_fffse_dec_fusion_rwkv_with2x4_model.pth'
    model_path = '/data/hongboye/projects/checkpoint2025/v_enc_256_fffse_dec_fusion_rwkv_with2x4/2025-02-15/244/isic18_3_BS_16_0.8227_v_enc_256_fffse_dec_fusion_rwkv_with2x4_model.pth'
    save_path = "visual_erf256rwkv0223"
    # 加载模型（带 Brwkv 模块）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_with_brwkv = load_model(model_path, device=device) #, use_brwkv=True
    
    # 计算带 Brwkv 模块的 ERF
    erf_with_brwkv = compute_erf(model_with_brwkv, input_size=(1024, 1024), device=device)
    
    # erf log norm
    erf_with_brwkv = np.log10(erf_with_brwkv + 1)
    erf_with_brwkv = erf_with_brwkv / np.max(erf_with_brwkv)
    
    print('======================= the high-contribution area ratio =====================')
    for thresh in [0.2, 0.3, 0.5, 0.99]:
        side_length, area_ratio = get_rectangle(erf_with_brwkv, thresh)
        print(f"thresh:{thresh}, rectangle side length {side_length}, area_ratio :{area_ratio}")
    
    # 保存带 Brwkv 模块的 ERF 图像（绿色背景）
    os.makedirs(save_path, exist_ok=True)
    heatmap(erf_with_brwkv, save_path=f'./{save_path}/PolyLoRA5x5_Brwkv_5_erf.png')# , camp='Greens'
    
    # # 加载模型（不带 Brwkv 模块）
    # model_without_brwkv = load_model(model_path, device=device, use_brwkv=False)
    
    # # 计算不带 Brwkv 模块的 ERF
    # erf_without_brwkv = compute_erf(model_without_brwkv, input_size=(1024, 1024), device=device)
    
    # # 保存不带 Brwkv 模块的 ERF 图像（绿色背景）
    # heatmap(erf_without_brwkv, save_path='LoRA5_woBrwkv_erf.png', camp='Greens')
# def main():
#     # 模型权重路径
#     model_path = '/data/hongboye/projects/checkpoint/LoRA__5/2024-12-28/225/poly_1_BS_4_0.8083_LoRA__5_model.pth'
    
#     # 加载模型（带 Brwkv 模块）
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model_with_brwkv = load_model(model_path, device=device) #, use_brwkv=True
    
#     # 计算带 Brwkv 模块的 ERF
#     erf_with_brwkv = compute_erf(model_with_brwkv, input_size=(1024, 1024), device=device)
    
#     # 保存带 Brwkv 模块的 ERF 图像（绿色背景）
#     heatmap(erf_with_brwkv, save_path='GLoRA5_wo_Brwkv_erf.png', camp='Greens')
    
#     # # 加载模型（不带 Brwkv 模块）
#     # model_without_brwkv = load_model(model_path, device=device, use_brwkv=False)
    
#     # # 计算不带 Brwkv 模块的 ERF
#     # erf_without_brwkv = compute_erf(model_without_brwkv, input_size=(1024, 1024), device=device)
    
#     # # 保存不带 Brwkv 模块的 ERF 图像（绿色背景）
#     # heatmap(erf_without_brwkv, save_path='LoRA5_woBrwkv_erf.png', camp='Greens')

if __name__ == '__main__':
    main()






# def compute_erf(model, input_size=(1024, 1024), device='cuda'):
#     model.eval()
#     model.to(device)
    
#     # 创建一个全零的输入图像
#     input_img = torch.zeros((1, 3, input_size[0], input_size[1]), device=device)
#     input_img.requires_grad = True
    
#     # 前向传播，获取输出
#     output = model(input_img)
    
#     # 选择一个输出点（通常是中心点）
#     output_point = output[0, :, output.shape[2] // 2, output.shape[3] // 2].sum()
    
#     # 反向传播计算梯度
#     output_point.backward()
    
#     # 获取输入图像的梯度
#     grad = input_img.grad.data.abs().sum(dim=1).squeeze().cpu().numpy()
    
#     return grad

# def plot_erf(erf, save_path, title='Effective Receptive Field'):
#     from matplotlib import pyplot as plt
#     plt.figure(figsize=(10, 10))
#     plt.imshow(erf, cmap='hot', interpolation='nearest')
#     plt.colorbar(label='Gradient Magnitude')
#     plt.title(title)
#     plt.axis('off')  # 不显示坐标轴
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 保存图像
#     plt.close()  # 关闭图像，避免内存泄漏

# # def main about opti  batch loss(criterion)

# # def main(args):
# #     import torch.nn as nn
# #     from torch import optim as optim
    
# #     model = load_model()
# #     meter = AverageMeter()
# #     optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)
    
# #     criterion = nn.BCEWithLogitsLoss()
# #     valloader = get_valloader()
# #     for samples in valloader:
        
   