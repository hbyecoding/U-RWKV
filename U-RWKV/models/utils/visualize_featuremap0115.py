import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize, Normalize
from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.compnet1225_enc_rwkv import LoRA__5

class LoRA__5_FMap(LoRA__5):
    def __init__(self, input_channel=3, num_classes=1, dims=[24, 48, 96, 192, 384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(LoRA__5_FMap, self).__init__(input_channel=3, num_classes=1, dims=[24, 48, 96, 192, 384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5)
        
    def forward(self, x):
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x5, p5 = self.e5(p4)
        x5_ = self.Brwkv_5(x5)
        p5 = self.Brwkv_5(p5)
        s1 = self.s1(p5, x5_)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        return out1, out2
    
class LoRA__5_FMap_woBrwkv(LoRA__5):
    def __init__(self, input_channel=3, num_classes=1, dims=[24, 48, 96, 192, 384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(LoRA__5_FMap_woBrwkv, self).__init__(input_channel=3, num_classes=1, dims=[24, 48, 96, 192, 384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5)
        
    def forward(self, x):
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x5, p5 = self.e5(p4)
        x5_ = self.Brwkv_5(x5)
        p5_ = self.Brwkv_5(p5)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        return out1, out2

def visualize_feature_maps(feature_maps, save_path):
    """
    保存 Feature Map 为图像文件
    :param feature_maps: 输入的 Feature Map（Tensor）
    :param save_path: 保存路径
    """
    feature_maps = feature_maps.detach().cpu().numpy()
    single_channel = feature_maps[0, 0, :, :]  # 取第一个样本的第一个通道
    single_channel = (single_channel - single_channel.min()) / (single_channel.max() - single_channel.min()) * 255
    single_channel = single_channel.astype(np.uint8)
    single_channel[single_channel > 127] = 255
    single_channel[single_channel <= 127] = 0
    cv2.imwrite(save_path, single_channel)

def get_dataloader(base_dir, dataset_num=1, img_size=256):
    """
    加载数据集
    :param base_dir: 数据集根目录
    :param dataset_num: 数据集编号（例如 1）
    :param img_size: 图像尺寸（模型期望的输入尺寸）
    :return: DataLoader
    """
    val_transform = Compose([Resize(img_size, img_size), Normalize()])
    db_val = MedicalDataSets(
        base_dir=base_dir,
        split="val",
        dataset_pre="poly",
        _ext=".jpg",
        transform=val_transform,
        train_file_dir=f"poly_train1.txt",
        val_file_dir=f"poly_val1.txt"
    )
    # print(f"Val num: {len(db_val)}")
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    return valloader

def main():
    # 加载数据集
    base_dir = '/data/hongboye/projects/Tan9/data'
    # val_transform = Compose([Resize(256, 256)])  # 只进行 Resize
    # val_dataset = MedicalDataSets(base_dir=base_dir, split="val", transform=val_transform)
    # valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    valloader = get_dataloader(base_dir=base_dir)
    # 加载多个模型
    models = {
        'LoRA__5': LoRA__5_FMap(dims=[24, 48, 96, 192, 384]).to('cuda'),
        # 添加其他模型
        'LoRA__5_woBrwkv':LoRA__5_FMap_woBrwkv(dims=[24, 48, 96, 192, 384]).to('cuda')
    }
    for model_name, model in models.items():
        model_path = f'/data/hongboye/projects/checkpoint/LoRA__5/2024-12-28/225/poly_1_BS_4_0.8083_LoRA__5_model.pth'
        # model_path = f'/data/hongboye/projects/checkpoint/{model_name}/2024-12-28/225/poly_1_BS_4_0.8083_{model_name}_model.pth'
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
        model.eval()
        
    # 遍历数据集
    for idx, data in enumerate(valloader):
        if idx >= 5:  # 只处理前 5 张图像
            break

        # 获取 Resize 后的图像和 GT（Tensor）
        image_resized = data['image'].to('cuda')
        label_resized = data['label'].to('cuda')

        # 获取原始图像和 GT 的路径
        image_path = data['image_path'][0]
        label_path = data['label_path'][0]

        # 读取原始图像和 GT
        original_image = cv2.imread(image_path)
        original_gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # Resize 原始图像和 GT（如果需要）
        resized_image = cv2.resize(original_image, (256, 256))
        resized_gt = cv2.resize(original_gt, (256, 256))

        # 保存 Resize 后的图像和 GT
        os.makedirs("visual_results", exist_ok=True)
        cv2.imwrite(f'visual_results/{idx}_input.png', resized_image)
        cv2.imwrite(f'visual_results/{idx}_gt.png', resized_gt)

        # 将 Resize 后的图像转换为 Tensor 并送入模型
        image_tensor = torch.from_numpy(resized_image.transpose(2, 0, 1)).float().unsqueeze(0).to('cuda') / 255.0

        # 对模型进行推理并保存结果
        for model_name, model in models.items():
 
            with torch.no_grad():
                out1, out2 = model(image_resized)
            visualize_feature_maps(out1, f'visual_results/{idx}_{model_name}_out1.png')
            visualize_feature_maps(out2, f'visual_results/{idx}_{model_name}_out2.png')

if __name__ == '__main__':
    main()