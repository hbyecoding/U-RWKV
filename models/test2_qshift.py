import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from test_q_shift_global_SparseAttn import q_shift_with_global_sparse_attn  # 假设 q_shift 函数在 q_shift.py 中定义

# 定义 MedicalDataSets 类
class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            train_file_dir="busi_train1.txt",
            val_file_dir="busi_val1.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                print("PATH", train_file_dir, os.path.join(self._base_dir, train_file_dir))
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        # 读取图像和标签
        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
        label = cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '.png'), cv2.IMREAD_GRAYSCALE)

        # 将标签扩展为 3 通道
        label = label[..., None]

        # 数据增强
        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        # 归一化
        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        sample = {"image": image, "label": label, "idx": idx}
        return sample


# 测试函数
def test_q_shift_on_dataset(base_dir, train_file_dir, val_file_dir, transform):
    # 创建数据集
    db_train = MedicalDataSets(base_dir=base_dir, split="train", transform=transform, train_file_dir=train_file_dir, val_file_dir=val_file_dir)

    # 从数据集中取出一张图片
    sample = db_train[0]  # 取出第一张图片
    image = sample["image"]  # 取出图像
    label = sample["label"]  # 取出标签

    # 将图像转换为适合 q_shift 的格式
    image = image.transpose(1, 2, 0)  # 从 (C, H, W) 转换为 (H, W, C)
    image = (image * 255).astype(np.uint8)  # 反归一化到 [0, 255]

    # 将图像转换为 PyTorch 张量
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # 添加批次维度 (B, H, W, C)
    image_tensor = image_tensor.permute(0, 3, 1, 2)  # 转换为 (B, C, H, W)

    # 应用 q_shift
    output_tensor = q_shift_with_global_sparse_attn(image_tensor)

    # 将输出转换回图像格式
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)
    output_image = (output_image * 255).astype(np.uint8)  # 反归一化到 [0, 255]

    # 返回原始图像和处理后的图像
    return image, output_image


# 示例调用
if __name__ == "__main__":
    # 定义参数
    base_dir = "./data/busi"
    train_file_dir = "busi_train.txt"
    val_file_dir = "busi_val.txt"
    transform = None  # 这里可以传入数据增强的 transform

    # 测试 q_shift 在数据集上的效果
    original_image, processed_image = test_q_shift_on_dataset(base_dir, train_file_dir, val_file_dir, transform)

    # 保存图像
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(processed_image)
    plt.savefig("results/q_shift_result.png")
    plt.close()