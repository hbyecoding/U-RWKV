import torch
import torch.nn as nn

class ProProcessWithDWConv(nn.Module):
    def __init__(self, ch_in, ch_out, depth=2, k=3):
        """
        Args:
            ch_in (int): 输入通道数
            ch_out (int): 输出通道数
            depth (int): 重复的深度（下采样次数）
            k (int): 深度可分离卷积的核大小
        """
        super(ProProcessWithDWConv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth = depth
        self.k = k

        # 定义下采样模块
        self.downsample_layers = nn.Sequential(
            *[self._make_downsample_layer(ch_in if i == 0 else ch_out) for i in range(depth)]
        )

    def _make_downsample_layer(self, ch):
        """
        创建一个下采样层，包含深度可分离卷积和逐点卷积
        """
        return nn.Sequential(
            # 深度可分离卷积
            nn.Conv2d(ch, ch, kernel_size=(self.k, self.k), stride=2, padding=(self.k // 2, self.k // 2), groups=ch),
            nn.GELU(),
            nn.BatchNorm2d(ch),
            # 逐点卷积
            nn.Conv2d(ch, self.ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(self.ch_out)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (BS, C, H, W)
        Returns:
            torch.Tensor: 输出张量，形状为 (BS, C, H//(2^depth), W//(2^depth))
        """
        return self.downsample_layers(x)


# 测试用例 1
def test_case_1():
    # 输入参数
    BS = 1
    C_in = 96
    H, W = 64, 64
    C_out = 384
    depth = 2  # 下采样两次，64 -> 32 -> 16

    # 初始化模块
    model = ProProcessWithDWConv(ch_in=C_in, ch_out=C_out, depth=depth)

    # 输入张量
    x = torch.randn(BS, C_in, H, W)

    # 前向传播
    output = model(x)

    # 打印结果
    print(f"Test Case 1:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")


# 测试用例 2
def test_case_2():
    # 输入参数
    BS = 1
    C_in = 192
    H, W = 32, 32
    C_out = 384
    depth = 1  # 下采样一次，32 -> 16

    # 初始化模块
    model = ProProcessWithDWConv(ch_in=C_in, ch_out=C_out, depth=depth)

    # 输入张量
    x = torch.randn(BS, C_in, H, W)

    # 前向传播
    output = model(x)

    # 打印结果
    print(f"Test Case 2:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")


# 运行测试
if __name__ == "__main__":
    test_case_1()
    test_case_2()