import torch
import torch.nn as nn

class GSC(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()

        # 第一个卷积块
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(in_channels)
        self.nonlinear = nn.ReLU()

        # 第二个卷积块
        self.proj2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.nonlinear2 = nn.ReLU()

        # 门控卷积
        self.gate_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.gate_norm = nn.InstanceNorm2d(in_channels)
        self.gate_activation = nn.Sigmoid()  # 使用Sigmoid作为门控激活函数

        # 最后的卷积块
        self.proj3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.InstanceNorm2d(in_channels)
        self.nonlinear3 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        # 第一个卷积块
        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonlinear(x1)

        # 第二个卷积块
        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonlinear2(x1)

        # 门控机制
        gate = self.gate_conv(x)
        gate = self.gate_norm(gate)
        gate = self.gate_activation(gate)

        # 应用门控
        x1 = x1 * gate

        # 最后的卷积块
        x = self.proj3(x1)
        x = self.norm3(x)
        x = self.nonlinear3(x)

        # 残差连接
        return x + x_residual

# 测试代码
if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)  # 假设输入是1个样本，64个通道，32x32的空间尺寸
    gsc = GSC(in_channels=64)
    output = gsc(x)
    print(output.shape)  # 输出形状应该与输入形状相同