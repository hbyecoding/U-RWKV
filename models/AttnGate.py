import torch
import torch.nn as nn
from thop import profile, clever_format

class MSAG(nn.Module):
    """
    Multi-scale attention gate
    """
    def __init__(self, channel):
        super(MSAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 3, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        _x = self.relu(torch.cat((x1, x2, x3), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x


# class MSAG(nn.Module):
#     """
#     Multi-scale attention gate
#     """
#     def __init__(self, channel):
#         super(MSAG, self).__init__()
#         self.channel = channel
#         assert self.channel % 3 == 0, "Channel must be divisible by 3"
        
#         # 普通 3x3 卷积
#         self.conv_3x3 = nn.Sequential(
#             nn.Conv2d(self.channel // 3, self.channel // 3, kernel_size=3, padding=1, stride=1, bias=True),
#             nn.BatchNorm2d(self.channel // 3),
#         )
        
#         # 水平方向的 1x11 卷积
#         self.conv_h = nn.Sequential(
#             nn.Conv2d(self.channel // 3, self.channel // 3, kernel_size=(1, 11), padding=(0, 5), stride=1, bias=True),
#             nn.BatchNorm2d(self.channel // 3),
#         )
        
#         # 垂直方向的 11x1 卷积
#         self.conv_v = nn.Sequential(
#             nn.Conv2d(self.channel // 3, self.channel // 3, kernel_size=(11, 1), padding=(5, 0), stride=1, bias=True),
#             nn.BatchNorm2d(self.channel // 3),
#         )
        
#         # Vote 机制
#         self.voteConv = nn.Sequential(
#             nn.Conv2d(self.channel, self.channel, kernel_size=(1, 1)),
#             nn.BatchNorm2d(self.channel),
#             nn.Sigmoid()
#         )
        
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         # 将输入通道分成三部分
#         x1, x2, x3 = torch.split(x, self.channel // 3, dim=1)
        
#         # 分别通过不同的卷积操作
#         x1 = self.conv_3x3(x1)  # 普通 3x3 卷积
#         x2 = self.conv_h(x2)    # 水平方向的 1x11 卷积
#         x3 = self.conv_v(x3)    # 垂直方向的 11x1 卷积
        
#         # 将三个分支的结果拼接
#         _x = self.relu(torch.cat((x1, x2, x3), dim=1))
        
#         # 通过 Vote 机制生成注意力权重
#         _x = self.voteConv(_x)
        
#         # 残差连接
#         x = x + x * _x
#         return x

if __name__ == "__main__":
    # 定义输入参数
    batch_size = 1  # 批量大小
    channel = 384  # 输入通道数
    height = 256  # 输入特征图高度
    width = 256  # 输入特征图宽度

    # 创建输入张量
    x = torch.randn(batch_size, channel, height, width).cuda()

    # 创建 MSAG 模型
    model = MSAG(channel=channel).cuda()

    # 打印模型结构
    print(model)

    # 前向传播测试
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format((flops, params), "%.2f")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")