def q_shift(input, shift_pixel=1, gamma=1/4, resolution=None,groups=8):
    """
    对 4D 张量 (NCHW) 进行分组八面移位操作。
    :param x: 输入张量，形状为 (N, C, H, W)
    :param groups: 分组数，默认为 8
    :return: 位移后的特征图，形状为 (N, C * 8, H, W)
    """
    N, C, H, W = x.shape
    assert C % groups == 0, "通道数必须能被分组数整除"
    channels_per_group = C // groups

    # 将输入张量按通道分组
    x_grouped = x.view(N, groups, channels_per_group, H, W)

    # 定义 8 种位移方向
import torch

def q_shift(input, shift_pixel=1, gamma=1/4, resolution=None, groups=8):
    """
    对 4D 张量 (NCHW) 进行分组八面移位操作。
    :param input: 输入张量，形状为 (N, C, H, W)
    :param groups: 分组数，默认为 8
    :return: 位移后的特征图，形状为 (N, C * 8, H, W)
    """
    N, C, H, W = input.shape
    assert C % groups == 0, "通道数必须能被分组数整除"
    channels_per_group = C // groups

    # 将输入张量按通道分组
    x_grouped = input.view(N, groups, channels_per_group, H, W)

    # 定义 8 种位移方向
    shifts = [
        (0, 0),    # 不位移
        (0, 1),    # 宽度方向右移
        (0, -1),   # 宽度方向左移
        (1, 0),    # 高度方向下移
        (-1, 0),   # 高度方向上移
        (1, 1),    # 右下方向位移
        (1, -1),   # 左下方向位移
        (-1, 1),   # 右上方向位移
        (-1, -1),  # 左上方向位移
    ]

    # 对每一组进行八面移位
    shifted_features = []
    for h_shift, w_shift in shifts:
        shifted_x = torch.roll(x_grouped, shifts=(h_shift, w_shift), dims=(-2, -1))
        shifted_features.append(shifted_x)

    # 将位移结果按通道维度拼接
    shifted_features = torch.cat(shifted_features, dim=2)  # (N, groups, C * 8, H, W)
    shifted_features = shifted_features.view(N, -1, H, W)  # (N, C * 8, H, W)

    return shifted_features

# 测试示例
if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 创建一个随机的 4D 张量 (N, C, H, W)
    N, C, H, W = 2, 16, 8, 8  # 批量大小为 2，通道数为 16，高度和宽度为 8
    input_tensor = torch.randn(N, C, H, W)

    print("输入张量形状:", input_tensor.shape)
    print("输入张量内容 (第一个样本的第一个通道):")
    print(input_tensor[0, 0])  # 打印第一个样本的第一个通道

    # 调用 q_shift 函数
    output_tensor = q_shift(input_tensor)

    print("\n输出张量形状:", output_tensor.shape)
    print("输出张量内容 (第一个样本的前 8 个通道):")
    print(output_tensor[0, :8])  # 打印第一个样本的前 8 个通道