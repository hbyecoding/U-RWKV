# import torch
# from thop import profile


# def q_shift(input, shift_pixel=1, gamma=1/4, resolution=None):
#     assert gamma <= 1/4
#     B, N, C = input.shape
#     if resolution is None:
#         sqrt_N = int(N ** 0.5)
#         if sqrt_N * sqrt_N == N:
#             resolution = (sqrt_N, sqrt_N)
#         else:
#             raise ValueError("Cannot infer a valid resolution for the given input shape.")

#     input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
#     input = input
#     B, C, H, W = input.shape
#     output = torch.zeros_like(input)
#     output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
#     output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
#     output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
#     output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
#     output[:, int(C*gamma*4):,...] = input[:, int(C*gamma*4):,...]
#     return output.flatten(2).transpose(1, 2)


# # 定义输入张量，根据你的需求修改形状和数据类型
# input_tensor = torch.randn(1, 16, 64)  

# # 调用 q_shift 函数
# output_tensor = q_shift(input_tensor)

# # 使用 profile 函数计算 FLOPs 和 Params
# flops, params = profile(q_shift, inputs=(input_tensor,))
# print(f"FLOPs: {flops}")
# print(f"Params: {params}")

# import torch
# from torch import nn
# from thop import profile


# class QShiftModule(nn.Module):
#     def __init__(self, shift_pixel=1, gamma=1/4):
#         super().__init__()
#         self.shift_pixel = shift_pixel
#         self.gamma = gamma

#     def forward(self, input):
#         assert self.gamma <= 1/4
#         B, N, C = input.shape
#         sqrt_N = int(N ** 0.5)
#         if sqrt_N * sqrt_N == N:
#             resolution = (sqrt_N, sqrt_N)
#         else:
#             raise ValueError("Cannot infer a valid resolution for the given input shape.")

#         input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
#         B, C, H, W = input.shape
#         output = torch.zeros_like(input)
#         output[:, 0:int(C*self.gamma), :, self.shift_pixel:W] = input[:, 0:int(C*self.gamma), :, 0:W-self.shift_pixel]
#         output[:, int(C*self.gamma):int(C*self.gamma*2), :, 0:W-self.shift_pixel] = input[:, int(C*self.gamma):int(C*self.gamma*2), :, self.shift_pixel:W]
#         output[:, int(C*self.gamma*2):int(C*self.gamma*3), self.shift_pixel:H, :] = input[:, int(C*self.gamma*2):int(C*self.gamma*3), 0:H-self.shift_pixel, :]
#         output[:, int(C*self.gamma*3):int(C*self.gamma*4), 0:H-self.shift_pixel, :] = input[:, int(C*self.gamma*3):int(C*self.gamma*4), self.shift_pixel:H, :]
#         output[:, int(C*self.gamma*4):,...] = input[:, int(C*self.gamma*4):,...]
#         return output.flatten(2).transpose(1, 2)


# 定义输入张量，根据你的需求修改形状和数据类型
# input_tensor = torch.randn(1, 16, 64)  

# # 创建 QShiftModule 实例
# q_shift_module = QShiftModule()

# # 使用 profile 函数计算 FLOPs 和 Params
# flops, params = profile(q_shift_module, inputs=(input_tensor,))
# print(f"FLOPs: {flops}")
# print(f"Params: {params}")
import torch
import torch.nn as nn
from einops import rearrange


def in_vertical_scan(input_tensor):
    """
    对输入张量进行垂直向内扫描
    """
    B, C, H, W = input_tensor.shape
    mid = H // 2
    # 将 NCHW 张量转置为 NHWC 格式，以便对倒数第二维（垂直方向）进行操作
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    top_half = input_tensor[:, :mid, :, :]
    bottom_half = torch.flip(input_tensor[:, mid:, :, :], dims=[1])
    # 拼接上下两部分
    print("top_half",top_half.flatten())
    print("bottom", bottom_half.flatten())    
    combined = torch.cat([top_half, bottom_half], dim=1)
    # 将张量转置为 (B * W, H, C) 形状
    transposed = rearrange(combined, 'B H W C -> (B W) H C').contiguous()
    return transposed.view(B, W * H, C) #.permute(0, 2, 1)

def qshift(input, shift_pixel=1, gamma=1 / 4, resolution=None, scan_type='horizontal_forward'):
    """
    对输入张量进行 q_shift 操作，并根据 scan_type 选择不同的扫描方式
    """
    assert gamma <= 1 / 4
    if len(input.shape) == 3:
        B, N, C = input.shape
    N = input.shape[1]
    if resolution is None:
        sqrt_N = int(N ** 0.5)
        if sqrt_N * sqrt_N == N:
            resolution = (sqrt_N, sqrt_N)
        else:
            raise ValueError("Cannot infer a valid resolution for the given input shape.")
    if len(input.shape) == 3:
        input = input.transpose(1, 2).reshape(B, C, resolution[0], resolution[1])
    # B, C, H, W = input.shape
    # output = torch.zeros_like(input)

    # if scan_type == 'horizontal_forward':
    #     output = horizontal_forward_scan(input)
    # elif scan_type == 'horizontal_backward':
    #     output = horizontal_backward_scan(input)
    # elif scan_type == 'vertical_forward':
    #     output = vertical_forward_scan(input)
    # elif scan_type == 'vertical_backward':
    #     output = vertical_backward_scan(input)
    # elif scan_type == 'in_horizontal':
    #     output = in_horizontal_scan(input)
    # elif scan_type == 'out_horizontal':
    #     output = out_horizontal_scan(input)
    # elif scan_type == 'in_vertical':
    #     output = in_vertical_scan(input)
    # elif scan_type == 'out_vertical':
    #     output = out_vertical_scan(input)
    # else:
    #     raise ValueError(f"Invalid scan_type: {scan_type}")

    # # output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    # # output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :, shift_pixel:W]
    # # output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3), 0:H - shift_pixel, :]
    # # output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:, int(C * gamma * 3):int(C * gamma * 4), shift_pixel:H, :]
    # # output[:, int(C * gamma * 4):,...] = input[:, int(C * gamma * 4):,...]
    output = in_vertical_scan(input)
    return output #.flatten(2).transpose(1, 2)


# 生成一个 3x3 的矩阵，假设 batch size 为 1，channel 为 1
input_tensor = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float32)
# input_tensor = input_tensor.unsqueeze(0)
print(input_tensor.shape)
# # 进行各种扫描操作
# scan_types = ['horizontal_forward', 'horizontal_backward', 'vertical_forward', 'vertical_backward',
#              'in_horizontal', 'out_horizontal', 'in_vertical', 'out_vertical']

# for scan_type in scan_types:
#     result = qshift(input_tensor, scan_type=scan_type)
#     print(f"Scan type: {scan_type}")
#     print(result)

print(in_vertical_scan(input_tensor))

print(qshift(input_tensor))


