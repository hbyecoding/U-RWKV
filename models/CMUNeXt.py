import torch
import torch.nn as nn
import math
from typing import Optional
# from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x



class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), #nearest 上采样 
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# ##########_________
# class ConvInOutModule(nn.Module):
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: int,
#             stride: int = 1,
#             padding: int = 0,
#             groups: int = 1,
#             norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
#             act_cfg: Optional[dict] = dict(type='GELU')):
#         super().__init__()
#         layers = []
#         # Convolution Layer
#         layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
#         # Normalization Layer
#         if norm_cfg:
#             norm_layer = self._get_norm_layer(out_channels, norm_cfg)
#             layers.append(norm_layer)
#         # Activation Layer
#         if act_cfg:
#             act_layer = self._get_act_layer(act_cfg)
#             layers.append(act_layer)
#         # Combine all layers
#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.block(x)

#     def _get_norm_layer(self, num_features, norm_cfg):
#         if norm_cfg['type'] == 'BN':
#             return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
#         # Add more normalization types if needed
#         raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

#     def _get_act_layer(self, act_cfg):
#         if act_cfg['type'] == 'ReLU':
#             return nn.ReLU(inplace=True)
#         if act_cfg['type'] == 'SiLU':
#             return nn.SiLU(inplace=True)
#         if act_cfg['type'] == 'GELU':
#             return nn.GELU(inplace=True)
#         # Add more activation types if needed
#         raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")


# class CMUNeXtBlock_Depth_Conv(nn.Module):
#     def __init__(self, ch_in, ch_out, depth=1, k=3,
#                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
#                 act_cfg: Optional[dict] = dict(type='GELU')):
#         super(CMUNeXtBlock_Depth_Conv, self).__init__()
#         # self.block = nn.Sequential(
#         #     *[nn.Sequential(
#         #         Residual(nn.Sequential(
#         #             # deep wise
#         #             DepthwiseConv2d(ch_in, ch_in, k, groups=ch_in , padding=(k // 2, k // 2)), #
#         #             # nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
#         #             nn.GELU(),
#         #             nn.BatchNorm2d(ch_in)
#         #         )),
#         #         ConvInOutModule(ch_in, ch_in * 4, kernel_size = (1, 1),stride=1, padding=(0,0), groups=ch_in,norm_cfg=norm_cfg, act_cfg=act_cfg),
#         #         ConvInOutModule(ch_in * 4, ch_in, kernel_size = (1, 1),stride=1, padding=(0,0), groups=ch_in,norm_cfg=norm_cfg, act_cfg=act_cfg),
#         #     ) for i in range(depth)]
#         # )
#         self.block = nn.Sequential(
#             *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     # deep wise
#                     # nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
#                     DepthwiseConv2d(ch_in, ch_in, k, groups=ch_in , padding=(k // 2, k // 2)),
#                     nn.GELU(),
#                     nn.BatchNorm2d(ch_in)
#                 )),
#                 nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
#                 nn.GELU(),
#                 nn.BatchNorm2d(ch_in * 4),
#                 nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
#                 nn.GELU(),
#                 nn.BatchNorm2d(ch_in)
#             ) for i in range(depth)]
#         )
#         self.up = DWConvBlock(ch_in, ch_out)

#     def forward(self, x):
#         x = self.block(x)
#         x = self.up(x)
#         return x


# class DWConvBlock(nn.Module):
#     def __init__(self, ch_in, ch_out,k = 3):
#         super(DWConvBlock, self).__init__()
        
#         self.conv = nn.Sequential(
#             # DepthwiseConv2d(ch_in, ch_out, k, groups=ch_in , padding=(k // 2, k // 2)),
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x
    
# class CMUNeXt_Depth_Conv(nn.Module):
#     def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
#         """
#         Args:
#             input_channel : input channel.
#             num_classes: output channel.
#             dims: length of channels
#             depths: length of cmunext blocks
#             kernels: kernal size of cmunext blocks
#         """
#         super(CMUNeXt_Depth_Conv, self).__init__()
#         # Encoder
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.stem = DWConvBlock(ch_in=input_channel, ch_out=dims[0])
#         self.encoder1 = CMUNeXtBlock_Depth_Conv(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
#         self.encoder2 = CMUNeXtBlock_Depth_Conv(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
#         self.encoder3 = CMUNeXtBlock_Depth_Conv(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
#         self.encoder4 = CMUNeXtBlock_Depth_Conv(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
#         self.encoder5 = CMUNeXtBlock_Depth_Conv(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
#         # Decoder
#         self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
#         self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
#         self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
#         self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
#         self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
#         self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
#         self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
#         self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
#         self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         x1 = self.stem(x)
#         x1 = self.encoder1(x1)
#         x2 = self.Maxpool(x1)
#         x2 = self.encoder2(x2)
#         x3 = self.Maxpool(x2)
#         x3 = self.encoder3(x3)
#         x4 = self.Maxpool(x3)
#         x4 = self.encoder4(x4)
#         x5 = self.Maxpool(x4)
#         x5 = self.encoder5(x5)

#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)
#         d1 = self.Conv_1x1(d2)

#         return d1

##########_________


class CMUNeXt(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(CMUNeXt, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        
        x5 = self.encoder5(x5)
        # import pdb;pdb.set_trace()
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


import pywt
from thop import profile, clever_format

from torch.nn import functional as F

# 小波变换下采样模块
class DWT_2D(nn.Module):
    def __init__(self, wave='haar'):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])  # 高通滤波器
        dec_lo = torch.Tensor(w.dec_lo[::-1])  # 低通滤波器

        # 构造四个滤波器核
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)  # 低频子带
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)  # 水平高频子带
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)  # 垂直高频子带
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # 对角线高频子带

        # 注册滤波器核为模型的缓冲区
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        dim = x.shape[1]  # 输入通道数
        groups = dim // 4  # 每组输入通道数为 4

        # 对输入通道进行分组卷积
        x_ll = torch.nn.functional.conv2d(x, self.w_ll.expand(groups, 4, -1, -1), stride=2, groups=groups)
        x_lh = torch.nn.functional.conv2d(x, self.w_lh.expand(groups, 4, -1, -1), stride=2, groups=groups)
        x_hl = torch.nn.functional.conv2d(x, self.w_hl.expand(groups, 4, -1, -1), stride=2, groups=groups)
        x_hh = torch.nn.functional.conv2d(x, self.w_hh.expand(groups, 4, -1, -1), stride=2, groups=groups)

        # 将四个子带拼接在一起
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x


# CMUNeXt 模型
class CMUNeXt(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24, 48, 96, 192, 384], depths=[1, 1, 1, 3, 1],
                 kernels=[3, 3, 7, 7, 7], use_wavelet=False):
        super(CMUNeXt, self).__init__()
        self.use_wavelet = use_wavelet
        if use_wavelet:
            self.dwt = DWT_2D(wave='haar')  # 小波变换下采样

        # Encoder
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)

        if self.use_wavelet:
            x2 = self.dwt(x1)  # 小波变换下采样
        else:
            x2 = F.max_pool2d(x1, kernel_size=2, stride=2)  # 传统卷积下采样
        x2 = self.encoder2(x2)

        if self.use_wavelet:
            x3 = self.dwt(x2)
        else:
            x3 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x3 = self.encoder3(x3)

        if self.use_wavelet:
            x4 = self.dwt(x3)
        else:
            x4 = F.max_pool2d(x3, kernel_size=2, stride=2)
        x4 = self.encoder4(x4)

        if self.use_wavelet:
            x5 = self.dwt(x4)
        else:
            x5 = F.max_pool2d(x4, kernel_size=2, stride=2)
        
        # x2 = F.max_pool2d(x1, kernel_size=2, stride=2)  # 传统卷积下采样
        # x2 = self.encoder2(x2)

        # x3 = F.max_pool2d(x2, kernel_size=2, stride=2)
        # x3 = self.encoder3(x3)

        # x4 = F.max_pool2d(x3, kernel_size=2, stride=2)
        # x4 = self.encoder4(x4)

        # x5 = F.max_pool2d(x4, kernel_size=2, stride=2)        
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1

    
# class CMUNeXt_1_3_128_256_384(nn.Module):
#     def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 256, 384], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
#         """
#         Args:
#             input_channel : input channel.
#             num_classes: output channel.
#             dims: length of channels
#             depths: length of cmunext blocks
#             kernels: kernal size of cmunext blocks
#         """
#         super(CMUNeXt_1_3_128_256_384, self).__init__()
#         # Encoder
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
#         self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
#         self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
#         self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
#         self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
#         self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
#         # Decoder
#         self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
#         self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
#         self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
#         self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
#         self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
#         self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
#         self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
#         self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
#         self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         x1 = self.stem(x)
#         x1 = self.encoder1(x1)
#         x2 = self.Maxpool(x1)
#         x2 = self.encoder2(x2)
#         x3 = self.Maxpool(x2)
#         x3 = self.encoder3(x3)
#         x4 = self.Maxpool(x3)
#         x4 = self.encoder4(x4)
#         x5 = self.Maxpool(x4)
#         x5 = self.encoder5(x5)

#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)
#         d1 = self.Conv_1x1(d2)

#         return d1


#  改 encoder
# conv  autopad

# # 自动填充函数
# def autopad(k, p=None, d=1):
#     """
#     k: kernel
#     p: padding
#     d: dilation
#     """
#     if d > 1:
#         # 实际的卷积核大小
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         # 自动填充
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p

# # 标准卷积模块
# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#     default_act = nn.GELU()
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         return self.act(self.conv(x))

# # 深度卷积模块
# class DWConv(Conv):
#     """Depth-wise convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""
#     def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
#         super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

# import torch

# # depthwise_conv2d (x, k, padding=1) 函数和 DepthwiseConv2d (5, 5, kernel_size=3, padding=1) 类的接口与 PyTorch 的 F.conv2d 和 nn.Conv2d 类似。
# # 多尺度多感受野模块
# class CMRF(nn.Module):
#     """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""
#     def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
#         super().__init__()
        
#         self.N         = N
#         self.c         = int(c2 * e / self.N)
#         self.add       = shortcut and c1 == c2
        
#         self.pwconv1   = Conv(c1, c2//self.N, 1, 1)
#         self.pwconv2   = Conv(c2//2, c2, 1, 1)
#         self.m         = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N-1))

#     def forward(self, x):
#         """Forward pass through CMRF Module."""
#         x_residual = x
#         x          = self.pwconv1(x)

#         x          = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
#         x.extend(m(x[-1]) for m in self.m)
#         x[0]       = x[0] +  x[1] 
#         x.pop(1)
        
#         y          = torch.cat(x, dim=1) 
#         y          = self.pwconv2(y)
#         return x_residual + y if self.add else y
    
    
# class CMUNeXt_cmrf(nn.Module):
#     def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
#         """
#         Args:
#             input_channel : input channel.
#             num_classes: output channel.
#             dims: length of channels
#             depths: length of cmunext blocks
#             kernels: kernal size of cmunext blocks
#         """
#         super(CMUNeXt_cmrf, self).__init__()
#         # Encoder
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
#         self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
#         self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
#         self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
#         self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
#         self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
#         # Decoder
#         # self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
#         # self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
#         # self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
#         # self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
#         # self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
#         # self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
#         # self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
#         # self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
#         # self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        
#         self.cmrf1 = CMRF(dims[0],c2 = dims[0], N = 1)
#         self.cmrf2 = CMRF(dims[0],c2 = dims[1], N = 2)
#         self.cmrf3 = CMRF(dims[1],c2 = dims[2], N = 4)
#         self.cmrf4 = CMRF(dims[2],c2 = dims[3], N = 4)
#         self.cmrf5 = CMRF(dims[3],c2 = dims[4], N = 8)
        
        
#         self.Up5 = up_conv(dims[4], ch_out=dims[3])
#         self.Up_conv5 = CMRF(dims[3] * 2, c2=dims[3], N = 8)
#         self.Up4 = up_conv(dims[3], ch_out=dims[2])
#         self.Up_conv4 = CMRF(dims[2] * 2, c2=dims[2], N = 4)
#         self.Up3 = up_conv(dims[2], ch_out=dims[1])
#         self.Up_conv3 = CMRF(dims[1] * 2, c2=dims[1], N = 2)
#         self.Up2 = up_conv(dims[1], ch_out=dims[0])
#         self.Up_conv2 = CMRF(dims[0] * 2, c2=dims[0], N = 1)
#         self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)        
        
        
#     def forward(self,x):
#         x1 = self.stem(x)
#         x1 = self.cmrf1(x1)
#         x2 = self.Maxpool(x1)
#         x2 = self.cmrf2(x2)
#         x3 = self.Maxpool(x2)
#         x3 = self.cmrf3(x3)
#         x4 = self.Maxpool(x3)
#         x4 = self.cmrf4(x4)
#         x5 = self.Maxpool(x4)
#         x5 = self.cmrf5(x5)
        
        
#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)
#         d1 = self.Conv_1x1(d2)
        
#         return d1


# class ConvModule(nn.Module):
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: int,
#             stride: int = 1,
#             padding: int = 0,
#             groups: int = 1,
#             norm_cfg: Optional[dict] = None,
#             act_cfg: Optional[dict] = None):
#         super().__init__()
#         layers = []
#         # Convolution Layer
#         layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
#         # Normalization Layer
#         if norm_cfg:
#             norm_layer = self._get_norm_layer(out_channels, norm_cfg)
#             layers.append(norm_layer)
#         # Activation Layer
#         if act_cfg:
#             act_layer = self._get_act_layer(act_cfg)
#             layers.append(act_layer)
#         # Combine all layers
#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.block(x)

#     def _get_norm_layer(self, num_features, norm_cfg):
#         if norm_cfg['type'] == 'BN':
#             return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
#         # Add more normalization types if needed
#         raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

#     def _get_act_layer(self, act_cfg):
#         if act_cfg['type'] == 'ReLU':
#             return nn.ReLU(inplace=True)
#         if act_cfg['type'] == 'SiLU':
#             return nn.SiLU(inplace=True)
#         # Add more activation types if needed
#         raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")


# class CAA(nn.Module):
#     """Context Anchor Attention"""
#     from typing import Optional
#     def __init__(
#             self,
#             ch_in: int,
#             ch_out: int,
#             N= 8,
#             h_kernel_size: int = 11,
#             v_kernel_size: int = 11,
#             norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
#             act_cfg: Optional[dict] = dict(type='SiLU')):
#         super().__init__()
#         self.avg_pool = nn.AvgPool2d(7, 1, 3)
#         self.conv1 = ConvModule(ch_in, ch_in // N, 1, 1, 0,
#                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
#         self.h_conv = ConvModule(ch_in // N, ch_in // N, (1, h_kernel_size), 1,
#                                  (0, h_kernel_size // 2), groups=ch_in,
#                                  norm_cfg=None, act_cfg=None)
#         self.v_conv = ConvModule(ch_in // N, ch_in // N, (v_kernel_size, 1), 1,
#                                  (v_kernel_size // 2, 0), groups=ch_in,
#                                  norm_cfg=None, act_cfg=None)
#         self.conv2 = ConvModule(ch_in // N, ch_out, 1, 1, 0,
#                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
#         self.act = nn.Sigmoid()

#     def forward(self, x):
#         attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
#         return attn_factor

# class CMUNeXt_caa(nn.Module):
#     def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
#         """
#         Args:
#             input_channel : input channel.
#             num_classes: output channel.
#             dims: length of channels
#             depths: length of cmunext blocks
#             kernels: kernal size of cmunext blocks
#         """
#         super(CMUNeXt_cmrf, self).__init__()
#         # Encoder
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
#         self.encoder1 = CAA(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
#         self.encoder2 = CAA(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
#         self.encoder3 = CAA(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
#         self.encoder4 = CAA(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
#         self.encoder5 = CAA(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
#         # Decoder
#         # self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
#         # self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
#         # self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
#         # self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
#         # self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
#         # self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
#         # self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
#         # self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
#         # self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        
#         self.cmrf1 = CMRF(dims[0],c2 = dims[0], N = 1)
#         self.cmrf2 = CMRF(dims[0],c2 = dims[1], N = 2)
#         self.cmrf3 = CMRF(dims[1],c2 = dims[2], N = 4)
#         self.cmrf4 = CMRF(dims[2],c2 = dims[3], N = 4)
#         self.cmrf5 = CMRF(dims[3],c2 = dims[4], N = 8)
        
        
#         self.Up5 = up_conv(dims[4], ch_out=dims[3])
#         self.Up_conv5 = CMRF(dims[3] * 2, c2=dims[3], N = 8)
#         self.Up4 = up_conv(dims[3], ch_out=dims[2])
#         self.Up_conv4 = CMRF(dims[2] * 2, c2=dims[2], N = 4)
#         self.Up3 = up_conv(dims[2], ch_out=dims[1])
#         self.Up_conv3 = CMRF(dims[1] * 2, c2=dims[1], N = 2)
#         self.Up2 = up_conv(dims[1], ch_out=dims[0])
#         self.Up_conv2 = CMRF(dims[0] * 2, c2=dims[0], N = 1)
#         self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)        
        
        
#     def forward(self,x):
#         x1 = self.stem(x)
#         x1 = self.cmrf1(x1)
#         x2 = self.Maxpool(x1)
#         x2 = self.cmrf2(x2)
#         x3 = self.Maxpool(x2)
#         x3 = self.cmrf3(x3)
#         x4 = self.Maxpool(x3)
#         x4 = self.cmrf4(x4)
#         x5 = self.Maxpool(x4)
#         x5 = self.cmrf5(x5)
        
        
#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)
#         d1 = self.Conv_1x1(d2)
        
#         return d1



# def cmunext_cmrf(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
#     return CMUNeXt_cmrf(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext_s(input_channel=3, num_classes=1, dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext_l(input_channel=3, num_classes=1, dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

# # 示例使用
# if __name__ == "__main__":
#     model = CMUNeXt_cmrf(input_channel=3, num_classes=1)
#     x = torch.randn(2, 3, 256, 256)
#     output = model(x)
#     print(count_params(model))

#     print(output.shape)


# if __name__ == "__main__":
#     # Define dimensions and kernels
#     dims = [48, 96, 192, 384, 768]
#     kernels = [[3, 5]]*5

    
#     # Create the U-Net model
#     model = CMUNeXt(dims= [48, 96, 192, 384, 768])
    
#     # Print number of parameters
#     print(f"Number of trainable parameters: {count_params(model)}")
    
#     # Test the model with a sample input
#     x = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
#     output = model(x)
    
#     # Print the shape of the output
#     print(f"Output shape: {output.shape}")
    
#     print("END")



if __name__ == "__main__":
    from thop import profile,clever_format
    x = torch.randn(1, 3, 256, 256).cuda()
    # model = LoRD(dims=[32, 64, 128, 256,768]).cuda()
    model = CMUNeXt(dims=[24, 48, 96, 192,384]).cuda()

    flops, params = profile(model,inputs=(x,))
    flops, params  = clever_format((flops,params),"%.2f")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")