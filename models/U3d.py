import torch
import torch.nn as nn
from timm.models.layers import DropPath

class Conv3d(nn.Module):
    """Standard 3D convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv3d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm3d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv3d(Conv3d):
    """Depth-wise 3D convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class CMUNeXtBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv3d(ch_in, ch_in, k, groups=ch_in, padding=k // 2),
                    nn.GELU(),
                    nn.BatchNorm3d(ch_in)
                )),
                nn.Conv3d(ch_in, ch_in * 4, 1),
                nn.GELU(),
                nn.BatchNorm3d(ch_in * 4),
                nn.Conv3d(ch_in * 4, ch_in, 1),
                nn.GELU(),
                nn.BatchNorm3d(ch_in)
            ) for _ in range(depth)]
        )
        self.up = Conv3d(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class up_conv3d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True),
            nn.Conv3d(ch_in, ch_out, 3, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class fusion_conv3d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_in, 3, padding=1, groups=2),
            nn.GELU(),
            nn.BatchNorm3d(ch_in),
            nn.Conv3d(ch_in, ch_out * 4, 1),
            nn.GELU(),
            nn.BatchNorm3d(ch_out * 4),
            nn.Conv3d(ch_out * 4, ch_out, 1),
            nn.GELU(),
            nn.BatchNorm3d(ch_out)
        )

    def forward(self, x):
        return self.conv(x)


class CMUNeXt3D(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 256, 768], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        super().__init__()
        self.Maxpool = nn.MaxPool3d(2, 2)
        self.stem = Conv3d(input_channel, dims[0])
        self.encoder1 = CMUNeXtBlock3D(dims[0], dims[0], depths[0], kernels[0])
        self.encoder2 = CMUNeXtBlock3D(dims[0], dims[1], depths[1], kernels[1])
        self.encoder3 = CMUNeXtBlock3D(dims[1], dims[2], depths[2], kernels[2])
        self.encoder4 = CMUNeXtBlock3D(dims[2], dims[3], depths[3], kernels[3])
        self.encoder5 = CMUNeXtBlock3D(dims[3], dims[4], depths[4], kernels[4])

        self.Up5 = up_conv3d(dims[4], dims[3])
        self.Up_conv5 = fusion_conv3d(dims[3] * 2, dims[3])
        self.Up4 = up_conv3d(dims[3], dims[2])
        self.Up_conv4 = fusion_conv3d(dims[2] * 2, dims[2])
        self.Up3 = up_conv3d(dims[2], dims[1])
        self.Up_conv3 = fusion_conv3d(dims[1] * 2, dims[1])
        self.Up2 = up_conv3d(dims[1], dims[0])
        self.Up_conv2 = fusion_conv3d(dims[0] * 2, dims[0])
        self.Conv_1x1 = nn.Conv3d(dims[0], num_classes, 1)

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
    
if __name__ == "__main__":
    x = torch.randn(4, 3, 64,64,64)