import torch
import torch.nn as nn


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
            nn.Upsample(scale_factor=2, mode='nearest'),
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

class GSC(nn.Module):
    def __init__(self, in_channels):
        super(GSC, self).__init__()
        self.proj3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.proj1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.norm = nn.InstanceNorm2d(in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_residual = x
        x1 = self.proj3x3(x)
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        x1g = self.norm(self.relu(self.proj1x1(x)))
        x1 = x1 * x1g  # 点乘操作
        x2 = self.norm(self.bn(self.proj3x3(x1)))
        return x2 + x_residual
    
    
class GSC_no_pm(nn.Module):
    def __init__(self, in_channels):
        super(GSC_no_pm, self).__init__()
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(in_channels)
        self.nonlinear = nn.ReLU()

        self.proj2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.nonlinear2 = nn.ReLU()

        self.proj3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.norm3 = nn.InstanceNorm2d(in_channels)
        self.nonlinear3 = nn.ReLU()

        self.proj4 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.norm4 = nn.InstanceNorm2d(in_channels)
        self.nonlinear4 = nn.ReLU()

    def forward(self, x):
        x_residual = x
        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonlinear(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonlinear2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonlinear3(x2)

        x = x1 + x2  # 没有点乘操作
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonlinear4(x)

        return x + x_residual

class CMUNeXt_GSC(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        super(CMUNeXt_GSC, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        
        # 在 encoder 之前添加 GSC
        self.gsc1 = GSC(dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        
        self.gsc2 = GSC(dims[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        
        self.gsc3 = GSC(dims[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        
        self.gsc4 = GSC(dims[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        
        self.gsc5 = GSC(dims[3])
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
        x1 = self.gsc1(x1)  # 添加 GSC
        x1 = self.encoder1(x1)
        
        x2 = self.Maxpool(x1)
        x2 = self.gsc2(x2)  # 添加 GSC
        x2 = self.encoder2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.gsc3(x3)  # 添加 GSC
        x3 = self.encoder3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.gsc4(x4)  # 添加 GSC
        x4 = self.encoder4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.gsc5(x5)  # 添加 GSC
        x5 = self.encoder5(x5)

        # Decoder 部分保持不变
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
    

class CMUNeXt_GSC_no_pm(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        super(CMUNeXt_GSC_no_pm, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        
        # 在 encoder 之前添加 GSC_no_pm
        self.gsc1 = GSC_no_pm(dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        
        self.gsc2 = GSC_no_pm(dims[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        
        self.gsc3 = GSC_no_pm(dims[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        
        self.gsc4 = GSC_no_pm(dims[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        
        self.gsc5 = GSC_no_pm(dims[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        
        # Decoder 部分保持不变
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
        x1 = self.gsc1(x1)  # 添加 GSC_no_pm
        x1 = self.encoder1(x1)
        
        x2 = self.Maxpool(x1)
        x2 = self.gsc2(x2)  # 添加 GSC_no_pm
        x2 = self.encoder2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.gsc3(x3)  # 添加 GSC_no_pm
        x3 = self.encoder3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.gsc4(x4)  # 添加 GSC_no_pm
        x4 = self.encoder4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.gsc5(x5)  # 添加 GSC_no_pm
        x5 = self.encoder5(x5)

        # Decoder 部分保持不变
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

def cmunext(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext_s(input_channel=3, num_classes=1, dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext_l(input_channel=3, num_classes=1, dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext_gsc(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt_GSC(input_channel=input_channel, num_classes=num_classes, dims=dims, depths=depths, kernels=kernels)

def cmunext_gsc_no_pm(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt_GSC_no_pm(input_channel=input_channel, num_classes=num_classes, dims=dims, depths=depths, kernels=kernels)


def cmunext_no_chin_4():
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
                    # nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                    # nn.GELU(),
                    # nn.BatchNorm2d(ch_in * 4),
                    # nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                    # nn.GELU(),
                    # nn.BatchNorm2d(ch_in)
                ) for i in range(depth)]
            )
            self.up = conv_block(ch_in, ch_out)

        def forward(self, x):
            x = self.block(x)
            x = self.up(x)
            return x
        
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
    
    return CMUNeXt()

    
def cmunext_no_chin_4_withSE(reduction = 4):
    class SELayer(nn.Module):
        def __init__(self, channel, reduction=4):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, int(channel / reduction), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(channel / reduction), channel, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

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
                ) for i in range(depth)]
            )
            self.up = conv_block(ch_in, ch_out)
            self.se = SELayer(ch_out, reduction=reduction) # here hbye #TODO

        def forward(self, x):
            x = self.block(x)
            x = self.up(x)
            x = self.se(x)
            return x
        
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
    
    return CMUNeXt()

def cmunext_no_chout_4_fusion():
    class fusion_conv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(fusion_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1)),
                # nn.GELU(),
                # nn.BatchNorm2d(ch_out * 4),
                # nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )

        def forward(self, x):
            x = self.conv(x)
            return x
        
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
    
    return CMUNeXt()

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def cmunext_chin_4_no_fusion_dec_se(reduction):
    class fusion_conv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(fusion_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1)),
                # nn.GELU(),
                # nn.BatchNorm2d(ch_out * 4),
                # nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out)
            )
            self.se = SELayer(ch_out, reduction=reduction)
        def forward(self, x):
            x = self.conv(x)
            x = self.se(x)
            return x

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

    return CMUNeXt()    

if __name__ == "__main__":
    from thop import profile,clever_format
    x = torch.randn(1, 3, 256, 256).cuda()
    # model = LoRD(dims=[32, 64, 128, 256,768]).cuda()
    # model = CMUNeXt(dims=[24, 48, 96, 192,384]).cuda()
    # model = cmunext().cuda()
    # model = cmunext_no_chin_4().cuda()
    model = cmunext_no_chout_4_fusion().cuda()
    # model = cmunext_no_chout_4_fusion().cuda()


    print(model(x).shape)
    flops, params = profile(model,inputs=(x,))
    flops, params  = clever_format((flops,params),"%.2f")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")