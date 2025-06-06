import torch
import torch.nn as nn
import torchvision.models as models

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

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderBlock, self).__init__()

        self.r1 = ResidualBlock(in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.r1(x)
        x = self.r2(x)
        p = self.pool(x)

        return x, p

# class ResidualGateBlock(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(ResidualGateBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_c)

#         self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_c)

#         self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(out_c)
#         self.se = SELayer(out_c)

#         # Gate module
#         self.gate_conv = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
#         self.gate_bn = nn.BatchNorm2d(out_c)

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x1 = self.bn1(x1)
#         x1 = self.relu(x1)

#         x2 = self.conv2(x1)
#         x2 = self.bn2(x2)

#         x3 = self.conv3(x)
#         x3 = self.bn3(x3)
#         x3 = self.se(x3)

#         # Gate module
#         gate = self.gate_conv(x2)
#         gate = self.gate_bn(gate)
#         gate = torch.tanh(gate)  # Apply tanh activation

#         # Apply gate to x2
#         x2 = x2 * gate  # Element-wise multiplication

#         x4 = x2 + x3
#         x4 = self.relu(x4)

#         return x4

# class EncoderBlock(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(EncoderBlock, self).__init__()

#         self.r1 = ResidualGateBlock(in_c, out_c)  # Replaced ResidualBlock with ResidualGateBlock
#         self.r2 = ResidualGateBlock(out_c, out_c)  # Replaced ResidualBlock with ResidualGateBlock
#         self.pool = nn.MaxPool2d(2, stride=2)

#     def forward(self, x):
#         x = self.r1(x)
#         x = self.r2(x)
#         p = self.pool(x)

#         return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c+out_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, x, s):
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        return x


class CompNet(nn.Module):
    def __init__(self):
        super(CompNet, self).__init__()

        """ Shared Encoder """
        self.e1 = EncoderBlock(3, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)

        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(256, 128)
        self.s2 = DecoderBlock(128, 64)
        self.s3 = DecoderBlock(64, 32)
        self.s4 = DecoderBlock(32, 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(256, 128)
        self.a2 = DecoderBlock(128, 64)
        self.a3 = DecoderBlock(64, 32)
        self.a4 = DecoderBlock(32, 16)

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        self.output1 = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)

        """ Decoder 1 """
        s1 = self.s1(p4, x4)
        a1 = self.a1(p4, x4)
        m1 = self.m1(a1)
        x5 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x5, x3)
        a2 = self.a2(a1, x3)
        m2 = self.m2(a2)
        x6 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x6, x2)
        a3 = self.a3(a2, x2)
        m3 = self.m3(a3)
        x7 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x7, x1)
        a4 = self.a4(a3, x1)
        m4 = self.m4(a4)
        x8 = s4 * m4

        """ Output """
        out1 = self.output1(x8)
        out2 = self.output2(a4)

        return out1, out2


import torch
import torch.nn as nn

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

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderBlock, self).__init__()

        self.r1 = ResidualBlock(in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.r1(x)
        x = self.r2(x)
        p = self.pool(x)

        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c + out_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, x, s):
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        return x

def compnet_full():
    """完整的 CompNet 模型"""
    class CompNet(nn.Module):
        def __init__(self):
            super(CompNet, self).__init__()

            """ Shared Encoder """
            self.e1 = EncoderBlock(3, 32)
            self.e2 = EncoderBlock(32, 64)
            self.e3 = EncoderBlock(64, 128)
            self.e4 = EncoderBlock(128, 256)

            """ Decoder: Segmentation """
            self.s1 = DecoderBlock(256, 128)
            self.s2 = DecoderBlock(128, 64)
            self.s3 = DecoderBlock(64, 32)
            self.s4 = DecoderBlock(32, 16)

            """ Decoder: Autoencoder """
            self.a1 = DecoderBlock(256, 128)
            self.a2 = DecoderBlock(128, 64)
            self.a3 = DecoderBlock(64, 32)
            self.a4 = DecoderBlock(32, 16)

            """ Autoencoder attention map """
            self.m1 = nn.Sequential(
                nn.Conv2d(128, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m2 = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m3 = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m4 = nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )

            """ Output """
            self.output1 = nn.Conv2d(16, 1, kernel_size=1, padding=0)
            self.output2 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            x1, p1 = self.e1(x)
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)
            x4, p4 = self.e4(p3)

            """ Decoder 1 """
            s1 = self.s1(p4, x4)
            a1 = self.a1(p4, x4)
            m1 = self.m1(a1)
            x5 = s1 * m1

            """ Decoder 2 """
            s2 = self.s2(x5, x3)
            a2 = self.a2(a1, x3)
            m2 = self.m2(a2)
            x6 = s2 * m2

            """ Decoder 3 """
            s3 = self.s3(x6, x2)
            a3 = self.a3(a2, x2)
            m3 = self.m3(a3)
            x7 = s3 * m3

            """ Decoder 4 """
            s4 = self.s4(x7, x1)
            a4 = self.a4(a3, x1)
            m4 = self.m4(a4)
            x8 = s4 * m4

            """ Output """
            out1 = self.output1(x8)
            out2 = self.output2(a4)

            return out1, out2

    return CompNet()



def compnet_no_se():
    """移除 SELayer 的 CompNet 模型"""
    class ResidualBlockNoSE(nn.Module):
        def __init__(self, in_c, out_c):
            super(ResidualBlockNoSE, self).__init__()

            self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_c)

            self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_c)

            self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
            self.bn3 = nn.BatchNorm2d(out_c)

            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x1 = self.conv1(x)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)

            x2 = self.conv2(x1)
            x2 = self.bn2(x2)

            x3 = self.conv3(x)
            x3 = self.bn3(x3)

            x4 = x2 + x3
            x4 = self.relu(x4)

            return x4
        

    class EncoderBlockNoSE(nn.Module):
        def __init__(self, in_c, out_c):
            super(EncoderBlockNoSE, self).__init__()

            self.r1 = ResidualBlockNoSE(in_c, out_c)
            self.r2 = ResidualBlockNoSE(out_c, out_c)
            self.pool = nn.MaxPool2d(2, stride=2)

        def forward(self, x):
            x = self.r1(x)
            x = self.r2(x)
            p = self.pool(x)

            return x, p

    class DecoderBlockNoSE(nn.Module):
        def __init__(self, in_c, out_c):
            super(DecoderBlockNoSE, self).__init__()

            self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
            self.r1 = ResidualBlockNoSE(in_c + out_c, out_c)
            self.r2 = ResidualBlockNoSE(out_c, out_c)

        def forward(self, x, s):
            x = self.upsample(x)
            x = torch.cat([x, s], axis=1)
            x = self.r1(x)
            x = self.r2(x)

            return x

    class CompNetNoSE(nn.Module):
        def __init__(self):
            super(CompNetNoSE, self).__init__()

            """ Shared Encoder """
            self.e1 = EncoderBlockNoSE(3, 32)
            self.e2 = EncoderBlockNoSE(32, 64)
            self.e3 = EncoderBlockNoSE(64, 128)
            self.e4 = EncoderBlockNoSE(128, 256)

            """ Decoder: Segmentation """
            self.s1 = DecoderBlockNoSE(256, 128)
            self.s2 = DecoderBlockNoSE(128, 64)
            self.s3 = DecoderBlockNoSE(64, 32)
            self.s4 = DecoderBlockNoSE(32, 16)

            """ Decoder: Autoencoder """
            self.a1 = DecoderBlockNoSE(256, 128)
            self.a2 = DecoderBlockNoSE(128, 64)
            self.a3 = DecoderBlockNoSE(64, 32)
            self.a4 = DecoderBlockNoSE(32, 16)

            """ Autoencoder attention map """
            self.m1 = nn.Sequential(
                nn.Conv2d(128, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m2 = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m3 = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )
            self.m4 = nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )

            """ Output """
            self.output1 = nn.Conv2d(16, 1, kernel_size=1, padding=0)
            self.output2 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            x1, p1 = self.e1(x)
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)
            x4, p4 = self.e4(p3)

            """ Decoder 1 """
            s1 = self.s1(p4, x4)
            a1 = self.a1(p4, x4)
            m1 = self.m1(a1)
            x5 = s1 * m1

            """ Decoder 2 """
            s2 = self.s2(x5, x3)
            a2 = self.a2(a1, x3)
            m2 = self.m2(a2)
            x6 = s2 * m2

            """ Decoder 3 """
            s3 = self.s3(x6, x2)
            a3 = self.a3(a2, x2)
            m3 = self.m3(a3)
            x7 = s3 * m3

            """ Decoder 4 """
            s4 = self.s4(x7, x1)
            a4 = self.a4(a3, x1)
            m4 = self.m4(a4)
            x8 = s4 * m4

            """ Output """
            out1 = self.output1(x8)
            out2 = self.output2(a4)

            return out1, out2

    return CompNetNoSE()

def compnet_no_attention():
    """移除 Autoencoder Attention Map 的 CompNet 模型"""
    class CompNetNoAttention(nn.Module):
        def __init__(self):
            super(CompNetNoAttention, self).__init__()

            """ Shared Encoder """
            self.e1 = EncoderBlock(3, 32)
            self.e2 = EncoderBlock(32, 64)
            self.e3 = EncoderBlock(64, 128)
            self.e4 = EncoderBlock(128, 256)

            """ Decoder: Segmentation """
            self.s1 = DecoderBlock(256, 128)
            self.s2 = DecoderBlock(128, 64)
            self.s3 = DecoderBlock(64, 32)
            self.s4 = DecoderBlock(32, 16)

            """ Decoder: Autoencoder """
            self.a1 = DecoderBlock(256, 128)
            self.a2 = DecoderBlock(128, 64)
            self.a3 = DecoderBlock(64, 32)
            self.a4 = DecoderBlock(32, 16)

            """ Output """
            self.output1 = nn.Conv2d(16, 1, kernel_size=1, padding=0)
            self.output2 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            x1, p1 = self.e1(x)
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)
            x4, p4 = self.e4(p3)

            """ Decoder 1 """
            s1 = self.s1(p4, x4)
            a1 = self.a1(p4, x4)
            x5 = s1

            """ Decoder 2 """
            s2 = self.s2(x5, x3)
            a2 = self.a2(a1, x3)
            x6 = s2

            """ Decoder 3 """
            s3 = self.s3(x6, x2)
            a3 = self.a3(a2, x2)
            x7 = s3

            """ Decoder 4 """
            s4 = self.s4(x7, x1)
            a4 = self.a4(a3, x1)
            x8 = s4

            """ Output """
            out1 = self.output1(x8)
            out2 = self.output2(a4)

            return out1, out2

    return CompNetNoAttention()

def compnet_single_decoder():
    """仅保留 Segmentation Decoder 的 CompNet 模型"""
    class CompNetSingleDecoder(nn.Module):
        def __init__(self):
            super(CompNetSingleDecoder, self).__init__()

            """ Shared Encoder """
            self.e1 = EncoderBlock(3, 32)
            self.e2 = EncoderBlock(32, 64)
            self.e3 = EncoderBlock(64, 128)
            self.e4 = EncoderBlock(128, 256)

            """ Decoder: Segmentation """
            self.s1 = DecoderBlock(256, 128)
            self.s2 = DecoderBlock(128, 64)
            self.s3 = DecoderBlock(64, 32)
            self.s4 = DecoderBlock(32, 16)

            """ Output """
            self.output1 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            x1, p1 = self.e1(x)
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)
            x4, p4 = self.e4(p3)

            """ Decoder """
            s1 = self.s1(p4, x4)
            s2 = self.s2(s1, x3)
            s3 = self.s3(s2, x2)
            s4 = self.s4(s3, x1)

            """ Output """
            out1 = self.output1(s4)

            return out1

    return CompNetSingleDecoder()

def compnet_shallow_encoder():
    """减少 Encoder 深度的 CompNet 模型"""
    class CompNetShallowEncoder(nn.Module):
        def __init__(self):
            super(CompNetShallowEncoder, self).__init__()

            """ Shared Encoder """
            self.e1 = EncoderBlock(3, 32)
            self.e2 = EncoderBlock(32, 64)
            self.e3 = EncoderBlock(64, 128)

            """ Decoder: Segmentation """
            self.s1 = DecoderBlock(128, 64)
            self.s2 = DecoderBlock(64, 32)
            self.s3 = DecoderBlock(32, 16)

            """ Output """
            self.output1 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            x1, p1 = self.e1(x)
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)

            """ Decoder """
            s1 = self.s1(p3, x3)
            s2 = self.s2(s1, x2)
            s3 = self.s3(s2, x1)

            """ Output """
            out1 = self.output1(s3)

            return out1

    return CompNetShallowEncoder()

def compnet_add_encoder():
    """减少 Encoder 深度的 CompNet 模型"""
    class CompNetAddEncoder(nn.Module):
        def __init__(self):
            super(CompNetAddEncoder, self).__init__()

            """ Shared Encoder """
            self.e1 = EncoderBlock(3, 32)
            self.e2 = EncoderBlock(32, 64)
            self.e3 = EncoderBlock(64, 128)
            self.e4 = EncoderBlock(128, 256)
            self.e5 = EncoderBlock(256, 512)

            """ Decoder: Segmentation """
            self.s1 = DecoderBlock(512, 256)
            self.s2 = DecoderBlock(256, 128)
            self.s3 = DecoderBlock(128, 64)
            self.s4 = DecoderBlock(64, 32)
            self.s5 = DecoderBlock(32, 16)

            """ Output """
            self.output1 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            x1, p1 = self.e1(x)
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)
            x4, p4 = self.e4(p3)
            x5, p5 = self.e5(p4)
            """ Decoder """
            s1 = self.s1(p5, x5)
            s2 = self.s2(s1, x4)
            s3 = self.s3(s2, x3)
            s4 = self.s4(s3, x2)
            s5 = self.s5(s4, x1)

            """ Output """
            out1 = self.output1(s5)

            return out1

    return CompNetAddEncoder()

def compnet_l():
    class CompNetAddEncoder(nn.Module):
        def __init__(self):
            super(CompNetAddEncoder, self).__init__()

            """ Shared Encoder """
            self.e1 = EncoderBlock(3, 32)
            self.e1_fix = EncoderBlock(32, 32)
            
            self.e2 = EncoderBlock(32, 64)
            self.e3 = EncoderBlock(64, 128)
            self.e4 = EncoderBlock(128, 256)
            self.e5 = EncoderBlock(256, 512)

            """ Decoder: Segmentation """
            self.s1 = DecoderBlock(512, 256)
            self.s2 = DecoderBlock(256, 128)
            self.s3 = DecoderBlock(128, 64)
            self.s4 = DecoderBlock(64, 32)
            self.s4_fix = DecoderBlock(32,32)
            self.s5 = DecoderBlock(32, 16)

            """ Output """
            self.output1 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        def forward(self, x):
            """ Encoder """
            x1, p1 = self.e1(x)
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)
            x4, p4 = self.e4(p3)
            x5, p5 = self.e5(p4)
            """ Decoder """
            s1 = self.s1(p5, x5)
            s2 = self.s2(s1, x4)
            s3 = self.s3(s2, x3)
            s4 = self.s4(s3, x2)
            s5 = self.s5(s4, x1)

            """ Output """
            out1 = self.output1(s5)

            return out1



if __name__ == "__main__":
    from thop import profile, clever_format
    model = compnet_add_encoder()
    from loguru import logger as log
    log.add("./vs_ablation.log")
    log.info("+"*20)
    log.info(model.named_modules)
    x = torch.zeros((1,3,256,256))
    flops, params = profile(model, inputs=(x, ))  
    flops, params = clever_format([flops, params], '%.2f')
    log.info("*"*20)
    log.info("compnet_add_encoder")
    log.info(f"FLOPs: {flops}")
    log.info(f"Params: {params}")
    # log.info("model(x).shape", model(x).shape)
