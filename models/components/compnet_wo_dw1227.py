import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
    

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
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

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
    
    

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x, skip):
#         x = self.up(x)
#         x = torch.cat([x, skip], dim=1)
#         x = self.conv(x)
#         return x

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
    def __init__(self, input_channel=3, num_classes=1, dims=[24, 48, 96, 192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(CompNet, self).__init__()
        self.nemb = dims[-1]
        self.nemb_by_2 = dims[-2]
        self.nemb_by_4 = dims[-3]
        """ Shared Encoder """
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])

        # """fusion"""    
        # self.withDW_by_2 = ProProcessWithDWConv(ch_in=self.nemb_by_2, ch_out=self.nemb)
        # self.withDW_by_4 = ProProcessWithDWConv(ch_in=self.nemb_by_4, ch_out=self.nemb)
        
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """ Output """
        # # bot2_conv: 一次 Conv2d + BatchNorm2d + GELU
        # self.bot2_conv = nn.Sequential(
        #     nn.Conv2d(self.nemb * 2, self.nemb, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.nemb),
        #     nn.GELU()
        # )

        # # bot4_conv: 两次 Conv2d + BatchNorm2d + GELU
        # self.bot4_conv = nn.Sequential(
        #     nn.Conv2d(self.nemb * 4, self.nemb *2, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.nemb* 2),
        #     nn.GELU(),
        #     nn.Conv2d(self.nemb*2, self.nemb, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.nemb),
        #     nn.GELU()
        # )
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x) # x1是 C24
        x2, p2 = self.e2(p1) # C48
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        """"fusion"""
        
        # Xemb_by_4 = self.withDW_by_4(x3)
        # Xemb_by_2 = self.withDW_by_2(x4) # C 384 
        # X_pre = torch.cat([Xemb_by_4, Xemb_by_2], dim=1) # C 2*384 
        # X_pre = self.bot2_conv(X_pre)
        # X_pre = torch.cat([X_pre, x5], dim=1)
        # x5 = self.bot2_conv(X_pre)
        
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1

class CompNet_with_DW(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[24, 48, 96, 192,384], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super().__init__()
        self.nemb = dims[-1]
        self.nemb_by_2 = dims[-2]
        self.nemb_by_4 = dims[-3]
        """ Shared Encoder """
        self.e1 = EncoderBlock(input_channel, dims[0])
        self.e2 = EncoderBlock(dims[0], dims[1])
        self.e3 = EncoderBlock(dims[1], dims[2])
        self.e4 = EncoderBlock(dims[2], dims[3])
        self.e5 = EncoderBlock(dims[3], dims[4])
        
        """ Decoder: Segmentation """
        self.s1 = DecoderBlock(dims[4], dims[3])
        self.s2 = DecoderBlock(dims[3], dims[2])
        self.s3 = DecoderBlock(dims[2], dims[1])
        self.s4 = DecoderBlock(dims[1], dims[0])
        self.s5 = DecoderBlock(dims[0], 16)

        """ Decoder: Autoencoder """
        self.a1 = DecoderBlock(dims[4], dims[3])
        self.a2 = DecoderBlock(dims[3], dims[2])
        self.a3 = DecoderBlock(dims[2], dims[1])
        self.a4 = DecoderBlock(dims[1], dims[0])
        self.a5 = DecoderBlock(dims[0], 16)

        """ Autoencoder attention map """
        self.m1 = nn.Sequential(
            nn.Conv2d(dims[3], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(dims[2], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(dims[1], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m4 = nn.Sequential(
            nn.Conv2d(dims[0], 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.m5 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        """fusion"""    
        self.withDW_by_2 = ProProcessWithDWConv(ch_in=self.nemb_by_2, ch_out=self.nemb,depth=1)
        self.withDW_by_4 = ProProcessWithDWConv(ch_in=self.nemb_by_4, ch_out=self.nemb,depth=2)

        """ Output """
        # bot2_conv: 一次 Conv2d + BatchNorm2d + GELU
        self.bot2_conv = nn.Sequential(
            nn.Conv2d(self.nemb * 2, self.nemb, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nemb),
            nn.GELU()
        )

        # bot4_conv: 两次 Conv2d + BatchNorm2d + GELU
        self.bot4_conv = nn.Sequential(
            nn.Conv2d(self.nemb * 4, self.nemb *2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nemb* 2),
            nn.GELU(),
            nn.Conv2d(self.nemb*2, self.nemb, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nemb),
            nn.GELU()
        )
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x) # x1是 C24
        x2, p2 = self.e2(p1) # C48
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x5, p5 = self.e5(p4)
        # print(x4.shape, p4.shape)
        # print(x5.shape, p5.shape)
        """"fusion"""
        # import pdb;pdb.set_trace()
        Xemb_by_4 = self.withDW_by_4(x3)
        Xemb_by_2 = self.withDW_by_2(x4) # C 384 
        X_pre = torch.cat([Xemb_by_4, Xemb_by_2], dim=1) # C 2*384 
        X_pre = self.bot2_conv(X_pre)
        X_pre = torch.cat([X_pre, x5], dim=1)
        x5 = self.bot2_conv(X_pre)
        
        """ Decoder 1 """
        # print(self.s1)
        s1 = self.s1(p5, x5)
        a1 = self.a1(p5, x5)
        m1 = self.m1(a1)
        x6 = s1 * m1

        """ Decoder 2 """
        s2 = self.s2(x6, x4)
        a2 = self.a2(a1, x4)
        m2 = self.m2(a2)
        x7 = s2 * m2

        """ Decoder 3 """
        s3 = self.s3(x7, x3)
        a3 = self.a3(a2, x3)
        m3 = self.m3(a3)
        x8 = s3 * m3

        """ Decoder 4 """
        s4 = self.s4(x8, x2)
        a4 = self.a4(a3, x2)
        m4 = self.m4(a4)
        x9 = s4 * m4

        """ Decoder 5 """
        s5 = self.s5(x9, x1)
        a5 = self.a5(a4, x1)
        m5 = self.m5(a5)
        x10 = s5 * m5

        """ Output """
        out1 = self.output1(x10)
        out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2
    
    
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    # DW = ProProcessWithDWConv(ch_in=3,ch_out=16, depth=1)
    # i1 = DW(x)
    # print(i1.shape)
    model = CompNet_with_DW()
    y1= model(x)#, y2
    # print(y1.shape, y2.shape)
    print(count_params(model))
    print("END")