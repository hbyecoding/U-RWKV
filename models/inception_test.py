import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch 16 * 1/8 = 2 32 128 16
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class DWNeXtStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, k=3):
        super(DWNeXtStage, self).__init__()
        # self.inception = InceptionDWConv2d(in_channels=in_channels, square_kernel_size=11, branch_ratio=0.125)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                Residual(
                    nn.Sequential(InceptionDWConv2d(in_channels=in_channels, square_kernel_size=11, branch_ratio=0.125),
                                nn.GELU(),
                                nn.BatchNorm2d(in_channels))
            ),
                nn.Conv2d(in_channels, in_channels * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(in_channels * 4),
                nn.Conv2d(in_channels * 4, in_channels, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(in_channels)
            ) for i in range(depth)
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)

class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class FusionConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FusionConv, self).__init__()
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
    
class DecoderBlock(nn.Module):
    def __init__(self,ch_in, ch_out):
        super().__init__()
        
        self.up = UpConv(ch_in=ch_in, ch_out=ch_out)
        self.fusion = FusionConv(ch_in=ch_out * 2, ch_out=ch_out)
        

class LOA(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims = [16, 32, 128 ,160, 256], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]] * 5, expansion_factors=[4] * 5):
        super(LOA, self).__init__()

        """ Shared Encoder """
        # self.e1 = LoRDBlockEnc(input_channel, dims[0])
        # self.e2 = LoRDBlockEnc(dims[0], dims[1])
        # self.e3 = LoRDBlockEnc(dims[1], dims[2])
        # self.e4 = LoRDBlockEnc(dims[2], dims[3])
        # self.e5 = LoRDBlockEnc(dims[3], dims[4])
        self.n_emb = dims[-1]
        self.e1 = DWNeXtStage(input_channel, dims[0], depth=depths[0])
        self.e2 = DWNeXtStage(dims[0], dims[1], depth=depths[1])
        self.e3 = DWNeXtStage(dims[1], dims[2], depth=depths[2])
        self.e4 = DWNeXtStage(dims[2], dims[3], depth=depths[3])
        self.e5 = DWNeXtStage(dims[3], dims[4], depth=depths[4])
        
        self.up1 = UpConv(dims[4], dims[3])
        self.fusion1 = FusionConv(dims[3]*2, dims[3])
        self.up2 = UpConv(dims[3], dims[2])
        self.fusion2 = FusionConv(dims[2]*2, dims[2])
        self.up3 = UpConv(dims[1], dims[1])
        self.fusion3 = FusionConv(dims[1]*2, dims[1])
        self.up4 = UpConv(dims[1], dims[0])
        self.fusion4 = FusionConv(dims[0]*2, dims[0])
        out_dim=16
        if dims[0] <= 8:
            out_dim = 8
        self.up5 = UpConv(dims[0], out_dim)
        self.fusion5 = FusionConv(out_dim*2,out_dim)    
        
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
        
        # self.Brwkv = BinaryOrientatedRWKV2D(n_embd=self.n_emb,
        #                                     n_layer=12, 
        #                                     shift_mode='q_shift',
        #                                     channel_gamma=1/4,
        #                                     shift_pixel=1,
        #                                     hidden_rate=4,
        #                                     init_mode="fancy",
        #                                     drop_path=0,
        #                                     key_norm=True)

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
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        """ Encoder """
        x1, p1 = self.e1(x)
        
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        x5, p5 = self.e5(p4)
        # # print(x4.shape, p4.shape)
        # # print(x5.shape, p5.shape)
        # x5 = self.Brwkv(x5)
        # p5 = self.Brwkv(p5)
        # """ Decoder 1 """
        # # print(self.s1)
        # s1 = self.s1(p5, x5)
        # a1 = self.a1(p5, x5)
        # m1 = self.m1(a1)
        # x6 = s1 * m1

        # """ Decoder 2 """
        # s2 = self.s2(x6, x4)
        # a2 = self.a2(a1, x4)
        # m2 = self.m2(a2)
        # x7 = s2 * m2

        # """ Decoder 3 """
        # s3 = self.s3(x7, x3)
        # a3 = self.a3(a2, x3)
        # m3 = self.m3(a3)
        # x8 = s3 * m3

        # """ Decoder 4 """
        # s4 = self.s4(x8, x2)
        # a4 = self.a4(a3, x2)
        # m4 = self.m4(a4)
        # x9 = s4 * m4

        # """ Decoder 5 """
        # s5 = self.s5(x9, x1)
        # a5 = self.a5(a4, x1)
        # m5 = self.m5(a5)
        # x10 = s5 * m5

        u1 = self.up1(x5)
        u1 = torch.cat((x4,u1),dim=1)
        u1 = self.fusion1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat((x3,u2),dim=1)
        u2 = self.fusion1(u2)

        u3 = self.up3(u2)
        u3 = torch.cat((x2,u3),dim=1)
        u3 = self.fusion1(u3)

        u4 = self.up3(u3)
        u4 = torch.cat((x1,u4),dim=1)
        u4 = self.fusion1(u4)
        
        u5 = self.up3(u4)

        
        """ Output """
        out1 = self.output1(u5)
        # out2 = self.output2(a5)
        # print("pause")
        return out1 #, out2
    

if __name__ == "__main__":
    from thop import profile, clever_format
    def count_params_and_flops(module, input_shape):
        input_tensor = torch.randn(input_shape)
        flops, params = profile(module, inputs=(input_tensor,))
        flops, params = clever_format((flops, params), "%.2f")
        print(f"FLOPs: {flops}")
        print(f"Params: {params}")
        return flops, params

  
    # 测试 EncoderBlock
    encoder_block = EncoderBlock(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 256, 256)
    out, pooled = encoder_block(x)
    print("EncoderBlock output shape:", out.shape, pooled.shape)

    # 测试 InceptionDWConv2d
    inception_dwconv = InceptionDWConv2d(in_channels=32,band_kernel_size=7)
    x = torch.randn(1, 32, 128, 128)
    out = inception_dwconv(x)
    print("InceptionDWConv2d output shape:", out.shape)

    # 测试 ConvMlp
    conv_mlp = ConvMlp(in_features=64)
    x = torch.randn(1, 64, 128, 128)
    out = conv_mlp(x)
    print("ConvMlp output shape:", out.shape)

    # 测试 DWNeXtStage
    dwnext_stage = DWNeXtStage(in_channels=64, out_channels=128)
    x = torch.randn(1, 64, 128, 128)
    out, pooled = dwnext_stage(x)
    print("DWNeXtStage output shape:", out.shape, pooled.shape)
    
    # 测试 EncoderBlock 的 FLOPs 和参数量
    print("EncoderBlock FLOPs and Params:")
    count_params_and_flops(encoder_block, (1, 3, 256, 256))

    # 测试 InceptionDWConv2d 的 FLOPs 和参数量
    print("InceptionDWConv2d FLOPs and Params:")
    count_params_and_flops(inception_dwconv, (1, 64, 128, 128))

    # 测试 ConvMlp 的 FLOPs 和参数量
    print("ConvMlp FLOPs and Params:")
    count_params_and_flops(conv_mlp, (1, 64, 128, 128))

    # 测试 DWNeXtStage 的 FLOPs 和参数量
    print("DWNeXtStage FLOPs and Params:")
    count_params_and_flops(dwnext_stage, (1, 64, 128, 128))      
    
    # 测试 DWNeXt 的 FLOPs 和参数量
    print("DWNeXt FLOPs and Params:")
    count_params_and_flops(LOA(dims=[24,48,96,192,384]), (1, 3, 256, 256))      