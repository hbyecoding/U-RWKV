import torch
import torch.nn as nn
import math
import argparse

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# bk 
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        if patch_size <=4:
            self.patcher = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=3, stride=1,padding=1),
                nn.Flatten(2)
            )
        else:
            self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        y = x
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x




class VitBottleNeck(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout=.001,
                 num_heads = 8, activation= "gelu", num_encoders = 4, num_classes=10):
        super(VitBottleNeck, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                   activation=activation,
                                                   batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
    
    def forward(self, x):
        y = x
        B0 = x.shape[0]
        x = self.patch_embedding(x)
        x = self.encoder_blocks(x)
        
        # Assuming x shape is (BS, L, dim) where L = (z*z) + 1 and dim = c * w * w
        BS, L, dim = x.shape
        
        # Extracting z and w from dim assuming dim = c * w * w
        c = self.patch_embedding.patcher[0].out_channels
        w = int(math.sqrt(dim // c))
        z = int(math.sqrt(L - 1))
        
        # Reshaping x to (BS, c, z*w, z*w)
        x = x[:, 1:, :]  # Remove the cls token
        x = x.reshape(BS, z, z, c, w, w)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # Rearrange dimensions
        x = x.view(BS, c, z * w, z * w)
        
        return x

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Vision Transformer (ViT) Training Script")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_df_dir', type=str, default="./dataset/train.csv", help='Path to train data')
    parser.add_argument('--test_df_dir', type=str, default="./dataset/test.csv", help='Path to test data')
    parser.add_argument('--submission_df_dir', type=str, default="./dataset/sample_submission.csv", help='Path to submission data')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--out_img_size', type=int, default=16, help='Input image size 256 /2/2/2/2 = 256/16')
    parser.add_argument('--patch_size', type=int, default=1, help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension') # in_channels * p1 * p2  cpp  = 1* 16 * 16
    parser.add_argument('--dropout', type=float, default=0.001, help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--activation', type=str, default="gelu", help='Activation function')
    parser.add_argument('--num_encoders', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd"], help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--datasetname', type=str, default="MNIST", help='Name of the dataset')
    parser.add_argument('--modelname', type=str, default="vit", help='Name of the model')
    parser.add_argument('--medmnist_name', type=str, default="breastmnist", help='Name of the medmnist data')
    
    return parser

        
# if __name__ == "__main__":# Parse command line arguments
#     parser = get_arg_parser()  # 由于是大通道 小hw 所以 我们还是 设置 patchsize 1
#     args = parser.parse_args()# Calculate num_patches
#     num_patches = (args.img_size // args.patch_size) ** 2# Initialize ViT model
#     embed_dim = args.in_channels * args.patch_size *args.patch_size
#     model = VitBottleNeck(args.in_channels, args.patch_size, embed_dim, num_patches, args.dropout,
#                           args.num_heads, args.activation, args.num_encoders, args.num_classes)# Print model parametersprint(f"Model Parameters: {count_params(model)}")# Create a random input tensor
#     x = torch.randn(size=(args.batch_size, args.in_channels, args.img_size, args.img_size))# Forward pass
#     output = model(x)# Print output shapeprint(f"Output shape: {output.shape}")
    

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
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 256, 768], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7], args_vit = None):
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
        if args_vit == None:
            PATCH_SIZE=1
            NUM_PATCHES = (16 //PATCH_SIZE) **2
        else:
            PATCH_SIZE = args_vit.patch_size
            NUM_PATCHES = (args_vit.out_img_size // PATCH_SIZE) ** 2
        IN_CHANNELS = dims[4]
        EMBED_DIM =dims[4]
        
        # Bottleneck
        # self.vit = VitBottleNeck(image_size=16, patch_size=4, num_classes=dims[4], dim=dims[4], depth=6, heads=8, mlp_dim=dims[4] * 4, channels=dims[4])
        # self.vit = VitBottleNeck(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT, NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASSES)
        self.vit = VitBottleNeck(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES) # EMBED_DIM = 768 ,
        
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
        

        # Bottleneck
        x5 = self.vit(x5)
        # x5 = x5.view(x5.size(0), -1, 8, 8)  # Reshape to (BS, dims[4], 8, 8)

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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# if __name__ == '__main__':
#     from loguru import logger as log
#     x = torch.randn(3, 3, 64, 64)  # 输入 B C H W
#     model = TinyUNet(in_channels=3, num_classes=1) # 这个就是 num——classes 的魅力
#     # print(model)
#     log.add('/data/hongboye/projects/model/plugs/segmentation/tinyUnet.log')
#     log.info("tinyUNet")
#     log.info(count_params(model))
#     log.info(model.named_modules)
#     output = model(x)
    
#     log.info("input shape", x.shape, "out.shape", output.shape)
    
#     print(output.shape)  # 输出应为 (3, 1, 64, 64)
# 示例使用
if __name__ == "__main__":
    parser = get_arg_parser()
    args_vit =  parser.parse_args()
    model = CMUNeXt(input_channel=3,args_vit=args_vit,num_classes=1 )
    x = torch.randn(2, 3, 256, 256)
    print(count_params(model))
    
    output = model(x)
    print(output.shape)