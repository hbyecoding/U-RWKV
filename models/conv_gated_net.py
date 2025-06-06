import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    

class DWConvWithGate(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[3, 3], expansion_factor=4, drop_path_rate=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor

        # Ensure that there are exactly two branches
        assert len(kernel_sizes) == 2, "There should be exactly two kernel sizes"

        # Branches with Depthwise Separable Convolutions and Bottleneck Structure
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2  # Maintain input size
            branch = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.GELU(),  # Use GELU instead of ReLU for consistency
                
                # Pointwise Convolution (expansion)
                nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1),
                nn.BatchNorm2d(in_channels * expansion_factor),  
                nn.GELU(),
                
                # Pointwise Convolution (projection)
                nn.Conv2d(in_channels * expansion_factor, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),  
                nn.GELU()
            )
            self.branches.append(branch)
            
        # Gating mechanism: 1x1 convolution to control information flow
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels*2),
            nn.Sigmoid()  # Sigmoid activation for gating
        )
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels*2, in_channels,kernel_size=1,stride=1, padding=0),
                         nn.BatchNorm2d(in_channels),
                         nn.GELU())
        self.normx1 = nn.BatchNorm2d(in_channels)  # Normalization
        self.normx2 = nn.BatchNorm2d(in_channels*2)  # Normalization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # self.scale = nn.Parameter(torch.ones((1, in_channels, 1, 1)), requires_grad=True)
        # self.scale = nn.Parameter(torch.randn((1, in_channels, 1, 1)) * 0.01, requires_grad=True)
        self.scale = nn.Parameter(torch.ones((1, in_channels, 1, 1))*0.01, requires_grad=True)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        shortcut = x
        # Apply multiple convolutional branches
        branch_outputs = [branch(x) for branch in self.branches]
        # Concatenate branch outputs along the channel dimension
        concat_output = torch.cat(branch_outputs, dim=1)
        gate_output = self.out_conv(self.normx2(concat_output))
        # Apply gating mechanism
        # gate_output = self.gate(concat_output)
        # gated_output = concat_output * gate_output  # Element-wise multiplication
        # out = 
        # # Sum the gated output to match the original channel size
        # output = gated_output.sum(dim=1).unsqueeze(1)
        # return output + x
        
        return gate_output * self.scale + x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class GatedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, kernel_sizes=[3, 5], expansion_factor=4):
        super(GatedBlock, self).__init__()
        self.block = nn.Sequential(
            *[Residual(DWConvWithGate(ch_in, kernel_sizes, expansion_factor)) for _ in range(depth)],
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )
        self.up = UpConv(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        # x = self.up(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
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

class LoG(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 256,768], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3,3]]*5, expansion_factors=[4]*5):
        ##[[3,5],[5,7],[5,7],[3,7],[3, 5]]
        super(LoG, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = ConvBlock(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = GatedBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], kernel_sizes=kernel_sizes[0], expansion_factor=expansion_factors[0])
        self.encoder2 = GatedBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], kernel_sizes=kernel_sizes[1], expansion_factor=expansion_factors[1])
        self.encoder3 = GatedBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], kernel_sizes=kernel_sizes[2], expansion_factor=expansion_factors[2])
        self.encoder4 = GatedBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], kernel_sizes=kernel_sizes[3], expansion_factor=expansion_factors[3])
        self.encoder5 = GatedBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], kernel_sizes=kernel_sizes[4], expansion_factor=expansion_factors[4])
        
        # Decoder
        self.Up5 = UpConv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = FusionConv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = UpConv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = FusionConv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = UpConv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = FusionConv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = UpConv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = FusionConv(ch_in=dims[0] * 2, ch_out=dims[0])
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


def _cmunext(dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernel_sizes=[[3, 3]]*5, expansion_factors=[4]*5):
    return LoG(dims=dims, depths=depths, kernel_sizes=kernel_sizes, expansion_factors=expansion_factors)

def _cmunext_s(dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernel_sizes=[[3, 3]]*5, expansion_factors=[4]*5):
    return LoG(dims=dims, depths=depths, kernel_sizes=kernel_sizes, expansion_factors=expansion_factors)

def _cmunext_l(dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernel_sizes=[[3, 3]]*5, expansion_factors=[4]*5):
    return LoG(dims=dims, depths=depths, kernel_sizes=kernel_sizes, expansion_factors=expansion_factors)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    # Define dimensions and kernels
    dims = [48, 96, 192, 384, 768]
    kernels = [[3, 3]]*5

    
    # Create the U-Net model
    model = LoG(kernel_sizes=kernels)
    
    # Print number of parameters
    print(f"Number of trainable parameters: {count_params(model)}")
    
    # Test the model with a sample input
    x = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 128x128 image
    output = model(x)
    
    # Print the shape of the output
    print(f"Output shape: {output.shape}")
    
    print("END")
    
    
    
# self gate
# class DWConvWithGate(nn.Module):
#     def __init__(self, in_channels, kernel_sizes=[3, 3], expansion_factor=4, drop_path_rate=0.0):
#         super().__init__()
#         self.in_channels = in_channels
#         self.kernel_sizes = kernel_sizes
#         self.expansion_factor = expansion_factor

#         # Ensure that there are exactly two branches
#         assert len(kernel_sizes) == 2, "There should be exactly two kernel sizes"

#         # Branches with Depthwise Separable Convolutions and Bottleneck Structure
#         self.branches = nn.ModuleList()
#         for kernel_size in kernel_sizes:
#             padding = kernel_size // 2  # Maintain input size
#             branch = nn.Sequential(
#                 # Depthwise Convolution
#                 nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
#                 nn.BatchNorm2d(in_channels),
#                 nn.GELU(),  # Use GELU instead of ReLU for consistency
                
#                 # Pointwise Convolution (expansion)
#                 nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1),
#                 nn.BatchNorm2d(in_channels * expansion_factor),  
#                 nn.GELU(),
                
#                 # Pointwise Convolution (projection)
#                 nn.Conv2d(in_channels * expansion_factor, in_channels, kernel_size=1),
#                 nn.BatchNorm2d(in_channels),  
#                 nn.GELU()
#             )
#             self.branches.append(branch)
            
#         # Gating mechanism: 1x1 convolution to control information flow
#         self.gate = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels*2, kernel_size=1),
#             nn.BatchNorm2d(in_channels*2),
#             nn.GELU()
#             # nn.Sigmoid()  # Sigmoid activation for gating
#         )
#         self.out_conv = nn.Conv2d(in_channels*2, in_channels,kernel_size=1,stride=1, padding=0)
#         self.normx1 = nn.BatchNorm2d(in_channels)  # Normalization
#         self.normx2 = nn.BatchNorm2d(in_channels*2)  # Normalization
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
#         self.scale = nn.Parameter(torch.ones((1, in_channels, 1, 1))*0.01, requires_grad=True)
#         # self.scale = nn.Parameter(torch.randn((1, in_channels, 1, 1)) * 0.01, requires_grad=True)
        
#     def forward(self, x):
#         shortcut = x
#         # Apply multiple convolutional branches
#         branch_outputs = [branch(x) for branch in self.branches]
#         # Concatenate branch outputs along the channel dimension
#         concat_output = torch.cat(branch_outputs, dim=1)
#         gate_output = self.normx1(self.drop_path(self.out_conv(self.normx2(concat_output))))
#         # Apply gating mechanism
#         gate_output = self.gate(x)
#         gated_output = concat_output * gate_output  # Element-wise multiplication
#         out = self.normx1(self.drop_path(self.out_conv(gated_output)))
#         # # Sum the gated output to match the original channel size
#         # output = gated_output.sum(dim=1).unsqueeze(1)
#         # return output + x
        
#         return out
# class DWConvWithGate(nn.Module):
#     def __init__(self, in_channels, kernel_sizes=[3, 3], expansion_factor=4):
#         super().__init__()
#         self.in_channels = in_channels
#         self.kernel_sizes = kernel_sizes
#         self.expansion_factor = expansion_factor

#         # Ensure that there are exactly two branches
#         assert len(kernel_sizes) == 2, "There should be exactly two kernel sizes"

#         # Branches with Depthwise Separable Convolutions and Bottleneck Structure
#         self.branches = nn.ModuleList()
#         for kernel_size in kernel_sizes:
#             padding = kernel_size // 2  # Maintain input size
#             branch = nn.Sequential(
#                 # Depthwise Convolution
#                 nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
#                 nn.BatchNorm2d(in_channels),
#                 nn.GELU(),  # Use GELU instead of ReLU for consistency
                
#                 # Pointwise Convolution (expansion)
#                 nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1),
#                 nn.BatchNorm2d(in_channels * expansion_factor),
#                 nn.GELU(),
                
#                 # Pointwise Convolution (projection)
#                 nn.Conv2d(in_channels * expansion_factor, in_channels//2, kernel_size=1),
#                 nn.BatchNorm2d(in_channels //2),
#                 nn.GELU()
#             )
#             self.branches.append(branch)

#         # Gating mechanism: 1x1 convolution to control information flow
#         self.gate = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.Sigmoid()  # Sigmoid activation for gating
#         )

#     def forward(self, x):
#         # Apply multiple convolutional branches
#         branch_outputs = [branch(x) for branch in self.branches]
#         # Concatenate branch outputs along the channel dimension
#         concat_output = torch.cat(branch_outputs, dim=1)

#         # Apply gating mechanism
#         gate_output = self.gate(concat_output)
#         gated_output = concat_output * gate_output  # Element-wise multiplication

#         # Sum the gated output to match the original channel size
#         output = gated_output.sum(dim=1).unsqueeze(1)
#         return output