import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConvWithGate(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[3, 3], expansion_factor=4):
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
                nn.Conv2d(in_channels * expansion_factor, in_channels //2, kernel_size=1),
                nn.BatchNorm2d(in_channels//2),
                nn.GELU()
            )
            self.branches.append(branch)

        # Gating mechanism: 1x1 convolution to control information flow
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # Sigmoid activation for gating
        )

    def forward(self, x):
        # Apply multiple convolutional branches
        branch_outputs = [branch(x) for branch in self.branches]
        # Concatenate branch outputs along the channel dimension
        concat_output = torch.cat(branch_outputs, dim=1)

        # Apply gating mechanism
        gate_output = self.gate(concat_output)
        gated_output = concat_output * gate_output  # Element-wise multiplication

        # Sum the gated output to match the original channel size
        # output = gated_output.view(-1, self.in_channels,  gated_output.size(2), gated_output.size(3)).sum(dim=2)
        output = gated_output
        return output

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernels=[3, 3]):
        super(ConvBlock, self).__init__()
        self.dw_conv_with_gate = DWConvWithGate(ch_in, kernel_sizes=kernels)
        self.up = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.dw_conv_with_gate(x)
        x = self.up(x)
        return x

class UNet(nn.Module):
    def __init__(self, dims=[48, 96, 192, 384, 768, 768], kernels=[3, 3, 7, 7, 7, 7]):
        super(UNet, self).__init__()
        self.dims = dims
        self.kernels = kernels
        self.inin_conv = nn.Conv2d(3, dims[0],kernel_size=3,stride=1,padding=1)
        # Down path
        self.down_blocks = nn.ModuleList([
            ConvBlock(dims[i], dims[i+1], kernels=[kernels[i], kernels[i+1]]) for i in range(len(dims) -1)
            
        ])
        
        # Up path
        self.up_blocks = nn.ModuleList([
            ConvBlock(dims[i+1]+dims[i], dims[i], kernels=[kernels[i+1], kernels[i]]) for i in range(len(dims)-1, -1, -1)
        ])
        
        # Final output layer
        self.out_conv = nn.Conv2d(dims[0], 1, kernel_size=1)  # Assuming binary segmentation

    def forward(self, x):
        skip_connections = []
        
        # Down path
        x = self.inin_conv(x)
        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)
        
        # Bottom block
        x = self.down_blocks[-1](x)
        
        # Up path
        for i, block in enumerate(self.up_blocks):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, skip_connections.pop()], dim=1)
            x = block(x)
        
        # Final output
        x = self.out_conv(x)
        return x

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Define dimensions and kernels
    dims = [48, 96, 192, 384, 768]
    kernels = [3, 3, 7, 7, 7]
    
    # Create the U-Net model
    model = UNet(dims=dims, kernels=kernels)
    
    # Print number of parameters
    print(f"Number of trainable parameters: {count_params(model)}")
    
    # Test the model with a sample input
    x = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
    output = model(x)
    
    # Print the shape of the output
    print(f"Output shape: {output.shape}")
    
    print("END")