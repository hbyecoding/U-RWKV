import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class WTConv(nn.Module):
    """Wavelet Transform Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.dwt = DWT()
        self.idwt = IDWT()
        
    def forward(self, x):
        # Apply DWT
        ll, lh, hl, hh = self.dwt(x)
        # Process each component
        ll = self.conv(ll)
        lh = self.conv(lh)
        hl = self.conv(hl)
        hh = self.conv(hh)
        # Combine using IDWT
        return self.idwt(ll, lh, hl, hh)

class AttnGate(nn.Module):
    """Attention Gate"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class RWKV_TimeMix(nn.Module):
    """RWKV Time Mixing Module"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.receptance = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Time-shift
        shifted = self.time_shift(x)
        
        # Compute R, K, V
        r = torch.sigmoid(self.receptance(x))
        k = self.key(shifted)
        v = self.value(shifted)
        
        # Time mixing
        out = r * k * v
        
        out = self.output(out)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return out

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Add other components as needed... 