import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out = self.out(out)
        return out

class TransformerBlock(nn.Module):
    """Transformer 模块"""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        print("dim", embed_dim)
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class PatchEmbedding(nn.Module):
    """将图像分割为 Patch 并嵌入"""
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W) -> (B, embed_dim, n_patches^0.5, n_patches^0.5)
        x = x.flatten(2).transpose(1, 2)  # (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        return x

class TransUNet(nn.Module):
    """TransUNet 模型"""
    def __init__(self, img_size=224, in_chans=3, num_classes=1, embed_dims=[64, 128, 256, 512], depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], patch_size=16):
        super(TransUNet, self).__init__()
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.patch_size = patch_size

        self.Conv1 = conv_block(ch_in=in_chans, ch_out=embed_dims[0])
        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dims[0])

        # Encoder
        self.encoder_layers = nn.ModuleList()
        for i in range(len(embed_dims)):
            print("TTTTTTHis",i, embed_dims[i])
            layer = nn.ModuleList([
                TransformerBlock(embed_dims[i], num_heads[i], embed_dims[i] * mlp_ratios[i])
                for _ in range(depths[i])
            ])
            self.encoder_layers.append(layer)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(len(embed_dims) - 1, 0, -1):
            layer = nn.Sequential(
                nn.ConvTranspose2d(embed_dims[i], embed_dims[i - 1], kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )
            self.decoder_layers.append(layer)

        # Final Output
        self.final_conv = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)

    # def forward(self, x):
    #     # Patch Embedding
    #     x = self.patch_embed(x)  # (B, n_patches, embed_dim)

    #     # Encoder
    #     encoder_features = []
    #     for i, layer in enumerate(self.encoder_layers):
    #         for block in layer:
    #             x = block(x)
    #         encoder_features.append(x)

    #     # Decoder
    #     for i, layer in enumerate(self.decoder_layers):
    #         x = layer(x)
    #         x = torch.cat([x, encoder_features[-(i + 2)]], dim=1)

    #     # Final Output
    #     x = self.final_conv(x)
    #     return x
    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Encoder
        encoder_features = []
        for i, layer in enumerate(self.encoder_layers):
            for block in layer:
                print("i before x shape",i, x.shape)
                print(block)
                x = block(x)
            encoder_features.append(x)

        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            x = torch.cat([x, encoder_features[-(i + 2)]], dim=1)

        # Final Output
        x = self.final_conv(x)
        return x    


if __name__ == "__main__":
    from thop import profile,clever_format
    x = torch.randn(1, 3, 224, 224).cuda()
    # model = LoRD(dims=[32, 64, 128, 256,768]).cuda()
    dims=[48, 96, 192, 384]    
    model = TransUNet(embed_dims= dims).cuda()

    flops, params = profile(model,inputs=(x,))
    flops, params  = clever_format((flops,params),"%.2f")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
        