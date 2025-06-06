

import torch
import torch.nn as nn
import argparse
        
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x


class Vit(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout,
                 num_heads, activation, num_encoders, num_classes):
        super(Vit, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                   activation=activation,
                                                   batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_blocks(x)
        x = self.MLP(x[:, 0, :])
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Transformer (ViT) Model")
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.001, help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--activation', type=str, default="gelu", help='Activation function')
    parser.add_argument('--num_encoders', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--hidden_layer', type=int, default=768, help='Hidden layer size')

    args = parser.parse_args()

    IMG_SIZE = args.img_size
    IN_CHANNELS = args.in_channels
    PATCH_SIZE = args.patch_size
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 49
    EMBED_DIM = args.embed_dim
    DROPOUT = args.dropout

    NUM_HEADS = args.num_heads
    ACTIVATION = args.activation
    NUM_ENCODERS = args.num_encoders
    NUM_CLASSES = args.num_classes
    HIDDEN_LAYER = args.hidden_layer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Vit(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT, NUM_HEADS, ACTIVATION, NUM_ENCODERS,
                NUM_CLASSES).to(device)
    x = torch.randn(size=(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)).to(device)
    prediction = model(x)
    print(prediction.shape)