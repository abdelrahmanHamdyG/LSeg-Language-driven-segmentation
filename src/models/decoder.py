import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    """
    Projects input features to embedding_dim using 1x1 conv.
    Optionally upsamples to target size.
    """
    def __init__(self, in_channels, embedding_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embedding_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, in_channels, H, W]
        x = self.conv(x)  # [B, embedding_dim, H, W]
        return x

class FusionBlock(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1)
        )

    def forward(self, x, skip):
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        out = x + skip
        out = self.conv(out) + out
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels=768, embedding_dim=512):
        super().__init__()
        self.proj_blocks = nn.ModuleList([
            DecoderBlock(in_channels, embedding_dim) for _ in range(4)
        ])
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(embedding_dim) for _ in range(3)
        ])
        self.refine = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1)
        )

    def forward(self, features):
        keys = ["res12", "res9", "res6", "res3"]

        feats = [self.proj_blocks[i](features[k]) for i, k in enumerate(keys)]
        
        x = feats[0]  # deepest feature
        for i in range(1, len(feats)):
            x = self.fusion_blocks[i-1](x, feats[i])  # every time size doubles
        x = self.refine(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        return x