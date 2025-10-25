#The pixel–text similarity computation: normalize pixel and text features, compute dot-product, apply temperature scaling, produce logits.
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSegHead(nn.Module):
    def __init__(self, embedding_dim=512, temperature=0.07):
        super(LSegHead, self).__init__()
        self.temperature = temperature

    def forward(self, text_features, img_features):
        # img_shape is [B,512,H,W] C=512 H=1/2 original image W=1/2 original image
        # text_features is [N,512] N=number of text prompts
        B, C, H, W = img_features.shape
        N = text_features.shape[0]
        

        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        # print("img_features shape after norm: ", img_features.shape)
        # print("img_features stats after norm: ", img_features.min().item(), img_features.max().item(), img_features.mean().item(), img_features.std().item())


        img_features = img_features.flatten(2).permute(0, 2, 1)  # [B,512,H,W]-> [B,512,H*W]-> [B,H*W,512]
        text_features = text_features.T  # [N,512]-> [512,N]
        logits = torch.matmul(img_features, text_features)  # [B,H*W,N]
        logits = logits / self.temperature
        logits = logits.permute(0, 2, 1).reshape(B, N, H, W)  # [B,N,H,W]
        # print("logits shape: ", logits.shape)
        # print("logits stats: ", logits.min().item(), logits.max().item(), logits.mean().item(), logits.std().item())
        return logits