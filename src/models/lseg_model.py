#Combines everything: image encoder (DPT + decoder), text encoder, and head. Handles forward pass, loss, and output maps.
# lseg_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

import clip
from src.models.clip_text_encoder import ClipTextEncoder
from src.models.dpt_backbone import DPTBackbone
from src.models.decoder import Decoder
from src.models.lseg_head import LSegHead
from src.models.spatial_blocks import SpatialRegularizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSegModel(nn.Module):
    def __init__(self,model_name="ViT-B/32"):
        super(LSegModel, self).__init__()
        self.model, _ = clip.load(model_name,device=device)

        clip_image_encoder = self.model.visual
        
        self.text_encoder = ClipTextEncoder(self.model)
        self.dpt_backbone = DPTBackbone(clip_image_encoder)
        self.decoder = Decoder()
        self.lseg_head = LSegHead()
        # Initialize later after first forward pass
        self.spatial_regularizer = SpatialRegularizer(
                    num_blocks=2,             
                    mode='bottleneck',        
                    activation='relu',
                    pool='max',
                    upsample_scale=2          #
                ).to(device)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = True
        for param in self.lseg_head.parameters():
            param.requires_grad = True
        for param in self.dpt_backbone.parameters():
            param.requires_grad = True
        



    def forward(self, image, text_features=None, text_prompts=None):
        # 1 Encode image
        image_features = self.dpt_backbone(image)  # multi-scale dict
        # print("image_features shape: ", image_features.shape)

        # 2 Encode text labels
        if text_features is None and text_prompts is not None:
            text_features = self.text_encoder(text_prompts)  # [N, 512]

        # 3 Decode backbone outputs
        image_features = self.decoder(image_features)  # [B, 512, H/2, W/2]
        # print("image_features shape: ", image_features.shape)

        #  Pixel-text correlation  
        logits = self.lseg_head(text_features, image_features)  # [B, N, H/2, W/2]

        
        
            

        output = self.spatial_regularizer(logits)  # [B, N, H, W]
        
        
        return output