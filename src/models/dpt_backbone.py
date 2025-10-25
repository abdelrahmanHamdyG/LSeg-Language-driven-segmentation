#DPT encoder — takes image, divides into patches, applies transformer blocks (ViT-like). Outputs multi-scale features.
import torch
import torch.nn as nn
import clip

class DPTBackbone(nn.Module):
    def __init__(self, visual):
        super(DPTBackbone, self).__init__()
        
        self.visual = visual
        self.hook_layers = [3, 6, 9, 12]
        self.layers_outputs = {}
        for i, blk in enumerate(self.visual.transformer.resblocks):
            if (i+1) in self.hook_layers:
                blk.register_forward_hook(self.save_output(i+1))
        self.visual.train()

    def save_output(self, idx):
        def hook(__, _, output):
            self.layers_outputs[idx] = output
        return hook

    def forward(self, x):
        _ = self.visual(x)



        backbone_output = {}
        for layer_id in self.hook_layers:
            out = self.layers_outputs[layer_id].permute(1, 0, 2)
            B, N, C = out.shape
            num_patches = N - 1
            h = w = int((num_patches)**0.5)
            out = out[:, 1:, :].permute(0, 2, 1)  # so it's B,C,N because convolution usually expect channel first format
            out = out.reshape(B, C, h, w)  # so it's B,C,H,W
            
            backbone_output[f"res{layer_id}"] = out
        
        return backbone_output