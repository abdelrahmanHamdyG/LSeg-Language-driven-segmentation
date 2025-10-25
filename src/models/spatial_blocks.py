import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv(nn.Module):
    """Per-channel 3×3 convolution (no cross-channel mixing)."""
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        out = x.reshape(B * C, 1, H, W)
        out = self.conv(out)
        return out.reshape(B, C, H, W)


class DepthwiseBlock(nn.Module):
    """Depthwise convolution + activation (no spatial bias)."""
    def __init__(self, activation='lrelu'):
        super().__init__()
        self.depthwise = DepthwiseConv()
        self.act = {
            'relu': nn.ReLU(inplace=True),
            'lrelu': nn.LeakyReLU(0.1, inplace=True),
            'tanh': nn.Tanh()
        }[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.depthwise(x))


class BottleneckBlock(nn.Module):
    """
    LSeg-style bottleneck refinement:
    depthwise conv + spatial summary (max over channels) + activation.
    """
    def __init__(self, activation='lrelu', pool='max'):
        super().__init__()
        self.depthwise = DepthwiseConv()
        self.act = {
            'relu': nn.ReLU(inplace=True),
            'lrelu': nn.LeakyReLU(0.1, inplace=True),
            'tanh': nn.Tanh()
        }[activation]
        self.pool_fn = torch.max if pool == 'max' else torch.mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # spatial summary over channels (1×HxW)
        if self.pool_fn == torch.max:
            summary = x.max(dim=1, keepdim=True)[0]
        else:
            summary = x.mean(dim=1, keepdim=True)

        refined = self.depthwise(x) + summary
        return self.act(refined)


class SpatialRegularizer(nn.Module):
    """
    Multi-block spatial refinement with residual scaling and optional upsample.
    Mathematically equivalent to LSeg's "Spatial Regularizer" refinement stage.
    """
    def __init__(self,
                 num_blocks: int = 1,
                 mode: str = 'bottleneck',   # 'bottleneck' or 'depthwise'
                 activation: str = 'lrelu',
                 pool: str = 'max',
                 upsample_scale: int = 2):
        super().__init__()
        Block = BottleneckBlock if mode == 'bottleneck' else DepthwiseBlock
        self.blocks = nn.Sequential(*[
            Block(activation=activation, pool=pool)
            if mode == 'bottleneck' else
            Block(activation=activation)
            for _ in range(num_blocks)
        ])
        self.scale = nn.Parameter(torch.tensor(0.0))  # learnable α ∈ (-1,1)
        self.upsample_scale = upsample_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.blocks(x)
        out = x + torch.tanh(self.scale) * refined

        if self.upsample_scale != 1:
            out = F.interpolate(out, scale_factor=self.upsample_scale,
                                mode='bilinear', align_corners=False)
        return out
