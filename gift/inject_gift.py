from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from time import time
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (1,) * x.ndim
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ResidualGIFTInjectedLinear(nn.Module):

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 6,
        drop_path: float = 0.1
    ):
        super().__init__()

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r

        # freeze pretrained
        self.weight = base_linear.weight
        self.weight.requires_grad = False

        if base_linear.bias is not None:
            self.bias = base_linear.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        # rotation vectors
        self.hra_u = nn.Parameter(torch.randn(r, self.out_features) * 1e-3)

        # rotation scale
        self.alpha = nn.Parameter(torch.zeros(1))

        # channel gate
        self.gate = nn.Parameter(torch.ones(self.out_features))

        # diagonal scaling (per output channel)
        self.diag_scale = nn.Parameter(torch.zeros(self.out_features))
        #---------------------------------------------
        #self.beta = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.ones(1))  # 2026-10-23
        # ---------------------------------------------

        self.drop_path = DropPath(drop_path)

    def forward(self, x):

        W = self.weight

        u = F.normalize(self.hra_u, dim=1)

        W_rot = W
        for i in range(self.r):
            ui = u[i].unsqueeze(1)
            W_rot = W_rot - 2 * ui @ (ui.T @ W_rot)

        delta = W_rot - W

        delta = self.gate.unsqueeze(1) * delta
        delta = self.drop_path(delta)

        W_new = W + self.alpha * delta

        scale_matrix = self.diag_scale.unsqueeze(1) * W
        W_new = W_new + self.beta * scale_matrix

        return F.linear(x, W_new, self.bias)

def inject_GIFT(
    model,
    r=6,
    last_n_layers=16,
    drop_path=0.1
):
    total_layers = len(model.blocks)  # 24
    start_layer = total_layers - last_n_layers

    for i in range(start_layer, total_layers):

        blk = model.blocks[i]

        blk.attn.qkv = ResidualGIFTInjectedLinear(
            blk.attn.qkv,
            r=r,
            drop_path=drop_path
        )

        blk.attn.proj = ResidualGIFTInjectedLinear(
            blk.attn.proj,
            r=r,
            drop_path=drop_path
        )

    return model
