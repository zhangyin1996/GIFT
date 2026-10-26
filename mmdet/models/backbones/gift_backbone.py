import torch
from mmdet.registry import MODELS
from .dino_v2 import DinoVisionTransformer

#-----------------------------------------------------
from gift.inject_gift import inject_GIFT


@MODELS.register_module()
class GIFTDINOv2(DinoVisionTransformer):    
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        inject_GIFT(self,)  # init
        self._freeze_except_GIFT()
        

    def _freeze_except_GIFT(self):
        for name, p in self.named_parameters():
            if any(k in name for k in [
                "hra_u",
                "alpha",
                "gate",
                "diag_scale",
                "beta"
            ]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return outs

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)

        for name, p in self.named_parameters():
            if any(k in name for k in [
                "hra_u",
                "alpha",
                "gate",
                "diag_scale",
                "beta"
            ]):
                p.requires_grad = True
            else:
                p.requires_grad = False
    
