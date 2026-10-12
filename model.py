import timm
import torch
import torch.nn as nn


class FusionNet(nn.Module):
    def __init__(self, meta_dim, backbone="efficientnet_b0", pretrained=True,
                 meta_hidden=64, fusion_hidden=256, dropout=0.3, meta_dropout_p=0.15):
        super().__init__()
        
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        img_dim = self.backbone.num_features
        self.meta_dim = meta_dim
        self.meta_dropout_p = meta_dropout_p  

        if meta_dim > 0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, meta_hidden), nn.BatchNorm1d(meta_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(meta_hidden, meta_hidden), nn.BatchNorm1d(meta_hidden), nn.ReLU(),
            )
            fused_dim = img_dim + meta_hidden
        else:
            fused_dim = img_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden), nn.BatchNorm1d(fusion_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),  )

    def fuse_and_classify(self, pooled, meta):
        if self.meta_dim > 0:
            if self.training and self.meta_dropout_p > 0:
                keep = (torch.rand(meta.size(0), 1, device=meta.device) > self.meta_dropout_p).float()
                meta = meta * keep
            pooled = torch.cat([pooled, self.meta_mlp(meta)], dim=1)
        return self.head(pooled).squeeze(1)

    def forward(self, img, meta, return_fmap=False):
        fmap = self.backbone.forward_features(img)          
        pooled = self.backbone.forward_head(fmap, pre_logits=True)
        logit = self.fuse_and_classify(pooled, meta)
        return (logit, fmap) if return_fmap else logit
