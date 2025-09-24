import torch
import torch.nn as nn

class PrototypeEmbeddingLoss(nn.Module):
    def __init__(self, prototypes):
        super().__init__()
        self.prototypes = prototypes  # nn.Parameter

    def forward(self, feature_emb, x_idx, specific_idx=None):
        # feature_emb: (batch, num_features, d_model)
        # x_idx: (batch, num_features)  # feature indices
        # specific_idx: int or None

        # Gather prototype for each feature
        proto = self.prototypes[x_idx]  # (batch, num_features, d_model)
        if specific_idx is not None:
            mask = (x_idx == specific_idx)
            feature_emb = feature_emb[mask]
            proto = proto[mask]
        loss = ((feature_emb - proto) ** 2).sum(-1).mean()
        return loss