import torch
from torch import nn

from diffpep.modules.common.layers import DistanceToBins

class ContactEmbedder(nn.Module):
    def __init__(self, dist_min=0.0, dist_max=20.0, num_bins=64, embed_dim=64, use_onehot=True):
        super(ContactEmbedder, self).__init__()
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.num_bins = num_bins
        self.use_onehot = use_onehot
        self.distance_to_bins = DistanceToBins(dist_min, dist_max, num_bins, use_onehot)
        self.distance_embedding = nn.Embedding(num_bins, embed_dim)

    def forward(self, dist):
        dist_one_hot = self.distance_to_bins(dist, dim=-1, normalize=not self.use_onehot)  # (N, *, num_bins, *)
        return 