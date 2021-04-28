import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import SAGEConv

class VGAE(nn.Module):
    def __init__(self, nb_nodes_features):
        super(VGAE, self).__init__()

        self.sage1 = SAGEConv(nb_nodes_features, 32, aggregator_type="mean")
        self.sage_mu = SAGEConv(32, 32, aggregator_type="mean")
        self.sage_logstd = SAGEConv(32, 32, aggregator_type="mean")

    def forward(self, g, n_feats, e_types=None):
        z, mu, logstd = self.encode(g, n_feats, e_types)
        return self.decode(z), mu, logstd

    def reparameterize(self, mu, logstd):
        return mu + torch.randn_like(logstd) * torch.exp(logstd)

    def encode(self, g, n_feats, e_types=None):
        h = self.sage1(g, n_feats, e_types)
        h = F.relu(h)

        mu = self.sage_mu(g, h, e_types)
        logstd = self.sage_logstd(g, h, e_types)

        z = self.reparameterize(mu, logstd)
        return z, mu, logstd

    def decode(self, z):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj)