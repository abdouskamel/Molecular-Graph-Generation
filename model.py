import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import RelGraphConv
from utils import zinc_atoms_info

n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_z = 128

class VGAE(nn.Module):
    def __init__(self, nb_node_types, nb_edge_types):
        super(VGAE, self).__init__()

        # For the encoder
        self.conv1 = RelGraphConv(nb_node_types, n_hidden_1, nb_edge_types)
        self.conv2 = RelGraphConv(n_hidden_1, n_hidden_2, nb_edge_types)
        
        self.conv_mu = RelGraphConv(n_hidden_2, n_hidden_z, nb_edge_types)
        self.conv_logstd = RelGraphConv(n_hidden_2, n_hidden_z, nb_edge_types)

        # For the decoder
        self.node_types_fc = nn.Linear(n_hidden_z, nb_node_types)
        self.edge_existence_fc = nn.Linear(3 * n_hidden_z + 2 * nb_node_types, 1)
        self.edge_types_fc = nn.Linear(3 * n_hidden_z + 2 * nb_node_types, nb_edge_types)

    def forward(self, g, n_feats, e_types=None):
        z, gz, mu, logstd = self.encode(g, n_feats, e_types)
        return self.decode(z, gz, g.batch_num_nodes()), mu, logstd

    def reparameterize(self, mu, logstd):
        return mu + torch.randn_like(logstd) * torch.exp(logstd)

    def encode(self, g, n_feats, e_types=None):
        # Run R-GCN to get mu and logstd for each node.
        h = F.relu(self.conv1(g, n_feats, e_types))
        h = F.relu(self.conv2(g, h, e_types))

        mu = self.conv_mu(g, h, e_types)
        logstd = self.conv_logstd(g, h, e_types)

        # Get the node embeddings.
        z = self.reparameterize(mu, logstd)

        # Aggregate to get the graph embeddings.
        g.ndata["z"] = z
        gz = dgl.readout_nodes(g, "z", op="mean")

        return z, gz, mu, logstd

    def decode(self, batch_z_init, batch_gz_init, batch_nb_nodes):
        decoded_graphs = []

        # Compute node types
        batch_node_types = self.node_types_fc(batch_z_init)

        # The embeddings and node types are batched, so we iterate through batches and decode them.
        nb_nodes_cumsum = 0
        for i, nb_nodes in enumerate(batch_nb_nodes):
            z_init = batch_z_init[nb_nodes_cumsum:nb_nodes_cumsum + nb_nodes]
            gz_init = batch_gz_init[i]
            node_types = batch_node_types[nb_nodes_cumsum:nb_nodes_cumsum + nb_nodes]

            positive_edges, negative_edges = self.__decode(z_init, gz_init, node_types)
            decoded_graphs.append((node_types, positive_edges, negative_edges))
            nb_nodes_cumsum += nb_nodes

        return decoded_graphs

    def __decode(self, z, gz, node_types):
        # We return a list of positive (ie existent) and negative (ie non-existent) edges.
        positive_edges = []
        negative_edges = []

        nb_nodes = len(z)
        node_types_categorical = torch.argmax(node_types, dim=1)
        in_graph = [True] + [False] * (nb_nodes - 1)
        nodes_masked = [False] * nb_nodes
        nodes_degree = [0] * nb_nodes

        # We start with a graph containing only node 0.
        focus_queue = [0]
        while len(focus_queue) > 0:
            u = focus_queue.pop(0)

            # We don't add edges to this node if it has been masked.
            if nodes_masked[u]:
                continue

            u_z = z[u]
            u_type = node_types[u]
            u_type_categorical = node_types_categorical[u].item()
            u_max_valence = zinc_atoms_info["maximum_valence"][u_type_categorical]
            
            # Try to add an edge from u to all other nodes.
            for v in range(nb_nodes):
                # We don't add (u, v) if v is masked.
                if u == v or nodes_masked[v]:
                    continue

                v_z = z[v]
                v_type = node_types[v]
                v_type_categorical = node_types_categorical[v].item()
                v_max_valence = zinc_atoms_info["maximum_valence"][v_type_categorical]

                # Get the probability of existence of (u, v).
                x = torch.cat((u_z, u_type, v_z, v_type, gz))
                edge_exists = torch.sigmoid(self.edge_existence_fc(x))

                if edge_exists >= 0.5:
                    # Get the type of edge (u, v).
                    edge_type = self.edge_types_fc(x)
                    edge_type_categorical = torch.argmax(edge_type).item()

                    # Check if we don't exceed the maximum valence of both u and v.
                    if nodes_degree[u] + edge_type_categorical + 1 > u_max_valence:
                        continue

                    if nodes_degree[v] + edge_type_categorical + 1 > v_max_valence:
                        continue

                    positive_edges.append((u, v, edge_exists, edge_type))
                    nodes_degree[u] += edge_type_categorical + 1
                    nodes_degree[v] += edge_type_categorical + 1

                    # Mask u and v if the maximum valence has been reached.
                    if nodes_degree[u] == u_max_valence:
                        nodes_masked[u] = True

                    if nodes_degree[v] == v_max_valence:
                        nodes_masked[v] = True

                    # Add v to the queue if it's a new node.
                    if not in_graph[v]:
                        focus_queue.append(v)
                        in_graph[v] = True

                else:
                    negative_edges.append((u, v, edge_exists))

            # We have finished adding edges to this node, so we mask it.
            nodes_masked[u] = True

        return positive_edges, negative_edges