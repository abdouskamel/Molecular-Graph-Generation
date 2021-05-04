import random
import torch
import dgl

from model import VGAE, n_hidden_z
from utils import postprocess_decoded_graph, graph_to_smiles

nb_molecules_to_generate = 1
min_nb_atoms = 5
max_nb_atoms = 20

nb_node_types = 14
nb_edge_types = 3

model = VGAE(nb_node_types, nb_edge_types)
model.load_state_dict(torch.load("trained_model.pkl"))
model.eval()

with torch.no_grad():
    for i in range(nb_molecules_to_generate):
        nb_atoms = random.randint(min_nb_atoms, max_nb_atoms)
        z = torch.randn(nb_atoms, n_hidden_z)
        gz = torch.mean(z, dim=0).reshape(1, -1)

        decoded_node_types, decoded_edges, _ = model.decode(z, gz, [nb_atoms])[0]
        node_types, edges_src_nodes, edges_dst_nodes, edges_types = postprocess_decoded_graph(decoded_node_types, decoded_edges)
        smiles = graph_to_smiles(node_types, edges_src_nodes, edges_dst_nodes, edges_types)