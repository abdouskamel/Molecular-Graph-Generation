import torch
from model import VGAE

nb_molecules_to_generate = 10
min_nb_atoms = 5
max_nb_atoms = 20

nb_node_types = 14
nb_edge_types = 3

model = VGAE(nb_node_types, nb_edge_types)
model.load_state_dict(torch.load("trained_model.pkl"))
model.eval()

for i in range(nb_molecules_to_generate):
    pass