import torch
from model import VGAE

# The number of molecules to generate
nb_molecules_to_generate = 10

nb_nodes_features = 14
model = VGAE(nb_nodes_features)
model.load_state_dict(torch.load("trained_model.pkl"))
model.eval()

