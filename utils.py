import numpy as np
import torch
import dgl

import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import rdmolops

# A mapping between the bond types and the number of bonds.
bond_dict = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}

# Information about atoms in the ZINC dataset
# Taken from : https://github.com/microsoft/constrained-graph-variational-autoencoder/blob/master/utils.py#L37
zinc_atoms_info = { 
    "atom_types": ["Br1(0)", "C4(0)", "Cl1(0)", "F1(0)", "H1(0)", "I1(0)", "N2(-1)", "N3(0)", "N4(1)", "O1(-1)", "O2(0)", "S2(0)","S4(0)", "S6(0)"],
    "maximum_valence": {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5:1, 6:2, 7:3, 8:4, 9:1, 10:2, 11:2, 12:4, 13:6, 14:3},
    "number_to_atom": {0: "Br", 1: "C", 2: "Cl", 3: "F", 4: "H", 5:"I", 6:"N", 7:"N", 8:"N", 9:"O", 10:"O", 11:"S", 12:"S", 13:"S"},
    "bucket_sizes": np.array([28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58, 84])
}

# The number of different atom types in the ZINC dataset.
# An atom type is defined by the symbol of the atom, its valence, and its charge.
nb_atom_types = len(zinc_atoms_info["atom_types"])

# Return true if the given molecule must be kekulized.
# Taken from : https://github.com/microsoft/constrained-graph-variational-autoencoder/blob/master/utils.py#L297
def need_kekulize(mol):
    for bond in mol.GetBonds():
        bond_type = str(bond.GetBondType())
        if bond_dict[bond_type] >= 3:
            return True

    return False

# Transform a molecule represented with SMILES into a graph.
# Taken from : https://github.com/microsoft/constrained-graph-variational-autoencoder/blob/master/utils.py#L341
def smiles_to_graph(smiles):
    edges = []
    nodes = []

    mol = Chem.MolFromSmiles(smiles)
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
    
    # Remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    # Get nodes and their features
    for atom in mol.GetAtoms():
        # We use symbol, valence and charge as the node features
        symbol = atom.GetSymbol()
        valence = atom.GetTotalValence()
        charge = atom.GetFormalCharge()

        # Do a one-hot encoding of (symbol, valence, charge)
        atom_type_str = "%s%i(%i)" % (symbol, valence, charge)
        if atom_type_str not in zinc_atoms_info["atom_types"]:
            return [], []
            
        one_hot_encoded_atom = [0] * nb_atom_types
        atom_type_index = zinc_atoms_info["atom_types"].index(atom_type_str)
        one_hot_encoded_atom[atom_type_index] = 1

        nodes.append(one_hot_encoded_atom)

    # Get edges (source, bond type, destination)
    for bond in mol.GetBonds():
        bond_type = str(bond.GetBondType())
        edges.append((bond.GetBeginAtomIdx(), bond_dict[bond_type], bond.GetEndAtomIdx()))

    return nodes, edges

# Generate a DGLGraph from our dataset graphs.
# We do this in order to train graph models with the Deep Graph Library (DGL).
def to_dgl_graph(graph):
    nb_nodes =  len(graph["nodes"])

    edges_src_nodes = []
    edges_dst_nodes = []
    edges_types = []

    for edge in graph["edges"]:
        edges_src_nodes.append(edge[0])
        edges_dst_nodes.append(edge[2])
        edges_types.append(edge[1])

    dgl_graph = dgl.graph((edges_src_nodes, edges_dst_nodes), num_nodes=nb_nodes)

    dgl_graph.ndata["feats"] = torch.tensor(graph["nodes"]).float()
    dgl_graph.ndata["feats_categorical"] = torch.argmax(dgl_graph.ndata["feats"], dim=1)
    dgl_graph.edata["type"] = torch.tensor(edges_types)

    return dgl_graph

# Compute the KL loss for the given mu and logstd.
def kl_loss(mu, logstd):
    return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

# Compute the reconstruction loss between the predicted graphs and the true graphs.
def recon_loss(pred_graphs, true_graphs, device):
    total_loss = 0
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    for pred_g, true_g in zip(pred_graphs, true_graphs):
        pred_node_types, pred_pos_edges, pred_neg_edges = pred_g
        true_node_types = true_g.ndata["feats_categorical"]

        # Cross-entropy loss for node types
        total_loss += ce_loss(pred_node_types, true_node_types)

        # Compute the loss for edge existence
        pred_edge_existence = []
        true_edge_existence = []

        for u, v, probability, _ in pred_pos_edges:
            pred_edge_existence.append(probability)
            if true_g.has_edges_between(u, v):
                true_edge_existence.append(1.0)
            else:
                true_edge_existence.append(0.0)

        for u, v, probability in pred_neg_edges:
            pred_edge_existence.append(probability)
            if true_g.has_edges_between(u, v):
                true_edge_existence.append(1.0)
            else:
                true_edge_existence.append(0.0)

        if len(pred_edge_existence) > 0:
            true_edge_existence = torch.tensor(true_edge_existence).to(device)
            pred_edge_existence = torch.cat(pred_edge_existence)

            total_loss += bce_loss(pred_edge_existence, true_edge_existence)

        # Compute the loss for edge types
        pred_edge_types = []
        true_edge_types = []

        for u, v, _, pred_type in pred_pos_edges:
            if true_g.has_edges_between(u, v):
                pred_edge_types.append(pred_type.reshape(1, -1))

                e_id = true_g.edge_ids(u, v)
                true_edge_types.append(true_g.edata["type"][e_id])

        if len(pred_edge_types) > 0:
            true_edge_types = torch.tensor(true_edge_types).to(device)
            pred_edge_types = torch.cat(pred_edge_types, dim=0)

            total_loss += ce_loss(pred_edge_types, true_edge_types)

    return total_loss / len(pred_graphs)

