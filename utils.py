import numpy as np
import torch
import dgl

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
    dgl_graph.edata["type"] = torch.tensor(edges_types).reshape(-1, 1).float()

    return dgl_graph

# Compute the KL loss for the given mu and logstd.
def kl_loss(mu, logstd):
    return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

# Compute the reconstruction loss using positive edges and sampled negative edges.
def recon_loss(z, bg):
    pos_edges_src, pos_edges_dst = bg.edges()
    neg_edges_src, neg_edges_dst = sample_negative_edges(bg)

    pos_edges_probas = torch.sigmoid((z[pos_edges_src] * z[pos_edges_dst]).sum(dim=1))
    neg_edges_probas = torch.sigmoid((z[neg_edges_src] * z[neg_edges_dst]).sum(dim=1))

    return -torch.log(pos_edges_probas + 1e-15).mean() - torch.log(1 - neg_edges_probas + 1e-15).mean()

# Given a batch of graphs, sample a negative edge for every positive edge in each graph
def sample_negative_edges(bg):
    neg_edges_src, neg_edges_dst = [], []
    total_nb_nodes = 0

    for graph in dgl.unbatch(bg):
        nb_nodes = graph.number_of_nodes()
        pos_edges_src, pos_edges_dst = graph.edges()

        new_neg_edges_src = torch.randint(nb_nodes, (nb_nodes,)) + total_nb_nodes
        new_neg_edges_dst = torch.randint(nb_nodes, (nb_nodes,)) + total_nb_nodes

        neg_edges_src.extend(new_neg_edges_src.tolist())
        neg_edges_dst.extend(new_neg_edges_dst.tolist())

        total_nb_nodes += nb_nodes

    return neg_edges_src, neg_edges_dst