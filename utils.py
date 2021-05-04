import numpy as np
import torch
import dgl

import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import rdmolops

# A mapping between the bond types and the number of bonds.
bond_dict = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}

# A mapping between edge types and bond types.
e_type_to_bond = {
    0 : Chem.rdchem.BondType.SINGLE, 
    1 : Chem.rdchem.BondType.DOUBLE, 
    2 : Chem.rdchem.BondType.TRIPLE,
    3 : Chem.rdchem.BondType.AROMATIC
}

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

# Transform a molecule represented with a graph into a SMILES.
def graph_to_smiles(node_types, edges_src_nodes, edges_dst_nodes, edges_types):
    mol = Chem.MolFromSmiles("")
    mol = Chem.rdchem.RWMol(mol)
    
    # Add atoms
    for n_type in node_types:
        atom = Chem.Atom(zinc_atoms_info["number_to_atom"][n_type])
        charge_num = int(zinc_atoms_info["atom_types"][n_type].split("(")[1].strip(")"))
        atom.SetFormalCharge(charge_num)
        a = mol.AddAtom(atom)

    # Add bonds
    for i in range(len(edges_src_nodes)):
        src = edges_src_nodes[i]
        dst = edges_dst_nodes[i]
        e_type = edges_types[i]

        a = mol.AddBond(src, dst, e_type_to_bond[e_type])

    smiles = Chem.MolToSmiles(mol)
    return smiles

# Generate a DGLGraph from our dataset graphs.
# We do this in order to train graph models with the Deep Graph Library (DGL).
def to_dgl_graph(graph):
    nb_nodes = len(graph["nodes"])

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
def recon_loss(pred_graphs, true_graphs):
    total_loss = 0
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    for pred_g, true_g in zip(pred_graphs, true_graphs):
        # Compute the loss for node types.
        pred_node_types, pred_pos_edges, pred_neg_edges = pred_g
        true_node_types = true_g.ndata["feats_categorical"]

        total_loss += ce_loss(pred_node_types, true_node_types)

        # Compute the loss for edge existence and edge types.
        pred_edge_existence = []
        true_edge_existence = []

        pred_edge_types = []
        true_edge_types = []

        # We keep track of the number of positive and negative edges that we have,
        # so as not to have too many negative edges.
        nb_negatives = 0
        nb_positives = 0

        # Process the predicted positive edges.
        for u, v, probability, pred_type in pred_pos_edges:
            if true_g.has_edges_between(u, v):
                pred_edge_existence.append(probability)
                true_edge_existence.append(1.0)

                # We have a positive edge, so get its type to compute the loss for edge types.
                pred_edge_types.append(pred_type.reshape(1, -1))
                e_id = true_g.edge_ids(u, v)
                true_edge_types.append(true_g.edata["type"][e_id])

                nb_positives += 1

            elif nb_negatives < nb_positives:
                pred_edge_existence.append(probability)
                true_edge_existence.append(0.0)
                nb_negatives += 1

        # Process the predicted negative edges.
        for u, v, probability in pred_neg_edges:
            if true_g.has_edges_between(u, v):
                pred_edge_existence.append(probability)
                true_edge_existence.append(1.0)
                nb_positives += 1

            elif nb_negatives < nb_positives:
                pred_edge_existence.append(probability)
                true_edge_existence.append(0.0)
                nb_negatives += 1

        if len(pred_edge_existence) > 0:
            true_edge_existence = torch.tensor(true_edge_existence)
            pred_edge_existence = torch.cat(pred_edge_existence)

            total_loss += bce_loss(pred_edge_existence, true_edge_existence)

        if len(pred_edge_types) > 0:
            true_edge_types = torch.tensor(true_edge_types)
            pred_edge_types = torch.cat(pred_edge_types, dim=0)

            total_loss += ce_loss(pred_edge_types, true_edge_types)

    return total_loss / len(pred_graphs)

# Postprocess the decoded graph by adding hydrogen atoms.
def postprocess_decoded_graph(decoded_node_types, decoded_edges):
    nb_nodes = len(decoded_node_types)
    node_valences = [0] * nb_nodes

    node_types = []
    edges_src_nodes = []
    edges_dst_nodes = []
    edges_types = []

    for type_distribution in decoded_node_types:
        node_types.append(torch.argmax(type_distribution).item())

    for src, dst, _, type_distribution in decoded_edges:
        edges_src_nodes.append(src)
        edges_dst_nodes.append(dst)

        e_type = torch.argmax(type_distribution).item()
        edges_types.append(e_type)

        node_valences[src] += e_type + 1
        node_valences[dst] += e_type + 1

    # Remove atoms with a valence of 0.
    to_remove = []

    for u, valence in enumerate(node_valences):
        if valence > 0:
            continue

        to_remove.append(u)
        for i in range(len(edges_src_nodes)):
            if edges_src_nodes[i] > u - len(to_remove):
                edges_src_nodes[i] -= 1

            if edges_dst_nodes[i] > u - len(to_remove):
                edges_dst_nodes[i] -= 1

    node_types = [x for i, x in enumerate(node_types) if i not in to_remove]
    node_valences = [x for x in node_valences if x > 0]
    nb_nodes = len(node_types)

    # Link hydrogen atoms to atoms that did not reach their maximum valence.
    for u, valence in enumerate(node_valences):
        max_valence = zinc_atoms_info["maximum_valence"][node_types[u]]
        for _ in range(0, max_valence - valence):
            v = nb_nodes
            nb_nodes += 1

            node_types.append(4)
            edges_src_nodes.append(u)
            edges_dst_nodes.append(v)
            edges_types.append(0)

    return node_types, edges_src_nodes, edges_dst_nodes, edges_types