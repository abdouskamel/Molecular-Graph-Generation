import pandas as pd
import json
import pickle

from utils import smiles_to_graph

csv_path = "data/zinc_250k.csv"
valid_idx_path = "data/valid_idx_zinc_250k.json"

train_save_path = "data/zinc_250k_train.pkl"
valid_save_path = "data/zinc_250k_valid.pkl"
train_smiles_save_path = "data/zinc_250k_train_smiles.pkl"

# Split the given dataset into train and validation
def train_valid_split(dataset):
    with open(valid_idx_path, "r") as f:
        valid_idx = json.load(f)

    train_df = dataset.iloc[~dataset.index.isin(valid_idx), :]
    valid_df = dataset.iloc[valid_idx, :]

    return train_df, valid_df

# Transform the SMILES dataset into a graph dataset
def parse_smiles_to_graphs(smiles_df):
    graphs = []

    for _, molecule in smiles_df.iterrows():
        nodes, edges = smiles_to_graph(molecule["smiles"])
        if len(nodes) == 0 or len(edges) == 0:
            continue

        graphs.append({
            "nodes": nodes,
            "edges": edges,
            "logP": molecule["logP"],
            "qed": molecule["qed"],
            "SAS": molecule["SAS"],
            "smiles": molecule["smiles"]
        })

    return graphs

# Open the CSV file containing ZINC-250K
dataset = pd.read_csv(csv_path)
dataset["smiles"] = dataset["smiles"].apply(lambda x: x.strip("\n"))

# Split to train/valid, and transform SMILES into graphs
train_df, valid_df = train_valid_split(dataset)

train_graphs = parse_smiles_to_graphs(train_df)
valid_graphs = parse_smiles_to_graphs(valid_df)

print("Number of molecules in the training dataset :", len(train_graphs))
print("Number of molecules in the validation dataset :", len(valid_graphs))

# Save the graph datasets
print("Save training and validation datasets to disk...")
with open(train_save_path, "wb") as f:
    pickle.dump(train_graphs, f)

with open(valid_save_path, "wb") as f:
    pickle.dump(valid_graphs, f)

# Save to disk the SMILES used for training
train_smiles = []
for g in train_graphs:
    train_smiles.append(g["smiles"])

with open(train_smiles_save_path, "wb") as f:
    pickle.dump(train_smiles, f)