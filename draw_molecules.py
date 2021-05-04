import pickle
import numpy as np

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Draw

nb_molecules_to_draw = 20
train_smiles_path = "data/zinc_250k_train_smiles.pkl"
generated_smiles_path = "data/zinc_250k_generated_smiles.pkl"

train_mols_images_save_path = "data/draw_molecules/training/"
generated_mols_images_save_path = "data/draw_molecules/generated/"

# Compute the QED score of the given molecules
def compute_qed(mols_list):
    qed_scores = []

    for mol in mols_list:
        try:
            qed_scores.append(QED.qed(mol))
        except:
            continue

    return qed_scores

# Load the training molecules
with open(train_smiles_path, "rb") as f:
    train_smiles = pickle.load(f)

    train_mols = []
    for i in np.random.randint(len(train_smiles), size=nb_molecules_to_draw):
        train_mols.append(Chem.MolFromSmiles(train_smiles[i]))

# Load the generated molecules
with open(generated_smiles_path, "rb") as f:
    generated_smiles = pickle.load(f)

    generated_mols = []
    for smiles in generated_smiles:
        generated_mols.append(Chem.MolFromSmiles(smiles))

# Keep the generated molecules with the best QED score
generated_qed_scores = compute_qed(generated_mols)
generated_mols = list(zip(generated_mols, generated_qed_scores))
generated_mols.sort(reverse=True, key=lambda x: x[1])
generated_mols = [mol for mol, qed in generated_mols[:nb_molecules_to_draw]]

# Draw molecules and save to disk
Path(train_mols_images_save_path).mkdir(parents=True, exist_ok=True)
Path(generated_mols_images_save_path).mkdir(parents=True, exist_ok=True)

for i, mol in enumerate(train_mols):
    Draw.MolToFile(mol, train_mols_images_save_path + "{}.png".format(i + 1))

for i, mol in enumerate(generated_mols):
    Draw.MolToFile(mol, generated_mols_images_save_path + "{}.png".format(i + 1))