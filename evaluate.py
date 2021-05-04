import pickle
import numpy as np

from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import QED
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs

train_smiles_path = "data/zinc_250k_train_smiles.pkl"
generated_smiles_path = "data/zinc_250k_generated_smiles.pkl"

# Compute the portion of generated molecules not present in the training molecules
def compute_novelty(generated_smiles_set, train_smiles_set):
    return len(generated_smiles_set.difference(train_smiles_set)) / len(generated_smiles_set)

# Compute the average Tanimoto distance between the given molecules.
# This gives a measure of diversity in the given molecules list.
def compute_diversity(mols_list):
    fps = [FingerprintMols.FingerprintMol(mol) for mol in mols_list]
    similarity_list = []

    for i in range(0, len(fps) - 1):
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        similarity_list.extend(s)

    return np.mean(similarity_list)

# Compute the logp scores of the given molecules
def compute_logp(mols_list):
    logp_scores = []

    for mol in mols_list:
        try:
            logp_scores.append(Crippen.MolLogP(mol))
        except:
            continue

    return logp_scores

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
    train_smiles_list = pickle.load(f)
    train_smiles_set = set(train_smiles_list)

    train_mols = []
    for smiles in train_smiles_set:
        train_mols.append(Chem.MolFromSmiles(smiles))

# Load the generated molecules
with open(generated_smiles_path, "rb") as f:
    generated_smiles_list = pickle.load(f)
    generated_smiles_set = set(generated_smiles_list)

    generated_mols = []
    for smiles in generated_smiles_set:
        mol = Chem.MolFromSmiles(smiles)

        # We check if the molecule is valid
        if mol is not None:
            generated_mols.append(mol)

uniqueness = len(generated_smiles_set) / len(generated_smiles_list)
validity = len(generated_mols) / len(generated_smiles_set)
novelty = compute_novelty(generated_smiles_set, train_smiles_set)
diversity = compute_diversity(generated_mols)

generated_logp_scores = compute_logp(generated_mols)
generated_qed_scores = compute_qed(generated_mols)

train_logp_scores = compute_logp(train_mols)
train_qed_scores = compute_qed(train_mols)

print("Number of generated molecules :", len(generated_smiles_list))
print("Uniqueness :", round(uniqueness * 100, 2))
print("Validity :", round(validity * 100, 2))
print("Novelty :", round(novelty * 100, 2))
print("Diversity :", round(diversity, 2))
print("")

print("Average LogP score :", round(np.mean(generated_logp_scores), 2))
print("LogP score standard deviation :", round(np.std(generated_logp_scores), 2))
print("")

print("Average QED score :", round(np.mean(generated_qed_scores), 2))
print("QED score standard deviation :", round(np.std(generated_qed_scores), 2))
print("")