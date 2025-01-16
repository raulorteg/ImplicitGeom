import pandas as pd
import torch


def get_thresholds(atom_symbols: list, bondlenghts_file: str):

    data = pd.read_csv(bondlenghts_file, sep=",", index_col=0).fillna(0.0)
    num_atoms = len(atom_symbols)
    thresholds = torch.zeros(len(atom_symbols), len(atom_symbols))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            thresholds[i, j] = data.loc[atom_symbols[i], atom_symbols[j]]
            thresholds[j, i] = thresholds[i, j]  # Note: bij=bji

    return thresholds


def smooth_step(x, k: float = 1.0):
    # sigmoid function
    return 1 / (1 + torch.exp(-k * x))
