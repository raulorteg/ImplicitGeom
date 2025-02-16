import math

import pandas as pd
import torch


def get_thresholds(atom_symbols: list, bondlenghts_file: str):

    data = pd.read_csv(bondlenghts_file, sep=",", index_col=0).fillna(0.0)
    num_atoms = len(atom_symbols)
    thresholds = torch.zeros(len(atom_symbols), len(atom_symbols))
    for i in range(num_atoms):
        for j in range(i, num_atoms):
            thresholds[i, j] = data.loc[atom_symbols[i], atom_symbols[j]]
            thresholds[j, i] = thresholds[i, j]  # Note: bij=bji

    return thresholds


def sigmoid(d, b, k: float = 1.0):
    """Sigmoid function, which can be used to get the approximate the heaviside function while being differentiable (so the gradients can flow back)
    :param d: torch.tensor of pairwise distances
    :param b: torch.tensor of pairwise empirical bond-lenghts"""
    alpha = torch.tensor(2.0).to(b.device)
    x = (
        torch.ones_like(b).to(b.device) - alpha * torch.eye(b.shape[0]).to(b.device)
    ) * b - d
    return 1 / (1 + torch.exp(-k * x))


def heavyside(d, b):
    """Heavyside function, which can be used to get the correct loss. Note this is not used in the optimization because
    this function is not differentiable and we instead approximate it with a sigmoid. Also note I will take the convention H(0) = 1.
    :param d: torch.tensor of pairwise distances
    :param b: torch.tensor of pairwise empirical bond-lenghts"""
    alpha = torch.tensor(2.0)
    return torch.heaviside(
        (torch.ones_like(b).to(b.device) - alpha * torch.eye(b.shape[0]).to(b.device))
        * b
        - d,
        values=torch.ones_like(b).to(b.device),
    )


def write_xyz_file(atom_symbols: list, xyz_coordinates, filename):
    num_atoms = len(atom_symbols)

    with open(filename, "w") as f:
        f.write(f"{num_atoms}\n")
        f.write("Generated XYZ file\n")

        for symbol, (x, y, z) in zip(atom_symbols, xyz_coordinates):
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")


def scheduler(iter_num, max_iters, a, b, mode="linear", exp_base=2):
    if iter_num >= max_iters:
        return b

    if mode == "linear":
        if iter_num < 0.1 * max_iters:
            return a

        elif iter_num < 0.9 * max_iters:
            return a + (b - a) * ((iter_num - 0.1 * max_iters) / (0.8 * max_iters))
        else:
            return b

    else:
        raise ValueError("Invalid mode")
