import argparse

import imageio
import numpy as np
import rdkit
import torch
import torch.nn as nn
from rdkit import Chem
from tqdm import tqdm

from implicitgeometry.utils import get_thresholds, smooth_step
from implicitgeometry.visualize import create_molecule_object, plot_3dmolecule

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # command line arguments
    parser.add_argument(
        "--smiles",
        type=str,
        default="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        help="SMILES string of a molecule. (default is Caffeine)",
    )
    parser.add_argument(
        "--addHs",
        action="store_true",
        help="Add hydrogens to the molecule? (default: False).",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=50000,
        help="Maximum number of iterations (default: 1000).",
    )
    parser.add_argument(
        "--plot_freq",
        type=int,
        default=500,
        help="Frequency of plot (plot every x iterations) (default: 500).",
    )

    args = parser.parse_args()

    bondlenghts_file = "configuration/bondlenghts.csv"

    # extract the list symbols and adjacency matrix
    mol = rdkit.Chem.MolFromSmiles(args.smiles)
    if args.addHs:
        mol = Chem.AddHs(mol)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    adjacency_matrix = torch.tensor(rdkit.Chem.rdmolops.GetAdjacencyMatrix(mol))

    # given the list of symbols and adj matrix get the standard bond distances
    thresholds = get_thresholds(
        atom_symbols=atom_symbols, bondlenghts_file=bondlenghts_file
    )

    # initial guess of positions
    param = torch.nn.Parameter(
        torch.normal(mean=0.0, std=0.1, size=(len(atom_symbols), 3))
    )
    optimizer = torch.optim.Adam([param], lr=0.001)

    buffers = []
    azim = 55
    for iter in tqdm(range(args.max_iters)):
        optimizer.zero_grad()

        # predictions of connectivity
        pairwise_distances = torch.cdist(param, param)
        x = thresholds - pairwise_distances
        x = smooth_step(x)  # to approximate differentiable heavyside function

        # compute element-wise squared differences with masked diagonal
        squared_diff = (x - adjacency_matrix.float()) ** 2
        masked_diff = squared_diff * torch.ones_like(squared_diff).fill_diagonal_(0.0)
        loss = masked_diff.sum()

        loss.backward()
        optimizer.step()

        # to save time and memory save frames only every some number of iterations
        if (iter % args.plot_freq == 0) or (iter == args.max_iters):

            molecule_info = create_molecule_object(
                atom_symbols=atom_symbols,
                positions=param.detach().numpy(),
                adjacency_matrix=x.detach().numpy(),
            )

            azim += 5
            buffers.append(
                plot_3dmolecule(
                    molecule_info=molecule_info,
                    azim=azim,
                    elev=20,
                    bond_resolution=15,
                    atom_resolution=20,
                    dpi=50,
                    step=iter,
                    loss=round(float(loss.detach().item()), 3),
                )
            )

    # create the visualization gif
    with imageio.get_writer("output.gif", mode="I") as writer:
        for buffer in buffers:
            # read the buffer as an image
            image = imageio.v2.imread(buffer)

            # append the image to the GIF
            writer.append_data(image)
