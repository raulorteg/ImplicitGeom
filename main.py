import imageio
import numpy as np
import rdkit
import torch
import torch.nn as nn
from rdkit import Chem

from implicitgeometry.utils import BondPerception, smooth_step
from implicitgeometry.visualize import create_molecule_object, plot_3dmolecule

if __name__ == "__main__":

    bondlenghts_file = "configuration/bondlenghts.csv"
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine
    # smiles = 'C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12C5=C%11C4=C3C3=C5C(=C81)C%10=C23' # fullerene
    # smiles = 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C' # penicilin
    # smiles = "CC(C)(O1)C[C@@H](O)[C@@]1(O2)[C@@H](C)[C@@H]3CC=C4[C@]3(C2)C(=O)C[C@H]5[C@H]4CC[C@@H](C6)[C@]5(C)Cc(n7)c6nc(C[C@@]89(C))c7C[C@@H]8CC[C@@H]%10[C@@H]9C[C@@H](O)[C@@]%11(C)C%10=C[C@H](O%12)[C@]%11(O)[C@H](C)[C@]%12(O%13)[C@H](O)C[C@@]%13(C)CO"

    mol = rdkit.Chem.MolFromSmiles(smiles)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    adjacency_matrix = torch.tensor(rdkit.Chem.rdmolops.GetAdjacencyMatrix(mol))

    perceptor = BondPerception(
        atom_symbols=atom_symbols, bondlenghts_file=bondlenghts_file
    )

    # initial guess of positions
    param = torch.nn.Parameter(
        torch.normal(mean=0.0, std=0.1, size=(len(atom_symbols), 3))
    )
    optimizer = torch.optim.Adam([param], lr=0.001)

    buffers = []
    azim = 55
    k = 1
    for iter in range(8000):
        optimizer.zero_grad()

        # predictions of connectivity
        pairwise_distances = torch.cdist(param, param)
        x = perceptor.thresholds - pairwise_distances
        x = smooth_step(x, k=1)  # to approximate differentiable heavyside function

        # compute element-wise squared differences with masked diagonal
        squared_diff = (x - adjacency_matrix.float()) ** 2
        masked_diff = squared_diff * torch.ones_like(squared_diff).fill_diagonal_(0.0)
        loss = masked_diff.sum()

        loss.backward()
        optimizer.step()

        # to save time and memory save frames only every some number of iterations
        if iter % 50 == 0:
            print(iter, loss.item(), k)
            azim += 5

            molecule_info = create_molecule_object(
                atom_symbols=atom_symbols,
                positions=param.detach().numpy(),
                adjacency_matrix=x.detach().numpy(),
            )

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
    with imageio.get_writer("test.gif", mode="I") as writer:
        for buffer in buffers:
            # read the buffer as an image
            image = imageio.imread(buffer)

            # append the image to the GIF
            writer.append_data(image)
