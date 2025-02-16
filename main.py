import argparse
import logging

import imageio
import numpy as np
import rdkit
import torch
import torch.nn as nn
from rdkit import Chem
from tqdm import tqdm

from implicitgeometry.utils import (
    get_thresholds,
    heavyside,
    scheduler,
    sigmoid,
    write_xyz_file,
)
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
        "--visualize",
        action="store_true",
        help="Visualize the process? Default False",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20000,
        help="Maximum number of iterations (default: 1000).",
    )
    parser.add_argument(
        "--plot_freq",
        type=int,
        default=500,
        help="Frequency of plot (plot every x iterations) (default: 500).",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="molecule",
        help="Filename prefix to save outputs to.",
    )

    args = parser.parse_args()

    bondlenghts_file = "configuration/bondlenghts.csv"
    k = 1

    logging.basicConfig(
        filename=f"{args.filename}.log",
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.visualize == True:

        buffer_losses = []
        buffer_real_losses = []
        buffer_iters = []
        buffers = []
        buffer_k = []
        azim = 55

    print(f"Using device: {device}")

    # extract the list symbols and adjacency matrix
    mol = rdkit.Chem.MolFromSmiles(args.smiles)
    if args.addHs:
        mol = Chem.AddHs(mol)
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    adjacency_matrix = torch.tensor(rdkit.Chem.rdmolops.GetAdjacencyMatrix(mol)).to(
        device
    )

    # given the list of symbols and adj matrix get the standard bond distances
    thresholds = get_thresholds(
        atom_symbols=atom_symbols, bondlenghts_file=bondlenghts_file
    ).to(device)

    # initial guess of positions
    param = torch.nn.Parameter(
        torch.normal(mean=0.0, std=0.1, size=(len(atom_symbols), 3), device=device)
    )
    optimizer = torch.optim.Adam([param], lr=0.001)

    solved_flag = False

    for iter in tqdm(range(args.max_iters)):
        k = scheduler(
            iter_num=iter, max_iters=args.max_iters, a=1, b=3, mode="linear", exp_base=2
        )

        if solved_flag == True:
            break

        param = param.to(device)
        optimizer.zero_grad()

        # predictions of connectivity
        pairwise_distances = torch.cdist(param, param)

        x = sigmoid(d=pairwise_distances, b=thresholds, k=k)  # the differentiable loss
        x_real = heavyside(
            d=pairwise_distances, b=thresholds
        )  # the real non-differentiable loss

        # compute element-wise squared differences
        squared_diff = (x - adjacency_matrix.float()) ** 2
        loss = squared_diff.sum()

        squared_diff = (x_real - adjacency_matrix.float()) ** 2
        loss_real = squared_diff.sum()

        loss.backward()
        optimizer.step()

        # to save time and memory save frames only every some number of iterations
        if (
            (iter % args.plot_freq == 0)
            or (iter == args.max_iters)
            or (solved_flag == True)
        ):

            wrong_bonds = squared_diff.sum().item()

            if wrong_bonds == 0.0:
                solved_flag = True

            printout_msg = f"Step {iter}, diff_loss={round(float(loss.detach().item()), 3)}, real_loss={round(float(loss_real.detach().item()), 3)}, wrong_bonds={wrong_bonds/2}, k={round(k,3)}"
            logger.info(printout_msg)

            if args.visualize == True:

                buffer_losses.append(loss.item())
                buffer_real_losses.append(loss_real.item())
                buffer_iters.append(iter)
                buffer_k.append(k)

                molecule_info = create_molecule_object(
                    atom_symbols=atom_symbols,
                    positions=param.detach().cpu().numpy(),
                    adjacency_matrix=x_real.detach().cpu().numpy(),
                )

                azim += 5
                buffers.append(
                    plot_3dmolecule(
                        molecule_info=molecule_info,
                        pred_pairwise_distances=pairwise_distances.detach().cpu(),
                        pred_adjacency_matrix=x_real.detach().cpu(),
                        iters=buffer_iters,
                        k=buffer_k,
                        diff_losses=buffer_losses,
                        real_losses=buffer_real_losses,
                        azim=azim,
                        elev=20,
                        bond_resolution=15,
                        atom_resolution=20,
                        dpi=50,
                        title=printout_msg,
                    )
                )

    write_xyz_file(
        atom_symbols=atom_symbols,
        xyz_coordinates=param.detach().cpu().numpy(),
        filename=f"{args.filename}.xyz",
    )

    if args.visualize == True:
        # create the visualization gif
        with imageio.get_writer(f"{args.filename}.gif", mode="I") as writer:
            for buffer in buffers:
                # read the buffer as an image
                image = imageio.v2.imread(buffer)

                # append the image to the GIF
                writer.append_data(image)
